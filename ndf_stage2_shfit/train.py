"""Fit a scalar SH 3DGS field to downsampled Mesh--GT error maps."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
import torch

from gaussolve_package.gaussolve_render import gaussolve_render
from mesh_renderer import MeshRenderer
from mesh_stage1_environment import Stage1Appearance
from pretrain_mesh import load_cameras, render as render_stage1
from ndf_stage2_shfit.field import ScalarSHGaussianField
from ndf_stage2_shfit.native_field import NativeScalarSHGaussianField
from ndf_stage2_shfit.targets import (
    binary_downsample_gaussian_target,
    downsample_mesh_error,
    masked_gaussian_blur,
    smooth_threshold,
)
from utils.loss_utils import ssim


@dataclass
class ErrorSample:
    name: str
    camera: object
    raw_error: torch.Tensor
    valid: torch.Tensor
    mesh_depth: torch.Tensor
    ground_truth_u8: torch.Tensor
    mesh_rgb_u8: torch.Tensor
    fixed_target: torch.Tensor | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_initial_points(path: Path) -> torch.Tensor:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"initial point cloud has no vertex element: {path}")
    vertex = ply["vertex"].data
    for name in ("x", "y", "z"):
        if name not in vertex.dtype.names:
            raise ValueError(f"initial point cloud has no {name} property: {path}")
    points = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype(
        np.float32, copy=False
    )
    if not np.isfinite(points).all():
        raise ValueError("initial point cloud contains non-finite XYZ")
    return torch.from_numpy(points.copy())


def low_resolution_camera(camera, width: int, height: int):
    return SimpleNamespace(
        image_name=camera.image_name,
        image_width=int(width),
        image_height=int(height),
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        world_view_transform=camera.world_view_transform,
        full_proj_transform=camera.full_proj_transform,
        camera_center=camera.camera_center,
    )


def load_stage1(checkpoint: Path):
    mesh_path = checkpoint / "mesh_pretrained.glb"
    appearance_path = checkpoint / "environment.pt"
    if not mesh_path.is_file() or not appearance_path.is_file():
        raise FileNotFoundError("Stage-1 checkpoint needs mesh_pretrained.glb and environment.pt")
    renderer = MeshRenderer(
        str(mesh_path),
        texture_path=None,
        device="cuda",
        use_sigmoid_rgb=True,
        use_sigmoid_alpha=True,
        mesh_alpha_mode=False,
        alpha_from_scratch=False,
        force_vertex_alpha=False,
    )
    appearance = Stage1Appearance.load(str(appearance_path), device="cuda").eval()
    if appearance.variant not in {"uv_sh", "texture"}:
        raise RuntimeError(
            "Stage 2 expects UV-SH or ordinary-texture Stage 1, "
            f"got {appearance.variant}"
        )
    for parameter in appearance.parameters():
        parameter.requires_grad_(False)
    if isinstance(renderer.texture, torch.Tensor):
        renderer.texture.requires_grad_(False)
    return renderer, appearance, mesh_path, appearance_path


@torch.no_grad()
def precompute_error_samples(
    renderer: MeshRenderer,
    appearance: Stage1Appearance,
    cameras: list,
    background: torch.Tensor,
    downsample: int,
    target_mode: str,
    binary_threshold: float,
    binary_blur_kernel: int,
    binary_blur_sigma: float,
) -> list[ErrorSample]:
    samples: list[ErrorSample] = []
    for camera in cameras:
        pos, verts_view, rast, mask = renderer.rasterize(camera)
        base = renderer.color_from_rast(
            camera,
            pos,
            rast,
            mask,
            bg_color=background,
            enable_antialiasing=True,
        )
        mesh_rgb = appearance(renderer, camera, rast, mask, base)
        ground_truth = camera.original_image.to("cuda", non_blocking=True)
        mesh_depth, _ = renderer.depth_from_rast(
            camera, verts_view, rast, mask, normalize=False
        )
        low = downsample_mesh_error(
            mesh_rgb,
            ground_truth,
            mask,
            mesh_depth,
            factor=downsample,
        )
        fixed_target = None
        if target_mode == "binary_blur":
            binary_target = binary_downsample_gaussian_target(
                mesh_rgb,
                ground_truth,
                mask,
                factor=downsample,
                threshold=binary_threshold,
                kernel_size=binary_blur_kernel,
                sigma=binary_blur_sigma,
            )
            if not torch.equal(binary_target["valid"], low["valid"]):
                raise RuntimeError("binary and continuous target masks disagree")
            fixed_target = binary_target["target"].half().cpu()
        height, width = low["raw_error"].shape[-2:]
        samples.append(
            ErrorSample(
                name=camera.image_name,
                camera=low_resolution_camera(camera, width, height),
                raw_error=low["raw_error"].half().cpu(),
                valid=low["valid"].cpu(),
                mesh_depth=low["mesh_depth"].float().cpu(),
                ground_truth_u8=(low["ground_truth"].clamp(0, 1).mul(255).byte().cpu()),
                mesh_rgb_u8=(low["mesh_rgb"].clamp(0, 1).mul(255).byte().cpu()),
                fixed_target=fixed_target,
            )
        )
    return samples


def sample_to_cuda(sample: ErrorSample, low: float, high: float):
    raw = sample.raw_error.cuda(non_blocking=True).float()
    valid = sample.valid.cuda(non_blocking=True)
    depth = sample.mesh_depth.cuda(non_blocking=True)
    if sample.fixed_target is None:
        target = smooth_threshold(raw, low, high) * valid
    else:
        target = sample.fixed_target.cuda(non_blocking=True).float() * valid
    return raw, target, valid, depth


def set_band_targets(
    samples: list[ErrorSample],
    low: float,
    high: float,
    target_mode: str,
    blur_kernel: int,
    blur_sigma: float,
) -> None:
    """Materialize the selected low-resolution target once on CPU."""

    if target_mode == "binary_blur":
        return
    for sample in samples:
        if target_mode == "smooth_band":
            sample.fixed_target = None
        elif target_mode == "smooth_band_blur":
            target = smooth_threshold(sample.raw_error.float(), low, high)
            sample.fixed_target = masked_gaussian_blur(
                target, sample.valid, blur_kernel, blur_sigma
            ).half()
        else:
            raise ValueError(f"unsupported target mode: {target_mode}")


PIPE = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)


def render_field_package(
    field: ScalarSHGaussianField,
    sample: ErrorSample,
    mesh_depth: torch.Tensor,
    mesh_depth_tolerance: float,
    use_mesh_depth: bool = True,
) -> dict:
    result = gaussolve_render(
        sample.camera,
        field,
        PIPE,
        torch.zeros(1, device="cuda"),
        gs_depth_is_inverse=False,
        mesh_depth=mesh_depth if use_mesh_depth else None,
        mesh_depth_tolerance=float(mesh_depth_tolerance),
        clamp_output=False,
    )
    return result


def render_field(
    field: ScalarSHGaussianField,
    sample: ErrorSample,
    mesh_depth: torch.Tensor,
    mesh_depth_tolerance: float,
    use_mesh_depth: bool = True,
) -> torch.Tensor:
    return render_field_package(
        field, sample, mesh_depth, mesh_depth_tolerance, use_mesh_depth
    )["render"]


def fit_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    field: ScalarSHGaussianField,
    positive_weight: float,
    opacity_sparse: float,
):
    prediction = prediction * valid
    weight = 1.0 + float(positive_weight) * target
    l1 = ((prediction - target).abs() * weight)[valid].mean()
    # L1 must see raw overshoot; otherwise a clamp at 1 gives false white blobs
    # zero gradient.  SSIM remains on the display range.
    ssim_value = ssim(prediction.clamp(0.0, 1.0), target)
    opacity_l1 = field.get_opacity.mean()
    loss = 0.8 * l1 + 0.2 * (1.0 - ssim_value) + float(opacity_sparse) * opacity_l1
    return loss, l1, ssim_value, opacity_l1


def gradient_contract(field: ScalarSHGaussianField) -> dict[str, dict[str, float | bool]]:
    result = {}
    for name, parameter in (
        ("xyz", field._xyz),
        ("f_dc", field._features_dc),
        ("f_rest", field._features_rest),
        ("opacity", field._opacity),
        ("scaling", field._scaling),
        ("rotation", field._rotation),
    ):
        gradient = parameter.grad
        result[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "nonzero": bool(gradient is not None and (gradient != 0).any()),
            "l2": float(gradient.norm().item()) if gradient is not None else 0.0,
        }
    return result


def train_field(
    initial_points: torch.Tensor,
    samples: list[ErrorSample],
    low: float,
    high: float,
    args,
    scene_extent: float,
    output_dir: Path,
):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    densification_config = {
        "enabled": bool(args.densify),
        "from_iter": int(args.densify_from_iter),
        "until_iter": int(args.densify_until_iter),
        "interval": int(args.densification_interval),
        "grad_threshold": float(args.densify_grad_threshold),
        "min_opacity": float(args.densify_min_opacity),
        "percent_dense": float(args.percent_dense),
        "max_points": int(args.densify_max_points),
        "prune_from_iter": int(args.prune_from_iter),
        "prune_interval": int(args.prune_interval),
        "size_prune_from_iter": int(args.size_prune_from_iter),
        "max_screen_size": float(args.max_screen_size),
        "max_world_scale_fraction": float(args.max_world_scale_fraction),
        "opacity_reset_interval": int(args.opacity_reset_interval),
        "opacity_reset_value": float(args.opacity_reset_value),
        "position_lr_init": float(args.position_lr_init),
        "position_lr_final": float(args.position_lr_final),
        "position_lr_delay_mult": float(args.position_lr_delay_mult),
    }
    native_backend = args.field_backend in {"native", "gaussolve"}
    field_class = (
        NativeScalarSHGaussianField
        if native_backend
        else ScalarSHGaussianField
    )
    field_metadata = {
        "target_low": low,
        "target_high": high,
        "downsample": args.downsample,
        "densification": densification_config,
        "source": args.initialization_label,
        "training_goal": "learn_spatial_distribution_not_exact_error_values",
        "mesh_depth_gate": bool(args.mesh_depth_gate),
    }
    if args.field_backend == "gaussolve":
        field = field_class.from_gaussolve_vertices(
            initial_points,
            sh_degree=args.sh_degree,
            initial_opacity=args.initial_opacity,
            initial_value=args.initial_value,
            metadata=field_metadata,
        )
    elif (
        args.field_backend == "native"
        and args.preserve_initial_geometry
        and args.initial_point_cloud
    ):
        field = field_class.from_ply_geometry(
            args.initial_point_cloud,
            sh_degree=args.sh_degree,
            initial_opacity=args.initial_opacity,
            initial_value=args.initial_value,
            metadata=field_metadata,
        )
    else:
        field = field_class.from_points(
            initial_points,
            sh_degree=args.sh_degree,
            initial_opacity=args.initial_opacity,
            initial_value=args.initial_value,
            metadata=field_metadata,
        )
    field.active_sh_degree = int(args.start_sh_degree)
    total_lr_iterations = (
        args.position_lr_max_steps
        if args.position_lr_max_steps > 0
        else args.iterations
    )
    if native_backend:
        optimizer = field.make_optimizer(
            scene_extent,
            total_iterations=total_lr_iterations,
            position_lr_init=args.position_lr_init,
            position_lr_final=args.position_lr_final,
            position_lr_delay_mult=args.position_lr_delay_mult,
            feature_rest_lr_scale=args.feature_rest_lr_scale,
            percent_dense=args.percent_dense,
        )
    else:
        optimizer = field.optimizer(
            scene_extent,
            total_iterations=total_lr_iterations,
            position_lr_init=args.position_lr_init,
            position_lr_final=args.position_lr_final,
            position_lr_delay_mult=args.position_lr_delay_mult,
            feature_rest_lr_scale=args.feature_rest_lr_scale,
        )
    # Statistics remain active for the whole run: densification stops halfway,
    # while opacity/screen/world-size pruning continues until the final step.
    field.reset_densification_stats()
    history = []
    densification_history = []
    first_gradient = None
    camera_order = list(range(len(samples)))
    order_rng = random.Random(args.seed)
    order_rng.shuffle(camera_order)
    start = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        if iteration % 1000 == 0:
            field.oneupSHdegree()
        if native_backend:
            xyz_lr = field.update_learning_rate(iteration)
        else:
            xyz_lr = field.update_learning_rate(optimizer, iteration)
        order_position = (iteration - 1) % len(samples)
        if iteration > 1 and order_position == 0:
            order_rng.shuffle(camera_order)
        sample = samples[camera_order[order_position]]
        _raw, target, valid, depth = sample_to_cuda(sample, low, high)
        optimizer.zero_grad(set_to_none=True)
        render_package = render_field_package(
            field,
            sample,
            depth,
            args.mesh_depth_tolerance,
            args.mesh_depth_gate,
        )
        prediction = render_package["render"]
        loss, l1, ssim_value, opacity_l1 = fit_loss(
            prediction,
            target,
            valid,
            field,
            args.positive_weight,
            args.opacity_sparse,
        )
        loss.backward()
        if iteration == 1:
            first_gradient = gradient_contract(field)
            if not all(row["finite"] for row in first_gradient.values()):
                raise RuntimeError("non-finite Stage-2 gradient")
            if not first_gradient["f_dc"]["nonzero"]:
                raise RuntimeError("DC coefficient did not receive a gradient")
        if iteration == 1000:
            progressive_gradient = gradient_contract(field)
            if not progressive_gradient["f_rest"]["nonzero"]:
                raise RuntimeError(
                    "higher-order SH did not receive a gradient after degree activation"
                )
        field.add_densification_stats(
            render_package["viewspace_points"],
            render_package["visibility_filter"],
            render_package["radii"],
        )
        optimizer.step()
        if args.stabilize:
            field.stabilize(
                min_scale_factor=args.stabilize_scale_min_factor,
                max_scale_factor=args.stabilize_scale_max_factor,
            )
        should_densify = (
            args.densify
            and iteration >= args.densify_from_iter
            and iteration <= args.densify_until_iter
            and iteration % args.densification_interval == 0
        )
        should_prune = (
            iteration >= args.prune_from_iter
            and iteration % args.prune_interval == 0
        )
        size_threshold = (
            args.max_screen_size if iteration > args.size_prune_from_iter else None
        )
        if should_densify:
            densification_row = {
                "iteration": iteration,
                "mode": "densify_and_prune",
                "opacity_capped": 0,
                **field.densify_and_prune(
                    optimizer,
                    grad_threshold=args.densify_grad_threshold,
                    min_opacity=args.densify_min_opacity,
                    scene_extent=scene_extent,
                    percent_dense=args.percent_dense,
                    max_points=args.densify_max_points,
                    max_screen_size=size_threshold,
                    max_world_scale_fraction=args.max_world_scale_fraction,
                ),
            }
            densification_history.append(densification_row)
            print(json.dumps({"structure_update": densification_row}), flush=True)
        elif should_prune:
            prune_row = {
                "iteration": iteration,
                "mode": "prune_only",
                "opacity_capped": 0,
                **field.prune_only(
                    optimizer,
                    min_opacity=args.densify_min_opacity,
                    scene_extent=scene_extent,
                    max_screen_size=size_threshold,
                    max_world_scale_fraction=args.max_world_scale_fraction,
                ),
            }
            densification_history.append(prune_row)
            print(json.dumps({"structure_update": prune_row}), flush=True)
        # Match the original 3DGS schedule: reset opacity only while
        # densification is active. Pruning itself continues afterwards.
        if (
            args.opacity_reset_interval > 0
            and iteration <= args.densify_until_iter
            and iteration % args.opacity_reset_interval == 0
        ):
            count = int(field.get_xyz.shape[0])
            reset_row = {
                "iteration": iteration,
                "mode": "opacity_reset",
                "opacity_capped": field.reset_opacity(
                    optimizer, max_opacity=args.opacity_reset_value
                ),
                "before": count,
                "eligible": 0,
                "clone_parents": 0,
                "split_parents": 0,
                "created": 0,
                "pruned": 0,
                "after": count,
                "pruned_low_opacity": 0,
                "pruned_large_screen": 0,
                "pruned_large_world": 0,
            }
            densification_history.append(reset_row)
            print(json.dumps({"structure_update": reset_row}), flush=True)
        if iteration == 1 or iteration % 100 == 0 or iteration == args.iterations:
            row = {
                "iteration": iteration,
                "xyz_lr": xyz_lr,
                "active_sh_degree": int(field.active_sh_degree),
                "loss": float(loss.detach().item()),
                "weighted_l1": float(l1.detach().item()),
                "ssim": float(ssim_value.detach().item()),
                "mean_opacity": float(opacity_l1.detach().item()),
                "active_opacity_ge_0p01": int((field.get_opacity >= 0.01).sum().item()),
            }
            history.append(row)
            print(json.dumps({"band": [low, high], **row}), flush=True)
    elapsed = time.perf_counter() - start
    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    if densification_history:
        with (output_dir / "densification_log.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(densification_history[0]))
            writer.writeheader()
            writer.writerows(densification_history)
    return field, history, densification_history, first_gradient, elapsed


@torch.no_grad()
def evaluate_field(
    field: ScalarSHGaussianField,
    samples: list[ErrorSample],
    low: float,
    high: float,
    mesh_depth_tolerance: float,
    use_mesh_depth: bool = True,
):
    rows = []
    for sample in samples:
        _raw, target, valid, depth = sample_to_cuda(sample, low, high)
        prediction = render_field(
            field, sample, depth, mesh_depth_tolerance, use_mesh_depth
        ) * valid
        difference = prediction - target
        l1 = difference.abs()[valid].mean()
        mse = difference.square()[valid].mean()
        predicted_binary = (prediction >= 0.5) & valid
        target_binary = (target >= 0.5) & valid
        intersection = (predicted_binary & target_binary).sum()
        union = (predicted_binary | target_binary).sum()
        true_positive = intersection
        precision = true_positive / predicted_binary.sum().clamp_min(1)
        recall = true_positive / target_binary.sum().clamp_min(1)
        rows.append(
            {
                "image_name": sample.name,
                "l1": float(l1.item()),
                "mse": float(mse.item()),
                "psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()),
                "iou_at_0p5": float((intersection / union.clamp_min(1)).item()),
                "precision_at_0p5": float(precision.item()),
                "recall_at_0p5": float(recall.item()),
                "target_positive_fraction": float(target_binary.float()[valid].mean().item()),
                "prediction_positive_fraction": float(predicted_binary.float()[valid].mean().item()),
            }
        )
    summary = {"count": len(rows)}
    for key in rows[0]:
        if key != "image_name":
            summary[key] = float(np.mean([row[key] for row in rows]))
    return summary, rows


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    array = tensor.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


@torch.no_grad()
def save_panels(
    field: ScalarSHGaussianField,
    samples: list[ErrorSample],
    low: float,
    high: float,
    mesh_depth_tolerance: float,
    output_dir: Path,
    use_mesh_depth: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    rows = []
    target_prediction_rows = []
    for sample in samples[:5]:
        raw, target, valid, depth = sample_to_cuda(sample, low, high)
        prediction = render_field(
            field, sample, depth, mesh_depth_tolerance, use_mesh_depth
        ) * valid
        images = [
            tensor_to_pil(sample.ground_truth_u8),
            tensor_to_pil(sample.mesh_rgb_u8),
            tensor_to_pil((raw.clamp(0, 0.15) / 0.15 * 255).byte()),
            tensor_to_pil((target * 255).byte()),
            tensor_to_pil((prediction.clamp(0.0, 1.0) * 255).byte()),
        ]
        width, height = images[0].size
        label_height = 28
        canvas = Image.new("RGB", (5 * width, height + label_height), "white")
        draw = ImageDraw.Draw(canvas)
        labels = ["GT", "UV-SH Mesh", "raw error / 0.15", "soft target", "SH-GS NDF"]
        for column, (image, label) in enumerate(zip(images, labels)):
            canvas.paste(image, (column * width, label_height))
            draw.text((column * width + 5, 5), label, fill="black", font=font)
        row_path = output_dir / f"{sample.name}.png"
        canvas.save(row_path, optimize=True)
        rows.append(canvas)

        pair = Image.new("RGB", (2 * width, height + label_height), "white")
        pair_draw = ImageDraw.Draw(pair)
        for column, (image, label) in enumerate(
            zip(images[3:], ("soft target", "trained SH-GS NDF"))
        ):
            pair.paste(image, (column * width, label_height))
            pair_draw.text((column * width + 5, 5), label, fill="black", font=font)
        target_prediction_rows.append(pair)
    contact = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        contact.paste(row, (0, y))
        y += row.height
    contact_path = output_dir / "contact_sheet.png"
    contact.save(contact_path, optimize=True)
    target_prediction_contact = Image.new(
        "RGB",
        (
            target_prediction_rows[0].width,
            sum(row.height for row in target_prediction_rows),
        ),
        "white",
    )
    y = 0
    for row in target_prediction_rows:
        target_prediction_contact.paste(row, (0, y))
        y += row.height
    target_prediction_contact.save(
        output_dir / "target_vs_trained_ndf.png", optimize=True
    )
    return contact_path


def parse_bands(value: str):
    bands = []
    for index, item in enumerate(value.split(","), start=1):
        low_text, high_text = item.split(":", maxsplit=1)
        low, high = float(low_text), float(high_text)
        if not 0 <= low < high <= 1:
            raise ValueError(f"invalid band {item}")
        bands.append((f"band{index}_low{low:g}_high{high:g}".replace(".", "p"), low, high))
    return bands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--initial_point_cloud")
    parser.add_argument("--initialization_label", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--bands", default="0.01:0.04")
    parser.add_argument(
        "--target_mode",
        choices=("smooth_band", "smooth_band_blur", "binary_blur"),
        default="smooth_band_blur",
    )
    parser.add_argument("--binary_threshold", type=float, default=0.01)
    parser.add_argument("--binary_blur_kernel", type=int, default=5)
    parser.add_argument("--binary_blur_sigma", type=float, default=1.0)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument(
        "--field_backend",
        choices=("gaussolve", "native", "custom"),
        default="gaussolve",
    )
    parser.add_argument(
        "--preserve_initial_geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--start_sh_degree", type=int, choices=(0, 1, 2, 3), default=0
    )
    parser.add_argument("--initial_opacity", type=float, default=0.01)
    parser.add_argument("--initial_value", type=float, default=0.05)
    parser.add_argument("--prune_opacity", type=float, default=0.01)
    parser.add_argument("--opacity_sparse", type=float, default=0.0)
    # A residual target is already in [0, 1].  Extra positive-pixel weighting
    # rewards oversized splats for painting broad white regions, so the normal
    # objective intentionally uses equal per-pixel weight.
    parser.add_argument("--positive_weight", type=float, default=0.0)
    parser.add_argument("--mesh_depth_tolerance", type=float, default=0.05)
    parser.add_argument(
        "--mesh_depth_gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--export_initial_opacity", type=float, default=0.01)
    parser.add_argument("--export_initial_value", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--densify", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=5000)
    parser.add_argument("--densification_interval", type=int, default=500)
    parser.add_argument("--densify_grad_threshold", type=float, default=2e-4)
    parser.add_argument("--densify_min_opacity", type=float, default=0.005)
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--densify_max_points", type=int, default=120000)
    parser.add_argument("--prune_from_iter", type=int, default=500)
    parser.add_argument("--prune_interval", type=int, default=500)
    parser.add_argument("--size_prune_from_iter", type=int, default=20000)
    # The NDF is rendered at 1/4 resolution and intentionally represents broad
    # regions, so use 200 px as a catastrophic-footprint guard rather than the
    # RGB renderer's much tighter 20 px refinement threshold.
    parser.add_argument("--max_screen_size", type=float, default=200.0)
    parser.add_argument("--max_world_scale_fraction", type=float, default=0.1)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--feature_rest_lr_scale", type=float, default=1.0)
    parser.add_argument(
        "--position_lr_max_steps",
        type=int,
        default=0,
        help="XYZ LR schedule length; 0 uses --iterations",
    )
    parser.add_argument("--opacity_reset_interval", type=int, default=0)
    parser.add_argument("--opacity_reset_value", type=float, default=0.01)
    parser.add_argument(
        "--stabilize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply the original per-step scale/parameter guard",
    )
    parser.add_argument("--stabilize_scale_min_factor", type=float, default=0.25)
    parser.add_argument("--stabilize_scale_max_factor", type=float, default=4.0)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output: {output}")
    output.mkdir(parents=True)
    write_json(output / "status.json", {"status": "running"})
    try:
        source = Path(args.source_path).resolve()
        checkpoint = Path(args.stage1_checkpoint).resolve()
        bands = parse_bands(args.bands)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        background = torch.ones(3, device="cuda")
        train_cameras, test_cameras = load_cameras(source)
        renderer, appearance, mesh_path, appearance_path = load_stage1(checkpoint)
        with torch.no_grad():
            unique_vertices = torch.unique(renderer.verts.detach().cpu(), dim=0)
            train_samples = precompute_error_samples(
                renderer,
                appearance,
                train_cameras,
                background,
                args.downsample,
                args.target_mode,
                args.binary_threshold,
                args.binary_blur_kernel,
                args.binary_blur_sigma,
            )
            test_samples = precompute_error_samples(
                renderer,
                appearance,
                test_cameras,
                background,
                args.downsample,
                args.target_mode,
                args.binary_threshold,
                args.binary_blur_kernel,
                args.binary_blur_sigma,
            )

        if args.initial_point_cloud:
            initial_path = Path(args.initial_point_cloud).resolve()
            if not initial_path.is_file():
                raise FileNotFoundError(initial_path)
            initial_points_cpu = load_initial_points(initial_path)
            args.initialization_label = (
                args.initialization_label or "external initial point cloud"
            )
            initialization = {
                "kind": "point_cloud",
                "path": str(initial_path),
                "sha256": sha256_file(initial_path),
                "count": int(initial_points_cpu.shape[0]),
            }
        else:
            initial_points_cpu = unique_vertices
            args.initialization_label = "exact unique Stage-1 mesh vertices"
            initialization = {
                "kind": "mesh_vertices",
                "path": str(mesh_path),
                "sha256": sha256_file(mesh_path),
                "count": int(initial_points_cpu.shape[0]),
            }
        valid_errors = torch.cat(
            [sample.raw_error.float()[sample.valid] for sample in train_samples]
        )
        quantile_levels = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        quantiles = torch.quantile(valid_errors, quantile_levels)
        error_audit = {
            "valid_pixel_count": int(valid_errors.numel()),
            "mean": float(valid_errors.mean().item()),
            "quantiles": {
                str(float(level)): float(value)
                for level, value in zip(quantile_levels, quantiles)
            },
        }
        write_json(output / "error_distribution.json", error_audit)

        centers = torch.stack([camera.camera_center for camera in train_cameras])
        center = centers.mean(dim=0)
        scene_extent = float((centers - center).norm(dim=1).max().item() * 1.1)
        command = {
            **vars(args),
            "source_path": str(source),
            "stage1_checkpoint": str(checkpoint),
            "output_dir": str(output),
            "train_count": len(train_samples),
            "test_count": len(test_samples),
            "initial_unique_mesh_vertices": int(unique_vertices.shape[0]),
            "initialization": initialization,
            "scene_extent": scene_extent,
            "stage1_hashes": {
                "mesh": sha256_file(mesh_path),
                "appearance": sha256_file(appearance_path),
            },
            "code_hashes": {
                name: sha256_file(Path(__file__).resolve().parent / name)
                for name in ("train.py", "field.py", "native_field.py", "targets.py")
            },
        }
        write_json(output / "command.json", command)

        # Stage-1 is needed only to create detached error targets.
        for camera in train_cameras + test_cameras:
            camera.original_image = None
        del valid_errors, renderer, appearance, train_cameras, test_cameras
        gc.collect()
        torch.cuda.empty_cache()
        initial_points = initial_points_cpu.cuda().float().contiguous()

        results = {}
        for name, low, high in bands:
            band_dir = output / name
            band_dir.mkdir()
            set_band_targets(
                train_samples,
                low,
                high,
                args.target_mode,
                args.binary_blur_kernel,
                args.binary_blur_sigma,
            )
            set_band_targets(
                test_samples,
                low,
                high,
                args.target_mode,
                args.binary_blur_kernel,
                args.binary_blur_sigma,
            )
            torch.cuda.reset_peak_memory_stats()
            field, history, densification_history, first_gradient, elapsed = train_field(
                initial_points,
                train_samples,
                low,
                high,
                args,
                scene_extent,
                band_dir,
            )
            counts = {
                "opacity_ge_1_over_255": int((field.get_opacity >= 1 / 255).sum().item()),
                "opacity_ge_0p01": int((field.get_opacity >= 0.01).sum().item()),
                "opacity_ge_0p05": int((field.get_opacity >= 0.05).sum().item()),
            }
            unpruned_train, _ = evaluate_field(
                field,
                train_samples,
                low,
                high,
                args.mesh_depth_tolerance,
                args.mesh_depth_gate,
            )
            unpruned_test, _ = evaluate_field(
                field,
                test_samples,
                low,
                high,
                args.mesh_depth_tolerance,
                args.mesh_depth_gate,
            )
            pruned = field.pruned(args.prune_opacity)
            artifact = band_dir / "ndf_scalar_sh.pt"
            inspection_ply = band_dir / "ndf_scalar_sh_rgb_inspection.ply"
            pruned.save(str(artifact))
            pruned.save_rgb_ply(str(inspection_ply))
            initialization_field = pruned.reinitialized(
                initial_opacity=args.export_initial_opacity,
                initial_value=args.export_initial_value,
            )
            initialization_artifact = band_dir / "ndf_guided_initialization.pt"
            initialization_ply = band_dir / "ndf_guided_initialization.ply"
            initialization_field.save(str(initialization_artifact))
            initialization_field.save_rgb_ply(str(initialization_ply))
            train_summary, train_rows = evaluate_field(
                pruned,
                train_samples,
                low,
                high,
                args.mesh_depth_tolerance,
                args.mesh_depth_gate,
            )
            test_summary, test_rows = evaluate_field(
                pruned,
                test_samples,
                low,
                high,
                args.mesh_depth_tolerance,
                args.mesh_depth_gate,
            )
            train_panel = save_panels(
                pruned,
                train_samples,
                low,
                high,
                args.mesh_depth_tolerance,
                band_dir / "panels_train",
                args.mesh_depth_gate,
            )
            test_panel = save_panels(
                pruned,
                test_samples,
                low,
                high,
                args.mesh_depth_tolerance,
                band_dir / "panels_test",
                args.mesh_depth_gate,
            )
            metrics = {
                "band": {"low": low, "high": high},
                "initial_count": int(initial_points.shape[0]),
                "retained_count": int(pruned.get_xyz.shape[0]),
                "densification": {
                    "enabled": bool(args.densify),
                    "events": densification_history,
                },
                "opacity_counts_before_prune": counts,
                "unpruned": {"train": unpruned_train, "test": unpruned_test},
                "pruned": {
                    "train": train_summary,
                    "test": test_summary,
                    "per_view_train": train_rows,
                    "per_view_test": test_rows,
                },
                "first_step_gradient": first_gradient,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "artifacts": {
                    "exact": {"path": str(artifact), "sha256": sha256_file(artifact)},
                    "inspection_ply": {
                        "path": str(inspection_ply),
                        "sha256": sha256_file(inspection_ply),
                    },
                    "guided_initialization": {
                        "path": str(initialization_artifact),
                        "sha256": sha256_file(initialization_artifact),
                        "opacity": args.export_initial_opacity,
                        "sh_value": args.export_initial_value,
                    },
                    "guided_initialization_ply": {
                        "path": str(initialization_ply),
                        "sha256": sha256_file(initialization_ply),
                    },
                    "train_panel": str(train_panel),
                    "test_panel": str(test_panel),
                },
            }
            write_json(band_dir / "metrics.json", metrics)
            results[name] = metrics
            del field, pruned, initialization_field
            gc.collect()
            torch.cuda.empty_cache()

        summary = {
            "status": "complete",
            "method": (
                f"{args.target_mode} Mesh--GT RGB L1 target -> "
                "target -> trainable scalar degree-3 SH 3DGS initialized from "
                f"{args.initialization_label}; densification={'on' if args.densify else 'off'}"
            ),
            "error_distribution": error_audit,
            "results": results,
            "command": command,
        }
        write_json(output / "summary.json", summary)
        summary_hash = sha256_file(output / "summary.json")
        write_json(
            output / "status.json",
            {"status": "complete", "summary_sha256": summary_hash},
        )
    except Exception:
        write_json(
            output / "status.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
