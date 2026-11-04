"""Standalone Stage-1 tests extracted from the existing mesh pretrain loop.

This entry deliberately reuses ``MeshRenderer.training_setup``,
``MeshRenderer.rasterize``, ``MeshRenderer.color_from_rast`` and the original
L1+SSIM pretrain loss.  Mesh geometry is fixed.  The ordinary RGB texture is
optimized in every run, optionally together with one small appearance module
from ``mesh_stage1_environment.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from mesh_renderer import MeshRenderer
from mesh_stage1_environment import AppearanceConfig, Stage1Appearance, VARIANTS
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import get_opengl_matrices
from utils.loss_utils import l1_loss, ssim


DEFAULT_MESH_LR_INIT = 1e-3
DEFAULT_MESH_LR_FINAL = 1e-4
DEFAULT_APPEARANCE_LR_INIT = 5e-3
DEFAULT_APPEARANCE_LR_FINAL = 5e-4


def default_save_iterations(iterations: int) -> list[int]:
    """Return the historical-compatible default checkpoint set.

    The 20k sweep uses the explicitly requested dense set.  The old 10k
    invocation keeps its 3k/10k checkpoints (with the newly required initial
    checkpoint at iteration zero).  Other iteration counts get only the
    initial and final checkpoints unless the caller supplies an explicit set.
    """

    if iterations == 20_000:
        return [0, 3_000, 5_000, 10_000, 15_000, 20_000]
    if iterations == 10_000:
        return [0, 3_000, 10_000]
    return [0, iterations]


def normalize_save_iterations(values: list[int] | tuple[int, ...] | None, iterations: int) -> list[int]:
    """Validate and canonicalize checkpoint iterations.

    Values outside ``[0, iterations]`` are rejected.  Repeated values are
    harmlessly de-duplicated, and iteration zero plus the final iteration are
    always included in the returned sorted set.
    """

    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}")
    requested = default_save_iterations(iterations) if values is None else list(values)
    out: set[int] = set()
    for value in requested:
        if not isinstance(value, int):
            raise ValueError(f"save iteration must be an integer, got {value!r}")
        if value < 0 or value > iterations:
            raise ValueError(
                f"save iteration {value} is outside the valid range [0, {iterations}]"
            )
        out.add(value)
    out.update((0, iterations))
    return sorted(out)


def exponential_lr(init: float, final: float, step: int, max_steps: int) -> float:
    """Simple log-linear/exponential learning-rate interpolation."""

    if init <= 0.0 or final <= 0.0:
        raise ValueError(f"learning rates must be positive, got {init}, {final}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(init) * (1.0 - t) + math.log(final) * t)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def background_metadata(white_background: bool) -> dict:
    """Return the explicit renderer background contract for run metadata."""

    value = 1.0 if white_background else 0.0
    return {
        "name": "white" if white_background else "black",
        "rgb": [value, value, value],
    }


def make_background(white_background: bool) -> torch.Tensor:
    value = 1.0 if white_background else 0.0
    return torch.full((3,), value, device="cuda")


def load_cameras(source_path: Path):
    scene_info = readColmapSceneInfo(
        str(source_path),
        images="images",
        depths="",
        masks="",
        eval=True,
        train_test_exp=False,
        lack_memory=False,
    )
    camera_args = SimpleNamespace(data_device="cpu")
    train = cameraList_from_camInfos(
        scene_info.train_cameras, 1.0, camera_args, scene_info.is_nerf_synthetic, False
    )
    test = cameraList_from_camInfos(
        scene_info.test_cameras, 1.0, camera_args, scene_info.is_nerf_synthetic, True
    )
    train = sorted(train, key=lambda camera: camera.image_name)
    test = sorted(test, key=lambda camera: camera.image_name)
    if not train or not test:
        raise RuntimeError(f"Stage-1 requires non-empty train/test splits, got {len(train)}/{len(test)}")
    for camera in train + test:
        camera.open_gl_view, camera.open_gl_mvp = get_opengl_matrices(camera)
    return train, test


def scene_center_from_cameras(cameras) -> torch.Tensor:
    return torch.stack([camera.camera_center.detach() for camera in cameras]).mean(dim=0)


def renderer_training_args(
    iterations: int,
    mesh_lr_init: float = DEFAULT_MESH_LR_INIT,
    mesh_lr_final: float = DEFAULT_MESH_LR_FINAL,
) -> SimpleNamespace:
    # These fields are consumed by MeshRenderer.training_setup.  Geometry is
    # held in ordinary tensors; only the RGB/alpha material parameters enter
    # the optimizer.
    return SimpleNamespace(
        mesh_lr_init=mesh_lr_init,
        mesh_lr_final=mesh_lr_final,
        mesh_lr_delay_steps=0,
        mesh_lr_delay_mult=0.0,
        iterations=iterations,
    )


def build_renderer(
    mesh_path: Path,
    iterations: int,
    mesh_lr_init: float = DEFAULT_MESH_LR_INIT,
    mesh_lr_final: float = DEFAULT_MESH_LR_FINAL,
) -> MeshRenderer:
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
    renderer.training_setup(
        renderer_training_args(iterations, mesh_lr_init, mesh_lr_final)
    )
    renderer.unfreeze()
    if renderer.optimizer is None or renderer.texture is None:
        raise RuntimeError("Existing mesh pretrain did not expose a trainable RGB texture")
    return renderer


def render(
    renderer: MeshRenderer,
    appearance: Stage1Appearance,
    camera,
    background: torch.Tensor,
    antialiasing: bool = True,
):
    pos, _verts_view, rast, mask = renderer.rasterize(camera)
    base = renderer.color_from_rast(
        camera,
        pos,
        rast,
        mask,
        bg_color=background,
        enable_antialiasing=antialiasing,
    )
    image = appearance(renderer, camera, rast, mask, base)
    return image, mask


def training_loss(image: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    # This is the original mesh pretrain loss: both images are zeroed outside
    # the rasterized mesh before L1 and SSIM are evaluated.
    mask_chw = mask.float()
    if mask_chw.ndim == 3:
        mask_chw = mask_chw.squeeze(0)
    mask_chw = mask_chw.unsqueeze(0)
    image_masked = image * mask_chw
    target_masked = target * mask_chw
    l1 = l1_loss(image_masked, target_masked)
    ssim_value = ssim(image_masked, target_masked)
    loss = 0.8 * l1 + 0.2 * (1.0 - ssim_value)
    return loss, l1, ssim_value


def image_metrics(image: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    mask_hw = mask.float().squeeze(0)
    mask_chw = mask_hw.unsqueeze(0)
    absolute = (image - target).abs()
    squared = (image - target).square()
    valid_denominator = (mask_hw.sum() * 3.0).clamp_min(1.0)
    valid_l1 = (absolute * mask_chw).sum() / valid_denominator
    valid_mse = (squared * mask_chw).sum() / valid_denominator
    full_mse = squared.mean()
    masked_ssim = ssim(image * mask_chw, target * mask_chw)
    return {
        "full_l1": float(absolute.mean().item()),
        "full_psnr": float((-10.0 * torch.log10(full_mse.clamp_min(1e-12))).item()),
        "mesh_l1": float(valid_l1.item()),
        "mesh_psnr": float((-10.0 * torch.log10(valid_mse.clamp_min(1e-12))).item()),
        "mesh_ssim_zero_outside": float(masked_ssim.item()),
        "mesh_coverage": float(mask_hw.mean().item()),
    }


@torch.no_grad()
def evaluate(renderer, appearance, cameras, background) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    for camera in cameras:
        image, mask = render(renderer, appearance, camera, background, antialiasing=True)
        target = camera.original_image.to("cuda", non_blocking=True)
        metrics = image_metrics(image, target, mask)
        records.append({"image_name": camera.image_name, **metrics})
    summary = {"count": len(records)}
    for key in (
        "full_l1",
        "full_psnr",
        "mesh_l1",
        "mesh_psnr",
        "mesh_ssim_zero_outside",
        "mesh_coverage",
    ):
        values = [record[key] for record in records]
        summary[key] = sum(values) / len(values)
    return summary, records


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


@torch.no_grad()
def save_panels(renderer, appearance, cameras, background, output_dir: Path, title: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = cameras[:5]
    row_paths: list[str] = []
    rows: list[Image.Image] = []
    for camera in selected:
        image, mask = render(renderer, appearance, camera, background, antialiasing=True)
        target = camera.original_image.to("cuda", non_blocking=True)
        error = (image - target).abs().mean(dim=0, keepdim=True).expand(3, -1, -1)
        columns = (_to_pil(target), _to_pil(image), _to_pil((error / 0.15).clamp(0.0, 1.0)))
        width, height = columns[0].size
        header = 28
        canvas = Image.new("RGB", (width * 3, height + header), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (label, column) in enumerate(zip(("GT", title, "absolute error / 0.15"), columns)):
            canvas.paste(column, (index * width, header))
            draw.text((index * width + 7, 7), label, fill="black")
        path = output_dir / f"{Path(camera.image_name).stem}.png"
        canvas.save(path)
        rows.append(canvas)
        row_paths.append(str(path))
    contact = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        contact.paste(row, (0, y))
        y += row.height
    contact_path = output_dir / "contact_sheet.png"
    contact.save(contact_path)
    return {
        "camera_names": [camera.image_name for camera in selected],
        "rows": row_paths,
        "contact_sheet": str(contact_path),
    }


def save_checkpoint(
    renderer,
    appearance,
    variant_dir: Path,
    iteration: int,
    train_cameras,
    test_cameras,
    background,
    background_info: dict,
    checkpoint_metadata: dict,
    final_iteration: int,
) -> dict:
    checkpoint_dir = variant_dir / f"iteration_{iteration}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    mesh_path = checkpoint_dir / "mesh_pretrained.glb"
    renderer.save_model(str(mesh_path))
    appearance_path = checkpoint_dir / "environment.pt"
    appearance.save(
        str(appearance_path),
        extra={
            "iteration": iteration,
            "mesh_sha256": sha256_file(mesh_path),
            **checkpoint_metadata,
        },
    )
    train_summary, train_records = evaluate(
        renderer, appearance, train_cameras, background
    )
    test_summary, test_records = evaluate(renderer, appearance, test_cameras, background)
    metrics = {
        "iteration": iteration,
        "variant": appearance.variant,
        "background": background_info,
        **checkpoint_metadata,
        "train": train_summary,
        "test": test_summary,
        "per_view": {"train": train_records, "test": test_records},
        "artifacts": {
            "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
            "environment": {
                "path": str(appearance_path),
                "sha256": sha256_file(appearance_path),
            },
        },
    }
    metrics_path = checkpoint_dir / "metrics.json"
    # Keep the historical 10k visual panel while also making the final
    # checkpoint self-contained for longer runs.
    if iteration in {10_000, final_iteration}:
        metrics["panels"] = {
            "train": save_panels(
                renderer,
                appearance,
                train_cameras,
                background,
                checkpoint_dir / "panels_train",
                appearance.variant,
            ),
            "test": save_panels(
                renderer,
                appearance,
                test_cameras,
                background,
                checkpoint_dir / "panels_test",
                appearance.variant,
            ),
        }
    # Write one complete, non-self-referential metrics document.  The file
    # digest is returned to the caller only, so metrics.json cannot contain a
    # stale hash of an earlier version of itself.
    write_json(metrics_path, metrics)
    metrics["metrics_sha256"] = sha256_file(metrics_path)
    return metrics


def adapt_test_environment(
    renderer,
    appearance: Stage1Appearance,
    test_cameras,
    background,
    output_dir: Path,
    steps: int = 300,
) -> dict | None:
    if appearance.variant not in {"view_env_sh", "view_env_mlp"}:
        return None
    adapted = copy.deepcopy(appearance).to("cuda")
    for parameter in adapted.parameters():
        parameter.requires_grad_(False)
    gate_parameters = adapted.gate_parameters()
    for parameter in gate_parameters:
        parameter.requires_grad_(True)
    if not gate_parameters:
        raise RuntimeError("Dynamic environment has no test-adaptable gate")
    # Texture and environment maps stay frozen.  Only the shared SH/MLP gate
    # sees the 15 test images during this explicitly labelled adaptation.
    if isinstance(renderer.texture, torch.Tensor):
        renderer.texture.requires_grad_(False)
    optimizer = torch.optim.Adam(gate_parameters, lr=1e-3)
    rng = random.Random(0)
    history: list[dict] = []
    for step in range(1, steps + 1):
        camera = test_cameras[rng.randrange(len(test_cameras))]
        image, mask = render(renderer, adapted, camera, background, antialiasing=True)
        target = camera.original_image.to("cuda", non_blocking=True)
        loss, l1, ssim_value = training_loss(image, target, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 50 == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "l1": float(l1.item()),
                    "ssim": float(ssim_value.item()),
                }
            )
    summary, records = evaluate(renderer, adapted, test_cameras, background)
    artifact = output_dir / "test_adapted_environment.pt"
    adapted.save(
        str(artifact),
        extra={
            "mode": "shared_test_view_environment_adaptation",
            "steps": steps,
            "texture_updated": False,
            "geometry_updated": False,
        },
    )
    panels = save_panels(
        renderer,
        adapted,
        test_cameras,
        background,
        output_dir / "panels_test_adapted",
        f"{appearance.variant} adapted",
    )
    return {
        "label": "test-time environment adaptation; not zero-shot",
        "shared_across_all_15_test_views": True,
        "texture_updated": False,
        "geometry_updated": False,
        "steps": steps,
        "history": history,
        "test": summary,
        "per_view_test": records,
        "artifact": {"path": str(artifact), "sha256": sha256_file(artifact)},
        "panels": panels,
    }


def train_variant(
    variant: str,
    mesh_path: Path,
    variant_dir: Path,
    train_cameras,
    test_cameras,
    iterations: int,
    seed: int,
    background: torch.Tensor,
    background_info: dict,
    save_iterations: list[int],
    mesh_lr_init: float,
    mesh_lr_final: float,
    appearance_lr_init: float,
    appearance_lr_final: float,
    input_mesh_sha256: str,
) -> dict:
    variant_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    random.seed(seed)
    torch.manual_seed(seed)
    renderer = build_renderer(mesh_path, iterations, mesh_lr_init, mesh_lr_final)
    appearance = Stage1Appearance(
        AppearanceConfig(variant=variant),
        scene_center_from_cameras(train_cameras),
        seed=seed,
    ).to("cuda")
    appearance_groups = appearance.optimizer_groups()
    appearance_optimizer = (
        torch.optim.Adam(appearance_groups, eps=1e-8) if appearance_groups else None
    )
    appearance_lr_schedule = lambda step: exponential_lr(
        appearance_lr_init, appearance_lr_final, step, iterations
    )
    if appearance_optimizer is not None:
        initial_appearance_lr = appearance_lr_schedule(0)
        for group in appearance_optimizer.param_groups:
            group["lr"] = initial_appearance_lr
    rng = random.Random(seed)
    camera_stack = list(train_cameras)
    checkpoints: dict[str, dict] = {}
    checkpoint_metadata = {
        "mesh_lr_init": mesh_lr_init,
        "mesh_lr_final": mesh_lr_final,
        "appearance_lr_init": appearance_lr_init,
        "appearance_lr_final": appearance_lr_final,
        "input_mesh_sha256": input_mesh_sha256,
        "geometry_frozen": True,
    }
    csv_path = variant_dir / "train_log.csv"
    # Iteration zero is deliberately saved before entering the optimizer loop.
    # This provides a paired, untouched GLB/environment baseline.
    checkpoints["0"] = save_checkpoint(
        renderer,
        appearance,
        variant_dir,
        0,
        train_cameras,
        test_cameras,
        background,
        background_info,
        checkpoint_metadata,
        iterations,
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "iteration",
                "image_name",
                "loss",
                "l1",
                "ssim",
                "mesh_lr",
                "appearance_lr",
                "mesh_lr_init",
                "mesh_lr_final",
                "appearance_lr_init",
                "appearance_lr_final",
            )
        )
        for iteration in range(1, iterations + 1):
            if not camera_stack:
                camera_stack = list(train_cameras)
            camera = camera_stack.pop(rng.randrange(len(camera_stack)))
            mesh_lr = renderer.update_learning_rate(iteration)
            appearance_lr = appearance_lr_schedule(iteration)
            if appearance_optimizer is not None:
                for group in appearance_optimizer.param_groups:
                    group["lr"] = appearance_lr
            image, mask = render(renderer, appearance, camera, background, antialiasing=True)
            target = camera.original_image.to("cuda", non_blocking=True)
            loss, l1, ssim_value = training_loss(image, target, mask)
            loss.backward()
            renderer.optimizer.step()
            renderer.optimizer.zero_grad(set_to_none=True)
            if appearance_optimizer is not None:
                appearance_optimizer.step()
                appearance_optimizer.zero_grad(set_to_none=True)
            if iteration == 1 or iteration % 10 == 0:
                writer.writerow(
                    (
                        iteration,
                        camera.image_name,
                        float(loss.item()),
                        float(l1.item()),
                        float(ssim_value.item()),
                        float(mesh_lr),
                        float(appearance_lr),
                        mesh_lr_init,
                        mesh_lr_final,
                        appearance_lr_init,
                        appearance_lr_final,
                    )
                )
                stream.flush()
            if iteration == 1 or iteration % 500 == 0:
                print(
                    f"[{variant}] {iteration}/{iterations} "
                    f"loss={loss.item():.7f} l1={l1.item():.7f} ssim={ssim_value.item():.7f}",
                    flush=True,
                )
            if iteration in save_iterations:
                checkpoints[str(iteration)] = save_checkpoint(
                    renderer,
                    appearance,
                    variant_dir,
                    iteration,
                    train_cameras,
                    test_cameras,
                    background,
                    background_info,
                    checkpoint_metadata,
                    iterations,
                )

    adapted = adapt_test_environment(
        renderer,
        appearance,
        test_cameras,
        background,
        variant_dir / f"iteration_{iterations}",
        steps=300,
    )
    result = {
        "status": "complete",
        "variant": variant,
        "geometry_frozen": True,
        "texture_trained_with_existing_mesh_pretrain_optimizer": True,
        "iterations": iterations,
        "save_iterations": save_iterations,
        **checkpoint_metadata,
        "background": background_info,
        "appearance_lr": appearance_lr_schedule(iterations),
        "checkpoints": checkpoints,
        "test_environment_adaptation": adapted,
        "train_log": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024.0**2),
    }
    write_json(variant_dir / "result.json", result)
    del appearance, renderer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=None,
        help="checkpoint iterations; zero and the final iteration are always included",
    )
    parser.add_argument("--mesh_lr_init", type=float, default=DEFAULT_MESH_LR_INIT)
    parser.add_argument("--mesh_lr_final", type=float, default=DEFAULT_MESH_LR_FINAL)
    parser.add_argument(
        "--appearance_lr_init",
        type=float,
        default=DEFAULT_APPEARANCE_LR_INIT,
    )
    parser.add_argument(
        "--appearance_lr_final",
        type=float,
        default=DEFAULT_APPEARANCE_LR_FINAL,
    )
    parser.add_argument("--seed", type=int, default=0)
    background_group = parser.add_mutually_exclusive_group()
    background_group.add_argument(
        "--white_background",
        dest="white_background",
        action="store_true",
        help="render against white (the historical default)",
    )
    background_group.add_argument(
        "--black_background",
        dest="white_background",
        action="store_false",
        help="render against black",
    )
    parser.set_defaults(white_background=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be > 0")
    for name in (
        "mesh_lr_init",
        "mesh_lr_final",
        "appearance_lr_init",
        "appearance_lr_final",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name} must be > 0")
    try:
        save_iterations = normalize_save_iterations(args.save_iterations, args.iterations)
    except ValueError as error:
        parser.error(str(error))

    source_path = Path(args.source_path).resolve()
    mesh_path = Path(args.mesh_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Stage-1 output: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "status.json", {"status": "running"})
    command = {
        "source_path": str(source_path),
        "mesh_path": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "output_dir": str(output_dir),
        "iterations": args.iterations,
        "save_iterations": save_iterations,
        "seed": args.seed,
        "mesh_lr_init": args.mesh_lr_init,
        "mesh_lr_final": args.mesh_lr_final,
        "appearance_lr_init": args.appearance_lr_init,
        "appearance_lr_final": args.appearance_lr_final,
        "geometry_frozen": True,
        "background": background_metadata(args.white_background),
        "variants": args.variants,
        "code_sha256": {
            "pretrain_mesh.py": sha256_file(Path(__file__).resolve()),
            "mesh_stage1_environment.py": sha256_file(
                Path(__file__).resolve().parent / "mesh_stage1_environment.py"
            ),
            "mesh_renderer.py": sha256_file(Path(__file__).resolve().parent / "mesh_renderer.py"),
        },
    }
    write_json(output_dir / "command.json", command)
    try:
        train_cameras, test_cameras = load_cameras(source_path)
        background = make_background(args.white_background)
        background_info = background_metadata(args.white_background)
        command["train_camera_count"] = len(train_cameras)
        command["test_camera_count"] = len(test_cameras)
        write_json(output_dir / "command.json", command)
        results = {}
        for variant in args.variants:
            torch.cuda.reset_peak_memory_stats()
            results[variant] = train_variant(
                variant,
                mesh_path,
                output_dir / variant,
                train_cameras,
                test_cameras,
                args.iterations,
                args.seed,
                background,
                background_info,
                save_iterations,
                args.mesh_lr_init,
                args.mesh_lr_final,
                args.appearance_lr_init,
                args.appearance_lr_final,
                command["mesh_sha256"],
            )
        summary = {
            "status": "complete",
            "geometry_frozen": True,
            "ordinary_texture_jointly_optimized": True,
            "background": background_info,
            "input_mesh_sha256": command["mesh_sha256"],
            "iterations": args.iterations,
            "save_iterations": save_iterations,
            "mesh_lr_init": args.mesh_lr_init,
            "mesh_lr_final": args.mesh_lr_final,
            "appearance_lr_init": args.appearance_lr_init,
            "appearance_lr_final": args.appearance_lr_final,
            "specular_brdf_or_material_maps": False,
            "results": results,
            "command": command,
        }
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "status.json",
            {
                "status": "complete",
                "summary": str(output_dir / "summary.json"),
                "summary_sha256": sha256_file(output_dir / "summary.json"),
            },
        )
    except Exception:
        write_json(
            output_dir / "status.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
