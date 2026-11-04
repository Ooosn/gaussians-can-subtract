"""Create NDF seed positions from a multi-view UV loss texture.

This is deliberately a surface sampler, not a learned field.  Stage-1 mesh
geometry and appearance are frozen, Mesh--GT residuals are accumulated in a
fixed-resolution UV atlas, and selected texels are mapped back to their
interpolated world-space surface positions.  A final 3D voxel deduplication
makes the seed count independent of the input mesh vertex count and removes
UV-seam duplicates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nvdiffrast.torch as dr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F

from pretrain_mesh import load_cameras
from ndf_stage2_shfit.native_field import NativeScalarSHGaussianField
from ndf_stage2_shfit.targets import masked_gaussian_blur, smooth_threshold
from ndf_stage2_shfit.train import (
    load_stage1,
    precompute_error_samples,
    render_field,
    sample_to_cuda,
    set_band_targets,
    sha256_file,
    tensor_to_pil,
    write_json,
)
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import C0


def erode_visible_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Remove raster silhouettes and the fixed image border."""

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("visible mask must be [H,W] or [1,H,W]")
    if radius <= 0:
        return mask
    invalid = (~mask).float()[None, None]
    expanded_invalid = F.max_pool2d(
        invalid, kernel_size=2 * radius + 1, stride=1, padding=radius
    )[0, 0].bool()
    result = mask & ~expanded_invalid
    result[:radius] = False
    result[-radius:] = False
    result[:, :radius] = False
    result[:, -radius:] = False
    return result


def voxel_deduplicate(
    points: torch.Tensor,
    scores: torch.Tensor,
    supports: torch.Tensor,
    uv_indices: torch.Tensor,
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the strongest surface sample in each world-space voxel."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if points.shape[0] == 0:
        raise ValueError("cannot deduplicate an empty point set")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    origin = points.amin(dim=0)
    voxel = torch.floor((points - origin) / float(voxel_size)).to(torch.int64)
    _unique, inverse = torch.unique(voxel, dim=0, return_inverse=True)
    count = int(inverse.max().item()) + 1

    # Support is the primary evidence; smooth target strength breaks ties.
    rank_score = supports.float() * 2.0 + scores
    best_score = torch.full(
        (count,), -torch.inf, dtype=rank_score.dtype, device=rank_score.device
    )
    best_score.scatter_reduce_(
        0, inverse, rank_score, reduce="amax", include_self=True
    )
    row = torch.arange(points.shape[0], device=points.device, dtype=torch.int64)
    sentinel = torch.full_like(row, points.shape[0])
    eligible = rank_score >= best_score[inverse] - 1e-8
    candidate = torch.where(eligible, row, sentinel)
    best_row = torch.full(
        (count,), points.shape[0], dtype=torch.int64, device=points.device
    )
    best_row.scatter_reduce_(
        0, inverse, candidate, reduce="amin", include_self=True
    )
    if bool((best_row >= points.shape[0]).any()):
        raise RuntimeError("voxel winner selection failed")
    return (
        points[best_row],
        scores[best_row],
        supports[best_row],
        uv_indices[best_row],
    )


@torch.no_grad()
def accumulate_uv_loss(
    renderer,
    appearance,
    cameras: list,
    resolution: int,
    silhouette_radius: int,
    background: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Accumulate one mean residual and surface position per view and texel."""

    if resolution <= 1:
        raise ValueError("UV resolution must be greater than one")
    texel_count = resolution * resolution
    error_sum = torch.zeros(texel_count, device="cuda", dtype=torch.float32)
    xyz_sum = torch.zeros(texel_count, 3, device="cuda", dtype=torch.float32)
    view_support = torch.zeros(texel_count, device="cuda", dtype=torch.int32)
    for view_index, camera in enumerate(cameras, start=1):
        pos, _verts_view, rast, mask = renderer.rasterize(camera)
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
        residual = (mesh_rgb - ground_truth).abs().mean(dim=0)
        uv_map, _ = dr.interpolate(
            renderer.uv.contiguous(), rast, renderer.faces_contiguous
        )
        world_map, _ = dr.interpolate(
            renderer.verts.contiguous(), rast, renderer.faces_contiguous
        )
        uv_map = uv_map[0]
        world_map = world_map[0]
        valid = erode_visible_mask(mask, silhouette_radius)
        valid &= torch.isfinite(residual)
        valid &= torch.isfinite(uv_map).all(dim=-1)
        valid &= torch.isfinite(world_map).all(dim=-1)
        valid &= (uv_map[..., 0] >= 0.0) & (uv_map[..., 0] <= 1.0)
        valid &= (uv_map[..., 1] >= 0.0) & (uv_map[..., 1] <= 1.0)

        uv = uv_map[valid]
        xyz = world_map[valid]
        err = residual[valid]
        x = torch.round(uv[:, 0] * (resolution - 1)).long()
        y = torch.round(uv[:, 1] * (resolution - 1)).long()
        texel_index = y * resolution + x

        per_view_count = torch.zeros(texel_count, device="cuda")
        per_view_error = torch.zeros(texel_count, device="cuda")
        per_view_xyz = torch.zeros(texel_count, 3, device="cuda")
        per_view_count.index_add_(
            0, texel_index, torch.ones_like(err, dtype=torch.float32)
        )
        per_view_error.index_add_(0, texel_index, err.float())
        per_view_xyz.index_add_(0, texel_index, xyz.float())
        touched = per_view_count > 0
        count = per_view_count[touched]
        error_sum[touched] += per_view_error[touched] / count
        xyz_sum[touched] += per_view_xyz[touched] / count[:, None]
        view_support[touched] += 1
        print(
            json.dumps(
                {
                    "view": view_index,
                    "total_views": len(cameras),
                    "image": camera.image_name,
                    "valid_pixels": int(valid.sum().item()),
                    "touched_texels": int(touched.sum().item()),
                }
            ),
            flush=True,
        )

    supported = view_support > 0
    mean_error = torch.zeros_like(error_sum)
    mean_xyz = torch.zeros_like(xyz_sum)
    mean_error[supported] = error_sum[supported] / view_support[supported].float()
    mean_xyz[supported] = xyz_sum[supported] / view_support[supported].float()[:, None]
    return {
        "mean_error": mean_error.reshape(1, resolution, resolution),
        "mean_xyz": mean_xyz.reshape(resolution, resolution, 3),
        "view_support": view_support.reshape(resolution, resolution),
    }


def save_point_ply(
    path: Path,
    points: torch.Tensor,
    scores: torch.Tensor,
    supports: torch.Tensor,
    uv_indices: torch.Tensor,
    resolution: int,
) -> None:
    points_np = points.detach().cpu().numpy().astype(np.float32, copy=False)
    score_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
    support_np = supports.detach().cpu().numpy().astype(np.int32, copy=False)
    uv_np = uv_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    data = np.empty(
        points_np.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("loss_weight", "f4"),
            ("view_support", "i4"),
            ("u", "f4"),
            ("v", "f4"),
        ],
    )
    data["x"], data["y"], data["z"] = points_np.T
    data["loss_weight"] = score_np
    data["view_support"] = support_np
    data["u"] = (uv_np % resolution).astype(np.float32) / (resolution - 1)
    data["v"] = (uv_np // resolution).astype(np.float32) / (resolution - 1)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def grayscale_image(value: torch.Tensor) -> Image.Image:
    array = value.detach().float().cpu().clamp(0, 1).mul(255).byte().numpy()
    return Image.fromarray(array, mode="L").convert("RGB")


def save_uv_diagnostic(
    path: Path,
    mean_error: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    selected: torch.Tensor,
) -> None:
    error_display = (mean_error[0] / 0.15).clamp(0, 1)
    support_display = support.float() / support.max().clamp_min(1).float()
    images = [
        grayscale_image(error_display),
        grayscale_image(target[0]),
        grayscale_image(support_display),
        grayscale_image(selected.float()),
    ]
    labels = ["UV residual / 0.15", "smooth 0.015-0.04", "view support", "selected"]
    size = images[0].width
    label_height = 28
    canvas = Image.new("RGB", (4 * size, size + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image, (index * size, label_height))
        draw.text((index * size + 5, 5), label, fill="black", font=font)
    canvas.save(path, optimize=True)


@torch.no_grad()
def save_projection_panel(
    path: Path,
    field: NativeScalarSHGaussianField,
    samples: list,
    low: float,
    high: float,
    mesh_depth_tolerance: float,
) -> None:
    # Diagnostic-only visibility: geometry/scale come from gaussolve_init,
    # while a stronger white scalar and opacity make the seed support legible.
    field._features_dc.fill_((1.0 - 0.5) / C0)
    field._features_rest.zero_()
    field._opacity.copy_(inverse_sigmoid(torch.full_like(field.get_opacity, 0.03)))
    field.active_sh_degree = 0
    font = ImageFont.load_default(size=18)
    rows = []
    for sample in samples:
        _raw, target, valid, depth = sample_to_cuda(sample, low, high)
        prediction = render_field(
            field, sample, depth, mesh_depth_tolerance, use_mesh_depth=True
        ).clamp(0, 1) * valid
        left = tensor_to_pil((target * 255).byte())
        right = tensor_to_pil((prediction * 255).byte())
        width, height = left.size
        row = Image.new("RGB", (2 * width, height + 28), "white")
        draw = ImageDraw.Draw(row)
        row.paste(left, (0, 28))
        row.paste(right, (width, 28))
        draw.text((5, 5), f"{sample.name}: loss target", fill="black", font=font)
        draw.text((width + 5, 5), "direct seed support", fill="black", font=font)
        rows.append(row)
    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--uv_resolution", type=int, default=1024)
    parser.add_argument("--low", type=float, default=0.015)
    parser.add_argument("--high", type=float, default=0.04)
    parser.add_argument("--selection_threshold", type=float, default=0.5)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--blur_kernel", type=int, default=5)
    parser.add_argument("--blur_sigma", type=float, default=1.0)
    parser.add_argument("--silhouette_radius", type=int, default=2)
    parser.add_argument("--voxel_size", type=float, default=0.04)
    parser.add_argument("--diagnostic_downsample", type=int, default=4)
    parser.add_argument("--mesh_depth_tolerance", type=float, default=0.05)
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
    args = parser.parse_args()

    if not 0 <= args.low < args.high <= 1:
        raise ValueError("expected 0 <= low < high <= 1")
    if not 0 < args.selection_threshold <= 1:
        raise ValueError("selection_threshold must be in (0,1]")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output: {output}")
    output.mkdir(parents=True)
    write_json(output / "status.json", {"status": "running"})

    try:
        source = Path(args.source_path).resolve()
        checkpoint = Path(args.stage1_checkpoint).resolve()
        background_value = 1.0 if args.white_background else 0.0
        background = torch.full((3,), background_value, device="cuda")
        background_info = {
            "name": "white" if args.white_background else "black",
            "rgb": [background_value, background_value, background_value],
        }
        train_cameras, test_cameras = load_cameras(source)
        renderer, appearance, mesh_path, appearance_path = load_stage1(checkpoint)
        if renderer.uv is None:
            raise RuntimeError("Stage-1 mesh has no UV coordinates")
        accumulated = accumulate_uv_loss(
            renderer,
            appearance,
            train_cameras,
            args.uv_resolution,
            args.silhouette_radius,
            background,
        )
        mean_error = accumulated["mean_error"]
        mean_xyz = accumulated["mean_xyz"]
        support = accumulated["view_support"]
        supported = support >= int(args.min_views)
        smoothed = masked_gaussian_blur(
            mean_error, supported.unsqueeze(0), args.blur_kernel, args.blur_sigma
        )
        target = smooth_threshold(smoothed, args.low, args.high) * supported
        selected = supported & (target[0] >= float(args.selection_threshold))
        if int(selected.sum().item()) < 4:
            raise RuntimeError("UV loss selection produced fewer than four points")
        uv_index = torch.nonzero(selected.reshape(-1), as_tuple=False).squeeze(1)
        candidate_points = mean_xyz.reshape(-1, 3)[uv_index]
        candidate_scores = target.reshape(-1)[uv_index]
        candidate_support = support.reshape(-1)[uv_index]
        points, scores, supports, uv_indices = voxel_deduplicate(
            candidate_points,
            candidate_scores,
            candidate_support,
            uv_index,
            args.voxel_size,
        )

        point_path = output / "uv_loss_surface_points.ply"
        exact_path = output / "uv_loss_surface_points.pt"
        save_point_ply(
            point_path, points, scores, supports, uv_indices, args.uv_resolution
        )
        torch.save(
            {
                "format": "uv_loss_surface_samples",
                "version": 1,
                "xyz": points.detach().cpu(),
                "loss_weight": scores.detach().cpu(),
                "view_support": supports.detach().cpu(),
                "uv_index": uv_indices.detach().cpu(),
                "uv_resolution": int(args.uv_resolution),
            },
            exact_path,
        )
        save_uv_diagnostic(
            output / "uv_loss_selection.png",
            smoothed,
            target,
            support,
            selected,
        )

        diagnostic_cameras = train_cameras[:5] + test_cameras[:5]
        diagnostic_samples = precompute_error_samples(
            renderer,
            appearance,
            diagnostic_cameras,
            background,
            args.diagnostic_downsample,
            "smooth_band_blur",
            args.low,
            args.blur_kernel,
            args.blur_sigma,
        )
        set_band_targets(
            diagnostic_samples,
            args.low,
            args.high,
            "smooth_band_blur",
            args.blur_kernel,
            args.blur_sigma,
        )
        field = NativeScalarSHGaussianField.from_gaussolve_vertices(
            points.contiguous(), sh_degree=3
        )
        save_projection_panel(
            output / "target_vs_direct_seed_support.png",
            field,
            diagnostic_samples,
            args.low,
            args.high,
            args.mesh_depth_tolerance,
        )

        summary = {
            "status": "complete",
            "method": "multi-view UV loss texture -> mesh surface XYZ -> 3D voxel dedup",
            "source_path": str(source),
            "stage1_checkpoint": str(checkpoint),
            "background": background_info,
            "mesh_path": str(mesh_path),
            "appearance_path": str(appearance_path),
            "train_views": len(train_cameras),
            "test_views_used_for_selection": 0,
            "uv_resolution": int(args.uv_resolution),
            "thresholds": {"low": args.low, "high": args.high},
            "selection_threshold": float(args.selection_threshold),
            "min_views": int(args.min_views),
            "blur": {"kernel": args.blur_kernel, "sigma": args.blur_sigma},
            "silhouette_radius": int(args.silhouette_radius),
            "voxel_size": float(args.voxel_size),
            "supported_texels": int(supported.sum().item()),
            "selected_texels_before_3d_dedup": int(selected.sum().item()),
            "points_after_3d_dedup": int(points.shape[0]),
            "mesh_vertex_rows_not_used_for_sampling": int(renderer.verts.shape[0]),
            "unique_mesh_vertices_not_used_for_sampling": int(
                torch.unique(renderer.verts.detach().cpu(), dim=0).shape[0]
            ),
            "score_quantiles": {
                str(q): float(torch.quantile(scores, q).item())
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "support_quantiles": {
                str(q): float(torch.quantile(supports.float(), q).item())
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "artifacts": {
                "point_cloud": {"path": str(point_path), "sha256": sha256_file(point_path)},
                "exact": {"path": str(exact_path), "sha256": sha256_file(exact_path)},
                "uv_selection": str(output / "uv_loss_selection.png"),
                "projection_panel": str(output / "target_vs_direct_seed_support.png"),
            },
            "input_hashes": {
                "mesh": sha256_file(mesh_path),
                "appearance": sha256_file(appearance_path),
            },
            "code_hash": sha256_file(Path(__file__).resolve()),
        }
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"status": "complete"})
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
