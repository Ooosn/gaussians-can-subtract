"""Official Reduced-3DGS SH sparsity/culling adapted to the hybrid main field."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814
def _load_ops():
    try:
        from diff_gaussian_rasterization import _C as active_extension

        if hasattr(active_extension, "calculate_colours_variance"):
            return active_extension
    except (ImportError, OSError) as error:
        raise ImportError(
            "Reduced SH culling requires the bundled "
            "diff-gaussian-rasterization-reduced extension"
        ) from error
    raise ImportError(
        "The active diff_gaussian_rasterization extension does not expose "
        "calculate_colours_variance; install the bundled Reduced extension"
    )


def ensure_degrees(model) -> torch.Tensor:
    count = int(model.get_xyz.shape[0])
    degrees = getattr(model, "_reduced_sh_degrees", None)
    if degrees is None:
        degrees = torch.full(
            (count, 1),
            int(model.active_sh_degree),
            dtype=torch.int32,
            device=model.get_xyz.device,
        )
        model._reduced_sh_degrees = degrees
    if tuple(degrees.shape) != (count, 1):
        raise RuntimeError(
            f"Reduced SH degree shape mismatch: {tuple(degrees.shape)} vs {(count, 1)}"
        )
    return degrees


def install_prune_tracking(model) -> None:
    if getattr(model, "_reduced_sh_prune_tracking", False):
        return
    original_prune_points = model.prune_points

    def tracked_prune_points(mask):
        degrees = ensure_degrees(model)
        valid = (~mask.reshape(-1)).detach()
        kept_degrees = degrees[valid].clone()
        result = original_prune_points(mask)
        model._reduced_sh_degrees = kept_degrees
        return result

    model.prune_points = tracked_prune_points
    model._reduced_sh_prune_tracking = True


def visible_sh_sparsity_loss(
    model, visibility_filter: torch.Tensor, weight: float = 0.1
) -> torch.Tensor:
    """Match Reduced's lambda/(visible*15*3) L1 gradient on non-DC SH."""
    indices = visibility_filter.reshape(-1).long()
    if indices.numel() == 0 or float(weight) == 0.0:
        return model._features_rest.sum() * 0.0
    return float(weight) * model._features_rest[indices].abs().mean()


@torch.no_grad()
def enforce_degree_mask(model) -> None:
    degrees = ensure_degrees(model).reshape(-1).long()
    rest = model._features_rest
    coefficient_index = torch.arange(rest.shape[1], device=rest.device)
    allowed = (degrees + 1).square() - 1
    mask = coefficient_index.unsqueeze(0) >= allowed.unsqueeze(1)
    rest.masked_fill_(mask.unsqueeze(-1), 0.0)


@torch.no_grad()
def cull_sh_bands(
    model,
    cameras,
    cdist_threshold: float = 6.0,
    std_threshold: float = 0.04,
) -> dict[str, object]:
    """Run the official Reduced multi-view SH-band culling operation."""
    if int(model.active_sh_degree) != 3:
        raise RuntimeError(
            f"Reduced SH culling expects active degree 3, got {model.active_sh_degree}"
        )
    degrees = ensure_degrees(model)
    install_prune_tracking(model)
    ops = _load_ops()

    camera_positions = torch.stack([camera.camera_center for camera in cameras])
    view_matrices = torch.stack([camera.world_view_transform for camera in cameras])
    projection_matrices = torch.stack([camera.full_proj_transform for camera in cameras])
    fovx = torch.tensor([camera.FoVx for camera in cameras], device="cuda")
    fovy = torch.tensor([camera.FoVy for camera in cameras], device="cuda")
    heights = torch.tensor(
        [camera.image_height for camera in cameras], device="cuda", dtype=torch.int32
    )
    widths = torch.tensor(
        [camera.image_width for camera in cameras], device="cuda", dtype=torch.int32
    )

    def calculate():
        return ops.calculate_colours_variance(
            camera_positions,
            model.get_xyz,
            model._opacity,
            model.get_scaling,
            model.get_rotation,
            view_matrices,
            projection_matrices,
            torch.tan(fovx * 0.5),
            torch.tan(fovy * 0.5),
            heights,
            widths,
            model.get_features,
            degrees,
            int(model.active_sh_degree),
        )

    _, weighted_variance, weighted_mean = calculate()
    weighted_std = weighted_variance.sqrt()
    weighted_std[weighted_std.isnan()] = 0
    uniform_mask = weighted_std.mean(dim=2).squeeze() < float(std_threshold)
    model._features_dc[uniform_mask] = (
        weighted_mean[uniform_mask] - 0.5
    ) / SH_C0
    degrees[uniform_mask] = 0
    model._features_rest[uniform_mask] = 0

    colour_distances, _, _ = calculate()
    colour_distances[colour_distances.isnan()] = 0
    threshold = float(cdist_threshold) * np.sqrt(3.0) / 255.0
    for degree in range(int(model.active_sh_degree) - 1, 0, -1):
        coefficient_count = (degree + 1) ** 2 - 1
        mask = colour_distances[:, degree] < threshold
        degrees[mask] = torch.minimum(
            torch.tensor(degree, dtype=torch.int32, device=degrees.device),
            degrees[mask],
        )
        model._features_rest[mask, coefficient_count:] = 0

    enforce_degree_mask(model)
    counts = torch.bincount(degrees.reshape(-1).long(), minlength=4).cpu().tolist()
    return {
        "counts": [int(value) for value in counts],
        "total": int(sum(counts)),
        "cdist_threshold": float(cdist_threshold),
        "effective_cdist_threshold": float(threshold),
        "std_threshold": float(std_threshold),
        "camera_count": int(len(cameras)),
    }


@torch.no_grad()
def save_degree_sidecar(model, path: str | os.PathLike[str], extra=None) -> None:
    degrees = ensure_degrees(model).detach().cpu()
    payload = {
        "schema": "reduced_full_sh_degrees_v1",
        "degrees": degrees,
        "counts": torch.bincount(degrees.reshape(-1).long(), minlength=4).tolist(),
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def save_grouped_ply(model, path: str | os.PathLike[str]) -> None:
    degrees = ensure_degrees(model).reshape(-1)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    elements = []
    for degree in range(4):
        mask = degrees == degree
        coefficient_count = (degree + 1) ** 2 - 1
        xyz = model._xyz[mask].detach().cpu().numpy()
        dc = model._features_dc[mask].detach().cpu().reshape(-1, 3).numpy()
        rest = (
            model._features_rest[mask, :coefficient_count]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacity = model._opacity[mask].detach().cpu().numpy()
        scaling = model._scaling[mask].detach().cpu().numpy()
        rotation = model._rotation[mask].detach().cpu().numpy()
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
        names.extend(f"f_rest_{index}" for index in range(coefficient_count * 3))
        names.extend(
            [
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ]
        )
        array = np.empty(int(mask.sum()), dtype=[(name, "f4") for name in names])
        attributes = np.concatenate((xyz, dc, rest, opacity, scaling, rotation), axis=1)
        array[:] = list(map(tuple, attributes))
        elements.append(PlyElement.describe(array, f"vertex_{degree}"))
    PlyData(elements).write(str(output))
