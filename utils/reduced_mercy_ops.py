"""Reduced-3DGS redundancy/mercy pruning for the main Gaussian field.

The CUDA symbols are imported lazily from the isolated Reduced GS staging
environment.  This keeps CPU/static tests importable and avoids changing the
shared renderer or GaussianModel.  The metric and mask follow the official
Reduced GS implementation:

* minimum projected pixel size over all training cameras;
* 30-neighbour sphere/ellipsoid intersection redundancy;
* minimum redundancy propagation through intersecting neighbours;
* ``mean + lambda * std`` threshold with a minimum floor; and
* the official ``redundancy_opacity_opacity`` mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ReducedMercyConfig:
    """Configuration matching the official Reduced GS ``full_final`` arm."""

    lambda_mercy: float = 1.0
    mercy_minimum: int = 3
    box_size: float = 1.0
    mercy_interval: int = 10
    num_neighbours: int = 30
    mercy_type: str = "redundancy_opacity_opacity"

    def trigger_interval(self, densification_interval: int) -> int:
        if int(densification_interval) <= 0:
            raise ValueError("densification_interval must be positive")
        if int(self.mercy_interval) <= 0:
            raise ValueError("mercy_interval must be positive")
        return int(self.mercy_interval) * int(densification_interval)


def mercy_prune_mask(
    opacity: torch.Tensor,
    redundancy_values: torch.Tensor,
    *,
    lambda_mercy: float = 1.0,
    mercy_minimum: int = 3,
    mercy_type: str = "redundancy_opacity_opacity",
    random_generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return the exact Reduced GS mercy mask and threshold diagnostics.

    ``opacity`` and ``redundancy_values`` may be ``[P]`` or ``[P, 1]``.  The
    returned mask is one-dimensional and lives on the input device.  The
    default path mirrors ``GaussianModel.mercy_points`` in the official
    checkout, including the unbiased ``torch.var`` and opacity quantiles.
    """

    if not torch.is_tensor(opacity) or not torch.is_tensor(redundancy_values):
        raise TypeError("opacity and redundancy_values must be tensors")
    opacity = opacity.reshape(-1)
    redundancy_values = redundancy_values.reshape(-1)
    if opacity.numel() != redundancy_values.numel():
        raise ValueError("opacity and redundancy_values must have the same length")
    if opacity.numel() == 0:
        raise ValueError("mercy pruning requires at least one Gaussian")
    if not torch.isfinite(opacity).all() or not torch.isfinite(redundancy_values).all():
        raise ValueError("opacity and redundancy_values must be finite")
    if float(lambda_mercy) < 0.0:
        raise ValueError("lambda_mercy must be non-negative")
    if int(mercy_minimum) < 0:
        raise ValueError("mercy_minimum must be non-negative")
    valid_types = {
        "redundancy_opacity",
        "redundancy_random",
        "opacity",
        "redundancy_opacity_opacity",
    }
    if mercy_type not in valid_types:
        raise ValueError(f"unsupported mercy_type: {mercy_type!r}")

    redundancy_float = redundancy_values.float()
    mean = redundancy_float.mean(dim=0, keepdim=True)
    # Official Reduced GS uses Tensor.var() with the default unbiased=True.
    std = redundancy_float.var(dim=0, keepdim=True).sqrt()
    threshold_value = mean + float(lambda_mercy) * std
    if not torch.isfinite(threshold_value).all():
        raise ValueError("redundancy threshold is non-finite")
    redundancy_threshold = max(float(threshold_value.item()), float(mercy_minimum))

    mask = redundancy_float > redundancy_threshold
    opacity_threshold = 0.0
    if mercy_type == "redundancy_opacity":
        if mask.any():
            candidates = mask.clone()
            candidates[candidates.clone()] = (
                opacity[candidates] < opacity[candidates].median()
            )
            mask = candidates
    elif mercy_type == "redundancy_random":
        if mask.any():
            candidate_count = int(mask.sum().item())
            random_values = torch.rand(
                candidate_count,
                device=opacity.device,
                generator=random_generator,
            )
            mask[mask.clone()] = random_values < 0.5
    elif mercy_type == "opacity":
        opacity_threshold = float(opacity.quantile(0.045).item())
        mask = opacity < opacity_threshold
    else:
        # Official option 3: lower-opacity half among redundant points, plus
        # the global lowest-opacity 3% capped at 0.05.
        if mask.any():
            candidates = mask.clone()
            candidates[candidates.clone()] = (
                opacity[candidates] < opacity[candidates].median()
            )
            mask = candidates
        global_threshold = torch.minimum(
            opacity.quantile(0.03),
            torch.tensor(0.05, device=opacity.device, dtype=opacity.dtype),
        )
        opacity_threshold = float(global_threshold.item())
        mask = torch.logical_or(mask, opacity < global_threshold)

    return mask, {
        "redundancy_mean": float(mean.item()),
        "redundancy_std": float(std.item()),
        "redundancy_threshold": redundancy_threshold,
        "opacity_threshold": opacity_threshold,
    }


def reduced_mercy_trigger_iterations(
    hybrid_start_iteration: int,
    main_densify_until_iteration: int,
    *,
    iterations: int = 30_000,
    densification_interval: int = 100,
    mercy_interval: int = 10,
    opacity_reset_interval: int = 3_000,
) -> list[int]:
    """Return the official-order mercy trigger iterations for active main GS.

    Reduced GS calls mercy every ``mercy_interval * densification_interval``
    steps, but skips an opacity-reset boundary while still inside the active
    densification phase.  The remaining ``iteration <= fine_tune_start``
    condition is represented by ``iterations - 3000`` here; for the H9
    main-cutoff at 15k it is non-binding, while the 12k reset boundary is
    explicitly excluded.
    """

    if int(hybrid_start_iteration) < 0:
        raise ValueError("hybrid_start_iteration must be non-negative")
    if int(main_densify_until_iteration) <= int(hybrid_start_iteration):
        return []
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    if int(densification_interval) <= 0 or int(mercy_interval) <= 0:
        raise ValueError("densification_interval and mercy_interval must be positive")
    if int(opacity_reset_interval) <= 0:
        raise ValueError("opacity_reset_interval must be positive")
    trigger_interval = int(densification_interval) * int(mercy_interval)
    fine_tune_start = int(iterations) - 3_000
    return [
        iteration
        for iteration in range(int(hybrid_start_iteration) + 1, int(main_densify_until_iteration))
        if iteration % trigger_interval == 0
        and iteration <= fine_tune_start
        and (
            iteration >= fine_tune_start
            or iteration % int(opacity_reset_interval) != 0
        )
    ]


def calculate_reduced_redundancy_metric(
    gaussians: Any,
    cameras: Sequence[Any],
    *,
    pixel_scale: float = 1.0,
    num_neighbours: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the official Reduced GS redundancy metric for current tensors.

    The caller must run with the isolated Reduced GS extension directory first
    on ``PYTHONPATH``.  The current mesh3dgs Camera object does not cache an
    inverse projection, so this helper computes it from ``full_proj_transform``
    exactly once per camera before calling the staged CUDA helper.
    """

    if not cameras:
        raise ValueError("at least one training camera is required")
    if float(pixel_scale) <= 0.0:
        raise ValueError("pixel_scale must be positive")
    if int(num_neighbours) <= 0:
        raise ValueError("num_neighbours must be positive")

    means3d = gaussians.get_xyz
    if not torch.is_tensor(means3d) or means3d.ndim != 2 or means3d.shape[1] != 3:
        raise ValueError("gaussians.get_xyz must have shape [P, 3]")
    point_count = int(means3d.shape[0])
    if point_count <= int(num_neighbours):
        raise ValueError(
            f"Reduced GS neighbour metric requires P > num_neighbours; got {point_count} and {num_neighbours}"
        )

    # Lazy imports are intentional: CPU/static tests must not require a CUDA
    # extension, while runtime PYTHONPATH selects the isolated Reduced build.
    from diff_gaussian_rasterization._C import (  # type: ignore[import-not-found]
        allocate_minimum_redundancy_value,
        find_minimum_projected_pixel_size,
        sphere_ellipsoid_intersection,
    )
    from simple_knn._C import distIndex2  # type: ignore[import-not-found]

    device = means3d.device
    dtype = means3d.dtype
    projection = torch.stack(
        [camera.full_proj_transform.to(device=device, dtype=dtype) for camera in cameras],
        dim=0,
    ).contiguous()
    inverse_projection = torch.linalg.inv(projection).contiguous()
    image_height = torch.tensor(
        [int(camera.image_height) for camera in cameras],
        device=device,
        dtype=torch.int32,
    )
    image_width = torch.tensor(
        [int(camera.image_width) for camera in cameras],
        device=device,
        dtype=torch.int32,
    )

    cube_size = find_minimum_projected_pixel_size(
        projection,
        inverse_projection,
        means3d,
        image_height,
        image_width,
    )
    half_diagonal = cube_size * float(pixel_scale) * (3.0**0.5) / 2.0
    _, neighbour_indices = distIndex2(means3d, int(num_neighbours))
    neighbour_indices = neighbour_indices.reshape(-1, int(num_neighbours)).to(torch.int32)

    redundancy_values, intersection_mask = sphere_ellipsoid_intersection(
        means3d,
        gaussians.get_scaling,
        gaussians.get_rotation,
        neighbour_indices,
        half_diagonal,
        int(num_neighbours),
    )
    redundancy_values = redundancy_values + 1
    self_indices = torch.arange(point_count, device=device, dtype=torch.int32).view(-1, 1)
    neighbour_indices = torch.cat((self_indices, neighbour_indices), dim=1)
    self_intersection = torch.ones(
        (point_count, 1), device=device, dtype=torch.bool
    )
    intersection_mask = torch.cat((self_intersection, intersection_mask), dim=1)
    minimum_redundancy = allocate_minimum_redundancy_value(
        redundancy_values,
        neighbour_indices,
        intersection_mask,
        int(num_neighbours) + 1,
    )[0]
    return minimum_redundancy, cube_size


def apply_reduced_mercy(
    gaussians: Any,
    cameras: Sequence[Any],
    config: ReducedMercyConfig = ReducedMercyConfig(),
) -> Dict[str, float | int]:
    """Compute and apply main-GS Reduced mercy pruning, returning CSV stats."""

    n_before = int(gaussians.get_xyz.shape[0])
    redundancy_values, cube_size = calculate_reduced_redundancy_metric(
        gaussians,
        cameras,
        pixel_scale=float(config.box_size),
        num_neighbours=int(config.num_neighbours),
    )
    mask, threshold_stats = mercy_prune_mask(
        gaussians.get_opacity,
        redundancy_values,
        lambda_mercy=float(config.lambda_mercy),
        mercy_minimum=int(config.mercy_minimum),
        mercy_type=config.mercy_type,
    )
    mercied = int(mask.sum().item())
    if mercied:
        pruned = int(gaussians.prune_points(mask))
        if pruned != mercied:
            raise RuntimeError(
                f"Reduced mercy mask/prune mismatch: mask={mercied}, pruned={pruned}"
            )
    n_after = int(gaussians.get_xyz.shape[0])
    return {
        "n_before": n_before,
        "redundancy_threshold": threshold_stats["redundancy_threshold"],
        "opacity_threshold": threshold_stats["opacity_threshold"],
        "n_mercied": mercied,
        "n_after": n_after,
        "cube_size_mean": float(cube_size.float().mean().item()),
    }
