"""Pure helpers for the isolated seeding correction.

This module is deliberately independent from training losses and render code.
It converts hybrid premultiplied linear-depth moments to metric camera-z and
counts distinct-camera support for voxelized seed candidates.
"""

import json
import math
import os

import torch


def hybrid_metric_depth(
    front_depth_moment,
    front_transmittance,
    mesh_depth,
    mesh_mask,
    alpha_eff,
    total_depth_moment,
    total_transmittance,
    *,
    min_opacity_mass=1e-4,
    eps=1e-6,
):
    """Return ``(metric_z, effective_mass, valid)`` for the hybrid ray.

    ``front_depth_moment`` and ``total_depth_moment`` are raw linear-depth
    moments, i.e. sums of ``T_i * alpha_i * z_i``.  With ``a=alpha_eff``:

      M = a*(M_front + T_front*mesh_mask*z_mesh) + (1-a)*M_total
      A = a*((1-T_front) + T_front*mesh_mask) + (1-a)*(1-T_total)
      z = M/A

    Thus A includes the front Gaussians plus opaque mesh in the kept-mesh
    branch and the complete Gaussian stream in the dissolved branch.
    """
    dtype = front_depth_moment.dtype
    device = front_depth_moment.device
    front_transmittance = front_transmittance.clamp(0.0, 1.0)
    total_transmittance = total_transmittance.clamp(0.0, 1.0)
    mesh_depth = mesh_depth.to(device=device, dtype=dtype)
    mesh_mask = mesh_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    alpha_eff = torch.as_tensor(alpha_eff, device=device, dtype=dtype).clamp(0.0, 1.0)

    kept_moment = front_depth_moment + front_transmittance * mesh_mask * mesh_depth
    kept_mass = (1.0 - front_transmittance) + front_transmittance * mesh_mask
    total_mass = 1.0 - total_transmittance
    hybrid_moment = alpha_eff * kept_moment + (1.0 - alpha_eff) * total_depth_moment
    effective_mass = alpha_eff * kept_mass + (1.0 - alpha_eff) * total_mass

    valid = (
        torch.isfinite(hybrid_moment)
        & torch.isfinite(effective_mass)
        & (effective_mass >= float(min_opacity_mass))
    )
    metric_depth = hybrid_moment / effective_mass.clamp_min(float(eps))
    valid = valid & torch.isfinite(metric_depth) & (metric_depth > 0.0)
    metric_depth = torch.where(valid, metric_depth, torch.zeros_like(metric_depth))
    return metric_depth, effective_mass, valid


def hybrid_depth_for_seeding(
    front_depth_moment,
    front_transmittance,
    mesh_depth,
    mesh_mask,
    alpha_eff,
    total_depth_moment,
    total_transmittance,
    *,
    mode="conditional",
    min_opacity_mass=1e-4,
    eps=1e-6,
):
    """Select raw or corrected z while keeping one shared hybrid contract.

    ``raw`` reproduces the historical premultiplied hybrid moment.  It does
    not divide by mass.  ``conditional`` divides that exact same moment by
    the effective mass.  Returning both modes is for controlled 2x2 tests;
    only ``conditional`` is the correction.
    """
    if mode not in {"raw", "conditional"}:
        raise ValueError("seed_depth_mode must be 'raw' or 'conditional'")
    dtype = front_depth_moment.dtype
    device = front_depth_moment.device
    front_t = front_transmittance.clamp(0.0, 1.0)
    mesh_z = mesh_depth.to(device=device, dtype=dtype)
    mesh_valid = mesh_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    keep_weight = torch.as_tensor(alpha_eff, device=device, dtype=dtype).clamp(0.0, 1.0)
    raw_moment = (
        keep_weight * (front_depth_moment + front_t * mesh_valid * mesh_z)
        + (1.0 - keep_weight) * total_depth_moment
    )
    conditional, mass, conditional_valid = hybrid_metric_depth(
        front_depth_moment,
        front_transmittance,
        mesh_depth,
        mesh_mask,
        alpha_eff,
        total_depth_moment,
        total_transmittance,
        min_opacity_mass=min_opacity_mass,
        eps=eps,
    )
    if mode == "conditional":
        return conditional, mass, conditional_valid
    raw_valid = torch.isfinite(raw_moment) & (raw_moment > 0.0)
    raw_moment = torch.where(raw_valid, raw_moment, torch.zeros_like(raw_moment))
    return raw_moment, mass, raw_valid


def distinct_camera_support(voxel_inverse, camera_ids, num_voxels):
    """Count unique camera IDs supporting every voxel."""
    if voxel_inverse.numel() == 0 or int(num_voxels) == 0:
        return torch.zeros(
            (int(num_voxels),), device=voxel_inverse.device, dtype=torch.long
        )
    pairs = torch.stack(
        [voxel_inverse.to(torch.long), camera_ids.to(torch.long)], dim=1
    )
    unique_pairs = torch.unique(pairs, dim=0)
    return torch.bincount(unique_pairs[:, 0], minlength=int(num_voxels))


def backproject_sampled_mesh_depth(
    u_idx,
    v_idx,
    depth_linear,
    mesh_mask,
    *,
    width,
    height,
    fov_x,
    fov_y,
    camera_to_world,
    min_depth=1e-5,
):
    """Back-project the historical mesh-depth NDF candidates.

    This helper intentionally has no main-Gaussian validity/mass input.  NDF
    candidates therefore stay identical for raw and conditional main-GS modes.
    """
    if depth_linear.dim() == 3:
        depth_linear = depth_linear.squeeze(0)
    if mesh_mask.dim() == 3:
        mesh_mask = mesh_mask.squeeze(0)
    dtype = depth_linear.dtype
    device = depth_linear.device
    z = (depth_linear * mesh_mask)[v_idx, u_idx]
    valid = torch.isfinite(z) & (z > float(min_depth))
    if not valid.any():
        return torch.empty((0, 3), device=device, dtype=dtype), valid

    u = u_idx[valid]
    v = v_idx[valid]
    z = z[valid]
    u_ndc = (u.to(dtype) + 0.5) / int(width) * 2.0 - 1.0
    v_ndc = (v.to(dtype) + 0.5) / int(height) * 2.0 - 1.0
    x_cam = u_ndc * z * math.tan(float(fov_x) * 0.5)
    y_cam = v_ndc * z * math.tan(float(fov_y) * 0.5)
    p_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)
    c2w = camera_to_world.to(device=device, dtype=dtype)
    return (p_cam @ c2w)[:, :3], valid


def write_json_atomic(path, payload):
    """Write one diagnostic artifact without touching model/checkpoint files."""
    if not path:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
