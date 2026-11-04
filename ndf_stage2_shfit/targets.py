"""Low-resolution error targets for Stage-2 NDF fitting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def downsample_mesh_error(
    mesh_rgb: torch.Tensor,
    ground_truth: torch.Tensor,
    mesh_mask: torch.Tensor,
    mesh_depth: torch.Tensor,
    factor: int,
    minimum_coverage: float = 0.75,
) -> dict[str, torch.Tensor]:
    """Area-downsample Mesh--GT RGB L1 and valid mesh depth.

    Averaging before thresholding removes isolated pixel residuals.  Pixels
    whose low-resolution footprint is mostly outside the mesh are excluded so
    mesh silhouettes and holes are not mislabeled as surface NDF evidence.
    """

    if factor < 1:
        raise ValueError("factor must be positive")
    if mesh_rgb.shape != ground_truth.shape or mesh_rgb.ndim != 3:
        raise ValueError("mesh_rgb and ground_truth must share [3,H,W]")
    if mesh_mask.ndim == 2:
        mesh_mask = mesh_mask.unsqueeze(0)
    if mesh_depth.ndim == 2:
        mesh_depth = mesh_depth.unsqueeze(0)
    if mesh_mask.shape != mesh_depth.shape or mesh_mask.shape[0] != 1:
        raise ValueError("mesh mask/depth must share [1,H,W]")

    height = max(1, mesh_rgb.shape[-2] // factor)
    width = max(1, mesh_rgb.shape[-1] // factor)
    output_size = (height, width)
    support = mesh_mask.float()
    raw_error = (mesh_rgb - ground_truth).abs().mean(dim=0, keepdim=True)
    support_low = F.interpolate(support.unsqueeze(0), output_size, mode="area").squeeze(0)
    error_numerator = F.interpolate(
        (raw_error * support).unsqueeze(0), output_size, mode="area"
    ).squeeze(0)
    raw_low = error_numerator / support_low.clamp_min(1e-6)
    depth_numerator = F.interpolate(
        (mesh_depth * support).unsqueeze(0), output_size, mode="area"
    ).squeeze(0)
    depth_low = depth_numerator / support_low.clamp_min(1e-6)
    valid = support_low >= float(minimum_coverage)
    raw_low = raw_low.masked_fill(~valid, 0.0)
    depth_low = depth_low.masked_fill(~valid, 0.0)
    gt_low = F.interpolate(
        ground_truth.unsqueeze(0), output_size, mode="area"
    ).squeeze(0)
    mesh_low = F.interpolate(mesh_rgb.unsqueeze(0), output_size, mode="area").squeeze(0)
    return {
        "raw_error": raw_low,
        "valid": valid,
        "mesh_depth": depth_low,
        "ground_truth": gt_low,
        "mesh_rgb": mesh_low,
        "support_fraction": support_low,
    }


def _gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Blur a single-channel ``[1,H,W]`` image without changing its shape."""

    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("Gaussian blur expects [1,H,W]")
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("Gaussian kernel size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")
    if kernel_size == 1:
        return image
    coordinate = torch.arange(
        kernel_size, device=image.device, dtype=image.dtype
    ) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    padded = F.pad(image.unsqueeze(0), (padding,) * 4, mode="reflect")
    return F.conv2d(padded, kernel_2d).squeeze(0)


def masked_gaussian_blur(
    image: torch.Tensor,
    valid: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Blur a scalar target without bleeding across invalid mesh pixels."""

    if image.shape != valid.shape or image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("image and valid must share [1,H,W]")
    valid_float = valid.to(image.dtype)
    numerator = _gaussian_blur(image * valid_float, kernel_size, sigma)
    denominator = _gaussian_blur(valid_float, kernel_size, sigma)
    return (numerator / denominator.clamp_min(1e-6)).clamp(0.0, 1.0).masked_fill(
        ~valid.bool(), 0.0
    )


def binary_downsample_gaussian_target(
    mesh_rgb: torch.Tensor,
    ground_truth: torch.Tensor,
    mesh_mask: torch.Tensor,
    factor: int,
    threshold: float = 0.01,
    kernel_size: int = 5,
    sigma: float = 1.0,
    minimum_coverage: float = 0.75,
) -> dict[str, torch.Tensor]:
    """Build a simple denoised NDF target from a hard photometric-error mask.

    The convention is the one used by dissolve/NDF rendering: pixels whose
    Mesh--GT RGB L1 error is *above* ``threshold`` are 1 (NDF required), and
    low-error pixels are 0.  The full-resolution error is thresholded first;
    this hard 0/1 mask is then area-downsampled and Gaussian blurred.  Thus
    there is no smooth activation band, while isolated high-resolution errors
    contribute only their fractional coverage after downsampling.  A
    normalized masked blur prevents invalid mesh holes from bleeding into
    valid surface.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("binary threshold must be in [0,1]")
    if factor < 1:
        raise ValueError("factor must be positive")
    if mesh_rgb.shape != ground_truth.shape or mesh_rgb.ndim != 3:
        raise ValueError("mesh_rgb and ground_truth must share [3,H,W]")
    if mesh_mask.ndim == 2:
        mesh_mask = mesh_mask.unsqueeze(0)
    if mesh_mask.ndim != 3 or mesh_mask.shape[0] != 1:
        raise ValueError("mesh_mask must be [1,H,W]")

    height = max(1, mesh_rgb.shape[-2] // factor)
    width = max(1, mesh_rgb.shape[-1] // factor)
    output_size = (height, width)
    support = mesh_mask.float()
    raw_error = (mesh_rgb - ground_truth).abs().mean(dim=0, keepdim=True)
    binary_full = (raw_error > float(threshold)).to(raw_error.dtype) * support
    support_low = F.interpolate(support.unsqueeze(0), output_size, mode="area").squeeze(0)
    error_numerator = F.interpolate(
        (raw_error * support).unsqueeze(0), output_size, mode="area"
    ).squeeze(0)
    raw_low = error_numerator / support_low.clamp_min(1e-6)
    binary_numerator = F.interpolate(
        binary_full.unsqueeze(0), output_size, mode="area"
    ).squeeze(0)
    binary_low = binary_numerator / support_low.clamp_min(1e-6)
    valid = support_low >= float(minimum_coverage)
    binary_low = binary_low * valid

    target = masked_gaussian_blur(binary_low, valid, kernel_size, sigma)
    return {
        "target": target,
        "binary_low": binary_low.masked_fill(~valid, 0.0),
        "raw_error": raw_low.masked_fill(~valid, 0.0),
        "valid": valid,
        "support_fraction": support_low,
    }


def smooth_threshold(error: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Exact 0/1 outside a loss band and smoothstep inside it."""

    if not 0.0 <= low < high <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
    normalized = ((error - float(low)) / float(high - low)).clamp(0.0, 1.0)
    return normalized.square() * (3.0 - 2.0 * normalized)
