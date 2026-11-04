"""CPU-only COLMAP track-angle contract for one-shot Gaussian initialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


ANGLE_THRESHOLD_DEG = 1.5
XYZ_ATOL = 5e-6
XYZ_RTOL = 1e-6


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("COLMAP image contains an invalid zero/non-finite quaternion")
    qw, qx, qy, qz = qvec / norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def _read_image_centers_text(path: Path) -> Dict[int, np.ndarray]:
    centers: Dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 10:
                raise ValueError(f"Malformed COLMAP images.txt image record: {stripped!r}")
            image_id = int(fields[0])
            if image_id in centers:
                raise ValueError(f"Duplicate IMAGE_ID {image_id} in {path}")
            qvec = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
            tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
            if not np.isfinite(tvec).all():
                raise ValueError(f"Non-finite translation for IMAGE_ID {image_id}")
            rotation = _qvec_to_rotmat(qvec)  # COLMAP world-to-camera rotation.
            centers[image_id] = -(rotation.T @ tvec)
            # The next physical line is the POINTS2D record and may be empty.
            if stream.readline() == "":
                raise ValueError(f"Missing POINTS2D line after IMAGE_ID {image_id}")
    if not centers:
        raise ValueError(f"No image poses found in {path}")
    return centers


def _iter_points3d_text(path: Path) -> Iterable[Tuple[int, np.ndarray, Tuple[int, ...]]]:
    seen_ids = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 8 or (len(fields) - 8) % 2 != 0:
                raise ValueError(f"Malformed COLMAP points3D.txt record at line {line_number}")
            point_id = int(fields[0])
            if point_id in seen_ids:
                raise ValueError(f"Duplicate POINT3D_ID {point_id} at line {line_number}")
            seen_ids.add(point_id)
            xyz = np.asarray([float(value) for value in fields[1:4]], dtype=np.float64)
            if not np.isfinite(xyz).all():
                raise ValueError(f"Non-finite XYZ for POINT3D_ID {point_id}")
            track_image_ids = tuple(int(fields[index]) for index in range(8, len(fields), 2))
            if len(set(track_image_ids)) != len(track_image_ids):
                raise ValueError(f"Duplicate IMAGE_ID in track for POINT3D_ID {point_id}")
            yield point_id, xyz, track_image_ids


def _max_pairwise_triangulation_angle_deg(point: np.ndarray, centers: np.ndarray) -> float:
    rays = centers - point[None, :]
    lengths = np.linalg.norm(rays, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("Point coincides with a camera center or produces a non-finite viewing ray")
    directions = rays / lengths[:, None]
    cosines = np.clip(directions @ directions.T, -1.0, 1.0)
    upper = cosines[np.triu_indices(cosines.shape[0], k=1)]
    return float(np.degrees(np.arccos(upper)).max())


def build_colmap_initialization_stability_mask(
    source_xyz,
    sparse_dir,
    train_center,
    cameras_extent: float,
    angle_threshold_deg: float = ANGLE_THRESHOLD_DEG,
):
    """Validate source row identity and return a stable-point mask in that exact order.

    The locked high-precision rule rejects a point only when it is farther than
    ``2 * cameras_extent`` from the training-camera center *and* its maximum
    pairwise triangulation angle is below 1.5 degrees. A single-view track has
    angle zero. This follows COLMAP's world-to-camera pose convention and
    triangulation-angle geometry. The function performs only CPU NumPy/file
    operations and never recomputes any Gaussian parameter.
    """
    if float(angle_threshold_deg) != ANGLE_THRESHOLD_DEG:
        raise ValueError(f"This locked contract requires angle_threshold_deg={ANGLE_THRESHOLD_DEG}")
    sparse_dir = Path(sparse_dir).resolve()
    points_path = sparse_dir / "points3D.txt"
    images_path = sparse_dir / "images.txt"
    for required in (points_path, images_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required COLMAP text model file is missing: {required}")

    source = np.asarray(source_xyz, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source_xyz must have shape [N,3], got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("source_xyz contains non-finite values")
    train_center = np.asarray(train_center, dtype=np.float64)
    if train_center.shape != (3,) or not np.isfinite(train_center).all():
        raise ValueError(f"train_center must be a finite [3] vector, got {train_center}")
    cameras_extent = float(cameras_extent)
    if not math.isfinite(cameras_extent) or cameras_extent <= 0.0:
        raise ValueError(f"cameras_extent must be positive and finite, got {cameras_extent}")

    centers = _read_image_centers_text(images_path)
    records = list(_iter_points3d_text(points_path))
    colmap_xyz64 = np.asarray([record[1] for record in records], dtype=np.float64).reshape(-1, 3)
    colmap_xyz32 = colmap_xyz64.astype(np.float32)
    if source.shape[0] != colmap_xyz32.shape[0]:
        raise RuntimeError(
            "Initialization-filter row contract failed: source Gaussian count "
            f"{source.shape[0]} != points3D.txt count {colmap_xyz32.shape[0]}"
        )
    # points3D.ply is written from these values as float32. Exact equality is
    # therefore the row-order contract; no nearest-neighbour remapping is allowed.
    row_match = (source == colmap_xyz32).all(axis=1)
    if not row_match.all():
        first = int(np.flatnonzero(~row_match)[0])
        delta = float(np.max(np.abs(source[first].astype(np.float64) - colmap_xyz64[first])))
        raise RuntimeError(
            "Initialization-filter row contract failed: source XYZ is not in exact "
            f"points3D.txt row order; first mismatch row={first}, max_abs_delta={delta:.9g}, "
            f"source={source[first].tolist()}, colmap={colmap_xyz64[first].tolist()}"
        )

    mask = np.ones(source.shape[0], dtype=np.bool_)
    max_angles = np.zeros(source.shape[0], dtype=np.float64)
    distances = np.linalg.norm(colmap_xyz64 - train_center[None, :], axis=1)
    outside_2r = distances > (2.0 * cameras_extent)
    single_view_or_empty = 0
    for row, (point_id, point, track_image_ids) in enumerate(records):
        missing = [image_id for image_id in track_image_ids if image_id not in centers]
        if missing:
            raise RuntimeError(
                f"POINT3D_ID {point_id} references missing IMAGE_ID(s) {missing} in images.txt"
            )
        if len(track_image_ids) < 2:
            single_view_or_empty += 1
            continue
        angle = _max_pairwise_triangulation_angle_deg(
            point, np.stack([centers[image_id] for image_id in track_image_ids], axis=0)
        )
        max_angles[row] = angle
    low_angle = max_angles < ANGLE_THRESHOLD_DEG
    rejected = outside_2r & low_angle
    mask[rejected] = False

    report = {
        "contract_version": 1,
        "sparse_dir": str(sparse_dir),
        "angle_threshold_deg": ANGLE_THRESHOLD_DEG,
        "distance_threshold_in_cameras_extent": 2.0,
        "train_center": train_center.tolist(),
        "cameras_extent": cameras_extent,
        "xyz_atol": XYZ_ATOL,
        "xyz_rtol": XYZ_RTOL,
        "source_count": int(source.shape[0]),
        "stable_count": int(mask.sum()),
        "rejected_count": int((~mask).sum()),
        "track_length_lt_2_angle_zero_count": int(single_view_or_empty),
        "outside_2r_count": int(outside_2r.sum()),
        "outside_2r_low_angle_rejected_count": int(rejected.sum()),
        "outside_2r_high_angle_retained_count": int((outside_2r & ~low_angle).sum()),
        "inside_or_on_2r_low_angle_retained_count": int((~outside_2r & low_angle).sum()),
        "rejected_distance_min": float(distances[rejected].min()) if rejected.any() else None,
        "rejected_distance_max": float(distances[rejected].max()) if rejected.any() else None,
        "rejected_max_angle_deg_max": float(max_angles[rejected].max()) if rejected.any() else None,
        "points3D_txt_sha256": _sha256_file(points_path),
        "images_txt_sha256": _sha256_file(images_path),
        "source_xyz_float32_sha256": _sha256_array(source),
        "colmap_xyz_float32_sha256": _sha256_array(colmap_xyz32),
        "stable_mask_bool_sha256": _sha256_array(mask),
    }
    return mask, report


def write_initialization_filter_audit(mask: np.ndarray, report: dict, output_dir) -> Tuple[str, str]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mask_path = output / "initialization_stability_mask.npy"
    report_path = output / "filter_manifest.json"
    rejected_path = output / "rejected_source_indices.csv"
    np.save(mask_path, np.asarray(mask, dtype=np.bool_), allow_pickle=False)
    rejected = np.flatnonzero(~np.asarray(mask, dtype=np.bool_))
    np.savetxt(
        rejected_path,
        np.column_stack([rejected, rejected + 1]),
        delimiter=",",
        header="zero_based_source_row,one_based_source_row",
        comments="",
        fmt="%d",
    )
    report = dict(report)
    report["mask_npy"] = str(mask_path)
    report["rejected_indices_csv"] = str(rejected_path)
    report["rejected_indices_csv_sha256"] = _sha256_file(rejected_path)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return str(mask_path), str(report_path)
