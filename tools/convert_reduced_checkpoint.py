"""CPU-only conversion of an unquantised Reduced-GS grouped PLY.

Reduced 3DGS stores one PLY element per SH degree.  The main hybrid loader
expects one ordinary ``vertex`` element with full SH3 coefficients, so this
converter expands ``vertex_0`` ... ``vertex_3`` into that ABI.  It deliberately
rejects quantised/codebook PLYs: resolving a codebook is a separate semantic
operation and must never happen implicitly during a handoff conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement


BASE_PROPERTIES = [
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
]
TAIL_PROPERTIES = [
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]
STANDARD_PROPERTIES = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + BASE_PROPERTIES[3:]
    + [f"f_rest_{i}" for i in range(45)]
    + TAIL_PROPERTIES
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _band_degree(name: str) -> int:
    if not name.startswith("vertex_"):
        raise ValueError(f"unexpected PLY element {name!r}; expected vertex_0..vertex_3")
    try:
        degree = int(name.split("_", 1)[1])
    except ValueError as exc:
        raise ValueError(f"invalid grouped PLY element name: {name!r}") from exc
    if degree not in range(4):
        raise ValueError(f"grouped PLY element {name!r} is outside SH3")
    return degree


def _validate_element_properties(element: Any, degree: int) -> dict[str, Any]:
    names = [prop.name for prop in element.properties]
    expected_rest = 3 * ((degree + 1) ** 2 - 1)
    if names[: len(BASE_PROPERTIES)] != BASE_PROPERTIES:
        raise ValueError(
            f"{element.name}: base property order mismatch; got {names[:len(BASE_PROPERTIES)]}"
        )
    if len(names) < len(BASE_PROPERTIES) + len(TAIL_PROPERTIES):
        raise ValueError(f"{element.name}: missing opacity/scale/rotation properties")
    tail_start = len(names) - len(TAIL_PROPERTIES)
    if names[tail_start:] != TAIL_PROPERTIES:
        raise ValueError(
            f"{element.name}: tail property order mismatch; got {names[tail_start:]}"
        )
    rest_names = names[len(BASE_PROPERTIES) : tail_start]
    expected_names = [f"f_rest_{i}" for i in range(len(rest_names))]
    if rest_names != expected_names:
        raise ValueError(f"{element.name}: f_rest properties must be contiguous from f_rest_0")
    if len(rest_names) > expected_rest:
        raise ValueError(
            f"{element.name}: {len(rest_names)} f_rest values exceed SH{degree} capacity {expected_rest}"
        )
    for prop in element.properties:
        if np.dtype(prop.val_dtype) != np.dtype("f4"):
            raise ValueError(
                f"{element.name}.{prop.name}: expected float32 unquantised data, got {prop.val_dtype}"
            )
    return {"degree": degree, "count": int(element.count), "rest_count": len(rest_names)}


def _finite(element: Any, property_name: str) -> None:
    values = np.asarray(element[property_name])
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values in {element.name}.{property_name}")


def _validate_grouped_input(ply: PlyData) -> tuple[list[Any], list[dict[str, Any]]]:
    elements = list(ply.elements)
    if not elements:
        raise ValueError("grouped PLY has no elements")
    if any(element.name == "codebook_centers" for element in elements):
        raise ValueError(
            "quantised/codebook PLY is unsupported; convert the unquantised grouped point_cloud.ply"
        )
    if any(not element.name.startswith("vertex_") for element in elements):
        raise ValueError("grouped PLY may contain only vertex_0..vertex_3 elements")
    degrees = [_band_degree(element.name) for element in elements]
    if degrees != sorted(degrees) or len(set(degrees)) != len(degrees):
        raise ValueError("grouped PLY vertex elements must be unique and ordered by SH degree")
    metadata = [_validate_element_properties(element, degree) for element, degree in zip(elements, degrees)]
    total = sum(item["count"] for item in metadata)
    if total <= 0:
        raise ValueError("grouped PLY contains no Gaussian rows")
    for element in elements:
        for prop in element.properties:
            _finite(element, prop.name)
    return elements, metadata


def _standard_dtype() -> np.dtype:
    return np.dtype([(name, "f4") for name in STANDARD_PROPERTIES])


def validate_mesh3dgs_abi(
    path: Path,
    *,
    expected_count: int | None = None,
    loader_source: Path | None = None,
) -> dict[str, Any]:
    """Validate the CPU-visible contract consumed by mesh3dgs ``load_ply``.

    ``GaussianModel.load_ply`` allocates CUDA tensors, so this function does
    not instantiate it.  Instead it performs the complete PLY/property/finite
    check and verifies the current loader source reads the first ``vertex``
    element and requests the exact SH3 field names.
    """
    ply = PlyData.read(str(path), mmap=False)
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError("standard handoff must contain exactly one vertex element")
    element = ply.elements[0]
    names = [prop.name for prop in element.properties]
    if names != STANDARD_PROPERTIES:
        raise ValueError("standard handoff property order does not match mesh3dgs SH3 ABI")
    if expected_count is not None and int(element.count) != int(expected_count):
        raise ValueError(f"standard handoff count mismatch: {element.count} != {expected_count}")
    for prop in element.properties:
        if np.dtype(prop.val_dtype) != np.dtype("f4"):
            raise ValueError(f"standard handoff property {prop.name} is not float32")
        _finite(element, prop.name)
    loader_contract = None
    if loader_source is not None:
        source = loader_source.read_text(encoding="utf-8")
        start = source.find("def load_ply")
        if start < 0:
            raise ValueError(f"mesh3dgs loader has no load_ply: {loader_source}")
        body = source[start:]
        required = [
            "plydata.elements[0][\"x\"]",
            "plydata.elements[0][\"f_dc_0\"]",
            'startswith("f_rest_")',
            "expected_count = 3*(self.max_sh_degree + 1) ** 2 - 3",
            "plydata.elements[0][\"opacity\"]",
            "scale_names = [p.name for p in plydata.elements[0].properties",
            "rot_names = [p.name for p in plydata.elements[0].properties",
        ]
        missing = [token for token in required if token not in body]
        if missing:
            raise ValueError(f"mesh3dgs loader contract tokens missing: {missing}")
        loader_contract = {
            "loader_source": str(loader_source),
            "loader_reads_first_element": True,
            "cuda_runtime_instantiation": False,
        }
    return {
        "path": str(path),
        "element": element.name,
        "count": int(element.count),
        "properties": names,
        "sha256": sha256_file(path),
        "mesh3dgs_loader": loader_contract,
    }


def convert_grouped_to_standard(
    input_path: Path,
    output_path: Path,
    *,
    expected_count: int | None = None,
    report_path: Path | None = None,
    loader_source: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite handoff: {output_path}")
    ply = PlyData.read(str(input_path), mmap=False)
    elements, metadata = _validate_grouped_input(ply)
    total = sum(item["count"] for item in metadata)
    if expected_count is not None and total != int(expected_count):
        raise ValueError(f"grouped total count mismatch: {total} != {expected_count}")

    output = np.empty(total, dtype=_standard_dtype())
    offset = 0
    band_report = []
    for element, item in zip(elements, metadata):
        count = item["count"]
        rows = output[offset : offset + count]
        for name in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"):
            rows[name] = np.asarray(element[name], dtype=np.float32)
        rows["nx"] = 0.0
        rows["ny"] = 0.0
        rows["nz"] = 0.0
        for index in range(45):
            source_name = f"f_rest_{index}"
            if index < item["rest_count"]:
                rows[source_name] = np.asarray(element[source_name], dtype=np.float32)
            else:
                rows[source_name] = 0.0
        for name in TAIL_PROPERTIES:
            rows[name] = np.asarray(element[name], dtype=np.float32)
        for name in STANDARD_PROPERTIES:
            if not np.isfinite(rows[name]).all():
                raise ValueError(f"non-finite values after expanding {element.name}.{name}")
        band_report.append(
            {
                "element": element.name,
                "degree": item["degree"],
                "count": count,
                "input_rest_count": item["rest_count"],
                "output_rest_count": 45,
            }
        )
        offset += count
    if offset != total:
        raise AssertionError(f"internal row offset mismatch: {offset} != {total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(output, "vertex")], text=False).write(str(output_path))
    abi = validate_mesh3dgs_abi(
        output_path,
        expected_count=total,
        loader_source=loader_source,
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "output": abi,
        "grouped_total_count": total,
        "bands": band_report,
        "normals": "synthesised float32 zeros; Reduced grouped input has no normals",
        "codebook_policy": "rejected when embedded; unquantised grouped PLY only",
        "roundtrip": abi["mesh3dgs_loader"],
    }
    if report_path is not None:
        report_path = report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--mesh3dgs-loader", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = convert_grouped_to_standard(
        args.input,
        args.output,
        expected_count=args.expected_count,
        report_path=args.report,
        loader_source=args.mesh3dgs_loader,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
