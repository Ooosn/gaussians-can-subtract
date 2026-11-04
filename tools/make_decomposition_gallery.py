"""Build a train/test decomposition gallery from existing rendered PNGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLUMNS = (
    ("GT.png", "GT"),
    ("Mesh.png", "Mesh"),
    ("Ours.png", "Ours"),
    ("NDF.png", "Diss."),
    ("Residual.png", "Residual"),
)
SCENE_ORDER = ("HY01", "Playroom", "DrJohnson", "SaintAnne", "Bridge", "Museum-1", "Museum-2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf", "arialbd.ttf")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def view_metrics(view: Path) -> dict[str, float]:
    with (
        Image.open(view / "GT.png") as gt_image,
        Image.open(view / "Ours.png") as ours_image,
        Image.open(view / "Mesh.png") as mesh_image,
    ):
        gt = gt_image.convert("RGB")
        ours = ours_image.convert("RGB")
        mesh = mesh_image.convert("RGB")
        width = 256
        height = max(1, round(ours.height * width / ours.width))
        gt_array = np.asarray(gt.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        ours_array = np.asarray(ours.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        mesh_array = np.asarray(mesh.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    return {
        "activity": float(np.abs(ours_array - mesh_array).mean() / 255.0),
        "ours_gt_mae": float(np.abs(ours_array - gt_array).mean() / 255.0),
        "ours_std": float(ours_array.std() / 255.0),
    }


def select_view(split_root: Path) -> tuple[Path, dict[str, float]]:
    candidates = []
    for view in sorted(path for path in split_root.iterdir() if path.is_dir()):
        if all((view / filename).is_file() for filename, _label in COLUMNS):
            metrics = view_metrics(view)
            # Reject stale/failed renders such as uniform grey placeholders.
            if metrics["ours_std"] < 0.02 or metrics["ours_gt_mae"] > 0.15:
                continue
            candidates.append((metrics["activity"], view, metrics))
    if not candidates:
        raise FileNotFoundError(f"No complete decomposition views under {split_root}")
    _score, view, metrics = max(candidates, key=lambda item: (item[0], item[1].name))
    return view, metrics


def resize_rgb(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size == (width, height):
            return image.copy()
        return image.resize((width, height), Image.Resampling.LANCZOS)


def render_scene(scene: str, selected: dict[str, Path], output: Path) -> Path:
    with Image.open(selected["train"] / "GT.png") as reference:
        aspect = reference.height / reference.width
    cell_width = 460
    cell_height = max(1, round(cell_width * aspect))
    left = 150
    header = 58
    gap = 8
    row_gap = 10
    width = left + len(COLUMNS) * cell_width + (len(COLUMNS) - 1) * gap
    height = header + 2 * cell_height + row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(28, bold=True)
    row_font = font(25, bold=True)
    for column, (_filename, label) in enumerate(COLUMNS):
        x = left + column * (cell_width + gap)
        box = draw.textbbox((0, 0), label, font=header_font)
        draw.text((x + (cell_width - (box[2] - box[0])) / 2, 12), label, fill="black", font=header_font)
    for row, split in enumerate(("train", "test")):
        y = header + row * (cell_height + row_gap)
        row_label = f"{split.title()}  {selected[split].name}"
        box = draw.textbbox((0, 0), row_label, font=row_font)
        draw.text((left - 12 - (box[2] - box[0]), y + (cell_height - (box[3] - box[1])) / 2), row_label, fill="black", font=row_font)
        for column, (filename, _label) in enumerate(COLUMNS):
            x = left + column * (cell_width + gap)
            image = resize_rgb(selected[split] / filename, cell_width, cell_height)
            canvas.paste(image, (x, y))
    target = output / "scenes" / f"{scene}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, optimize=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.source_root).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    output.mkdir(parents=True)

    scene_records = []
    scene_images = []
    for scene in SCENE_ORDER:
        scene_root = source / scene
        if not scene_root.is_dir():
            raise FileNotFoundError(scene_root)
        train, train_metrics = select_view(scene_root / "train")
        test, test_metrics = select_view(scene_root / "test")
        selected = {"train": train, "test": test}
        copied = {}
        for split, view in selected.items():
            selected_root = output / "selected" / scene / split
            selected_root.mkdir(parents=True, exist_ok=True)
            copied[split] = {}
            for filename, _label in COLUMNS:
                destination = selected_root / filename
                shutil.copy2(view / filename, destination)
                copied[split][filename] = {
                    "path": str(destination),
                    "sha256": sha256(destination),
                }
        scene_image = render_scene(scene, selected, output)
        scene_images.append((scene, scene_image))
        scene_records.append(
            {
                "scene": scene,
                "train": {"view": train.name, **train_metrics, "source": str(train)},
                "test": {"view": test.name, **test_metrics, "source": str(test)},
                "selected_files": copied,
                "figure": {"path": str(scene_image), "sha256": sha256(scene_image)},
            }
        )

    opened = [(scene, Image.open(path).convert("RGB")) for scene, path in scene_images]
    title_height = 54
    gap = 18
    width = max(image.width for _scene, image in opened)
    height = sum(title_height + image.height for _scene, image in opened) + gap * (len(opened) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = font(30, bold=True)
    y = 0
    for index, (scene, image) in enumerate(opened):
        draw.text((10, y + 8), scene, fill="black", font=title_font)
        y += title_height
        sheet.paste(image, (0, y))
        y += image.height
        if index + 1 < len(opened):
            y += gap
        image.close()
    sheet_path = output / "gaussolve_decomposition_all_scenes.png"
    sheet.save(sheet_path, optimize=True)
    manifest = {
        "schema": "gaussolve_decomposition_gallery_v1",
        "status": "PASS",
        "selection": "maximum mean absolute RGB difference between Ours and Mesh within each split",
        "columns": [label for _filename, label in COLUMNS],
        "row_order": ["train", "test"],
        "scene_order": list(SCENE_ORDER),
        "scenes": scene_records,
        "contact_sheet": {"path": str(sheet_path), "sha256": sha256(sheet_path)},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "scenes": len(scene_records)}))


if __name__ == "__main__":
    main()
