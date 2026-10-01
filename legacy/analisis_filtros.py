"""Analisis y preprocesamiento del New Plant Diseases Dataset.

Uso:
    python analisis_filtros.py --dataset dataset --salida resultados

El script genera:
* dataset_stats.json con el resumen del conjunto de imagenes.
* resultados/filtros_01.png ... resultados/filtros_03.png con comparaciones.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(dataset: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def classify_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if "valid" in parts or "validation" in parts:
        return "valid"
    if "test" in parts:
        return "test"
    return "unknown"


def summarize(images: list[Path]) -> dict:
    split_counts = Counter()
    class_counts = Counter()
    formats = Counter()
    sizes = Counter()
    color_modes = Counter()
    errors = []

    for path in images:
        split = classify_split(path)
        split_counts[split] += 1
        formats[path.suffix.lower()] += 1
        class_name = path.parent.name if split != "test" else "test_unlabeled"
        class_counts[class_name] += 1

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            errors.append({"path": str(path), "error": "No se pudo leer"})
            continue
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        sizes[f"{width}x{height}"] += 1
        color_modes[f"channels_{channels}"] += 1

    return {
        "total": len(images),
        "by_split": dict(split_counts),
        "by_class": dict(class_counts),
        "by_format": dict(formats),
        "by_size": dict(sizes),
        "by_color_channels": dict(color_modes),
        "errors": errors,
    }


def equalize_color(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced_lightness, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )


def sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def make_comparison(image: np.ndarray) -> np.ndarray:
    median = cv2.medianBlur(image, 5)
    smooth = cv2.GaussianBlur(image, (5, 5), 0)
    enhanced = sharpen(equalize_color(median))
    panels = [
        ("Original", image),
        ("Ecualizacion CLAHE", equalize_color(image)),
        ("Filtro mediana", median),
        ("Suavizado Gaussiano", smooth),
        ("Realce unsharp", sharpen(image)),
        ("Combinacion recomendada", enhanced),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    labeled = []
    for title, panel in panels:
        panel = cv2.copyMakeBorder(
            panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35)
        )
        cv2.putText(panel, title, (8, 23), font, 0.55, (255, 255, 255), 1)
        labeled.append(panel)
    row_width = labeled[0].shape[1] * 3
    row_height = labeled[0].shape[0]
    rows = []
    for offset in (0, 3):
        rows.append(cv2.hconcat(labeled[offset : offset + 3]))
    return cv2.vconcat(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--salida", type=Path, default=Path("resultados"))
    args = parser.parse_args()

    images = find_images(args.dataset)
    if not images:
        raise FileNotFoundError(f"No se encontraron imagenes en {args.dataset}")

    args.salida.mkdir(parents=True, exist_ok=True)
    stats = summarize(images)
    (args.salida / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    examples = [path for path in images if classify_split(path) == "train"][:3]
    for index, path in enumerate(examples, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"No se pudo leer la imagen de ejemplo: {path}")
        output = make_comparison(image)
        output_path = args.salida / f"filtros_{index:02d}.png"
        if not cv2.imwrite(str(output_path), output):
            raise OSError(f"No se pudo guardar {output_path}")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Ejemplos generados: {len(examples)} en {args.salida}")


if __name__ == "__main__":
    main()
