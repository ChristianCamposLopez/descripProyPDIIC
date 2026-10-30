"""Punto de entrada — Parcial 2: Segmentación de imágenes foliares.

Ejecuta los cuatro algoritmos de segmentación (Canny, Umbral Global,
Otsu, K-Means) sobre tres imágenes representativas del dataset y genera:
  - Figuras comparativas por nivel de complejidad (Baja / Media / Alta).
  - Histogramas con umbrales superpuestos.
  - Figura comparativa global.
  - Diagramas de flujo por caso.
  - Métricas en JSON.

Uso:
    python -m parcial2.src.main --dataset dataset --salida parcial2/resultados
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from parcial2.src.segmentation import (
    preprocess_image,
    segment_canny,
    segment_global_threshold,
    segment_otsu,
    segment_kmeans,
)
from parcial2.src.visualization import (
    generate_case_figure,
    generate_histograms_figure,
    generate_comparative_table_figure,
)
from parcial2.src.flowcharts import generate_all_flowcharts


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    dirs = {
        "base":        base_dir,
        "plots":       base_dir / "graficas",
        "comparisons": base_dir / "comparaciones",
        "flowcharts":  base_dir / "diagramas",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def process_image(
    image_bgr: np.ndarray,
    label: str,
    species: str,
    complexity: str,
    canny_params: dict,
) -> Dict[str, Any]:
    """Aplica todos los algoritmos de segmentación a una imagen y devuelve resultados."""
    prep = preprocess_image(image_bgr)
    rgb = prep["rgb"]

    canny = segment_canny(
        prep["blurred_gray"],
        low_thresh=canny_params["low"],
        high_thresh=canny_params["high"],
        morph_kernel_size=canny_params.get("morph", 5),
    )
    # Canal óptimo para umbralización: a* del espacio Lab (separa verde/fondo)
    global_th = segment_global_threshold(prep["a_channel"], invert=True)
    otsu_th   = segment_otsu(prep["a_channel"], invert=True)
    kmeans    = segment_kmeans(rgb, k=3, use_lab=True)

    return {
        "label":      label,
        "species":    species,
        "complexity": complexity,
        "rgb":        rgb,
        "prep":       prep,
        "canny":      canny,
        "global":     global_th,
        "otsu":       otsu_th,
        "kmeans":     kmeans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parcial 2 PDI — Segmentación de imágenes foliares"
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--salida",  type=Path, default=Path("parcial2/resultados"))
    args = parser.parse_args()

    dirs = ensure_dirs(args.salida)

    # ─── Buscar imágenes de entrenamiento ─────────────────────────────────────
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted(
        p for p in args.dataset.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "train" in p.parts
    )

    if not all_images:
        print(f"[!] No se encontraron imágenes en {args.dataset}. "
              f"Asegúrate de tener el dataset descargado.")
        return

    # ─── Seleccionar 3 imágenes (una por nivel de complejidad) ────────────────
    # En producción podrías seleccionar manualmente rutas específicas.
    # Aquí seleccionamos aleatoriamente para demostración.
    random.seed(42)
    sample = random.sample(all_images, min(3, len(all_images)))

    complexity_configs = [
        {
            "label":   "Complejidad BAJA",
            "species": sample[0].parent.name if len(sample) > 0 else "Hoja sana",
            "canny":   {"low": 50, "high": 150, "morph": 5},
        },
        {
            "label":   "Complejidad MEDIA",
            "species": sample[1].parent.name if len(sample) > 1 else "Hoja con lesiones",
            "canny":   {"low": 50, "high": 150, "morph": 5},
        },
        {
            "label":   "Complejidad ALTA",
            "species": sample[2].parent.name if len(sample) > 2 else "Hoja severamente dañada",
            "canny":   {"low": 30, "high": 120, "morph": 7},
        },
    ]

    # ─── Procesar cada caso ───────────────────────────────────────────────────
    cases_data: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}

    for i, (img_path, cfg) in enumerate(zip(sample, complexity_configs), start=1):
        print(f"\n[{i}/3] Procesando: {img_path.name} ({cfg['label']})")
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  [!] No se pudo leer: {img_path}")
            continue

        complexity_key = f"caso_{i}"
        data = process_image(
            image_bgr,
            label=cfg["label"],
            species=cfg["species"],
            complexity=complexity_key,
            canny_params=cfg["canny"],
        )
        cases_data[complexity_key] = data

        # Figura por caso
        generate_case_figure(
            case_name=img_path.name,
            complexity_label=cfg["label"],
            data=data,
            out_path=dirs["comparisons"] / f"caso_{complexity_key}.png",
        )

        # Métricas
        metrics[complexity_key] = {
            "imagen": img_path.name,
            "complejidad": cfg["label"],
            "especie": cfg["species"],
            "canny": {
                "parametros": data["canny"]["params"],
                "pixeles_borde": int(np.sum(data["canny"]["edges"] > 0)),
            },
            "umbral_global": {
                "T": data["global"]["threshold"],
                "iteraciones": data["global"]["iterations"],
            },
            "otsu": {
                "T_star": data["otsu"]["threshold"],
                "varianza_max": round(data["otsu"]["max_variance"], 4),
            },
            "kmeans": {
                "K": data["kmeans"]["k"],
                "compactness": round(data["kmeans"]["compactness"], 2),
                "clusters_hoja": data["kmeans"]["leaf_clusters"],
                "espacio_color": data["kmeans"]["params"]["color_space"],
            },
        }

    # ─── Figuras globales ─────────────────────────────────────────────────────
    if len(cases_data) >= 2:
        generate_histograms_figure(
            cases_data,
            dirs["plots"] / "histogramas_umbrales.png",
        )

    if len(cases_data) >= 2:
        generate_comparative_table_figure(
            cases_data,
            dirs["comparisons"] / "comparativa_global.png",
        )

    # ─── Diagramas de flujo ───────────────────────────────────────────────────
    generate_all_flowcharts(dirs)

    # ─── Guardar métricas ─────────────────────────────────────────────────────
    metrics_path = args.salida / "metricas_segmentacion.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Métricas guardadas en: {metrics_path}")

    print(f"\n✓ Parcial 2 finalizado. Resultados en '{args.salida}'")


if __name__ == "__main__":
    main()

