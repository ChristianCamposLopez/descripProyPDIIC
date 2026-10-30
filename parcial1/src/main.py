"""Punto de entrada — Parcial 1: Preprocesamiento de imágenes foliares.

Genera grids comparativos aplicando todos los filtros e transformaciones
de intensidad definidos en el curso PDI sobre 3 imágenes de entrenamiento.

Uso:
    python -m parcial1.src.main --dataset dataset --salida parcial1/resultados
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from parcial1.src.dataset import DatasetAnalyzer
from parcial1.src.visualization import Visualizer
from parcial1.src.filters import (
    # Cat. 1 – Transformaciones de Intensidad
    NegativeTransform,
    LogTransform,
    GammaTransform,
    ContrastStretching,
    BinaryThreshold,
    IntensityBandHighlight,
    GlobalEqualization,
    CLAHEEqualization,
    # Cat. 2 – Filtros Estadísticos / De Orden
    MedianFilter,
    MaxFilter,
    MinFilter,
    MidpointFilter,
    ArithmeticMeanFilter,
    GeometricMeanFilter,
    # Cat. 3 – Filtros Suavizantes
    BoxFilter,
    WeightedAverageFilter,
    GaussianBlurFilter,
    BilateralFilter,
    # Cat. 4 – Filtros Realzantes
    SobelFilter,
    PrewittFilter,
    LaplacianFilter,
    HighboostFilter,
    UnsharpMaskFilter,
    GradientLaplacianFilter,
    # Cat. 5 – Segmentación / Detección avanzada
    LoGFilter,
    LineDetectionFilter,
    ExGLeafSegmenter,
    # Pipeline
    Pipeline,
)


def build_filters_map() -> dict:
    """Construye el mapa completo {Título: Filtro} para el grid comparativo."""
    return {
        # ── Referencia ────────────────────────────────────────────────────────
        "Original":                     None,

        # ── Categoría 1: Transformaciones de Intensidad ───────────────────────
        "Negativo  (s=255-r)":          NegativeTransform(),
        "Log (c·log(1+r))":             LogTransform(),
        "Gamma γ=0.5 (aclara)":         GammaTransform(gamma=0.5),
        "Gamma γ=2.0 (oscurece)":       GammaTransform(gamma=2.0),
        "Estiramiento Contraste":        ContrastStretching(r1=70, s1=0, r2=200, s2=255),
        "Umbral Binario m=128":          BinaryThreshold(m=128),
        "Banda Intensidad [100-200]":    IntensityBandHighlight(low=100, high=200),
        "Ecualización Global (L)":       GlobalEqualization(),
        "CLAHE (Lab, clip=2.0)":         CLAHEEqualization(),

        # ── Categoría 2: Filtros Estadísticos / De Orden ──────────────────────
        "Mediana 5×5":                   MedianFilter(kernel_size=5),
        "MAX 5×5 (elimina pimienta)":    MaxFilter(kernel_size=5),
        "MIN 5×5 (elimina sal)":         MinFilter(kernel_size=5),
        "Punto Medio 5×5":               MidpointFilter(kernel_size=5),
        "Media Aritmética 5×5":          ArithmeticMeanFilter(kernel_size=5),
        "Media Geométrica 5×5":          GeometricMeanFilter(kernel_size=5),

        # ── Categoría 3: Filtros Suavizantes ──────────────────────────────────
        "Box Filter 3×3 (1/9)":          BoxFilter(),
        "Promedio Pesado [1,2,1]/16":     WeightedAverageFilter(),
        "Gaussiano 5×5":                 GaussianBlurFilter(kernel_size=(5, 5)),
        "Bilateral (d=5, σ=35)":         BilateralFilter(d=5, sigma_color=35, sigma_space=35),

        # ── Categoría 4: Filtros Realzantes ───────────────────────────────────
        "Sobel ∇f":                      SobelFilter(ksize=3),
        "Prewitt ∇f":                    PrewittFilter(),
        "Laplaciano ∇²f (4-vec)":        LaplacianFilter(ksize=1, enhance=False),
        "Realce Laplaciano (g=f+∇²f)":   LaplacianFilter(ksize=1, enhance=True),
        "Highboost k=2.0":               HighboostFilter(k=2.0, blur_sigma=1.2),
        "Unsharp Mask (amount=1.5)":     UnsharpMaskFilter(sigma_x=1.2, amount=1.5),
        "Gradiente-Laplaciano 6 pasos":  GradientLaplacianFilter(gamma_correction=0.5),

        # ── Categoría 5: Segmentación / Detección avanzada ────────────────────
        "LoG / Marr-Hildreth σ=1":       LoGFilter(sigma=1.0, ksize=9),
        "Líneas 0° (horizontal)":        LineDetectionFilter(direction="0"),
        "Líneas 45°":                    LineDetectionFilter(direction="45"),
        "Líneas 90° (vertical)":         LineDetectionFilter(direction="90"),
        "Líneas -45°":                   LineDetectionFilter(direction="-45"),

        # ── Pipeline recomendado ──────────────────────────────────────────────
        "Pipeline P1 (ExG final)":       Pipeline([
            BilateralFilter(),
            CLAHEEqualization(),
            UnsharpMaskFilter(),
            ExGLeafSegmenter(),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parcial 1 PDI — Grids comparativos de todos los filtros"
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--salida", type=Path, default=Path("parcial1/resultados"))
    parser.add_argument("--n_imagenes", type=int, default=3,
                        help="Número de imágenes de muestra a procesar")
    args = parser.parse_args()

    # 1. Análisis del dataset
    analyzer = DatasetAnalyzer(args.dataset)
    images = analyzer.find_images()

    if not images:
        print(f"Advertencia: No se encontraron imágenes en {args.dataset}.")
        return

    args.salida.mkdir(parents=True, exist_ok=True)

    stats = analyzer.summarize(images)
    analyzer.save_stats(stats, args.salida / "dataset_stats.json")
    print(f"Dataset: {stats['total']} imágenes encontradas.")

    # 2. Selección de imágenes de entrenamiento
    train_images = [p for p in images if "train" in p.parts]
    n = min(args.n_imagenes, len(train_images))
    examples = random.sample(train_images, n) if n >= args.n_imagenes else train_images[:n]

    # 3. Construir mapa de filtros
    filters_map = build_filters_map()

    # 4. Generar grids comparativos
    visualizer = Visualizer()
    for idx, path in enumerate(examples, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [!] No se pudo leer: {path}")
            continue

        grid = visualizer.create_comparison_grid(image, filters_map)
        output_path = args.salida / f"filtros_{idx:02d}.png"
        visualizer.save_comparison(grid, output_path)
        print(f"  Guardado: {output_path}")

    print(f"\nProcesamiento P1 finalizado. {len(examples)} grids en '{args.salida}'")


if __name__ == "__main__":
    main()
