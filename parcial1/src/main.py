"""Punto de entrada — Parcial 1: Preprocesamiento y Filtrado de Imágenes Foliares.

Implementa la ejecución y visualización de todos los filtros del temario de PDI:
  - Modo 'individual': Tarjeta individual por cada filtro con su pie de página descriptivo
    (categoría, fórmula matemática y análisis de propósito/efecto).
  - Modo 'grid': Grid matricial comparativo global.
  - Modo 'ambos': Genera ambos formatos y el archivo figuras_filtros.tex para el reporte LaTeX.

Uso:
    python -m parcial1.src.main --modo ambos --salida parcial1/resultados
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from parcial1.src.dataset import DatasetAnalyzer
from parcial1.src.filters import FilterSpec, ImageFilter, get_all_filter_specs
from parcial1.src.visualization import Visualizer


def load_input_images(dataset_dir: Path, n_images: int) -> list[Path]:
    """Carga imágenes desde el dataset o desde la carpeta de muestras si no existe."""
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        analyzer = DatasetAnalyzer(dataset_dir)
        all_imgs = analyzer.find_images()
        train_images = [p for p in all_imgs if "train" in p.parts]
        pool = train_images if train_images else all_imgs
        if pool:
            n = min(n_images, len(pool))
            return random.sample(pool, n) if n >= n_images else pool[:n]

    # Fallback transparente a parcial1/muestras
    fallback_dir = Path("parcial1/muestras")
    if fallback_dir.exists():
        fallback_imgs = sorted(
            [p for p in fallback_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if fallback_imgs:
            print(f"[INFO] Utilizando {len(fallback_imgs)} imágenes base desde '{fallback_dir}'.")
            return fallback_imgs[:n_images]

    return []


def run_pipeline(
    images_paths: list[Path],
    salida_dir: Path,
    modo: str = "ambos",
) -> None:
    """Ejecuta el procesamiento de filtros y genera los artefactos visuales."""
    visualizer = Visualizer()
    specs = get_all_filter_specs()
    salida_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================================================================================")
    print(f" PIPELINE DE FILTROS PDI — PARCIAL 1")
    print(f" Total de filtros configurados: {len(specs)}")
    print(f" Modo de generación: {modo}")
    print(f" Directorio de salida: {salida_dir}")
    print(f"================================================================================\n")

    latex_items: list[tuple[str, FilterSpec]] = []

    for img_idx, img_path in enumerate(images_paths, start=1):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[!] Error al cargar imagen: {img_path}")
            continue

        sample_name = f"muestra_{img_idx:02d}"
        print(f"[*] Procesando {sample_name} ({img_path.name}) ...")

        # ---------------------------------------------------------------------
        # 1. MODO INDIVIDUAL: Tarjeta individual con pie de página descriptivo
        # ---------------------------------------------------------------------
        if modo in ("individual", "ambos"):
            indiv_dir = salida_dir / "individuales" / sample_name
            indiv_dir.mkdir(parents=True, exist_ok=True)

            for spec in specs:
                # Aplicar filtro o conservar original
                if spec.filter_obj is None:
                    filtered = image.copy()
                else:
                    filtered = spec.filter_obj.apply(image)

                # Generar tarjeta con pie de página
                card = visualizer.create_individual_card(filtered, spec, target_width=800)
                out_path = indiv_dir / f"{spec.id_name}.png"
                visualizer.save_individual_card(card, out_path)

                # Registrar para snippet LaTeX de la primera muestra
                if img_idx == 1:
                    rel_tex_path = f"resultados/individuales/{sample_name}/{spec.id_name}.png"
                    latex_items.append((rel_tex_path, spec))

            print(f"    -> {len(specs)} imágenes individuales guardadas en: {indiv_dir}")

        # ---------------------------------------------------------------------
        # 2. MODO GRID: Grid comparativo global
        # ---------------------------------------------------------------------
        if modo in ("grid", "ambos"):
            # Construir mapa {Título: ImageFilter}
            filters_map: dict[str, ImageFilter | None] = {
                spec.title: spec.filter_obj for spec in specs
            }
            grid = visualizer.create_comparison_grid(image, filters_map, cols=4)
            grid_path = salida_dir / f"filtros_{img_idx:02d}.png"
            visualizer.save_comparison(grid, grid_path)
            print(f"    -> Grid comparativo guardado en: {grid_path}")

    # Exportar archivo de figuras LaTeX para inclusión directa en reporte
    if latex_items and modo in ("individual", "ambos"):
        tex_out = salida_dir / "individuales" / "figuras_filtros.tex"
        visualizer.generate_latex_snippets(latex_items, tex_out)
        print(f"\n[+] Fragmentos LaTeX generados exitosamente en: {tex_out}")

    print("\n[OK] Pipeline finalizado con éxito.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parcial 1 PDI — Procesamiento individual y en grid de todos los filtros"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Ruta al directorio de imágenes o dataset",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("parcial1/resultados"),
        help="Ruta donde se guardarán los resultados",
    )
    parser.add_argument(
        "--n_imagenes",
        type=int,
        default=3,
        help="Número de imágenes a procesar",
    )
    parser.add_argument(
        "--modo",
        type=str,
        choices=["ambos", "individual", "grid"],
        default="ambos",
        help="Modo de generación: 'individual' (con pie de página), 'grid' o 'ambos'",
    )
    args = parser.parse_args()

    images = load_input_images(args.dataset, args.n_imagenes)
    if not images:
        print("[!] No se encontraron imágenes para procesar.")
        return

    run_pipeline(images, args.salida, modo=args.modo)


if __name__ == "__main__":
    main()
