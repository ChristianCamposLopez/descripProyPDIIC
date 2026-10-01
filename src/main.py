import argparse
import random
from pathlib import Path
import cv2

from src.dataset import DatasetAnalyzer
from src.visualization import Visualizer
from src.filters import (
    Pipeline,
    GlobalEqualization,
    CLAHEEqualization,
    MedianFilter,
    BilateralFilter,
    GaussianFilter, # Alias handled below
    GaussianBlurFilter,
    UnsharpMaskFilter,
    ExGLeafSegmenter
)

def main():
    parser = argparse.ArgumentParser(description="Procesamiento de imágenes foliares (PDI)")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--salida", type=Path, default=Path("resultados_solid"))
    args = parser.parse_args()

    # 1. Dataset Analysis
    analyzer = DatasetAnalyzer(args.dataset)
    images = analyzer.find_images()
    
    if not images:
        print(f"Advertencia: No se encontraron imágenes en {args.dataset}. Mostrando solo estructura.")
        return

    args.salida.mkdir(parents=True, exist_ok=True)
    
    # Analyze and save stats
    stats = analyzer.summarize(images)
    analyzer.save_stats(stats, args.salida / "dataset_stats.json")

    # 2. Filter Pipeline Definition
    # Usamos dependencias inyectadas para la visualización
    filters_map = {
        "Original": None,
        "Ecualizacion Global": GlobalEqualization(),
        "CLAHE (Lab)": CLAHEEqualization(),
        "Bilateral (Ruido)": BilateralFilter(d=5, sigma_color=35, sigma_space=35),
        "Unsharp Mask (Realce)": UnsharpMaskFilter(sigma_x=1.2, amount=1.5),
        "Pipeline Final (Segmentado)": Pipeline([
            BilateralFilter(),
            CLAHEEqualization(),
            UnsharpMaskFilter(),
            ExGLeafSegmenter()
        ])
    }

    # 3. Process Examples
    visualizer = Visualizer()
    train_images = [path for path in images if "train" in path.parts]
    
    # Seleccionar aleatoriamente para diversidad si hay suficientes, o tomar primeras 3
    if len(train_images) >= 3:
        examples = random.sample(train_images, 3)
    else:
        examples = train_images[:3]

    for index, path in enumerate(examples, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
            
        output = visualizer.create_comparison_grid(image, filters_map)
        output_path = args.salida / f"filtros_solid_{index:02d}.png"
        visualizer.save_comparison(output, output_path)

    print(f"Procesamiento finalizado. {len(examples)} comparaciones guardadas en {args.salida}")


if __name__ == "__main__":
    main()
