import cv2
import numpy as np
from pathlib import Path
from src.filters import ImageFilter

class Visualizer:
    """
    Responsable de renderizar y guardar los grids comparativos.
    """
    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55):
        self.font = font
        self.font_scale = font_scale

    def create_comparison_grid(self, image: np.ndarray, filters_map: dict[str, ImageFilter]) -> np.ndarray:
        """
        Toma un diccionario de {Título: Filtro} y genera un grid comparativo.
        'Original' se pasa como un ImageFilter que devuelve la imagen intacta,
        o se maneja explícitamente.
        """
        labeled_panels = []
        for title, img_filter in filters_map.items():
            if img_filter is None:
                panel = image.copy()
            else:
                panel = img_filter.apply(image)
                
            panel = cv2.copyMakeBorder(
                panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35)
            )
            cv2.putText(panel, title, (8, 23), self.font, self.font_scale, (255, 255, 255), 1)
            labeled_panels.append(panel)

        # Crear filas de 3 imágenes
        rows = []
        for offset in range(0, len(labeled_panels), 3):
            row_panels = labeled_panels[offset : offset + 3]
            # Si la última fila tiene menos de 3, rellenar
            while len(row_panels) < 3:
                blank = np.zeros_like(labeled_panels[0])
                row_panels.append(blank)
            rows.append(cv2.hconcat(row_panels))
            
        return cv2.vconcat(rows)

    def save_comparison(self, image_grid: np.ndarray, output_path: Path) -> None:
        if not cv2.imwrite(str(output_path), image_grid):
            raise OSError(f"No se pudo guardar {output_path}")
