"""Módulo de Visualización — Parcial 1 PDI.

Genera tanto grids comparativos como tarjetas individuales con pie de página
descriptivo (fórmula, categoría y justificación teórica) según el estándar
del curso de la Dra. Verónica Rodríguez López y Gonzalez & Woods.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
import cv2
import numpy as np

from parcial1.src.filters import FilterSpec, ImageFilter


class Visualizer:
    """Responsable del renderizado de imágenes individuales con pie de página y grids."""

    # Paleta de colores para barras de categorías (formato BGR)
    _CAT_COLORS = {
        "Referencia": (180, 180, 180),
        "Cat 1": (225, 185, 40),       # Cian/Aqua
        "Cat 2": (30, 140, 240),       # Ámbar/Naranja
        "Cat 3": (70, 205, 95),        # Verde esmeralda
        "Cat 4": (230, 115, 80),       # Azul cobalto
        "Cat 5": (75, 85, 235),        # Coral/Rojo
        "Pipeline": (210, 80, 200),    # Magenta
    }

    def __init__(
        self,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        grid_font_scale: float = 0.52,
    ) -> None:
        self.font = font
        self.grid_font_scale = grid_font_scale

    def _get_category_color(self, category: str) -> tuple[int, int, int]:
        for prefix, color in self._CAT_COLORS.items():
            if prefix in category:
                return color
        return (200, 200, 200)

    # =========================================================================
    # 1. TARJETAS INDIVIDUALES CON PIE DE PÁGINA DESCRIPTIVO
    # =========================================================================

    def create_individual_card(
        self,
        image: np.ndarray,
        spec: FilterSpec,
        target_width: int = 800,
    ) -> np.ndarray:
        """Genera una imagen individual autocontenida con pie de página explicativo.

        Estructura visual:
            ┌──────────────────────────────────────────────┐
            │                                              │
            │          IMAGEN PROCESADA / FILTRADA         │
            │                                              │
            ├──────────────────────────────────────────────┤  <- Barra de color por categoría
            │ [CATEGORÍA] • NOMBRE FORMAL DEL FILTRO       │
            │ Ecuación / Kernel: <fórmula>                 │
            │ Propósito: <descripción multilínea clara>     │
            └──────────────────────────────────────────────┘
        """
        # 1. Redimensionar proporcionalmente para nitidez tipográfica
        h, w = image.shape[:2]
        if w != target_width:
            scale = target_width / float(w)
            target_height = int(round(h * scale))
            img_resized = cv2.resize(
                image, (target_width, target_height), interpolation=cv2.INTER_CUBIC
            )
        else:
            img_resized = image.copy()

        # 2. Configuración de textos y saltos de línea (word wrap)
        cat_color = self._get_category_color(spec.category)
        margin_x = 24
        wrap_width = 86  # caracteres por línea aprox a 800px

        header_text = f"[{spec.category.upper()}]  {spec.title.upper()}"
        formula_text = f"Modelo / Formula: {spec.formula}"
        desc_lines = textwrap.wrap(f"Analisis: {spec.description}", width=wrap_width)

        # 3. Calcular altura dinámica del pie de página (footer)
        line_height = 24
        footer_height = 28 + 26 + (len(desc_lines) * line_height) + 24
        footer = np.full((footer_height, target_width, 3), (24, 20, 18), dtype=np.uint8)

        # 4. Dibujar barra de categoría divisoria (4px)
        cv2.line(footer, (0, 2), (target_width, 2), cat_color, 4)

        # 5. Renderizar textos en el footer
        curr_y = 30
        # Línea de Título
        cv2.putText(
            footer,
            header_text,
            (margin_x, curr_y),
            self.font,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Línea de Fórmula
        curr_y += 24
        cv2.putText(
            footer,
            formula_text,
            (margin_x, curr_y),
            self.font,
            0.48,
            (220, 200, 110),
            1,
            cv2.LINE_AA,
        )

        # Líneas de Descripción
        curr_y += 6
        for line in desc_lines:
            curr_y += line_height
            cv2.putText(
                footer,
                line,
                (margin_x, curr_y),
                self.font,
                0.46,
                (195, 200, 205),
                1,
                cv2.LINE_AA,
            )

        # 6. Concatenar verticalmente la imagen con su pie de página
        card = cv2.vconcat([img_resized, footer])
        return card

    def save_individual_card(self, card: np.ndarray, output_path: Path) -> None:
        """Guarda la tarjeta individual en el disco creando carpetas si es necesario."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), card):
            raise OSError(f"No se pudo guardar la tarjeta en {output_path}")

    # =========================================================================
    # 2. GRIDS COMPARATIVOS GLOBALES
    # =========================================================================

    def create_comparison_grid(
        self,
        image: np.ndarray,
        filters_map: dict[str, ImageFilter | None],
        cols: int = 3,
    ) -> np.ndarray:
        """Toma un diccionario {Título: Filtro} y genera un grid comparativo."""
        labeled_panels = []
        for title, img_filter in filters_map.items():
            if img_filter is None:
                panel = image.copy()
            else:
                panel = img_filter.apply(image)

            # Barra superior para el título en el grid
            panel = cv2.copyMakeBorder(
                panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35)
            )
            cv2.putText(
                panel,
                title,
                (8, 23),
                self.font,
                self.grid_font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            labeled_panels.append(panel)

        # Agrupar en filas de 'cols' paneles
        rows = []
        for offset in range(0, len(labeled_panels), cols):
            row_panels = labeled_panels[offset : offset + cols]
            while len(row_panels) < cols:
                blank = np.zeros_like(labeled_panels[0])
                row_panels.append(blank)
            rows.append(cv2.hconcat(row_panels))

        return cv2.vconcat(rows)

    def save_comparison(self, image_grid: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image_grid):
            raise OSError(f"No se pudo guardar {output_path}")

    # =========================================================================
    # 3. EXPORTACIÓN DE FRAGMENTOS LATEX
    # =========================================================================

    @staticmethod
    def generate_latex_snippets(
        items: list[tuple[str, FilterSpec]],
        output_tex_path: Path,
    ) -> None:
        """Escribe un archivo .tex con figuras individuales y captions estructurados."""
        output_tex_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "% ============================================================================",
            "% Figuras Individuales de Filtros de PDI con Pie de Página y Descripción",
            "% Generado automáticamente por Visualizer",
            "% ============================================================================\n",
        ]

        curr_cat = None
        for rel_path, spec in items:
            if spec.category != curr_cat:
                curr_cat = spec.category
                lines.append(f"\n\\subsection*{{{curr_cat}}}\n")

            fig_block = (
                "\\begin{figure}[htbp]\n"
                "  \\centering\n"
                f"  \\includegraphics[width=0.78\\linewidth]{{{rel_path}}}\n"
                f"  \\caption{{\\textbf{{{spec.title}:}} {spec.description} "
                f"\\textit{{Modelo:}} \\texttt{{{spec.formula}}}.}}\n"
                f"  \\label{{fig:{spec.id_name}}}\n"
                "\\end{figure}\n"
            )
            lines.append(fig_block)

        output_tex_path.write_text("\n".join(lines), encoding="utf-8")
