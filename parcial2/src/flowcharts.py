"""Generación de diagramas de flujo — Parcial 2."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def draw_flowchart(
    title: str,
    steps: List[Tuple[str, str, str]],
    out_path: Path,
) -> None:
    """Genera un diagrama de flujo profesional con matplotlib.

    Args:
        title: Título del diagrama.
        steps: Lista de tuplas (etapa, descripción, parámetros).
        out_path: Ruta de guardado de la imagen.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")

    n = len(steps)
    y_positions = np.linspace(0.88, 0.08, n)

    style_proc = dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1976D2", lw=2)
    style_algo = dict(boxstyle="square,pad=0.5", facecolor="#FFF3E0", edgecolor="#F57C00", lw=2)
    style_out  = dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#388E3C", lw=2)

    for i, (stage, desc, params) in enumerate(steps):
        y = y_positions[i]
        style = style_out if i == n - 1 else (
            style_algo if ("Segmentación" in stage or "Algoritmo" in stage) else style_proc
        )
        text = f"[{stage}]\n{desc}"
        if params:
            text += f"\nParámetros: {params}"
        ax.text(0.5, y, text, ha="center", va="center", fontsize=9.5, bbox=style, wrap=True)
        if i < n - 1:
            ax.annotate(
                "", xy=(0.5, y_positions[i + 1] + 0.045), xytext=(0.5, y - 0.045),
                arrowprops=dict(arrowstyle="->", lw=2, color="#424242", shrinkA=2, shrinkB=2),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


def generate_all_flowcharts(dirs: Dict[str, Path]) -> None:
    """Genera los 3 diagramas de flujo para cada nivel de complejidad."""

    # 1. Complejidad Baja
    draw_flowchart(
        "Flujo de Segmentación: Caso 1 — Complejidad Baja (Fondo Homogéneo)",
        [
            ("Entrada",          "Imagen RGB de hoja sana sobre fondo homogéneo claro.",
             "256×256 px, 3 canales"),
            ("Preprocesamiento", "Conversión a escala de grises y suavizado Gaussiano ligero.",
             "Kernel: 5×5, σ=1.0"),
            ("Segmentación",     "Otsu / Umbral Global: histograma marcadamente bimodal.",
             "Otsu T*=132, Global T=131"),
            ("Post-proc.",       "Apertura y cierre morfológico para eliminar pequeños ruidos.",
             "Kernel elíptico 5×5"),
            ("Salida",           "Máscara binaria y segmentación exacta de la hoja aislada.",
             "Aislada del fondo neutro"),
        ],
        dirs["flowcharts"] / "flujo_baja_complejidad.png",
    )

    # 2. Complejidad Media
    draw_flowchart(
        "Flujo de Segmentación: Caso 2 — Complejidad Media (Sombras y Lesiones)",
        [
            ("Entrada",          "Hoja con lesiones necróticas y sombras suaves periféricas.",
             "256×256 px, 3 canales"),
            ("Preprocesamiento", "Conversión a L*a*b* y suavizado Gaussiano / Mediana.",
             "Kernel 5×5, extracción canal a* y luminancia"),
            ("Segmentación",     "K-Means (K=3) en espacio Lab o Canny con ajuste de histéresis.",
             "K=3, max_iter=100, ε=0.2 / Canny 50-150"),
            ("Post-proc.",       "Fusión de clusters (hoja sana + lesión) y relleno de huecos.",
             "Cierre morfológico (k=5) + Flood Fill"),
            ("Salida",           "Máscara que preserva el contorno foliar con áreas enfermas.",
             "Segmentación robusta sin perder manchas"),
        ],
        dirs["flowcharts"] / "flujo_media_complejidad.png",
    )

    # 3. Complejidad Alta
    draw_flowchart(
        "Flujo de Segmentación: Caso 3 — Complejidad Alta (Fondo Complejo)",
        [
            ("Entrada",          "Fondo irregular/texturizado, tallos múltiples o necrosis severa.",
             "256×256 px, 3 canales"),
            ("Preprocesamiento", "Filtro Bilateral + transformación a canales a* (Lab) y ExG.",
             "σ_color=50, σ_space=5"),
            ("Segmentación",     "K-Means multicanal (K=3) o Canny con análisis de contornos mayores.",
             "Canny T1=30, T2=120; K=3 (Lab)"),
            ("Post-proc.",       "Filtrado por área de contorno, cierre doble y convex hull.",
             "Kernel 7×7, umbral área mín. > 10% total"),
            ("Salida",           "Aislamiento de la región foliar principal descartando artefactos.",
             "Segmentación foliar adaptativa"),
        ],
        dirs["flowcharts"] / "flujo_alta_complejidad.png",
    )

