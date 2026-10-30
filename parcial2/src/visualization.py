"""Visualizaciones y figuras comparativas — Parcial 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from parcial2.src.segmentation import apply_mask


def generate_case_figure(
    case_name: str,
    complexity_label: str,
    data: Dict[str, Any],
    out_path: Path,
) -> None:
    """Figura 2×4 con todos los pasos y métodos para un caso.

    Fila 1: Original, Preprocesamiento (Gauss), Canny bordes, Canny máscara.
    Fila 2: Umbral Global, Otsu, K-Means cuantizado, K-Means segmentado.
    """
    rgb = data["rgb"]
    gray = data["prep"]["blurred_gray"]
    canny_res = data["canny"]
    global_res = data["global"]
    otsu_res = data["otsu"]
    kmeans_res = data["kmeans"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    plt.suptitle(
        f"Segmentación de Hoja — Complejidad {complexity_label.upper()} ({case_name})\n"
        f"Proyecto PDI — Segundo Parcial",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Fila 1
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("(a) Original RGB", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("(b) Preproc. (Gauss 5×5)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(canny_res["edges"], cmap="gray")
    axes[0, 2].set_title(
        f"(c) Canny Bordes\n"
        f"(T1={canny_res['params']['low_thresh']}, T2={canny_res['params']['high_thresh']})",
        fontsize=10,
    )
    axes[0, 2].axis("off")

    axes[0, 3].imshow(apply_mask(rgb, canny_res["mask"]))
    axes[0, 3].set_title("(d) Segmentado Canny\n(Cierre + Contornos)", fontsize=10)
    axes[0, 3].axis("off")

    # Fila 2
    axes[1, 0].imshow(apply_mask(rgb, global_res["mask"]))
    axes[1, 0].set_title(
        f"(e) Umbral Global\n"
        f"(T={global_res['threshold']}, Iter={global_res['iterations']})",
        fontsize=10,
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(apply_mask(rgb, otsu_res["mask"]))
    axes[1, 1].set_title(
        f"(f) Umbral Otsu\n"
        f"(T*={otsu_res['threshold']}, σ²={otsu_res['max_variance']:.2f})",
        fontsize=10,
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(kmeans_res["segmented_colors"])
    axes[1, 2].set_title(
        f"(g) K-Means Cuantizado\n"
        f"(K={kmeans_res['k']}, {kmeans_res['params']['color_space']})",
        fontsize=10,
    )
    axes[1, 2].axis("off")

    axes[1, 3].imshow(apply_mask(rgb, kmeans_res["mask"]))
    axes[1, 3].set_title(
        f"(h) Segmentado K-Means\n(Clusters {kmeans_res['leaf_clusters']})",
        fontsize=10,
    )
    axes[1, 3].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


def generate_histograms_figure(
    cases_data: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Histogramas de intensidad con umbrales Global y Otsu superpuestos."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plt.suptitle(
        "Distribución de Intensidades y Umbrales Calculados (Global vs Otsu)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    for idx, (c_key, d) in enumerate(cases_data.items()):
        ax = axes[idx]
        gray = d["prep"]["blurred_gray"]
        t_global = d["global"]["threshold"]
        t_otsu = d["otsu"]["threshold"]
        hist, bins = np.histogram(gray, bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color="black", lw=1.5, label="Histograma Gris")
        ax.axvline(t_global, color="royalblue", linestyle="--", lw=2,
                   label=f"Global T={t_global}")
        ax.axvline(t_otsu, color="crimson", linestyle="-.", lw=2,
                   label=f"Otsu T*={t_otsu}")
        ax.set_title(f"{d['label']}\n{d['species']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Nivel de Intensidad (0-255)")
        ax.set_ylabel("Frecuencia de Píxeles")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")


def generate_comparative_table_figure(
    cases_data: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Figura resumen comparando los 4 métodos para todos los casos."""
    methods = ["Original", "Canny", "Global", "Otsu", "K-Means"]
    n_cases = len(cases_data)
    fig, axes = plt.subplots(n_cases, 5, figsize=(16, 3.2 * n_cases))
    plt.suptitle(
        "Comparativa de Segmentación por Método y Nivel de Complejidad",
        fontsize=14, fontweight="bold", y=0.99,
    )
    for row_idx, (c_key, d) in enumerate(cases_data.items()):
        rgb = d["rgb"]
        col_imgs = [
            rgb,
            apply_mask(rgb, d["canny"]["mask"]),
            apply_mask(rgb, d["global"]["mask"]),
            apply_mask(rgb, d["otsu"]["mask"]),
            apply_mask(rgb, d["kmeans"]["mask"]),
        ]
        for col_idx, img in enumerate(col_imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            if row_idx == 0:
                ax.set_title(methods[col_idx], fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{d['label']}\n({d['species']})", fontsize=10, fontweight="bold"
                )
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path}")

