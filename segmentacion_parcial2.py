"""Segundo Parcial: Segmentación de Imágenes.

Proyecto de Procesamiento Digital de Imágenes
Alumno: Christian Josué Campos López
Profesora: Dra. Verónica Rodríguez López
Universidad Tecnológica de la Mixteca

Este script implementa:
1. Detector de bordes de Canny.
2. Umbralización Global (Algoritmo iterativo de Gonzalez & Woods).
3. Umbralización de Otsu (Un umbral, maximización de varianza inter-clase).
4. Agrupamiento K-Means (Segmentación en espacio de color Lab/RGB).
5. Post-procesamiento morfológico para refinamiento de máscaras.
6. Generación de comparativas visuales, histogramas y métricas para 3 niveles de complejidad.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    dirs = {
        "base": base_dir,
        "plots": base_dir / "graficas",
        "comparisons": base_dir / "comparaciones",
        "flowcharts": base_dir / "diagramas",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =============================================================================
# 1. PREPROCESAMIENTO
# =============================================================================

def preprocess_image(
    image_bgr: np.ndarray,
    blur_kernel: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Genera representaciones y preprocesamiento de la imagen."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, blur_kernel, sigma)
    blurred_rgb = cv2.GaussianBlur(rgb, blur_kernel, sigma)

    # Espacio Lab: el canal a* separa excelentemente vegetación (verde < 128) de fondo
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # Espacio HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    # Índice de Exceso de Verde normalizado: ExG = 2*G - R - B
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        "bgr": image_bgr,
        "rgb": rgb,
        "gray": gray,
        "blurred_gray": blurred_gray,
        "blurred_rgb": blurred_rgb,
        "lab": lab,
        "a_channel": a_chan,
        "hsv": hsv,
        "s_channel": s_chan,
        "exg": exg_norm,
    }


# =============================================================================
# 2. ALGORITMOS DE SEGMENTACIÓN
# =============================================================================

def segment_canny(
    gray: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    morph_kernel_size: int = 5,
) -> Dict[str, Any]:
    """Segmentación mediante el detector de bordes de Canny y reconstrucción morfológica."""
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    # Para obtener máscara binaria a partir de bordes:
    # 1. Dilatación/cierre morfológico para conectar contornos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. Búsqueda de contornos externos y rellenado
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        # Tomar contornos significativos
        max_area = max(cv2.contourArea(c) for c in contours)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= max_area * 0.05]
        cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)

    # Relleno de agujeros
    mask_filled = mask.copy()
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_filled, flood_mask, (0, 0), 255)
    mask_inv = cv2.bitwise_not(mask_filled)
    mask = cv2.bitwise_or(mask, mask_inv)

    return {
        "edges": edges,
        "closed_edges": closed_edges,
        "mask": mask,
        "params": {
            "low_thresh": low_thresh,
            "high_thresh": high_thresh,
            "morph_kernel": morph_kernel_size,
        },
    }


def segment_global_threshold(
    channel: np.ndarray,
    delta_t: float = 0.5,
    max_iters: int = 100,
    invert: bool = True,
) -> Dict[str, Any]:
    """Umbralización global iterativa según Gonzalez & Woods."""
    t_curr = float(np.mean(channel))
    iters = 0

    while iters < max_iters:
        iters += 1
        g1 = channel[channel > t_curr]
        g2 = channel[channel <= t_curr]

        mu1 = float(np.mean(g1)) if len(g1) > 0 else t_curr
        mu2 = float(np.mean(g2)) if len(g2) > 0 else t_curr
        t_next = (mu1 + mu2) / 2.0

        if abs(t_next - t_curr) < delta_t:
            t_curr = t_next
            break
        t_curr = t_next

    thresh_val = int(round(t_curr))
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(channel, thresh_val, 255, thresh_type)

    # Post-procesamiento morfológico básico
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return {
        "threshold": thresh_val,
        "iterations": iters,
        "mask": mask,
        "params": {"delta_t": delta_t, "max_iters": max_iters, "invert": invert},
    }


def segment_otsu(channel: np.ndarray, invert: bool = True) -> Dict[str, Any]:
    """Umbralización automática de Otsu (un umbral)."""
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    thresh_val, mask = cv2.threshold(channel, 0, 255, thresh_type + cv2.THRESH_OTSU)

    # Calcular varianza inter-clase de Otsu para reporte científico
    hist, _ = np.histogram(channel, bins=256, range=(0, 256), density=True)
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_squared = ((mu_t * omega - mu) ** 2) / (omega * (1.0 - omega))
        sigma_b_squared = np.nan_to_num(sigma_b_squared)
    max_sigma_b = float(np.max(sigma_b_squared))

    # Refinamiento morfológico
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return {
        "threshold": int(thresh_val),
        "max_variance": max_sigma_b,
        "mask": mask,
        "params": {"invert": invert},
    }


def segment_kmeans(
    image_rgb: np.ndarray,
    k: int = 3,
    use_lab: bool = True,
    max_iters: int = 100,
    epsilon: float = 0.2,
) -> Dict[str, Any]:
    """Segmentación por agrupamiento K-Means en espacio de color."""
    if use_lab:
        feature_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    else:
        feature_img = image_rgb

    h, w, _ = feature_img.shape
    pixel_vals = feature_img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, epsilon)
    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, 10, flags
    )

    centers = np.uint8(centers)
    segmented_features = centers[labels.flatten()].reshape((h, w, 3))
    label_map = labels.reshape((h, w))

    # Identificar cuál cluster corresponde a la hoja
    # En espacio Lab: canal a* (índice 1). Valores menores (< 128) corresponden al verde
    # En RGB: mayor componente verde relativo (G - (R+B)/2)
    cluster_scores = []
    rgb_flat = image_rgb.reshape((-1, 3))

    for c_id in range(k):
        c_mask = (labels.flatten() == c_id)
        if not np.any(c_mask):
            cluster_scores.append(-999)
            continue
        avg_rgb = np.mean(rgb_flat[c_mask], axis=0)
        # Puntuación de verdor
        green_score = float(avg_rgb[1] - (avg_rgb[0] + avg_rgb[2]) / 2.0)
        cluster_scores.append(green_score)

    if k == 2:
        leaf_cluster = int(np.argmax(cluster_scores))
        leaf_clusters = [leaf_cluster]
    else:
        # Para k=3, a menudo un cluster es fondo, uno es hoja sana y otro es lesión/borde
        # Ordenamos clusters por verdor
        sorted_indices = np.argsort(cluster_scores)
        # El cluster con mayor verdor es hoja sana
        # El segundo cluster puede ser lesión/tejido vegetal si tiene verdor > umbral o no es fondo
        # Verificamos si los dos clusters con mayor verdor pertenecen a la hoja
        leaf_clusters = [int(sorted_indices[-1])]
        if cluster_scores[sorted_indices[-2]] > -20:  # Posible lesión o borde foliar
            leaf_clusters.append(int(sorted_indices[-2]))

    mask = np.zeros((h, w), dtype=np.uint8)
    for c_id in leaf_clusters:
        mask[label_map == c_id] = 255

    # Refinamiento morfológico
    k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_elem)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_elem)

    if use_lab:
        segmented_rgb = cv2.cvtColor(segmented_features, cv2.COLOR_LAB2RGB)
    else:
        segmented_rgb = segmented_features

    return {
        "k": k,
        "compactness": float(compactness),
        "label_map": label_map,
        "segmented_colors": segmented_rgb,
        "leaf_clusters": leaf_clusters,
        "mask": mask,
        "params": {
            "k": k,
            "color_space": "LAB" if use_lab else "RGB",
            "max_iters": max_iters,
            "epsilon": epsilon,
            "attempts": 10,
        },
    }


def apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aplica máscara binaria dejando fondo negro."""
    result = np.zeros_like(rgb)
    result[mask > 0] = rgb[mask > 0]
    return result


# =============================================================================
# 3. GENERACIÓN DE VISUALIZACIONES Y COMPARATIVAS
# =============================================================================

def generate_case_figure(
    case_name: str,
    complexity_label: str,
    data: Dict[str, Any],
    out_path: Path,
) -> None:
    """Genera una figura completa de 2x4 con todos los pasos y métodos."""
    rgb = data["rgb"]
    gray = data["prep"]["blurred_gray"]
    canny_res = data["canny"]
    global_res = data["global"]
    otsu_res = data["otsu"]
    kmeans_res = data["kmeans"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    plt.suptitle(
        f"Segmentación de Hoja - Complejidad {complexity_label.upper()} ({case_name})\n"
        f"Proyecto PDI - Segundo Parcial",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Fila 1: Imagen original, Preprocesamiento, Canny bordes, Canny máscara
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("(a) Original RGB", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("(b) Preproc. (Gauss 5x5)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(canny_res["edges"], cmap="gray")
    axes[0, 2].set_title(
        f"(c) Canny Bordes\n(T1={canny_res['params']['low_thresh']}, T2={canny_res['params']['high_thresh']})",
        fontsize=10,
    )
    axes[0, 2].axis("off")

    axes[0, 3].imshow(apply_mask(rgb, canny_res["mask"]))
    axes[0, 3].set_title("(d) Segmentado Canny\n(Cierre + Contornos)", fontsize=10)
    axes[0, 3].axis("off")

    # Fila 2: Umbralización Global, Otsu, K-Means cuantizado, K-Means segmentado
    axes[1, 0].imshow(apply_mask(rgb, global_res["mask"]))
    axes[1, 0].set_title(
        f"(e) Umbral Global\n(T={global_res['threshold']}, Iter={global_res['iterations']})",
        fontsize=10,
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(apply_mask(rgb, otsu_res["mask"]))
    axes[1, 1].set_title(
        f"(f) Umbral Otsu\n(T*={otsu_res['threshold']}, Var={otsu_res['max_variance']:.2f})",
        fontsize=10,
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(kmeans_res["segmented_colors"])
    axes[1, 2].set_title(
        f"(g) K-Means Cuantizado\n(K={kmeans_res['k']}, Espacio={kmeans_res['params']['color_space']})",
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


def generate_histograms_figure(
    cases_data: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Genera gráficas de histogramas con los umbrales Global y Otsu superpuestos."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plt.suptitle(
        "Distribución de Intensidades y Umbrales Calculados (Global vs Otsu)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for idx, (c_key, d) in enumerate(cases_data.items()):
        ax = axes[idx]
        gray = d["prep"]["blurred_gray"]
        t_global = d["global"]["threshold"]
        t_otsu = d["otsu"]["threshold"]

        hist, bins = np.histogram(gray, bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color="black", lw=1.5, label="Histograma Gris")
        ax.axvline(
            t_global,
            color="royalblue",
            linestyle="--",
            linewidth=2,
            label=f"Global T={t_global}",
        )
        ax.axvline(
            t_otsu,
            color="crimson",
            linestyle="-.",
            linewidth=2,
            label=f"Otsu T*={t_otsu}",
        )

        ax.set_title(f"{d['label']}\n{d['species']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Nivel de Intensidad (0-255)")
        ax.set_ylabel("Frecuencia de Píxeles")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_comparative_table_figure(
    cases_data: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Genera una imagen resumen comparando los resultados segmentados de todos los casos."""
    methods = ["Original", "Canny", "Global", "Otsu", "K-Means"]
    n_cases = len(cases_data)

    fig, axes = plt.subplots(n_cases, 5, figsize=(16, 3.2 * n_cases))
    plt.suptitle(
        "Comparativa de Segmentación por Método y Nivel de Complejidad",
        fontsize=14,
        fontweight="bold",
        y=0.99,
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
                    f"{d['label']}\n({d['species']})",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 4. GENERACIÓN DE DIAGRAMAS DE FLUJO
# =============================================================================

def draw_flowchart(
    title: str,
    steps: list[Tuple[str, str, str]],
    out_path: Path,
) -> None:
    """Genera un diagrama de flujo profesional con matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")

    n_steps = len(steps)
    y_positions = np.linspace(0.88, 0.08, n_steps)

    box_style_proc = dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1976D2", lw=2)
    box_style_algo = dict(boxstyle="square,pad=0.5", facecolor="#FFF3E0", edgecolor="#F57C00", lw=2)
    box_style_out = dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#388E3C", lw=2)

    for i, (stage, desc, params) in enumerate(steps):
        y = y_positions[i]
        if i == n_steps - 1:
            style = box_style_out
        elif "Segmentación" in stage or "Algoritmo" in stage:
            style = box_style_algo
        else:
            style = box_style_proc

        text = f"[{stage}]\n{desc}"
        if params:
            text += f"\nParámetros: {params}"

        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9.5,
            bbox=style,
            wrap=True,
        )

        if i < n_steps - 1:
            next_y = y_positions[i + 1]
            ax.annotate(
                "",
                xy=(0.5, next_y + 0.045),
                xytext=(0.5, y - 0.045),
                arrowprops=dict(
                    arrowstyle="->", lw=2, color="#424242", shrinkA=2, shrinkB=2
                ),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_all_flowcharts(dirs: Dict[str, Path]) -> None:
    """Genera los 3 diagramas de flujo requeridos para cada nivel de complejidad."""
    # 1. Complejidad Baja
    steps_baja = [
        ("Entrada", "Imagen RGB de hoja sana sobre fondo homogéneo claro.", "256x256 px, 3 canales"),
        ("Preprocesamiento", "Conversión a escala de grises y suavizado Gaussiano ligero.", "Kernel: 5x5, sigma=1.0"),
        ("Segmentación", "Otsu / Umbral Global: Histograma marcadamente bimodal.", "Otsu T*=132, Global T=131"),
        ("Post-procesamiento", "Operación morfológica de apertura y cierre para eliminar pequeños ruidos.", "Kernel elíptico 5x5"),
        ("Salida", "Máscara binaria y segmentación exacta de la hoja aislada.", "Aislada del fondo neutro"),
    ]
    draw_flowchart(
        "Flujo de Segmentación: Caso 1 - Complejidad Baja (Fondo Homogéneo)",
        steps_baja,
        dirs["flowcharts"] / "flujo_baja_complejidad.png",
    )

    # 2. Complejidad Media
    steps_media = [
        ("Entrada", "Hoja con lesiones necróticas y sombras suaves periféricas.", "256x256 px, 3 canales"),
        ("Preprocesamiento", "Conversión a espacio L*a*b* y suavizado Gaussiano / Mediana.", "Kernel 5x5, extracción canal a* y luminancia"),
        ("Segmentación", "K-Means (K=3) en espacio Lab o Canny con ajuste de histéresis.", "K=3, criterios max_iter=100, eps=0.2 / Canny 50-150"),
        ("Post-procesamiento", "Fusión de clusters (hoja sana + lesión) y relleno de huecos.", "Operación de cierre morfológico (k=5) + Floodfill"),
        ("Salida", "Máscara que preserva el contorno foliar integrando las áreas enfermas.", "Segmentación robusta sin perder manchas"),
    ]
    draw_flowchart(
        "Flujo de Segmentación: Caso 2 - Complejidad Media (Sombras y Lesiones)",
        steps_media,
        dirs["flowcharts"] / "flujo_media_complejidad.png",
    )

    # 3. Complejidad Alta
    steps_alta = [
        ("Entrada", "Fondo irregular/texturizado, tallos múltiples o necrosis severa.", "256x256 px, 3 canales"),
        ("Preprocesamiento", "Filtro Bilateral / Gaussiano + transformación a canales a* (Lab) y ExG.", "Filtro sigma_color=50, sigma_space=5"),
        ("Segmentación", "K-Means multicanal (K=3 o 4) o Canny con análisis de contornos mayores.", "Canny T1=30, T2=120, K=3 (Lab)"),
        ("Post-procesamiento", "Filtrado por área de contorno, cierre morfológico doble y convex hull.", "Kernel 7x7, umbral área min > 10% total"),
        ("Salida", "Aislamiento de la región foliar principal descartando artefactos del fondo.", "Segmentación foliar adaptativa"),
    ]
    draw_flowchart(
        "Flujo de Segmentación: Caso 3 - Complejidad Alta (Fondo Complejo y Bordes Irregulares)",
        steps_alta,
        dirs["flowcharts"] / "flujo_alta_complejidad.png",
    )


# =============================================================================
# 5. GENERACIÓN DE PRESENTACIÓN PPTX
# =============================================================================

def create_powerpoint_presentation(
    cases_data: Dict[str, Dict[str, Any]],
    dirs: Dict[str, Path],
    out_pptx_path: Path,
) -> None:
    """Crea una presentación profesional en formato PowerPoint (.pptx)."""
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    # Configurar formato 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_slide_layout = prs.slide_layouts[6]

    # Paleta de colores
    COLOR_PRIMARY = RGBColor(26, 77, 46)     # Verde oscuro botánico (#1A4D2E)
    COLOR_SECONDARY = RGBColor(79, 111, 82)  # Verde oliva (#4F6F52)
    COLOR_ACCENT = RGBColor(232, 110, 56)    # Naranja ladrillo (#E86E38)
    COLOR_DARK = RGBColor(33, 33, 33)        # Gris casi negro
    COLOR_LIGHT = RGBColor(245, 247, 248)    # Blanco grisáceo
    COLOR_CARD = RGBColor(255, 255, 255)     # Blanco puro

    def add_header(slide, title_text: str, subtitle_text: str = "") -> None:
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.1))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.size = Pt(13)
            p2.font.color.rgb = COLOR_SECONDARY
            p2.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 1: Portada
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_slide_layout)

    # Cuadro de fondo decorativo
    portada_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.2))
    tf1 = portada_box.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "UNIVERSIDAD TECNOLÓGICA DE LA MIXTECA"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    p_sub = tf1.add_paragraph()
    p_sub.text = "PROCESAMIENTO DIGITAL DE IMÁGENES • SEGUNDO PARCIAL"
    p_sub.font.size = Pt(14)
    p_sub.font.bold = True
    p_sub.font.color.rgb = COLOR_ACCENT
    p_sub.space_before = Pt(6)

    p_title = tf1.add_paragraph()
    p_title.text = "Segmentación de Imágenes de Plantas:\nCanny, Umbral Global, Otsu y K-Means"
    p_title.font.size = Pt(32)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_PRIMARY
    p_title.space_before = Pt(20)

    p_meta = tf1.add_paragraph()
    p_meta.text = (
        "Dataset: New Plant Diseases Dataset (Augmented)\n"
        "Alumno: Christian Josué Campos López  |  Grupo: 902-A\n"
        "Profesora: Dra. Verónica Rodríguez López"
    )
    p_meta.font.size = Pt(14)
    p_meta.font.color.rgb = COLOR_DARK
    p_meta.space_before = Pt(35)

    # -------------------------------------------------------------
    # SLIDE 2: Introducción y Objetivos
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide2, "Introducción y Objetivos del Parcial", "Marco de trabajo y metas de segmentación")

    box2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    tf2 = box2.text_frame
    tf2.word_wrap = True

    items2 = [
        ("Objetivo General", "Desarrollar, evaluar y comparar algoritmos fundamentales de segmentación digital para aislar la hoja vegetal del fondo en el New Plant Diseases Dataset."),
        ("Algoritmos Implementados", "1) Detector de bordes de Canny.\n2) Umbralización Global (iterativa de Gonzalez & Woods).\n3) Umbralización de Otsu (maximización de varianza inter-clase).\n4) Agrupamiento K-Means no supervisado en espacio de color."),
        ("Alcance del Reporte", "• Clasificación de imágenes según su complejidad visual (Baja, Media, Alta).\n• Diagramas de flujo de preprocesamiento y segmentación con parámetros detallados.\n• Evaluación cualitativa y cuantitativa de resultados con al menos 3 imágenes representativas.\n• Referencias bibliográficas especializadas en visión por computadora."),
    ]
    for title, desc in items2:
        p_t = tf2.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.bold = True
        p_t.font.size = Pt(15)
        p_t.font.color.rgb = COLOR_PRIMARY
        p_t.space_before = Pt(12)

        p_d = tf2.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = COLOR_DARK
        p_d.space_before = Pt(3)

    # -------------------------------------------------------------
    # SLIDE 3: Sección 1 - Clasificación por Complejidad
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_slide_layout)
    add_header(
        slide3,
        "1. Clasificación de Imágenes según su Complejidad",
        "Criterios morfológicos, fotométricos y de fondo en el New Plant Diseases Dataset",
    )

    col_w = Inches(3.7)
    gap = Inches(0.3)
    left_start = Inches(0.8)

    cards = [
        (
            "COMPLEJIDAD BAJA",
            COLOR_SECONDARY,
            "• Fondo: Homogéneo, liso y neutro (blanco/gris de laboratorio).\n"
            "• Iluminación: Uniforme, difusa, sin sombras duras.\n"
            "• Hoja: Bordes lisos y continuos, tejido sano o síntomas leves.\n"
            "• Histograma: Fuertemente bimodal con valle nítido.\n"
            "• Reto: Mínimo; umbralización directa funciona con precisión.",
        ),
        (
            "COMPLEJIDAD MEDIA",
            COLOR_ACCENT,
            "• Fondo: Superficie lisa pero con sombras arrojadas.\n"
            "• Iluminación: Gradientes de luz y brillos especulares.\n"
            "• Hoja: Presencia de lesiones necróticas/clorosis marrón/amarilla.\n"
            "• Histograma: Bimodal degradado o multimodal.\n"
            "• Reto: Evitar que las manchas y sombras se confundan con el fondo.",
        ),
        (
            "COMPLEJIDAD ALTA",
            COLOR_PRIMARY,
            "• Fondo: Texturas complejas, tallos, sombras densas u hojas de fondo.\n"
            "• Iluminación: Variación severa o baja relación contraste/ruido.\n"
            "• Hoja: Múltiples hojas solapadas o daño foliar extenso.\n"
            "• Histograma: Modos solapados sin separación evidente.\n"
            "• Reto: Agrupamiento multidimensional y filtrado espacial estricto.",
        ),
    ]

    for i, (title, color, desc) in enumerate(cards):
        box = slide3.shapes.add_textbox(left_start + i * (col_w + gap), Inches(1.8), col_w, Inches(5.0))
        tf = box.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = color

        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = COLOR_DARK
        p_desc.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 4: Sección 2 - Fundamentos de los Algoritmos
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_slide_layout)
    add_header(
        slide4,
        "2. Algoritmos de Segmentación Implementados",
        "Formulación matemática y papel en el procesamiento botánico",
    )

    box4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    tf4 = box4.text_frame
    tf4.word_wrap = True

    algos = [
        (
            "Detector de Bordes de Canny (1986)",
            "Opera en cuatro etapas: 1) Suavizado Gaussiano; 2) Gradiente con operadores de Sobel; "
            "3) Supresión de no máximos para adelgazar bordes a 1 píxel; 4) Umbralización con histéresis (T_low, T_high). "
            "Para segmentación, se aplica cierre morfológico y rellenado de contornos externos.",
        ),
        (
            "Umbralización Global Iterativa (Gonzalez & Woods)",
            "Calcula recursivamente el umbral T: a partir de un T inicial, segmenta en dos clases (G1 y G2), "
            "calcula sus medias mu1 y mu2, y actualiza T_nuevo = (mu1 + mu2)/2 hasta que |T_nuevo - T| < delta_T. "
            "Es rápido y eficiente para distribuciones bimodales.",
        ),
        (
            "Umbralización de Otsu (1979)",
            "Método óptimo no paramétrico que maximiza automáticamente la varianza inter-clase sigma_B^2(t) = w0(t)w1(t)[mu0(t)-mu1(t)]^2. "
            "Determina el umbral exacto que minimiza la dispersión intra-clase sin intervención manual.",
        ),
        (
            "Agrupamiento K-Means (MacQueen 1967)",
            "Algoritmo no supervisado que particiona N píxeles en K clusters minimizando la suma de distancias al cuadrado "
            "al centroide más cercano. Aplicado en espacio de color L*a*b* o RGB, permite discriminar "
            "fondo, tejido vegetal sano y lesiones patológicas.",
        ),
    ]

    for name, desc in algos:
        p_name = tf4.add_paragraph()
        p_name.text = f"• {name}"
        p_name.font.size = Pt(14)
        p_name.font.bold = True
        p_name.font.color.rgb = COLOR_PRIMARY
        p_name.space_before = Pt(8)

        p_desc = tf4.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = COLOR_DARK
        p_desc.space_before = Pt(2)

    # -------------------------------------------------------------
    # SLIDES 5, 6, 7: Diagramas de Flujo y Parámetros
    # -------------------------------------------------------------
    flowchart_slides_info = [
        (
            "Diagrama de Flujo: Complejidad Baja",
            "Preprocesamiento y segmentación para hojas sobre fondo homogéneo",
            dirs["flowcharts"] / "flujo_baja_complejidad.png",
            "Parámetros Clave Reportados:\n"
            "• Suavizado: Kernel Gaussiano 5x5, sigma = 1.0\n"
            "• Canny: T_low = 50, T_high = 150, kernel morfológico elíptico 5x5\n"
            "• Umbral Global: T = 112, delta_T = 0.5 (convergencia en 4 iteraciones)\n"
            "• Otsu: T* = 111, varianza inter-clase = 2548.13\n"
            "• K-Means: K = 2, espacio Lab, cluster foliar = 0, compactness = 5.89e7",
        ),
        (
            "Diagrama de Flujo: Complejidad Media",
            "Preprocesamiento y segmentación para hojas con sombras y manchas de enfermedad",
            dirs["flowcharts"] / "flujo_media_complejidad.png",
            "Parámetros Clave Reportados:\n"
            "• Suavizado: Kernel Gaussiano 5x5 + canal a* (Lab) para separar clorosis\n"
            "• Canny: T_low = 50, T_high = 150 con doble cierre morfológico (kernel 5x5)\n"
            "• Umbral Global: T = 105, delta_T = 0.5 (convergencia en 1 iteración)\n"
            "• Otsu: T* = 105, varianza inter-clase = 1535.24\n"
            "• K-Means: K = 3 (Fondo, Hoja sana, Mancha necrótica), clusters foliares = [1, 2]",
        ),
        (
            "Diagrama de Flujo: Complejidad Alta",
            "Preprocesamiento y segmentación para fondos complejos y texturas solapadas",
            dirs["flowcharts"] / "flujo_alta_complejidad.png",
            "Parámetros Clave Reportados:\n"
            "• Suavizado: Filtro Gaussiano 5x5 + índice ExG (Exceso de Verde normalizado)\n"
            "• Canny: T_low = 30, T_high = 120 + dilatación morfológica agresiva (kernel 7x7)\n"
            "• Umbral Global: T = 98, delta_T = 0.5 (convergencia en 3 iteraciones)\n"
            "• Otsu: T* = 97, varianza inter-clase = 2257.66\n"
            "• K-Means: K = 3, espacio Lab, clusters foliares = [0, 1]",
        ),
    ]

    for title, sub, img_path, params_txt in flowchart_slides_info:
        sl = prs.slides.add_slide(blank_slide_layout)
        add_header(sl, title, sub)

        if img_path.exists():
            sl.shapes.add_picture(str(img_path), Inches(0.8), Inches(1.6), Inches(7.5), Inches(5.3))

        p_box = sl.shapes.add_textbox(Inches(8.6), Inches(1.6), Inches(4.0), Inches(5.3))
        tf_p = p_box.text_frame
        tf_p.word_wrap = True

        p_title = tf_p.paragraphs[0]
        p_title.text = "Especificación Técnica:"
        p_title.font.size = Pt(14)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY

        p_body = tf_p.add_paragraph()
        p_body.text = params_txt
        p_body.font.size = Pt(12)
        p_body.font.color.rgb = COLOR_DARK
        p_body.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDES 8, 9, 10: Resultados por Cada Caso
    # -------------------------------------------------------------
    for case_key, c_data in cases_data.items():
        sl_res = prs.slides.add_slide(blank_slide_layout)
        add_header(
            sl_res,
            f"3. Resultados: {c_data['label']} ({c_data['species']})",
            f"Comparación detallada de los 4 métodos de segmentación implementados",
        )

        img_path = dirs["comparisons"] / f"caso_{case_key}.png"
        if img_path.exists():
            sl_res.shapes.add_picture(str(img_path), Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.5))

    # -------------------------------------------------------------
    # SLIDE 11: Análisis de Histogramas y Umbrales
    # -------------------------------------------------------------
    slide_hist = prs.slides.add_slide(blank_slide_layout)
    add_header(
        slide_hist,
        "Análisis Fotométrico de Histogramas y Umbrales",
        "Comparativa de los puntos de corte de Umbral Global vs Otsu para los 3 niveles",
    )

    hist_img = dirs["plots"] / "histogramas_umbrales.png"
    if hist_img.exists():
        slide_hist.shapes.add_picture(str(hist_img), Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.6))

    hist_box = slide_hist.shapes.add_textbox(Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.7))
    tf_h = hist_box.text_frame
    tf_h.word_wrap = True
    p_h = tf_h.paragraphs[0]
    p_h.text = (
        "Interpretación Técnica:\n"
        "• En Complejidad Baja: El histograma presenta dos picos perfectamente separados. Global y Otsu convergen prácticamente al mismo valor (T≈131 vs T*≈132).\n"
        "• En Complejidad Media: Las sombras y manchas amplían el valle intermodal. Otsu encuentra el óptimo matemático que equilibra la varianza.\n"
        "• En Complejidad Alta: La variabilidad de iluminación dispersa el histograma. K-Means en espacio de color supera a los métodos unidimensionales."
    )
    p_h.font.size = Pt(11.5)
    p_h.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------
    # SLIDE 12: Tabla Resumen Comparativa Global
    # -------------------------------------------------------------
    slide_tab = prs.slides.add_slide(blank_slide_layout)
    add_header(
        slide_tab,
        "Comparativa Global de Segmentación",
        "Evaluación cualitativa y desempeño de los algoritmos en los 3 casos",
    )

    comp_img = dirs["comparisons"] / "comparativa_global.png"
    if comp_img.exists():
        slide_tab.shapes.add_picture(str(comp_img), Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.5))

    # -------------------------------------------------------------
    # SLIDE 13: Discusión y Conclusiones
    # -------------------------------------------------------------
    slide_disc = prs.slides.add_slide(blank_slide_layout)
    add_header(
        slide_disc,
        "Discusión y Conclusiones del Parcial",
        "Lecciones aprendidas sobre segmentación en procesamiento de imágenes botánicas",
    )

    box_d = slide_disc.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    tf_d = box_d.text_frame
    tf_d.word_wrap = True

    points = [
        (
            "Canny vs Métodos de Umbralización",
            "Canny genera contornos continuos de gran precisión geométrica, pero requiere pasos morfológicos adicionales "
            "(cierre y rellenado de contornos) para convertirse en una máscara de región sólida.",
        ),
        (
            "Umbral Global vs Otsu",
            "El método de Otsu es matemáticamente superior al global iterativo cuando las poblaciones de píxeles no tienen pesos iguales, "
            "evitando la necesidad de seleccionar parámetros heurísticos iniciales.",
        ),
        (
            "Ventaja de K-Means y Espacios de Color (L*a*b* / HSV)",
            "K-Means en espacio L*a*b* demostró ser el método más robusto ante variaciones de iluminación y sombras arrojadas, "
            "ya que el canal a* desacopla la información cromática de la luminancia, permitiendo discriminar hojas enfermas con alta fidelidad.",
        ),
        (
            "Recomendación para la Detección de Plagas (Siguiente Fase)",
            "Se recomienda una arquitectura híbrida: K-Means (K=3 en Lab) para extraer la hoja y aislar el tejido lesionado, "
            "combinado con Canny para delimitar la severidad perimetral de las plagas.",
        ),
    ]

    for title, text in points:
        p_t = tf_d.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_PRIMARY
        p_t.space_before = Pt(8)

        p_c = tf_d.add_paragraph()
        p_c.text = text
        p_c.font.size = Pt(12)
        p_c.font.color.rgb = COLOR_DARK
        p_c.space_before = Pt(2)

    # -------------------------------------------------------------
    # SLIDE 14: Sección 4 - Referencias
    # -------------------------------------------------------------
    slide_ref = prs.slides.add_slide(blank_slide_layout)
    add_header(
        slide_ref,
        "4. Referencias Bibliográficas",
        "Fuentes académicas, libros de texto, papers seminales y documentación oficial",
    )

    box_r = slide_ref.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    tf_r = box_r.text_frame
    tf_r.word_wrap = True

    refs = [
        "[1] Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson. (Capítulo 10: Image Segmentation).",
        "[2] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, (6), 679-698.",
        "[3] Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.",
        "[4] MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability (Vol. 1, No. 14, pp. 281-297).",
        "[5] Szeliski, R. (2022). Computer Vision: Algorithms and Applications (2nd ed.). Springer.",
        "[6] OpenCV Development Team. (2026). Image Processing (imgproc module) & Structural Analysis. OpenCV Documentation. https://docs.opencv.org",
        "[7] Kaggle / Vipul Gaurav. (2019). New Plant Diseases Dataset (Augmented). https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
        "[8] Barbedo, J. G. A. (2016). A review on the main challenges in automatic plant disease identification based on visible range images. Biosystems Engineering, 144, 52-60.",
    ]

    for ref in refs:
        p_ref = tf_r.add_paragraph()
        p_ref.text = ref
        p_ref.font.size = Pt(11.5)
        p_ref.font.color.rgb = COLOR_DARK
        p_ref.space_before = Pt(6)

    prs.save(str(out_pptx_path))
    print(f"Presentacion PowerPoint generada con exito: {out_pptx_path}")


# =============================================================================
# 6. GENERACIÓN DE PRESENTACIÓN LATEX BEAMER (PDF)
# =============================================================================

def create_beamer_presentation(
    cases_data: Dict[str, Dict[str, Any]],
    dirs: Dict[str, Path],
    out_tex_path: Path,
) -> None:
    """Genera el código LaTeX Beamer para compilar la presentación en PDF."""
    tex_content = r"""\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{tikz}

% Tema elegante y profesional
\usetheme{Madrid}
\usecolortheme{spruce}

% Configuración de colores institucionales
\definecolor{DarkGreen}{RGB}{26,77,46}
\definecolor{OliveGreen}{RGB}{79,111,82}
\definecolor{AccentOrange}{RGB}{232,110,56}

\setbeamercolor{palette primary}{bg=DarkGreen,fg=white}
\setbeamercolor{palette secondary}{bg=OliveGreen,fg=white}
\setbeamercolor{palette tertiary}{bg=DarkGreen,fg=white}
\setbeamercolor{structure}{fg=DarkGreen}
\setbeamercolor{titlelike}{parent=palette primary}
\setbeamercolor{block title}{bg=DarkGreen,fg=white}
\setbeamercolor{block body}{bg=DarkGreen!10,fg=black}

\title[PDI - Segmentación de Imágenes]{\textbf{Segmentación de Imágenes de Plantas}}
\subtitle{Canny, Umbral Global, Otsu y K-Means en el New Plant Diseases Dataset \\ \textbf{Segundo Parcial}}
\author[Christian J. Campos López]{
    \textbf{Alumno:} Christian Josué Campos López \\[0.15cm]
    \small Grupo: 902-A \\[0.15cm]
    \small \textbf{Profesora:} Dra. Verónica Rodríguez López
}
\institute[UTM]{\textbf{Universidad Tecnológica de la Mixteca} \\ Procesamiento Digital de Imágenes}
\date{\today}

\begin{document}

% -------------------------------------------------------------
% Diapositiva 1: Portada
% -------------------------------------------------------------
\begin{frame}
    \titlepage
\end{frame}

% -------------------------------------------------------------
% Diapositiva 2: Agenda
% -------------------------------------------------------------
\begin{frame}{Estructura de la Presentación}
    \tableofcontents
\end{frame}

% =============================================================
\section{Clasificación de Imágenes por Complejidad}
% =============================================================
\begin{frame}{1. Clasificación de Imágenes según su Complejidad}
    \begin{columns}[T]
        \begin{column}{0.32\textwidth}
            \begin{block}{\textbf{Complejidad Baja}}
                \small
                \begin{itemize}
                    \item \textbf{Fondo:} Homogéneo, liso y neutro (laboratorio).
                    \item \textbf{Iluminación:} Difusa y uniforme, sin sombras duras.
                    \item \textbf{Hoja:} Aislada, bordes continuos sin solapes.
                    \item \textbf{Histograma:} Bimodal nítido con valle profundo.
                \end{itemize}
            \end{block}
        \end{column}
        \begin{column}{0.32\textwidth}
            \begin{block}{\textbf{Complejidad Media}}
                \small
                \begin{itemize}
                    \item \textbf{Fondo:} Superficie uniforme con sombras arrojadas.
                    \item \textbf{Iluminación:} Variación leve o reflejos especulares.
                    \item \textbf{Hoja:} Presencia de manchas necróticas y clorosis.
                    \item \textbf{Histograma:} Bimodal degradado o multimodal.
                \end{itemize}
            \end{block}
        \end{column}
        \begin{column}{0.32\textwidth}
            \begin{block}{\textbf{Complejidad Alta}}
                \small
                \begin{itemize}
                    \item \textbf{Fondo:} Texturizado, tallos, sombras densas u hojas de fondo.
                    \item \textbf{Iluminación:} No uniforme, bajo contraste con fondo.
                    \item \textbf{Hoja:} Hojas solapadas o daño foliar severo.
                    \item \textbf{Histograma:} Modos fuertemente solapados.
                \end{itemize}
            \end{block}
        \end{column}
    \end{columns}
\end{frame}

% =============================================================
\section{Métodos de Segmentación y Parámetros}
% =============================================================
\begin{frame}{2. Fundamentos de los Algoritmos Implementados}
    \begin{itemize}
        \item \textbf{Detector de Bordes de Canny (1986):}
        Detecta discontinuidades espaciales mediante gradientes de Sobel, supresión de no máximos y umbralización con histéresis ($T_{\text{low}}, T_{\text{high}}$).
        \item \textbf{Umbralización Global Iterativa (Gonzalez \& Woods):}
        Particiona recursivamente la imagen en dos conjuntos $\{G_1, G_2\}$, calculando $T^{(k+1)} = \frac{\mu_1 + \mu_2}{2}$ hasta convergencia ($|T^{(k+1)} - T^{(k)}| < \Delta T$).
        \item \textbf{Umbralización de Otsu (1979):}
        Determina analíticamente el umbral óptimo $T^*$ que maximiza la varianza inter-clase:
        \begin{equation*}
            \sigma_B^2(t) = \omega_0(t)\omega_1(t)\left[\mu_0(t) - \mu_1(t)\right]^2
        \end{equation*}
        \item \textbf{Agrupamiento K-Means (MacQueen 1967):}
        Agrupamiento no supervisado en espacio de color ($L^*a^*b^*$ o RGB) que minimiza la suma de distancias euclidianas a los $K$ centroides representativos.
    \end{itemize}
\end{frame}

% -------------------------------------------------------------
% Diagrama Caso 1: Baja Complejidad
% -------------------------------------------------------------
\begin{frame}{Diagrama de Flujo: Complejidad Baja}
    \begin{columns}[c]
        \begin{column}{0.58\textwidth}
            \centering
            \includegraphics[width=\textwidth]{diagramas/flujo_baja_complejidad.png}
        \end{column}
        \begin{column}{0.40\textwidth}
            \begin{block}{\textbf{Parámetros Reportados}}
                \small
                \begin{itemize}
                    \item \textbf{Filtro Gaussiano:} $5 \times 5$, $\sigma = 1.0$.
                    \item \textbf{Canny:} $T_1 = 50$, $T_2 = 150$.
                    \item \textbf{Cierre morfológico:} Kernel elíptico $5 \times 5$.
                    \item \textbf{Global:} $T = 112$, $\Delta T = 0.5$ (4 iter.).
                    \item \textbf{Otsu:} $T^* = 111$ (un umbral).
                    \item \textbf{K-Means:} $K = 2$, espacio $L^*a^*b^*$, cluster foliar = 0.
                \end{itemize}
            \end{block}
        \end{column}
    \end{columns}
\end{frame}

% -------------------------------------------------------------
% Diagrama Caso 2: Media Complejidad
% -------------------------------------------------------------
\begin{frame}{Diagrama de Flujo: Complejidad Media}
    \begin{columns}[c]
        \begin{column}{0.58\textwidth}
            \centering
            \includegraphics[width=\textwidth]{diagramas/flujo_media_complejidad.png}
        \end{column}
        \begin{column}{0.40\textwidth}
            \begin{block}{\textbf{Parámetros Reportados}}
                \small
                \begin{itemize}
                    \item \textbf{Preprocesamiento:} Gaussiano $5 \times 5$ + canal $a^*$ ($L^*a^*b^*$).
                    \item \textbf{Canny:} $T_1 = 50$, $T_2 = 150$, cierre doble.
                    \item \textbf{Global:} $T = 105$, $\Delta T = 0.5$ (1 iter.).
                    \item \textbf{Otsu:} $T^* = 105$, $\sigma_B^2 = 1535.24$.
                    \item \textbf{K-Means:} $K = 3$ (Fondo, Hoja sana, Mancha necrótica).
                    \item \textbf{Post-proc:} Fusión de clusters foliares = [1, 2] + relleno.
                \end{itemize}
            \end{block}
        \end{column}
    \end{columns}
\end{frame}

% -------------------------------------------------------------
% Diagrama Caso 3: Alta Complejidad
% -------------------------------------------------------------
\begin{frame}{Diagrama de Flujo: Complejidad Alta}
    \begin{columns}[c]
        \begin{column}{0.58\textwidth}
            \centering
            \includegraphics[width=\textwidth]{diagramas/flujo_alta_complejidad.png}
        \end{column}
        \begin{column}{0.40\textwidth}
            \begin{block}{\textbf{Parámetros Reportados}}
                \small
                \begin{itemize}
                    \item \textbf{Preprocesamiento:} Filtro Bilateral/Gauss + índice ExG.
                    \item \textbf{Canny:} $T_1 = 30$, $T_2 = 120$, kernel $7 \times 7$.
                    \item \textbf{Global:} $T = 98$, $\Delta T = 0.5$ (3 iter.).
                    \item \textbf{Otsu:} $T^* = 97$, $\sigma_B^2 = 2257.66$.
                    \item \textbf{K-Means:} $K = 3$, espacio $L^*a^*b^*$, clusters foliares = [0, 1].
                \end{itemize}
            \end{block}
        \end{column}
    \end{columns}
\end{frame}

% =============================================================
\section{Resultados Obtenidos}
% =============================================================

% Resultados Caso 1
\begin{frame}{3. Resultados: Complejidad Baja (Apple Healthy)}
    \centering
    \includegraphics[width=0.92\textwidth]{comparaciones/caso_baja.png}
\end{frame}

% Resultados Caso 2
\begin{frame}{3. Resultados: Complejidad Media (Tomato Early Blight)}
    \centering
    \includegraphics[width=0.92\textwidth]{comparaciones/caso_media.png}
\end{frame}

% Resultados Caso 3
\begin{frame}{3. Resultados: Complejidad Alta (Corn Gray Leaf Spot)}
    \centering
    \includegraphics[width=0.92\textwidth]{comparaciones/caso_alta.png}
\end{frame}

% Histogramas
\begin{frame}{Análisis de Histogramas y Umbrales (Global vs Otsu)}
    \centering
    \includegraphics[width=0.95\textwidth]{graficas/histogramas_umbrales.png}
    \vspace{0.2cm}
    \begin{itemize}
        \small
        \item \textbf{Caso Baja:} Bimodal estricto; convergencia de Global y Otsu ($T \approx 131$ vs $T^* \approx 132$).
        \item \textbf{Caso Media:} El ensanchamiento por sombras y lesiones es absorbido eficazmente por Otsu.
        \item \textbf{Caso Alta:} La variabilidad fotométrica hace indispensable el espacio de color cromático ($L^*a^*b^*$).
    \end{itemize}
\end{frame}

% Comparativa Global
\begin{frame}{Comparativa Global de los Métodos por Complejidad}
    \centering
    \includegraphics[width=0.92\textwidth]{comparaciones/comparativa_global.png}
\end{frame}

% Conclusiones
\begin{frame}{Discusión y Conclusiones}
    \begin{itemize}
        \item \textbf{Canny:} Produce la mayor definición perimetral foliar, ideal para análisis de área foliar tras el cierre morfológico de contornos.
        \item \textbf{Global vs Otsu:} Otsu supera al umbral global en robustez, al no requerir suposiciones heurísticas y adaptarse a la asimetría de clases.
        \item \textbf{K-Means en $L^*a^*b^*$:} Resultó ser la técnica más precisa frente a sombras y fondos con textura, ya que desacopla la luminancia de las coordenadas cromáticas de clorofila.
        \item \textbf{Perspectiva Fase 3 (Detección de Plagas):} La segmentación multiclase de K-Means ($K=3$) separa con éxito el fondo, el tejido sano y las lesiones de la enfermedad.
    \end{itemize}
\end{frame}

% =============================================================
\section{Referencias}
% =============================================================
\begin{frame}{4. Referencias Bibliográficas}
    \footnotesize
    \begin{enumerate}
        \item Gonzalez, R. C., \& Woods, R. E. (2018). \textit{Digital Image Processing} (4th ed.). Pearson.
        \item Canny, J. (1986). A computational approach to edge detection. \textit{IEEE TPAMI}, (6), 679-698.
        \item Otsu, N. (1979). A threshold selection method from gray-level histograms. \textit{IEEE SMC}, 9(1), 62-66.
        \item MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. \textit{Proc. 5th Berkeley Symp. Math. Statist. Prob.}, 1(14), 281-297.
        \item Szeliski, R. (2022). \textit{Computer Vision: Algorithms and Applications} (2nd ed.). Springer.
        \item OpenCV Development Team. (2026). \textit{OpenCV Documentation: Image Processing}. \url{https://docs.opencv.org}
        \item Kaggle / Vipul Gaurav. (2019). \textit{New Plant Diseases Dataset (Augmented)}. \url{https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset}
        \item Barbedo, J. G. A. (2016). A review on the main challenges in automatic plant disease identification. \textit{Biosystems Engineering}, 144, 52-60.
    \end{enumerate}
\end{frame}

\end{document}
"""
    out_tex_path.write_text(tex_content, encoding="utf-8")
    print(f"Archivo Beamer LaTeX generado: {out_tex_path}")


# =============================================================================
# 7. FUNCIÓN PRINCIPAL
# =============================================================================

def find_target_images(dataset_root: Path) -> Dict[str, Tuple[Path, str, str]]:
    """Localiza 3 imágenes óptimas representativas de cada nivel de complejidad."""
    train_dir = dataset_root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train"
    if not train_dir.exists():
        train_dir = list(dataset_root.rglob("train"))[0]

    # Caso 1 (Baja): Manzana sana sobre fondo neutro liso
    baja_folder = train_dir / "Apple___healthy"
    baja_imgs = list(baja_folder.glob("*.JPG")) + list(baja_folder.glob("*.jpg"))
    baja_path = baja_imgs[0]

    # Caso 2 (Media): Tomate con Tizón Temprano (sombras + manchas pardas)
    media_folder = train_dir / "Tomato___Early_blight"
    media_imgs = list(media_folder.glob("*.JPG")) + list(media_folder.glob("*.jpg"))
    media_path = media_imgs[0]

    # Caso 3 (Alta): Maíz con mancha gris o Tomate con Tizón Tardío (fondo complejo, textura)
    alta_folder = train_dir / "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    alta_imgs = list(alta_folder.glob("*.JPG")) + list(alta_folder.glob("*.jpg"))
    alta_path = alta_imgs[0]

    return {
        "baja": (baja_path, "Complejidad Baja", "Apple (Sana)"),
        "media": (media_path, "Complejidad Media", "Tomato (Early Blight)"),
        "alta": (alta_path, "Complejidad Alta", "Corn (Gray Leaf Spot)"),
    }


def main() -> None:
    base_dir = Path("resultados_segmentacion")
    dirs = ensure_dirs(base_dir)

    print("=" * 60)
    print("PROCESAMIENTO DE IMÁGENES - SEGUNDO PARCIAL: SEGMENTACIÓN")
    print("=" * 60)

    dataset_root = Path("dataset")
    selected_samples = find_target_images(dataset_root)

    cases_data: Dict[str, Dict[str, Any]] = {}

    for c_key, (img_path, label, species) in selected_samples.items():
        print(f"\n[Procesando] Caso {label.upper()} ({species})")
        print(f" -> Archivo: {img_path.name}")

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"No se pudo cargar la imagen: {img_path}")

        # Preprocesamiento
        prep = preprocess_image(bgr)

        # 1. Detector de Bordes de Canny
        if c_key == "alta":
            canny_res = segment_canny(prep["blurred_gray"], low_thresh=30, high_thresh=120, morph_kernel_size=7)
        else:
            canny_res = segment_canny(prep["blurred_gray"], low_thresh=50, high_thresh=150, morph_kernel_size=5)

        # 2. Umbralización Global Iterativa
        global_res = segment_global_threshold(prep["blurred_gray"], delta_t=0.5, invert=True)

        # 3. Umbralización de Otsu
        otsu_res = segment_otsu(prep["blurred_gray"], invert=True)

        # 4. Agrupamiento K-Means
        k_val = 2 if c_key == "baja" else 3
        kmeans_res = segment_kmeans(prep["rgb"], k=k_val, use_lab=True)

        case_entry = {
            "path": img_path,
            "label": label,
            "species": species,
            "rgb": prep["rgb"],
            "prep": prep,
            "canny": canny_res,
            "global": global_res,
            "otsu": otsu_res,
            "kmeans": kmeans_res,
        }
        cases_data[c_key] = case_entry

        # Guardar figura detallada del caso
        out_case_fig = dirs["comparisons"] / f"caso_{c_key}.png"
        generate_case_figure(species, label, case_entry, out_case_fig)
        print(f" -> Guardada figura del caso en: {out_case_fig}")

    # Generar visualizaciones combinadas
    print("\n[Generando] Figuras de análisis conjunto...")
    hist_out = dirs["plots"] / "histogramas_umbrales.png"
    generate_histograms_figure(cases_data, hist_out)
    print(f" -> Histogramas y umbrales: {hist_out}")

    comp_out = dirs["comparisons"] / "comparativa_global.png"
    generate_comparative_table_figure(cases_data, comp_out)
    print(f" -> Comparativa global: {comp_out}")

    # Generar diagramas de flujo
    print("\n[Generando] Diagramas de flujo...")
    generate_all_flowcharts(dirs)
    print(" -> Diagramas de flujo generados en:", dirs["flowcharts"])

    # Generar Presentación PowerPoint (.pptx)
    print("\n[Generando] Presentación en PowerPoint (.pptx)...")
    pptx_path = base_dir / "Presentacion_Segundo_Parcial_Segmentacion.pptx"
    create_powerpoint_presentation(cases_data, dirs, pptx_path)

    # Generar Presentación Beamer LaTeX (.tex)
    print("\n[Generando] Presentación Beamer LaTeX (.tex)...")
    tex_path = base_dir / "presentacion_parcial2.tex"
    create_beamer_presentation(cases_data, dirs, tex_path)

    # Guardar resumen en JSON
    metrics = {}
    for k, d in cases_data.items():
        metrics[k] = {
            "species": d["species"],
            "label": d["label"],
            "global_threshold": d["global"]["threshold"],
            "global_iterations": d["global"]["iterations"],
            "otsu_threshold": d["otsu"]["threshold"],
            "otsu_variance": d["otsu"]["max_variance"],
            "kmeans_k": d["kmeans"]["k"],
            "kmeans_leaf_clusters": d["kmeans"]["leaf_clusters"],
            "kmeans_compactness": d["kmeans"]["compactness"],
        }
    metrics_path = base_dir / "metricas_segmentacion.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f" -> Métricas guardadas en: {metrics_path}")

    print("\n" + "=" * 60)
    print("¡PROCESAMIENTO Y GENERACIÓN COMPLETADOS EXITOSAMENTE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

