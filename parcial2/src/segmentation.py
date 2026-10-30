"""Algoritmos de segmentación — Parcial 2.

Implementa los cuatro métodos requeridos por el PDF del proyecto:
  1. Detector de bordes de Canny
  2. Umbralización Global iterativa (Gonzalez & Woods)
  3. Umbralización de Otsu (un umbral, maximización de varianza inter-clase)
  4. Agrupamiento K-Means en espacio de color Lab / RGB

Cada función devuelve un diccionario con la máscara resultante,
los parámetros usados y métricas relevantes para el reporte.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


# =============================================================================
# UTILIDADES
# =============================================================================

def apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aplica una máscara binaria dejando el fondo en negro."""
    result = np.zeros_like(rgb)
    result[mask > 0] = rgb[mask > 0]
    return result


# =============================================================================
# 1. DETECTOR DE BORDES DE CANNY
# =============================================================================

def segment_canny(
    gray: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    morph_kernel_size: int = 5,
) -> Dict[str, Any]:
    """Segmentación mediante el detector de bordes de Canny.

    Proceso:
        1. Suavizado Gaussiano (incluido internamente por cv2.Canny).
        2. Cálculo de gradiente con Sobel.
        3. Supresión de no máximos (adelgazamiento a 1 px).
        4. Umbralización con histéresis (T_low, T_high).
        5. Cierre morfológico para conectar contornos.
        6. Rellenado de contornos externos para obtener máscara binaria.

    Args:
        gray: Imagen en escala de grises preprocesada.
        low_thresh: Umbral bajo de histéresis (T_low).
        high_thresh: Umbral alto de histéresis (T_high).
        morph_kernel_size: Tamaño del kernel de cierre morfológico.

    Returns:
        Diccionario con ``edges``, ``closed_edges``, ``mask`` y ``params``.
    """
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(gray)
    if contours:
        max_area = max(cv2.contourArea(c) for c in contours)
        valid = [c for c in contours if cv2.contourArea(c) >= max_area * 0.05]
        cv2.drawContours(mask, valid, -1, 255, thickness=cv2.FILLED)

    # Relleno de agujeros internos (flood fill desde esquina)
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


# =============================================================================
# 2. UMBRALIZACIÓN GLOBAL ITERATIVA (GONZALEZ & WOODS)
# =============================================================================

def segment_global_threshold(
    channel: np.ndarray,
    delta_t: float = 0.5,
    max_iters: int = 100,
    invert: bool = True,
) -> Dict[str, Any]:
    """Umbralización global iterativa según Gonzalez & Woods.

    Algoritmo:
        1. T₀ = media global del canal.
        2. Segmentar en G₁ (> T) y G₂ (≤ T).
        3. μ₁ = media(G₁), μ₂ = media(G₂).
        4. T_nuevo = (μ₁ + μ₂) / 2.
        5. Repetir hasta |T_nuevo - T| < δT.

    Eficiente y exacto para distribuciones de histograma **bimodales**.

    Args:
        channel: Canal de imagen 2D en uint8.
        delta_t: Criterio de convergencia (diferencia mínima de T).
        max_iters: Número máximo de iteraciones.
        invert: Si True aplica THRESH_BINARY_INV (hoja oscura en fondo claro).

    Returns:
        Diccionario con ``threshold``, ``iterations``, ``mask`` y ``params``.
    """
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

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return {
        "threshold": thresh_val,
        "iterations": iters,
        "mask": mask,
        "params": {"delta_t": delta_t, "max_iters": max_iters, "invert": invert},
    }


# =============================================================================
# 3. UMBRALIZACIÓN DE OTSU
# =============================================================================

def segment_otsu(channel: np.ndarray, invert: bool = True) -> Dict[str, Any]:
    """Umbralización automática de Otsu (un umbral).

    Maximiza la varianza inter-clase:
        σ_B²(t) = ω₀(t)·ω₁(t)·[μ₀(t) - μ₁(t)]²

    El umbral T* minimiza la varianza intra-clase sin intervención manual.

    Args:
        channel: Canal de imagen 2D en uint8.
        invert: Si True aplica THRESH_BINARY_INV.

    Returns:
        Diccionario con ``threshold``, ``max_variance``, ``mask`` y ``params``.
    """
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    thresh_val, mask = cv2.threshold(channel, 0, 255, thresh_type + cv2.THRESH_OTSU)

    # Calcular varianza inter-clase máxima (para reporte científico)
    hist, _ = np.histogram(channel, bins=256, range=(0, 256), density=True)
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_sq = ((mu_t * omega - mu) ** 2) / (omega * (1.0 - omega))
        sigma_b_sq = np.nan_to_num(sigma_b_sq)
    max_sigma_b = float(np.max(sigma_b_sq))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return {
        "threshold": int(thresh_val),
        "max_variance": max_sigma_b,
        "mask": mask,
        "params": {"invert": invert},
    }


# =============================================================================
# 4. AGRUPAMIENTO K-MEANS
# =============================================================================

def segment_kmeans(
    image_rgb: np.ndarray,
    k: int = 3,
    use_lab: bool = True,
    max_iters: int = 100,
    epsilon: float = 0.2,
) -> Dict[str, Any]:
    """Segmentación por agrupamiento K-Means en espacio de color.

    Particiona los N píxeles en K clusters minimizando la suma de
    distancias al cuadrado al centroide más cercano. Aplicado en espacio
    L*a*b* para mejor separación perceptual de colores.

    Identificación de la hoja: se selecciona el cluster con mayor
    puntuación de verdor (G - (R+B)/2 en espacio RGB original).

    Args:
        image_rgb: Imagen en espacio RGB (uint8).
        k: Número de clusters.
        use_lab: Si True, trabaja en espacio L*a*b* para mejor separación.
        max_iters: Máximo de iteraciones del algoritmo K-Means.
        epsilon: Criterio de convergencia de error.

    Returns:
        Diccionario con ``k``, ``compactness``, ``label_map``,
        ``segmented_colors``, ``leaf_clusters``, ``mask`` y ``params``.
    """
    if use_lab:
        feature_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    else:
        feature_img = image_rgb

    h, w, _ = feature_img.shape
    pixel_vals = feature_img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, epsilon)
    compactness, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    segmented_features = centers[labels.flatten()].reshape((h, w, 3))
    label_map = labels.reshape((h, w))

    # Identificar clusters foliares por puntuación de verdor en RGB
    rgb_flat = image_rgb.reshape((-1, 3))
    cluster_scores = []
    for c_id in range(k):
        c_mask = (labels.flatten() == c_id)
        if not np.any(c_mask):
            cluster_scores.append(-999.0)
            continue
        avg_rgb = np.mean(rgb_flat[c_mask], axis=0)
        green_score = float(avg_rgb[1] - (avg_rgb[0] + avg_rgb[2]) / 2.0)
        cluster_scores.append(green_score)

    sorted_indices = np.argsort(cluster_scores)
    leaf_clusters = [int(sorted_indices[-1])]
    if k >= 3 and cluster_scores[sorted_indices[-2]] > -20:
        leaf_clusters.append(int(sorted_indices[-2]))

    mask = np.zeros((h, w), dtype=np.uint8)
    for c_id in leaf_clusters:
        mask[label_map == c_id] = 255

    k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_elem)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_elem)

    segmented_rgb = (
        cv2.cvtColor(segmented_features, cv2.COLOR_LAB2RGB) if use_lab
        else segmented_features
    )

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


# =============================================================================
# PREPROCESAMIENTO AUXILIAR
# =============================================================================

def preprocess_image(
    image_bgr: np.ndarray,
    blur_kernel: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Genera representaciones y canales preprocesados de la imagen.

    Convierte la imagen a varios espacios de color útiles para
    seleccionar el canal óptimo según el método de segmentación.

    Returns:
        Diccionario con: ``bgr``, ``rgb``, ``gray``, ``blurred_gray``,
        ``blurred_rgb``, ``lab``, ``a_channel``, ``hsv``, ``s_channel``,
        ``exg``.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, blur_kernel, sigma)
    blurred_rgb = cv2.GaussianBlur(rgb, blur_kernel, sigma)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

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

