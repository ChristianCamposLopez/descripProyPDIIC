"""Filtros e Transformaciones de Imagen — Parcial 1.

Implementa las 5 categorías del curso de PDI:
  1. Transformaciones de Intensidad (punto a punto)
  2. Filtros Estadísticos / De Orden
  3. Filtros Suavizantes (pasabajas)
  4. Filtros Realzantes (derivadas)
  5. Filtros de Segmentación y Detección de Bordes Avanzados

Diseño: patrón Template Method con clase base ``ImageFilter`` (OCP).
Cada filtro es instanciable de forma independiente o dentro de un ``Pipeline``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


# =============================================================================
# BASE
# =============================================================================

class ImageFilter(ABC):
    """Clase base para todos los filtros (Open/Closed Principle)."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica el filtro sobre *image* (BGR uint8) y devuelve el resultado."""


# =============================================================================
# 1. TRANSFORMACIONES DE INTENSIDAD (PUNTO A PUNTO)
#    Mapeo s = T(r) aplicado píxel a píxel.
# =============================================================================

class NegativeTransform(ImageFilter):
    """Negativo de la imagen: s = L - 1 - r.

    Invierte los niveles de gris. Útil para resaltar detalles claros
    sobre fondos oscuros dominantes.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        # cv2.bitwise_not equivale a 255 - r para imágenes uint8
        return cv2.bitwise_not(image)


class LogTransform(ImageFilter):
    """Transformación logarítmica: s = c · log(1 + r).

    Comprime el rango dinámico de imágenes con variaciones extremas de luz,
    expandiendo los valores oscuros y reduciendo los valores claros.

    Args:
        c: Constante de escalado. Por defecto se calcula automáticamente para
           que la salida ocupe todo el rango [0, 255].
    """

    def __init__(self, c: float | None = None) -> None:
        self.c = c

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        log_img = np.log1p(gray)           # log(1 + r)
        c = self.c if self.c is not None else 255.0 / np.log1p(255.0)
        result = np.clip(c * log_img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


class GammaTransform(ImageFilter):
    """Transformación Gamma (Ley de Potencias): s = r^γ.

    * γ < 1 → **aclara** la imagen (expande tonos oscuros).
    * γ > 1 → **oscurece** la imagen (comprime tonos claros).
    * γ = 1 → identidad.

    Implementada con una LUT de 256 entradas para máxima eficiencia.

    Args:
        gamma: Exponente γ.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma
        # Pre-calcular LUT [0..255]
        table = np.array(
            [(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8
        )
        self._lut = table

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.LUT(image, self._lut)


class ContrastStretching(ImageFilter):
    """Estiramiento de contraste (transformación por tramos / piecewise lineal).

    Amplía el intervalo de intensidad [r1, r2] al rango completo [s1, s2],
    separando tonos similares en la zona de interés.

    Tramos:
        [0,   r1] → [0,   s1]  (compresión o identidad de sombras)
        [r1,  r2] → [s1,  s2]  (estiramiento del rango útil)
        [r2, 255] → [s2, 255]  (compresión o identidad de altas luces)

    Args:
        r1, r2: Límites del rango de entrada.
        s1, s2: Límites del rango de salida.
    """

    def __init__(
        self,
        r1: int = 70,
        s1: int = 0,
        r2: int = 200,
        s2: int = 255,
    ) -> None:
        self.r1, self.s1 = r1, s1
        self.r2, self.s2 = r2, s2
        self._lut = self._build_lut()

    def _build_lut(self) -> np.ndarray:
        lut = np.zeros(256, dtype=np.uint8)
        for r in range(256):
            if r <= self.r1:
                s = int(self.s1 * r / max(self.r1, 1))
            elif r <= self.r2:
                s = int(
                    self.s1
                    + (self.s2 - self.s1) * (r - self.r1) / max(self.r2 - self.r1, 1)
                )
            else:
                s = int(
                    self.s2
                    + (255 - self.s2) * (r - self.r2) / max(255 - self.r2, 1)
                )
            lut[r] = np.clip(s, 0, 255)
        return lut

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.LUT(image, self._lut)


class BinaryThreshold(ImageFilter):
    """Umbralización binaria: corte escalonado en valor m.

    s = 0   si r < m
    s = L-1 si r >= m

    Args:
        m: Valor umbral (0–255).
        invert: Si True, intercambia blanco y negro.
    """

    def __init__(self, m: int = 128, invert: bool = False) -> None:
        self.m = m
        self.invert = invert

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_type = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, self.m, 255, thresh_type)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


class IntensityBandHighlight(ImageFilter):
    """Resaltado / Corte de bandas de intensidad.

    Aísla el rango de grises [low, high]: los píxeles dentro del rango
    se muestran con su valor original; el resto se atenúa según
    ``background_scale``.

    Args:
        low: Límite inferior del rango de interés.
        high: Límite superior del rango de interés.
        background_scale: Factor de atenuación del fondo (0=negro, 1=original).
    """

    def __init__(
        self, low: int = 100, high: int = 200, background_scale: float = 0.2
    ) -> None:
        self.low = low
        self.high = high
        self.background_scale = background_scale

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, self.low, self.high)           # 255 en banda
        result = (image * self.background_scale).astype(np.uint8)
        result[mask > 0] = image[mask > 0]
        return result


class GlobalEqualization(ImageFilter):
    """Ecualización global del histograma en el canal L (espacio LAB).

    Fórmula discreta: s_k = (L-1)/(M·N) · Σⱼ₌₀ᵏ nⱼ

    Redistribuye de forma uniforme los niveles de intensidad para
    maximizar el contraste global sin afectar la crominancia.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        equalized_l = cv2.equalizeHist(l_channel)
        merged = cv2.merge((equalized_l, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class CLAHEEqualization(ImageFilter):
    """Ecualización CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Variante adaptativa que subdivide la imagen en teselas y aplica la
    ecualización localmente, limitando la amplificación del contraste
    con ``clip_limit`` para evitar amplificar el ruido.

    Args:
        clip_limit: Umbral de recorte del histograma local.
        tile_grid_size: Tamaño de la rejilla de teselas (cols, rows).
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# =============================================================================
# 2. FILTROS ESTADÍSTICOS / DE ORDEN (ESPACIALES)
#    Operan sobre la vecindad S_{x,y} reduciendo ruido.
# =============================================================================

class MedianFilter(ImageFilter):
    """Filtro de mediana.

    Reemplaza cada píxel por la **mediana** de su vecindad.
    Filtro principal para eliminar **ruido impulsivo (sal y pimienta)**
    preservando bordes mejor que los filtros de promedio.

    Args:
        kernel_size: Tamaño del kernel (debe ser impar).
    """

    def __init__(self, kernel_size: int = 3) -> None:
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, self.kernel_size)


class MaxFilter(ImageFilter):
    """Filtro MAX (dilatación morfológica).

    Reemplaza cada píxel por el **valor máximo** de su vecindad.
    Diseñado para eliminar **ruido de pimienta** (puntos oscuros).

    Args:
        kernel_size: Tamaño del elemento estructurante cuadrado.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        return cv2.dilate(image, kernel)


class MinFilter(ImageFilter):
    """Filtro MIN (erosión morfológica).

    Reemplaza cada píxel por el **valor mínimo** de su vecindad.
    Diseñado para eliminar **ruido de sal** (puntos claros).

    Args:
        kernel_size: Tamaño del elemento estructurante cuadrado.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        return cv2.erode(image, kernel)


class MidpointFilter(ImageFilter):
    """Filtro de punto medio.

    Promedia el valor máximo y mínimo de la vecindad:
    s = (max + min) / 2

    Eficaz para ruido uniformemente distribuido.

    Args:
        kernel_size: Tamaño del elemento estructurante cuadrado.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        max_img = cv2.dilate(image.astype(np.float32), kernel)
        min_img = cv2.erode(image.astype(np.float32), kernel)
        midpoint = ((max_img + min_img) / 2.0).astype(np.uint8)
        return midpoint


class ArithmeticMeanFilter(ImageFilter):
    """Filtro de media aritmética (promedio simple).

    s(x,y) = (1/mn) · Σ f(r,c)   para (r,c) en S_{x,y}

    Indicado para reducir **ruido Gaussiano**; produce emborronamiento
    proporcional al tamaño del kernel.

    Args:
        kernel_size: Tamaño del kernel cuadrado.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.blur(image, (self.kernel_size, self.kernel_size))


class GeometricMeanFilter(ImageFilter):
    """Filtro de media geométrica.

    s(x,y) = [ Π f(r,c) ]^(1/mn)   para (r,c) en S_{x,y}

    Atenúa ruido Gaussiano **preservando mejor los detalles** que la
    media aritmética (menos suavizado en bordes).

    Args:
        kernel_size: Tamaño del kernel cuadrado.
    """

    def __init__(self, kernel_size: int = 3) -> None:
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        img_f = image.astype(np.float64) + 1.0   # evitar log(0)
        log_img = np.log(img_f)
        kernel = np.ones(
            (self.kernel_size, self.kernel_size), np.float64
        ) / (self.kernel_size ** 2)
        log_mean = cv2.filter2D(log_img, cv2.CV_64F, kernel)
        geo_mean = np.exp(log_mean) - 1.0
        return np.clip(geo_mean, 0, 255).astype(np.uint8)


# =============================================================================
# 3. FILTROS SUAVIZANTES (PASABAJAS / PROMEDIO)
#    Convolución/correlación con máscaras de promedio ponderado.
# =============================================================================

class BoxFilter(ImageFilter):
    """Filtro de Caja (Box Filter).

    Máscara 3×3 con todos los coeficientes iguales a 1/9.
    Reduce ruido a costa de **emborronar** los bordes de interés.
    """

    _KERNEL = np.ones((3, 3), np.float32) / 9.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, self._KERNEL)


class WeightedAverageFilter(ImageFilter):
    """Promedio Pesado (Weighted Average).

    Máscara 3×3 que otorga mayor peso al píxel central (4) y a los
    vecinos ortogonales (2), dividiendo entre 16.

    Kernel:  [1 2 1]
             [2 4 2]  / 16
             [1 2 1]

    Suaviza y atenúa ruido **reduciendo el emborronamiento** de estructuras
    respecto al Box Filter.
    """

    _KERNEL = np.array(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32
    ) / 16.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, self._KERNEL)


class GaussianBlurFilter(ImageFilter):
    """Filtro Gaussiano suavizante G(x,y) = exp(-(x²+y²)/(2σ²)).

    Filtro pasabajas continuo isótropo. Ampliamente usado como etapa
    previa de suavizado en detectores de bordes (Canny, LoG).

    Args:
        kernel_size: Dimensiones del kernel (deben ser impares).
        sigma_x: Desviación estándar en x (0 = calcular automáticamente).
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (5, 5),
        sigma_x: float = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, self.kernel_size, self.sigma_x)


class BilateralFilter(ImageFilter):
    """Filtro bilateral.

    Suavizado que **preserva bordes**: combina una función Gaussiana
    espacial con otra de similitud de intensidad. Ideal para eliminar
    ruido en imágenes foliares sin difuminar los contornos de lesiones.

    Args:
        d: Diámetro del vecindario de píxeles.
        sigma_color: σ en el espacio de color (similitud de intensidad).
        sigma_space: σ espacial (distancia euclidiana entre píxeles).
    """

    def __init__(
        self, d: int = 5, sigma_color: float = 35.0, sigma_space: float = 35.0
    ) -> None:
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)


# =============================================================================
# 4. FILTROS REALZANTES (AGUZADO / DERIVADAS)
#    Acentúan cambios bruscos de intensidad (bordes y discontinuidades).
# =============================================================================

class SobelFilter(ImageFilter):
    """Máscaras de Sobel — primera derivada (gradiente ∇f).

    Aproximan la derivada parcial en x e y asignando peso 2 a los
    elementos centrales. Generan bordes más gruesos y suaves que Prewitt.
    Devuelve la magnitud del gradiente normalizada a [0, 255].

    Args:
        ksize: Tamaño del kernel (1, 3, 5 o 7).
    """

    def __init__(self, ksize: int = 3) -> None:
        self.ksize = ksize

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)
        magnitude = cv2.magnitude(gx, gy)
        result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


class PrewittFilter(ImageFilter):
    """Máscaras de Prewitt — primera derivada (gradiente ∇f).

    Aproximan las derivadas parciales con coeficientes uniformes de 1.
    Más sensibles al ruido que Sobel pero más rápidas de calcular.
    Devuelve la magnitud del gradiente normalizada a [0, 255].

    Kernels:
        Gx = [[-1,0,1],[-1,0,1],[-1,0,1]]
        Gy = [[-1,-1,-1],[0,0,0],[1,1,1]]
    """

    _KX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    _KY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.filter2D(gray, cv2.CV_32F, self._KX)
        gy = cv2.filter2D(gray, cv2.CV_32F, self._KY)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


class LaplacianFilter(ImageFilter):
    """Laplaciano ∇²f — segunda derivada.

    Genera respuesta fuerte ante detalles finos, puntos y cruces por cero.

    Modos:
        * ``enhance=False``: devuelve |∇²f| normalizado (visualización).
        * ``enhance=True``: aplica realce → g(x,y) = f(x,y) + c·∇²f(x,y).
          Restaura el fondo sumando los bordes detectados a la imagen original.

    Args:
        ksize: 1 → kernel 4-vecinos; 3 → kernel 8-vecinos extendido.
        enhance: Si True, suma el laplaciano a la imagen original.
        c: Constante de ponderación para el realce (positivo o negativo
           según la definición del laplaciano utilizado).
    """

    def __init__(self, ksize: int = 1, enhance: bool = False, c: float = 1.0) -> None:
        self.ksize = ksize
        self.enhance = enhance
        self.c = c

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=self.ksize)

        if self.enhance:
            # g(x,y) = f(x,y) + c * ∇²f(x,y)
            enhanced = gray.astype(np.float64) + self.c * lap
            result = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            abs_lap = np.abs(lap)
            result = cv2.normalize(abs_lap, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


class HighboostFilter(ImageFilter):
    """Filtrado Highboost.

    g(x,y) = f(x,y) + k · [f(x,y) − f̄(x,y)]

    con k > 1 como factor de ganancia. Resta la versión suavizada f̄
    para construir una máscara de aguzado y la amplifica sobre la imagen
    original. Con k = 1 equivale al filtro Unsharp Mask clásico.

    Args:
        k: Factor de ganancia (k > 1 amplifica bordes).
        blur_sigma: σ del Gaussiano usado para calcular f̄.
    """

    def __init__(self, k: float = 2.0, blur_sigma: float = 1.2) -> None:
        self.k = k
        self.blur_sigma = blur_sigma

    def apply(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.blur_sigma)
        mask = image.astype(np.float32) - blurred.astype(np.float32)
        result = image.astype(np.float32) + self.k * mask
        return np.clip(result, 0, 255).astype(np.uint8)


class UnsharpMaskFilter(ImageFilter):
    """Filtro Unsharp Mask (caso particular de Highboost con k = amount).

    Equivalente a ``HighboostFilter`` pero expresado como:
        result = amount * image - (amount - 1) * blurred

    Args:
        sigma_x: σ del Gaussiano para el suavizado previo.
        amount: Factor de realce (≥ 1.0).
    """

    def __init__(self, sigma_x: float = 1.2, amount: float = 1.5) -> None:
        self.sigma_x = sigma_x
        self.amount = amount

    def apply(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.sigma_x)
        return cv2.addWeighted(image, self.amount, blurred, -(self.amount - 1.0), 0)


class GradientLaplacianFilter(ImageFilter):
    """Filtrado Gradiente-Laplaciano (proceso en 6 pasos).

    Combina el realce por Laplaciano con la magnitud suavizada del
    gradiente como máscara multiplicativa, sumada a la imagen original
    y ajustada con corrección Gamma.

    Pasos:
        1. Calcular ∇²f (Laplaciano) de la imagen en grises.
        2. Realzar: h1 = f + ∇²f.
        3. Calcular |∇f| (magnitud de Sobel) de la imagen original.
        4. Suavizar la magnitud con Gaussiano → h2.
        5. Multiplicar: mask = h1 · h2 (máscara de bordes ponderada).
        6. Combinar con original: g = f + mask; aplicar corrección Gamma.

    Args:
        gamma_correction: Exponente γ para la corrección final.
    """

    def __init__(self, gamma_correction: float = 0.5) -> None:
        self.gamma_correction = gamma_correction

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Paso 1: Laplaciano
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)

        # Paso 2: Realce con Laplaciano
        h1 = gray + lap

        # Paso 3: Magnitud del gradiente (Sobel)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Paso 4: Suavizar la magnitud del gradiente
        grad_smooth = cv2.GaussianBlur(grad_mag, (5, 5), 0)

        # Paso 5: Máscara multiplicativa (normalizar a [0,1] antes)
        h1_norm = cv2.normalize(h1, None, 0.0, 1.0, cv2.NORM_MINMAX)
        grad_norm = cv2.normalize(grad_smooth, None, 0.0, 1.0, cv2.NORM_MINMAX)
        combined_mask = h1_norm * grad_norm

        # Paso 6: Combinar con original y aplicar Gamma
        gray_norm = gray / 255.0
        result = gray_norm + combined_mask
        result = np.clip(result, 0.0, 1.0)
        result = result ** self.gamma_correction       # corrección gamma
        result = (result * 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# =============================================================================
# 5. FILTROS DE SEGMENTACIÓN Y DETECCIÓN DE BORDES AVANZADOS
# =============================================================================

class LoGFilter(ImageFilter):
    """Laplaciano de la Gaussiana (LoG / Marr-Hildreth).

    Combina el suavizado Gaussiano con la segunda derivada para localizar
    bordes a diferentes escalas evaluando los **cruces por cero**.
    Permite ajustar la escala de detección a través de σ.

    Args:
        sigma: Desviación estándar del Gaussiano (controla escala de bordes).
        ksize: Tamaño del kernel Gaussiano (debe ser impar).
    """

    def __init__(self, sigma: float = 1.0, ksize: int = 9) -> None:
        self.sigma = sigma
        self.ksize = ksize

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Suavizado Gaussiano
        blurred = cv2.GaussianBlur(gray, (self.ksize, self.ksize), self.sigma)
        # Laplaciano sobre la imagen suavizada
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        abs_log = np.abs(log)
        result = cv2.normalize(abs_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


class LineDetectionFilter(ImageFilter):
    """Máscaras de detección de líneas (segunda derivada orientada).

    Detecta líneas de 1 píxel de grosor en direcciones específicas
    usando máscaras basadas en la segunda derivada.

    Máscaras disponibles:
        ``"0"``  → Líneas horizontales   (0°)
        ``"45"`` → Líneas diagonales     (+45°)
        ``"90"`` → Líneas verticales     (90°)
        ``"-45"``→ Líneas diagonales     (-45°)

    Args:
        direction: Dirección de detección ("0", "45", "90" o "-45").
    """

    _MASKS: dict[str, np.ndarray] = {
        "0":   np.array([[-1, -1, -1], [ 2,  2,  2], [-1, -1, -1]], np.float32),
        "45":  np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]], np.float32),
        "90":  np.array([[-1,  2, -1], [-1,  2, -1], [-1,  2, -1]], np.float32),
        "-45": np.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]], np.float32),
    }

    def __init__(self, direction: str = "0") -> None:
        if direction not in self._MASKS:
            raise ValueError(f"direction debe ser uno de {list(self._MASKS)}")
        self.direction = direction

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kernel = self._MASKS[self.direction]
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        abs_filtered = np.abs(filtered)
        result = cv2.normalize(abs_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


class ExGLeafSegmenter(ImageFilter):
    """Segmentador de hoja mediante Índice de Exceso de Verde (ExG).

    ExG = 2·G - R - B

    Extrae la máscara de la hoja usando el índice ExG seguido de
    umbralización de Otsu y apertura morfológica para limpiar ruido.
    Preserva la región de la hoja en color original; el fondo queda negro.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32)
        b, g, r = cv2.split(img_float)
        exg = 2.0 * g - r - b
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
        exg_u8 = exg_norm.astype(np.uint8)
        _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.bitwise_and(image, image, mask=mask_clean)


# =============================================================================
# PIPELINE (Patrón Composite)
# =============================================================================

class Pipeline(ImageFilter):
    """Ejecuta una secuencia ordenada de filtros (Patrón Composite/Pipeline).

    Permite componer cualquier combinación de ``ImageFilter`` y aplicarlos
    de forma encadenada sobre una imagen.

    Args:
        filters: Lista de instancias de ``ImageFilter`` a ejecutar en orden.
    """

    def __init__(self, filters: list[ImageFilter]) -> None:
        self.filters = filters

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = image
        for f in self.filters:
            result = f.apply(result)
        return result
