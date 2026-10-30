---
name: pdi-coder
description: >-
  Implementador especializado de código, algoritmos y pipelines para Procesamiento Digital de Imágenes (PDI) en Python, NumPy y OpenCV.
  Usar cuando se requiera programar, depurar, modularizar o refactorizar filtros espaciales, transformaciones de intensidad,
  algoritmos de segmentación (Canny, Otsu, K-Means), morfología matemática, generación de comparativas visuales y pipelines de procesamiento
  con principios SOLID, estabilidad numérica estricta y prevención de desbordamientos.
---

# Habilidad: Implementador de Código y Pipelines de PDI (Python / OpenCV / NumPy)

Esta guía define las directivas técnicas, patrones de diseño de software y salvaguardas de bajo nivel para que cualquier IA implemente código limpio, modular, numéricamente estable y libre de desbordamientos en Procesamiento Digital de Imágenes.

---

## 1. Arquitectura de Código y Principios de Diseño

### 1.1 Patrón Template Method & Open/Closed Principle (OCP)
Todo filtro o transformación debe heredar de una clase base abstracta común:

```python
from abc import ABC, abstractmethod
import numpy as np

class ImageFilter(ABC):
    """Interfaz base para todos los filtros de imagen."""
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica la transformación sobre una imagen (BGR o Gray uint8) y retorna el resultado."""
        pass
```

### 1.2 Patrón Composite / Pipeline
Permite encadenar transformaciones de manera ordenada y reproducible:

```python
class Pipeline(ImageFilter):
    """Ejecuta una secuencia ordenada de filtros de forma encadenada."""
    def __init__(self, filters: list[ImageFilter]) -> None:
        self.filters = filters

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = image
        for f in self.filters:
            result = f.apply(result)
        return result
```

---

## 2. Salvaguardas Numéricas Críticas (Prevención de Errores Comunes)

Cualquier código de PDI generado debe implementar estrictamente estas 6 reglas:

### Regla 1: Prevención de Desbordamiento Aritmético en `uint8`
* **Error común**: Operar directamente sobre matrices `np.uint8` (`img1 + img2` o `img * 1.5`), lo cual genera wrap-around cíclico ($250 + 10 = 4$).
* **Patrón correcto**:
  ```python
  # Promover a float32 -> operar -> recortar a [0, 255] -> castear a uint8
  img_f = image.astype(np.float32)
  processed = funcion_de_procesamiento(img_f)
  result = np.clip(processed, 0, 255).astype(np.uint8)
  ```

### Regla 2: Preservación de Valores Negativos en Derivadas
* **Error común**: `cv2.Laplacian(gray, cv2.CV_8U)` trunca todos los gradientes negativos a cero, perdiendo la mitad de la información del borde.
* **Patrón correcto**:
  ```python
  # Calcular derivadas en profundidad CV_64F
  lap_64f = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
  # Para visualización: valor absoluto y normalización a [0, 255]
  lap_abs = np.abs(lap_64f)
  lap_vis = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  ```

### Regla 3: Protección contra $\log(0)$
* **Error común**: `np.log(gray)` produce $-\infty$ o `NaN` en píxeles negros ($0$).
* **Patrón correcto**:
  ```python
  # Usar np.log1p(x) que calcula log(1 + x) de forma numéricamente exacta
  c = 255.0 / np.log1p(255.0)
  log_img = np.clip(c * np.log1p(gray.astype(np.float32)), 0, 255).astype(np.uint8)
  ```

### Regla 4: Protección de División por Cero en Varianza de Otsu
* Cuando $\omega_0 = 0$ o $\omega_0 = 1$, la fórmula de varianza inter-clase divide entre cero.
* **Patrón correcto**:
  ```python
  with np.errstate(divide="ignore", invalid="ignore"):
      sigma_b_sq = ((mu_t * omega - mu) ** 2) / (omega * (1.0 - omega))
      sigma_b_sq = np.nan_to_num(sigma_b_sq)
  ```

### Regla 5: Mapeo de Coordenadas (NumPy vs. OpenCV)
* En NumPy: indexación matricial `matriz[fila, columna]` $\rightarrow$ `matriz[y, x]`.
* En OpenCV: funciones geométricas `cv2.circle(img, (x, y), ...)`, donde $x$ es columna e $y$ es fila.

### Regla 6: Preservación de Espacios de Color
* OpenCV lee por defecto en **BGR**. Matplotlib requiere **RGB**.
* Al ecualizar o filtrar imágenes a color sin alterar el matiz, convertir a espacio **L\*a\*b\*** o **HSV**, procesar únicamente el canal de luminancia ($L$ o $V$) y re-ensamblar.

---

## 3. Catálogo de Implementaciones de Referencia

### 3.1 Transformaciones de Intensidad
* **Negativo**: `cv2.bitwise_not(image)`
* **Transformación Gamma (con LUT para máxima velocidad)**:
  ```python
  class GammaTransform(ImageFilter):
      def __init__(self, gamma: float = 1.0):
          self.gamma = gamma
          self._lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
      def apply(self, image: np.ndarray) -> np.ndarray:
          return cv2.LUT(image, self._lut)
  ```
* **Ecualización CLAHE sobre luminancia**:
  ```python
  class CLAHEEqualization(ImageFilter):
      def __init__(self, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)):
          self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
      def apply(self, image: np.ndarray) -> np.ndarray:
          lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
          l, a, b = cv2.split(lab)
          l_eq = self.clahe.apply(l)
          return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
  ```

### 3.2 Filtros Estadísticos y de Orden
* **Mediana**: `cv2.medianBlur(image, ksize)`
* **MAX (pimienta)**: `cv2.dilate(image, np.ones((k, k), np.uint8))`
* **MIN (sal)**: `cv2.erode(image, np.ones((k, k), np.uint8))`
* **Punto Medio**:
  ```python
  kernel = np.ones((k, k), np.uint8)
  max_img = cv2.dilate(image.astype(np.float32), kernel)
  min_img = cv2.erode(image.astype(np.float32), kernel)
  midpoint = ((max_img + min_img) / 2.0).astype(np.uint8)
  ```

### 3.3 Filtros Suavizantes y Realzantes
* **Box Filter**: `cv2.filter2D(image, -1, np.ones((3, 3), np.float32) / 9.0)`
* **Promedio Pesado**:
  ```python
  w = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
  return cv2.filter2D(image, -1, w)
  ```
* **Máscaras de Sobel (Magnitud del Gradiente)**:
  ```python
  gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
  mag = cv2.magnitude(gx, gy)
  result = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  ```
* **Laplaciano con Recuperación de Fondo ($g = f + c \cdot \nabla^2 f$)**:
  ```python
  lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
  enhanced = gray.astype(np.float64) + c * lap
  return np.clip(enhanced, 0, 255).astype(np.uint8)
  ```
* **Máscaras Direccionales de Líneas**:
  ```python
  MASKS = {
      "0":   np.array([[-1, -1, -1], [ 2,  2,  2], [-1, -1, -1]], np.float32),
      "45":  np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]], np.float32),
      "90":  np.array([[-1,  2, -1], [-1,  2, -1], [-1,  2, -1]], np.float32),
      "-45": np.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]], np.float32),
  }
  ```

### 3.4 Algoritmos de Segmentación

#### Canny con Extracción de Máscara Binaria
```python
def segment_canny(gray: np.ndarray, low_thresh: int = 50, high_thresh: int = 150, morph_k: int = 5):
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        max_area = max(cv2.contourArea(c) for c in contours)
        valid = [c for c in contours if cv2.contourArea(c) >= max_area * 0.05]
        cv2.drawContours(mask, valid, -1, 255, thickness=cv2.FILLED)
    # FloodFill para rellenar huecos internos
    h, w = mask.shape
    flood = mask.copy()
    f_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, f_mask, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))
```

#### Umbralización Global Iterativa (Gonzalez & Woods)
```python
def global_threshold(channel: np.ndarray, delta_t: float = 0.5, max_iters: int = 100):
    t = float(np.mean(channel))
    for _ in range(max_iters):
        g1 = channel[channel > t]
        g2 = channel[channel <= t]
        mu1 = float(np.mean(g1)) if len(g1) > 0 else t
        mu2 = float(np.mean(g2)) if len(g2) > 0 else t
        t_next = (mu1 + mu2) / 2.0
        if abs(t_next - t) < delta_t:
            break
        t = t_next
    _, mask = cv2.threshold(channel, int(round(t)), 255, cv2.THRESH_BINARY)
    return int(round(t)), mask
```

#### K-Means en Espacio L\*a\*b\* para Segmentación Vegetal
```python
def segment_kmeans_vegetation(rgb_image: np.ndarray, k: int = 3):
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    h, w, _ = lab.shape
    pixels = lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Evaluar verdor relativo de clusters en RGB: G - (R+B)/2
    rgb_flat = rgb_image.reshape((-1, 3))
    scores = [np.mean(rgb_flat[labels.flatten() == i, 1] - 0.5 * (rgb_flat[labels.flatten() == i, 0] + rgb_flat[labels.flatten() == i, 2])) for i in range(k)]
    leaf_cluster = np.argmax(scores)
    
    mask = (labels.reshape((h, w)) == leaf_cluster).astype(np.uint8) * 255
    k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_elem)
    return mask
```

---

## 4. Banco de Pruebas Rápidas (Smoke Testing)

Para validar que un filtro o pipeline recién codificado funciona correctamente sin romper dimensiones ni tipos de datos, ejecutar:

```python
def smoke_test(filter_instance: ImageFilter):
    dummy_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = filter_instance.apply(dummy_img)
    assert out.shape[:2] == dummy_img.shape[:2], f"Error de dimensiones: {out.shape} vs {dummy_img.shape}"
    assert out.dtype == np.uint8, f"Tipo de dato incorrecto: {out.dtype}"
    assert np.all((out >= 0) & (out <= 255)), "Valores fuera de rango [0, 255]"
    print(f"✓ {filter_instance.__class__.__name__} pasó el smoke test.")
```

