# Detalles de Mejoras - Proyecto PDI

Este documento detalla los cambios realizados en el proyecto para alcanzar el 100% en la rúbrica de evaluación del "Primer Parcial", de acuerdo a las sugerencias del profesor.

## 1. Gráfica de Distribución de Clases (Solución Rúbrica 1 y 3)
Se indicó que faltaba mostrar visualmente el número de imágenes por categoría en el reporte.

**Cambios realizados:**
- **Creación de script:** Se creó un script en Python (`src/generar_grafica.py`) que lee el archivo `dataset_stats.json` y genera un diagrama de barras horizontales mostrando el conteo por categoría.
- **Generación de imagen:** Se ejecutó el script, generando la gráfica en la ruta `resultados/distribucion_clases.png`.
- **Actualización del Reporte LaTeX:** Se editó el archivo `reporte_pdi.tex` (Sección 1) insertando la gráfica inmediatamente después de la Tabla 1, usando un entorno `figure`. Esto cumple el requerimiento del conteo por clase de forma estética y profesional.

## 2. Ajuste en la Segmentación ExG (Fondo Negro Puro)
Se sugirió mejorar el aislamiento de la hoja evitando desenfocar el fondo, sustituyéndolo completamente por negro puro (zeros) para no confundir futuros modelos.

**Cambios realizados:**
- **Modificación en Código:** Se editó el archivo `src/filters.py` específicamente en el método `apply` de la clase `ExGLeafSegmenter`.
- **Qué se cambió:** 
  Antes (líneas 100-108), el código atenuaba el fondo con un `GaussianBlur` y luego un `addWeighted` combinando el fondo con la hoja:
  ```python
  background = cv2.GaussianBlur(image, (15, 15), 0)
  background = cv2.addWeighted(background, 0.4, np.zeros_like(background), 0, 0)
  leaf_region = cv2.bitwise_and(image, image, mask=mask_clean)
  bg_region = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask_clean))
  return cv2.add(leaf_region, bg_region)
  ```
  Ahora, el código simplemente retorna la región de la hoja aislada (los píxeles donde no hay hoja, es decir, el fondo, quedan completamente en negro/0):
  ```python
  leaf_region = cv2.bitwise_and(image, image, mask=mask_clean)
  return leaf_region
  ```

Con estas dos mejoras, todos los entregables de la rúbrica se han completado satisfactoriamente.
