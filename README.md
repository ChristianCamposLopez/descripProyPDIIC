# Procesamiento de Imágenes de Plantas - Detección de Plagas

Implementación del proyecto de Procesamiento Digital de Imágenes (PDI) para la detección de plagas y enfermedades en hojas de plantas (Universidad Tecnológica de la Mixteca — Dra. Verónica Rodríguez López).

El proyecto utiliza técnicas de PDI para analizar, preprocesar y segmentar el conjunto de datos de hojas enfermas y sanas, aplicando principios de diseño modular para asegurar un código mantenible y reproducible.

## Estructura del Proyecto

```text
descripProyPDIIC/
├── src/                        # Código fuente modular
│   ├── main.py                 # Orquestador y punto de entrada (Filtros)
│   ├── filters.py              # Interfaces y clases de Filtros (Pipeline, ExG, CLAHE, etc.)
│   ├── dataset.py              # Carga y análisis del Dataset
│   └── visualization.py        # Generación de grillas comparativas
├── segmentacion_parcial2.py    # Algoritmos de segmentación del 2do Parcial (Canny, Global, Otsu, K-Means)
├── resultados_segmentacion/    # Resultados, gráficas y presentaciones del 2do Parcial
│   ├── Presentacion_Segundo_Parcial_Segmentacion.pptx # Presentación PowerPoint (16:9)
│   ├── presentacion_parcial2.pdf # Presentación Beamer (PDF)
│   ├── comparaciones/          # Paneles comparativos de segmentación (Baja, Media, Alta)
│   ├── diagramas/              # Diagramas de flujo por nivel de complejidad
│   └── graficas/               # Histogramas y umbrales de segmentación
├── resultados/                 # Salidas y metadatos JSON del 1er Parcial
├── pyproject.toml              # Definición del proyecto y dependencias
├── download_dataset.py         # Script para descarga automática desde Kaggle
├── reporte_pdi.tex / .pdf      # Reporte formal del primer parcial
└── reporte_parcial1.tex / .pdf # Reporte inicial en LaTeX
```

---

## Dataset

El proyecto utiliza el [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) de Kaggle.

- **Total:** 87,900 imágenes JPG (256×256 píxeles, RGB).
- **Categorías:** 38 clases de enfermedades + imágenes sin etiqueta.
- **Distribución:** 70,295 imágenes de entrenamiento, 17,572 de validación y 33 de prueba.

Para descargar automáticamente:
```bash
python download_dataset.py
```

---

## Primer Parcial: Preprocesamiento y Filtros

Pipeline de reducción de ruido y realce de características de enfermedad (CLAHE, Mediana, Gaussiano, Unsharp Masking, Exceso de Verde).

```bash
python -m src.main --dataset dataset --salida resultados
```

---

## Segundo Parcial: Segmentación de Imágenes

Implementación y comparación rigurosa de 4 algoritmos de segmentación sobre hojas clasificadas en 3 niveles de complejidad visual (Baja, Media y Alta):

1. **Detector de bordes de Canny (1986):** Detección de discontinuidades con gradientes de Sobel, supresión de no máximos, umbralización con histéresis y reconstrucción morfológica de contornos.
2. **Umbralización Global Iterativa:** Algoritmo de partición bimodal según Gonzalez & Woods.
3. **Umbralización de Otsu (1979):** Cálculo analítico del umbral óptimo $T^*$ mediante maximización de varianza inter-clase.
4. **Agrupamiento K-Means (1967):** Segmentación no supervisada en espacio de color $L^*a^*b^*$ discriminando fondo, hoja sana y tejido lesionado.

### Archivos del Segundo Parcial en `resultados_segmentacion/`

- **Presentaciones:**
  - `Presentacion_Segundo_Parcial_Segmentacion.pptx`: Presentación en PowerPoint lista para exponer (formato widescreen 16:9).
  - `presentacion_parcial2.pdf`: Presentación compilada con LaTeX Beamer (14 diapositivas estructuradas).
  - `presentacion_parcial2.tex`: Código fuente en LaTeX Beamer.
- **Visualizaciones y Métricas:**
  - `comparaciones/caso_baja.png`: Panel 2×4 para Manzana Sana (Complejidad Baja).
  - `comparaciones/caso_media.png`: Panel 2×4 para Tomate Early Blight (Complejidad Media).
  - `comparaciones/caso_alta.png`: Panel 2×4 para Maíz Gray Leaf Spot (Complejidad Alta).
  - `comparaciones/comparativa_global.png`: Tabla comparativa simultánea de los 4 métodos en los 3 casos.
  - `graficas/histogramas_umbrales.png`: Distribución de intensidades y puntos de corte (Global vs Otsu).
  - `diagramas/`: Diagramas de flujo de preprocesamiento y segmentación (`flujo_baja_complejidad.png`, `flujo_media_complejidad.png`, `flujo_alta_complejidad.png`).
  - `metricas_segmentacion.json`: Parámetros cuantitativos y umbrales calculados.

### Ejecución de la Segmentación

```powershell
python segmentacion_parcial2.py
```
