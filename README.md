# Proyecto PDI — Procesamiento de Imágenes de Plantas para Detección de Plagas

**Dataset:** New Plant Diseases Dataset (Kaggle)  
**Alumno:** Irving Tristán Pérez Zurita · **Grupo:** 902-A  
**Profesora:** Dra. Verónica Rodríguez López  
**Universidad Tecnológica de la Mixteca**

---

## Estructura del Proyecto

```
descripProyPDIIC/
├── parcial1/                        ← PRIMER PARCIAL: Preprocesamiento y filtros
│   ├── docs/                        ← Rúbrica P1, explicaciones teóricas y notas
│   │   ├── descripProyPDIIC_PARTE1.pdf
│   │   ├── Detalles_Irving.md
│   │   ├── explicacion_filtros.pdf
│   │   └── explicacion_filtros.tex
│   ├── legacy/                      ← Script monolítico inicial de referencia
│   │   └── analisis_filtros.py
│   ├── resultados/                  ← Grids comparativos, distribución de clases y stats
│   │   ├── dataset_stats.json
│   │   ├── distribucion_clases.png
│   │   └── filtros_01.png ... filtros_03.png
│   ├── src/                         ← Código modular (SOLID / Patrón Composite)
│   │   ├── dataset.py
│   │   ├── filters.py               ← 28 filtros e transformaciones (5 categorías)
│   │   ├── generar_grafica.py
│   │   ├── main.py
│   │   └── visualization.py
│   ├── reporte_pdi.pdf              ← Reporte oficial de Parcial 1 (compilado)
│   ├── reporte_pdi.tex              ← Código fuente LaTeX del reporte P1
│   ├── reporte_parcial1.pdf         ← Versión alternativa de reporte
│   └── reporte_parcial1.tex
│
├── parcial2/                        ← SEGUNDO PARCIAL: Segmentación de imágenes
│   ├── docs/                        ← Rúbrica y especificación oficial P2
│   │   └── descripProyPDIIC_PARTE2.pdf
│   ├── legacy/                      ← Script monolítico inicial antes de modularizar
│   │   └── segmentacion_parcial2.py
│   ├── presentacion/                ← Presentaciones requeridas (Beamer + PowerPoint)
│   │   ├── presentacion_parcial2.pdf
│   │   ├── presentacion_parcial2.tex
│   │   └── Presentacion_Segundo_Parcial_Segmentacion.pptx
│   ├── resultados/                  ← Comparativas, diagramas de flujo y métricas
│   │   ├── comparaciones/
│   │   ├── diagramas/
│   │   ├── graficas/
│   │   └── metricas_segmentacion.json
│   ├── src/                         ← Código modular (Canny, Global, Otsu, K-Means)
│   │   ├── flowcharts.py
│   │   ├── main.py
│   │   ├── segmentation.py
│   │   └── visualization.py
│   ├── reporte_parcial2.pdf         ← Reporte detallado de Parcial 2 (compilado)
│   └── reporte_parcial2.tex         ← Código fuente LaTeX del reporte P2
│
├── scripts/                         ← Scripts de utilidad general
│   └── download_dataset.py          ← Descarga del dataset desde Kaggle
├── pyproject.toml                   ← Configuración del proyecto y dependencias
├── uv.lock                          ← Bloqueo determinista de dependencias
├── README.md                        ← Documentación principal
└── .gitignore                       ← Reglas de exclusión de Git
```

---

## Ejecución

### Requisitos
```bash
# Instalar dependencias con uv (recomendado)
uv sync
```

### Descarga del Dataset (si es necesario)
```bash
uv run python3 scripts/download_dataset.py --dest dataset
```

### Parcial 1 — Grids de filtros
```bash
uv run python3 -m parcial1.src.main --dataset dataset --salida parcial1/resultados
```

### Parcial 2 — Segmentación
```bash
uv run python3 -m parcial2.src.main --dataset dataset --salida parcial2/resultados
```

### Compilar reportes LaTeX
```bash
# Reporte Parcial 1
cd parcial1 && pdflatex reporte_pdi.tex && cd ..

# Reporte Parcial 2
cd parcial2 && pdflatex reporte_parcial2.tex && cd ..
```
pdflatex reporte_parcial2.tex
```

---

## Filtros implementados (Parcial 1)

### Categoría 1 — Transformaciones de Intensidad
| Clase | Fórmula | Descripción |
|---|---|---|
| `NegativeTransform` | `s = 255 - r` | Inversión de niveles de gris |
| `LogTransform` | `s = c·log(1+r)` | Compresión de rango dinámico |
| `GammaTransform` | `s = r^γ` | γ<1 aclara; γ>1 oscurece |
| `ContrastStretching` | Piecewise lineal | Estiramiento de rango útil |
| `BinaryThreshold` | Corte en umbral m | División blanco/negro |
| `IntensityBandHighlight` | Rango [low, high] | Resaltado de banda de grises |
| `GlobalEqualization` | `s_k = (L-1)/MN·Σnⱼ` | Ecualización global del histograma |
| `CLAHEEqualization` | Adaptativa local | Ecualización limitada con CLAHE |

### Categoría 2 — Filtros Estadísticos / De Orden
| Clase | Operación | Tipo de ruido |
|---|---|---|
| `MedianFilter` | Mediana de vecindad | Sal y pimienta |
| `MaxFilter` | Máximo de vecindad | Pimienta (puntos oscuros) |
| `MinFilter` | Mínimo de vecindad | Sal (puntos claros) |
| `MidpointFilter` | (max + min) / 2 | Ruido uniforme |
| `ArithmeticMeanFilter` | Promedio simple | Ruido Gaussiano |
| `GeometricMeanFilter` | Producto^(1/N) | Gaussiano (mejor detalle) |

### Categoría 3 — Filtros Suavizantes
| Clase | Kernel | Efecto |
|---|---|---|
| `BoxFilter` | 3×3, coef. 1/9 | Promedio uniforme |
| `WeightedAverageFilter` | [1,2,1;2,4,2;1,2,1]/16 | Menos emborronamiento |
| `GaussianBlurFilter` | Gaussiano σ configurable | Suavizado isótropo |
| `BilateralFilter` | Espacial + intensidad | Preserva bordes |

### Categoría 4 — Filtros Realzantes (Derivadas)
| Clase | Tipo | Descripción |
|---|---|---|
| `SobelFilter` | 1ª derivada | Magnitud del gradiente ∇f |
| `PrewittFilter` | 1ª derivada | Coeficientes uniformes |
| `LaplacianFilter(enhance=False)` | 2ª derivada | Visualización ∇²f |
| `LaplacianFilter(enhance=True)` | 2ª derivada | g = f + c·∇²f |
| `HighboostFilter` | Highboost | g = f + k·(f - f̄), k>1 |
| `UnsharpMaskFilter` | Unsharp Mask | Caso k=1.5 del Highboost |
| `GradientLaplacianFilter` | 6 pasos | Laplaciano + Gradiente + Gamma |

### Categoría 5 — Segmentación y Detección Avanzada
| Clase | Técnica | Descripción |
|---|---|---|
| `LoGFilter` | LoG / Marr-Hildreth | Laplaciano de la Gaussiana |
| `LineDetectionFilter` | Máscaras orientadas | Detección de líneas 0°,45°,90°,-45° |
| `ExGLeafSegmenter` | ExG + Otsu | Aislamiento de región foliar |

---

## Algoritmos de Segmentación (Parcial 2)
- **Detector de bordes de Canny** (Suavizado → Sobel → NMS → Histéresis)
- **Umbralización Global iterativa** (Gonzalez & Woods, convergencia por δT)
- **Umbralización de Otsu** (maximización de varianza inter-clase σ²_B)
- **Agrupamiento K-Means** (espacio L*a*b*, K=3, identificación por verdor)
