# Proyecto PDI — Procesamiento de Imágenes de Plantas para Detección de Plagas

**Universidad Tecnológica de la Mixteca (UTM)**  
**Materia:** Procesamiento Digital de Imágenes (PDI)  
**Profesora:** Dra. Verónica Rodríguez López  
**Dataset Base:** [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 📁 Estructura del Proyecto

```
descripProyPDIIC/
├── parcial1/                        ← PRIMER PARCIAL: Preprocesamiento y filtros espaciales
│   ├── docs/                        ← Rúbrica P1, explicaciones teóricas y notas
│   │   ├── descripProyPDIIC_PARTE1.pdf
│   │   ├── Detalles_Irving.md
│   │   ├── explicacion_filtros.pdf
│   │   └── explicacion_filtros.tex
│   ├── muestras/                    ← Muestras foliares locales (fallback offline)
│   │   ├── hoja_01.jpg
│   │   ├── hoja_02.jpg
│   │   └── hoja_03.jpg
│   ├── resultados/                  ← Artefactos visuales generados
│   │   ├── dataset_stats.json
│   │   ├── distribucion_clases.png
│   │   ├── filtros_01.png ... filtros_03.png (grids globales)
│   │   └── individuales/            ← Tarjetas individuales con pie de página
│   │       ├── figuras_filtros.tex  ← Snippets LaTeX automáticos (\input)
│   │       ├── muestra_01/          ← 36 imágenes con footer explicativo
│   │       ├── muestra_02/
│   │       └── muestra_03/
│   ├── src/                         ← Código modular (SOLID / Patrón Template Method)
│   │   ├── dataset.py               ← Análisis estadístico del conjunto de datos
│   │   ├── filters.py               ← 36 filtros formales y especificaciones (5 categorías)
│   │   ├── generar_grafica.py       ← Gráfica de distribución de clases
│   │   ├── main.py                  ← Pipeline CLI con modos individual / grid / ambos
│   │   └── visualization.py         ← Renderizado de tarjetas con footer y exportador LaTeX
│   ├── reporte_parcial1.pdf         ← Reporte compilado del Parcial 1
│   └── reporte_parcial1.tex         ← Código fuente LaTeX del reporte
│
├── parcial2/                        ← SEGUNDO PARCIAL: Segmentación de imágenes
│   ├── docs/                        ← Rúbrica y especificación oficial P2
│   │   └── descripProyPDIIC_PARTE2.pdf
│   ├── presentacion/                ← Presentaciones requeridas (Beamer + PowerPoint)
│   │   ├── presentacion_parcial2.pdf
│   │   ├── presentacion_parcial2.tex
│   │   └── Presentacion_Segundo_Parcial_Segmentacion.pptx
│   ├── resultados/                  ← Comparativas, diagramas de flujo y métricas
│   │   ├── comparaciones/           ← Segmentación por niveles de complejidad
│   │   ├── diagramas/               ← Diagramas de flujo por caso
│   │   ├── graficas/                ← Histogramas con umbrales superpuestos
│   │   └── metricas_segmentacion.json
│   ├── src/                         ← Código modular (Canny, Global, Otsu, K-Means)
│   │   ├── flowcharts.py            ← Generación visual de diagramas de bloques
│   │   ├── main.py                  ← Pipeline de ejecución Parcial 2
│   │   ├── segmentation.py          ← Algoritmos formales de segmentación
│   │   └── visualization.py         ← Gráficas comparativas y tablas de métricas
│   ├── reporte_parcial2.pdf         ← Reporte detallado de Parcial 2
│   └── reporte_parcial2.tex         ← Código fuente LaTeX del reporte
│
├── scripts/                         ← Utilidades generales
│   └── download_dataset.py          ← Descarga del dataset desde Kaggle
├── pyproject.toml                   ← Configuración de dependencias (Hatch + UV)
├── uv.lock                          ← Bloqueo determinista de dependencias
└── README.md                        ← Documentación principal del repositorio
```

---

## 🚀 Instalación y Ejecución con `uv`

El proyecto utiliza [`uv`](https://docs.astral.sh/uv/) como gestor rápido y determinista de entornos virtuales de Python.

### 1. Sincronizar el entorno
```bash
uv sync
```

### 2. Descargar el dataset completo (opcional)
Si se desea procesar el conjunto completo de 87,900 imágenes:
```bash
uv run python3 scripts/download_dataset.py --dest dataset
```
*(Nota: Si no se descarga el dataset, el pipeline utiliza automáticamente las muestras locales en `parcial1/muestras/`).*

### 3. Ejecutar Parcial 1 — Filtrado y Tarjetas Individuales
El pipeline admite tres modos de salida (`--modo`):
* `individual`: Genera únicamente las 36 tarjetas individuales con pie de página descriptivo.
* `grid`: Genera únicamente los grids comparativos globales (4 columnas).
* `ambos` (por defecto): Genera tarjetas individuales, grids y el archivo de snippets LaTeX.

```bash
# Modo completo (recomendado)
uv run python3 -m parcial1.src.main --modo ambos

# Opcional: especificar ruta de dataset y salida
uv run python3 -m parcial1.src.main --dataset dataset --salida parcial1/resultados --modo ambos
```

### 4. Ejecutar Parcial 2 — Algoritmos de Segmentación
```bash
uv run python3 -m parcial2.src.main --salida parcial2/resultados
```

### 5. Compilar Reportes en LaTeX
```bash
# Compilar Reporte Parcial 1
cd parcial1 && pdflatex -interaction=nonstopmode reporte_parcial1.tex && cd ..

# Compilar Reporte Parcial 2
cd parcial2 && pdflatex -interaction=nonstopmode reporte_parcial2.tex && cd ..
```

---

## 🔬 Catálogo de Filtros Implementados (Parcial 1)

Cada filtro se implementa bajo el patrón **Template Method** (`ImageFilter`) con seguridad numérica contra desbordamientos (`uint8`/`float32`), normalización estricta y generación de metadatos formales para pies de página.

### 1. Transformaciones de Intensidad (Punto a Punto: $s = T(r)$)
| ID | Filtro | Formulación Formal | Propósito / Comportamiento |
|---|---|---|---|
| `01` | **Original** | $f(x, y)$ | Referencia base sin modificación. |
| `02` | **Negativo de una Imagen** | $s = L - 1 - r = 255 - r$ | Invierte niveles; resalta detalles claros sobre fondos oscuros. |
| `03` | **Transformación Logarítmica** | $s = c \log(1 + r)$ | Expande sombras oscuras y comprime altas luces. |
| `04` | **Gamma ($\gamma = 0.5$)** | $s = c \cdot r^\gamma, \; \gamma < 1$ | Aclara tonos medios y oscuros de la imagen. |
| `05` | **Gamma ($\gamma = 2.0$)** | $s = c \cdot r^\gamma, \; \gamma > 1$ | Oscurece y atenúa sobreexposición en altas luces. |
| `06` | **Estiramiento de Contraste** | Mapeo lineal $[r_1, r_2] \to [s_1, s_2]$ | Amplía el rango útil comprimido al intervalo $[0, 255]$. |
| `07` | **Umbralización Binaria** | Escalón en umbral $m = 128$ | Segmenta la imagen en blanco y negro puro. |
| `08` | **Resaltado de Bandas** | $s = r$ en $[a, b]$; $0.2 \cdot r$ en resto | Aísla un rango de grises específico atenuando el fondo. |
| `09` | **Ecualización Global** | $s_k = \frac{L-1}{MN} \sum n_j$ (Canal L) | Redistribuye el histograma maximizando el contraste global. |
| `10` | **Ecualización CLAHE** | Rejilla $8 \times 8$, Clip Limit $2.0$ | Ecualización local adaptativa sin amplificar ruido. |

### 2. Filtros Estadísticos / De Orden (Espaciales)
| ID | Filtro | Operación sobre vecindad $S_{x,y}$ | Tipo de Ruido Atenuado |
|---|---|---|---|
| `11` | **Filtro Mediana ($5 \times 5$)** | $g(x,y) = \text{mediana} \{ f(s,t) \}$ | Óptimo para ruido impulsivo (sal y pimienta); preserva bordes. |
| `12` | **Filtro MAX ($5 \times 5$)** | $g(x,y) = \max \{ f(s,t) \}$ | Elimina ruido de pimienta (puntos oscuros) por dilatación. |
| `13` | **Filtro MIN ($5 \times 5$)** | $g(x,y) = \min \{ f(s,t) \}$ | Elimina ruido de sal (puntos claros) por erosión. |
| `14` | **Filtro Punto Medio ($5 \times 5$)** | $g(x,y) = \frac{1}{2}[\max(f) + \min(f)]$ | Atenúa ruido con distribución uniforme. |
| `15` | **Media Aritmética ($5 \times 5$)** | $g(x,y) = \frac{1}{mn} \sum f(s,t)$ | Promedio simple; reduce varianza de ruido Gaussiano. |
| `16` | **Media Geométrica ($5 \times 5$)** | $g(x,y) = [ \prod f(s,t) ]^{1/mn}$ | Atenúa ruido Gaussiano preservando bordes mejor que la aritmética. |

### 3. Filtros Suavizantes (Pasabajas / Promedio)
| ID | Filtro | Estructura de Máscara / Modelo | Efecto Visual |
|---|---|---|---|
| `17` | **Filtro de Caja (Box Filter)** | Matriz $3 \times 3$ uniforme $\frac{1}{9}$ | Suavizado uniforme con emborronamiento isotrópico. |
| `18` | **Promedio Pesado** | $\frac{1}{16} [1, 2, 1; 2, 4, 2; 1, 2, 1]$ | Suavizado que reduce el desenfoque priorizando el centro. |
| `19` | **Filtro Gaussiano ($5 \times 5$)** | $G(x,y) = \exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$ | Pasabajas continuo sin artefactos angulares. |
| `20` | **Filtro Bilateral** | Espacial ($\sigma_s=35$) + Fotométrica ($\sigma_c=35$) | Suaviza texturas homogéneas preservando bordes nítidos. |

### 4. Filtros Realzantes (Derivadas y Aguzado)
| ID | Filtro | Formulación Matemática | Características |
|---|---|---|---|
| `21` | **Sobel (Gradiente $\nabla f$)** | $|\nabla f| \approx \sqrt{G_x^2 + G_y^2}$ | Primera derivada con peso central 2; bordes gruesos y robustos. |
| `22` | **Prewitt (Gradiente $\nabla f$)** | Coeficientes uniformes de 1 | Primera derivada por diferencias finitas directas. |
| `23` | **Laplaciano 4-vecinos** | $\nabla^2 f$ con centro $-4$ | Segunda derivada isotrópica; máxima respuesta a detalles finos. |
| `24` | **Laplaciano 8-vecinos** | $\nabla^2 f$ con centro $-8$ | Segunda derivada omnidireccional con conectividad completa. |
| `25` | **Realce Laplaciano (4-vec)** | $g(x,y) = f(x,y) - \nabla^2 f(x,y)$ | Suma bordes finos a la imagen original restaurando el fondo. |
| `26` | **Realce Laplaciano (8-vec)** | $g(x,y) = f(x,y) - \nabla^2 f(x,y)$ (8 vec) | Aguzado espacial intenso ortogonal y diagonal. |
| `27` | **Highboost ($k = 2.0$)** | $g = f + k \cdot [f - \bar{f}], \; k > 1$ | Amplifica fuertemente la máscara de altas frecuencias. |
| `28` | **Unsharp Mask ($k = 1.5$)** | $g = f + 1.5 \cdot [f - f_{\text{Gauss}}]$ | Máscara clásica fotográfica de nitidez aparente. |
| `29` | **Gradiente-Laplaciano** | 6 pasos (Laplaciano $\times$ Sobel + Gamma) | Combinación secuencial de bordes finos modulados por gradiente. |

### 5. Filtros de Segmentación y Detección de Bordes Avanzados
| ID | Filtro | Técnica / Formulación | Propósito |
|---|---|---|---|
| `30` | **LoG (Marr-Hildreth)** | $\nabla^2 [G(\sigma) * f]$ con $\sigma = 1.0$ | Suavizado multiescala previo para cruces por cero. |
| `31` | **Líneas Horizontales ($0^\circ$)** | Kernel $3 \times 3$ con fila central $+2$ | Aísla líneas horizontales de 1 píxel. |
| `32` | **Líneas Diagonales ($+45^\circ$)** | Kernel $3 \times 3$ orientado a $+45^\circ$ | Aísla estructuras lineales a 45 grados. |
| `33` | **Líneas Verticales ($90^\circ$)** | Kernel $3 \times 3$ con columna central $+2$ | Aísla nervaduras principales verticales. |
| `34` | **Líneas Diagonales ($-45^\circ$)** | Kernel $3 \times 3$ orientado a $-45^\circ$ | Aísla estructuras lineales a -45 grados. |
| `35` | **Detector Canny** | Gauss $\to$ Sobel $\to$ NMS $\to$ Histéresis | Contornos continuos delgados de 1 píxel sin ruido espurio. |
| `36` | **Pipeline Foliar P1** | Bilateral $\to$ CLAHE $\to$ Unsharp $\to$ ExG+Otsu | Pipeline integrado óptimo para segmentación foliar. |

---

## 🧩 Segundo Parcial — Métodos de Segmentación
1. **Detector de Bordes de Canny:** Segmentación por discontinuidad estructural con filtrado morfológico de cierre.
2. **Umbralización Global Iterativa:** Algoritmo heurístico de Gonzalez & Woods por partición bimodal hasta convergencia $\Delta T \le 0.5$.
3. **Algoritmo de Otsu:** Maximización exhaustiva de la varianza entre clases $\sigma_B^2(k)$ sobre el canal cromático $a^*$ ($L^*a^*b^*$).
4. **Agrupamiento K-Means ($K=3$):** Cuantización de color en espacio $L^*a^*b^*$ separando fondo, tejido sano y zonas lesionadas.

Los resultados incluyen análisis comparativo por nivel de complejidad (Baja, Media y Alta), diagramas de flujo metodológicos y tabla cuantitativa de métricas en formato JSON y PNG.
