# Procesamiento de Imágenes de Plantas - Detección de Plagas

Este proyecto implementa técnicas de Procesamiento Digital de Imágenes (PDI) para analizar y preprocesar el conjunto de datos de hojas enfermas y sanas, cumpliendo con los requerimientos del primer parcial.

El proyecto ha sido refactorizado aplicando **principios S.O.L.I.D.** para asegurar un código modular, mantenible y escalable.

## Estructura del Proyecto

```text
descripProyPDIIC/
├── src/                        # Código fuente refactorizado (S.O.L.I.D.)
│   ├── main.py                 # Orquestador y punto de entrada
│   ├── filters.py              # Interfaces y clases de Filtros (Pipeline, ExG, CLAHE, etc.)
│   ├── dataset.py              # Lógica de carga y análisis del Dataset
│   └── visualization.py        # Generación de grillas comparativas
├── legacy/                     # Scripts monolíticos anteriores
├── resultados/                 # Imágenes de salida y metadatos JSON generados
├── pyproject.toml              # Definición del proyecto y dependencias (uv)
├── download_dataset.py         # Script para descarga automática desde Kaggle
├── reporte_pdi.tex             # Reporte formal en formato LaTeX
└── reporte_pdi.pdf             # PDF compilado del reporte
```

## Requisitos y Entorno (con `uv`)

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) como gestor de paquetes y dependencias ultrarrápido.

1. **Instalar uv** (si no lo tienes):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Sincronizar el entorno y dependencias**:
   ```bash
   uv sync
   ```
   Esto creará un entorno virtual aislado `.venv` e instalará `opencv-python-headless`, `numpy` y `kaggle`.

## Descarga del Dataset

El proyecto utiliza el [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). Para automatizar su descarga (2.7 GB):

1. Asegúrate de tener tu archivo `kaggle.json` configurado en `~/.kaggle/kaggle.json`.
2. Ejecuta el script de descarga en el entorno de uv:
   ```bash
   uv run download_dataset.py
   ```
   Esto extraerá las imágenes en la carpeta `dataset/`.

## Ejecución del Pipeline PDI

Para analizar el dataset, calcular las métricas y generar las imágenes comparativas aplicando los filtros S.O.L.I.D. (incluyendo la máscara de hoja ExG, Filtro Bilateral, CLAHE y Unsharp Masking):

```bash
uv run -m src.main --dataset dataset --salida resultados
```

Esto generará en la carpeta `resultados/`:
- `dataset_stats.json`
- `filtros_solid_01.png`, `filtros_solid_02.png`, etc.
