# Procesamiento de imágenes de plantas

Implementación del primer parcial del proyecto de procesamiento de imágenes
para detección de plagas y enfermedades en plantas.

## Dataset

El proyecto utiliza el [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
de Kaggle. El dataset no se incluye en este repositorio por su tamaño
aproximado de 2.7 GB. Descárgalo y descomprímelo dentro de `dataset/`.

El análisis local realizado encontró:

- 87,900 imágenes JPG.
- 39 categorías.
- 70,295 imágenes de entrenamiento.
- 17,572 imágenes de validación.
- 33 imágenes de prueba.
- Imágenes RGB de 256x256 píxeles.

## Ejecución

Con Python y las dependencias `opencv-python-headless` y `numpy` instaladas:

```powershell
python analisis_filtros.py --dataset dataset --salida resultados
```

El script genera las estadísticas del conjunto y tres comparaciones visuales
con ecualización CLAHE, filtro de mediana, suavizado gaussiano, realce unsharp
y una combinación de filtros.

## Estructura

- `analisis_filtros.py`: análisis del dataset y aplicación de filtros.
- `resultados/dataset_stats.json`: estadísticas completas obtenidas.
- `resultados/filtros_01.png` a `filtros_03.png`: resultados visuales.
- `descripProyPDIIC_PARTE1.pdf`: descripción del primer parcial.
