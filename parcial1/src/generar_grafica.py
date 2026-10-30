import json
import matplotlib.pyplot as plt
from pathlib import Path

def generate_plot():
    # Resolver ruta del archivo de estadísticas
    possible_paths = [
        Path("parcial1/resultados/dataset_stats.json"),
        Path("resultados/dataset_stats.json"),
        Path(__file__).resolve().parent.parent / "resultados" / "dataset_stats.json",
    ]
    stats_file = next((p for p in possible_paths if p.exists()), None)
    if stats_file is None:
        raise FileNotFoundError("No se encontró dataset_stats.json en las rutas esperadas.")

    with open(stats_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    classes = data['by_class']
    names = list(classes.keys())
    values = list(classes.values())

    # Limpiar los nombres para la gráfica
    names_clean = [n.replace('___', ' - ').replace('_', ' ') for n in names]

    # Crear la figura (alta para que quepan 39 clases)
    plt.figure(figsize=(10, 12))
    plt.barh(names_clean, values, color='skyblue', edgecolor='black')
    plt.xlabel('Número de Imágenes')
    plt.title('Distribución de Imágenes por Categoría')

    for index, value in enumerate(values):
        plt.text(value, index, str(value), va='center', fontsize=8)

    plt.tight_layout()

    out_dir = stats_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "distribucion_clases.png"
    plt.savefig(out_file, dpi=300)
    print(f"Gráfica guardada en {out_file}")

if __name__ == '__main__':
    generate_plot()
