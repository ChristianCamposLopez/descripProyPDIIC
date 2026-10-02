import json
import matplotlib.pyplot as plt
import os

def generate_plot():
    # Cargar los datos
    with open('dataset_stats.json', 'r') as f:
        data = json.load(f)
    
    classes = data['by_class']
    
    # Ordenar por número de imágenes (opcional, ayuda a la visualización)
    # classes_sorted = dict(sorted(classes.items(), key=lambda item: item[1]))
    # En este caso, podemos dejarlos como están o ordenarlos alfabéticamente
    
    names = list(classes.keys())
    values = list(classes.values())
    
    # Limpiar los nombres para la gráfica
    names_clean = [n.replace('___', ' - ').replace('_', ' ') for n in names]
    
    # Crear la figura (alta para que quepan 39 clases)
    plt.figure(figsize=(10, 12))
    
    # Crear barras horizontales
    plt.barh(names_clean, values, color='skyblue', edgecolor='black')
    
    plt.xlabel('Número de Imágenes')
    plt.title('Distribución de Imágenes por Categoría')
    
    # Añadir el número al final de cada barra
    for index, value in enumerate(values):
        plt.text(value, index, str(value), va='center', fontsize=8)
        
    plt.tight_layout()
    
    # Asegurar que el directorio de resultados exista
    os.makedirs('resultados', exist_ok=True)
    
    # Guardar la figura
    plt.savefig('resultados/distribucion_clases.png', dpi=300)
    print("Gráfica guardada en resultados/distribucion_clases.png")

if __name__ == '__main__':
    generate_plot()
