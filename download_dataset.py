#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Descarga el New Plant Diseases Dataset desde Kaggle.")
    parser.add_argument("--dest", type=str, default="dataset", help="Carpeta de destino para el dataset")
    args = parser.parse_args()

    dest_path = Path(args.dest)
    
    print("Verificando credenciales de Kaggle...")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        print(f"ERROR: No se encontró el archivo de credenciales {kaggle_json}")
        print("Para descargar el dataset necesitas:")
        print("1. Crear una cuenta en https://www.kaggle.com/")
        print("2. Ir a 'Settings' > 'Create New Token' para descargar kaggle.json")
        print("3. Colocar el archivo en ~/.kaggle/kaggle.json")
        print("4. Ejecutar: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    print("Instalando/verificando la librería 'kaggle'...")
    os.system(f"{sys.executable} -m pip install -q kaggle")

    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDescargando 'vipoooool/new-plant-diseases-dataset' en la carpeta '{dest_path}'...")
    print("Esto puede tardar varios minutos dependiendo de tu conexión (aprox. 2.7 GB)...")
    
    # Descargar y descomprimir directamente
    command = f'kaggle datasets download -d vipoooool/new-plant-diseases-dataset --unzip -p "{dest_path}"'
    result = os.system(command)

    if result == 0:
        print("\n¡Descarga y extracción completadas con éxito!")
        print(f"El dataset está listo para usarse en la ruta: {dest_path.absolute()}")
    else:
        print("\nOcurrió un error al intentar descargar el dataset.")
        sys.exit(result)

if __name__ == "__main__":
    main()
