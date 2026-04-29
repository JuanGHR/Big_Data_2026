"""
Script para ejecutar los 4 notebooks de Iris
Ejecutar desde una terminal con Jupyter instalado:
    python EJECUTAR_NOTEBOOKS.py
"""

import os
import sys

# Verificar que estamos en el directorio correcto
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 60)
print("EJECUTOR DE NOTEBOOKS - IRIS DATASET")
print("=" * 60)

# Intentar usar papermill (más robusto) o nbconvert
try:
    from papermill.execute import execute_notebook
    print("\n✅ Usando papermill para ejecutar notebooks\n")
    
    notebooks = [
        "01_Bloque_I_Iris_EDA.ipynb",
        "02_Bloque_II_Iris_Regression.ipynb",
        "03_Bloque_III_Iris_Classification.ipynb",
        "04_Bloque_IV_Iris_Clustering.ipynb"
    ]
    
    for nb in notebooks:
        if os.path.exists(nb):
            print(f"Ejecutando: {nb}")
            try:
                execute_notebook(nb, nb)
                print(f"  ✅ {nb} - COMPLETADO\n")
            except Exception as e:
                print(f"  ❌ Error en {nb}: {str(e)}\n")
        else:
            print(f"  ⚠️  No encontrado: {nb}\n")
            
except ImportError:
    print("\n⚠️  papermill no instalado. Intentando con nbconvert...\n")
    
    try:
        import nbformat
        from nbclient import NotebookClient
        
        notebooks = [
            "01_Bloque_I_Iris_EDA.ipynb",
            "02_Bloque_II_Iris_Regression.ipynb",
            "03_Bloque_III_Iris_Classification.ipynb",
            "04_Bloque_IV_Iris_Clustering.ipynb"
        ]
        
        for nb in notebooks:
            if os.path.exists(nb):
                print(f"Ejecutando: {nb}")
                try:
                    notebook = nbformat.read(nb, as_version=4)
                    client = NotebookClient(notebook, timeout=300)
                    client.execute()
                    nbformat.write(notebook, nb)
                    print(f"  ✅ {nb} - COMPLETADO\n")
                except Exception as e:
                    print(f"  ❌ Error en {nb}: {str(e)}\n")
            else:
                print(f"  ⚠️  No encontrado: {nb}\n")
                
    except ImportError:
        print("\n❌ ERROR: No se encontró papermill ni nbclient")
        print("Instalar con: pip install papermill nbclient\n")
        sys.exit(1)

print("=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print(f"\nNotebooks en: {script_dir}")
print("Podés abrirlos también manualmente desde Jupyter Notebook")
