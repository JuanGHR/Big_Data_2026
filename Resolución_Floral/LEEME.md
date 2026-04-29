# Análisis Cuantificación Floral - 4 Bloques

Trabajo realizado siguiendo la estructura de los 4 primeros bloques del curso Big Data 2026, pero utilizando el dataset **Iris** en lugar de los datos originales de ventas.

## Archivos incluidos

1. **01_Bloque_I_Iris_EDA.ipynb** - Análisis Descriptivo (EDA)
   - Exploración del dataset Iris (150 muestras, 3 especies, 4 características)
   - Visualizaciones: histogramas, boxplots, scatter plots, pairplot
   - Estadísticas por especie

2. **02_Bloque_II_Iris_Regression.ipynb** - Regresión
   - Predicción de `PetalLengthCm` (variable continua)
   - Modelos: Linear Regression, Ridge, Lasso, Random Forest
   - Métricas: MAE, RMSE, R²
   - Comparación de modelos

3. **03_Bloque_III_Iris_Classification.ipynb** - Clasificación
   - Clasificación multiclase (3 especies: setosa, versicolor, virginica)
   - Modelos: Logistic Regression, Decision Tree, Random Forest
   - Métricas por clase: Precision, Recall, F1-score
   - Matriz de confusión y Curvas ROC (One-vs-Rest)

4. **04_Bloque_IV_Iris_Clustering.ipynb** - Clustering
   - K-Means (k=2, 3, 4, 5)
   - Métricas: Silhouette Score, Elbow method
   - PCA para visualización en 2D
   - Comparación con etiquetas reales

5. **Iris.csv** - Dataset utilizado
   - 150 filas × 6 columnas (Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species)

6. **EJECUTAR_NOTEBOOKS.py** - Script auxiliar
   - Ejecuta los 4 notebooks automáticamente
   - Requiere: `papermill` o `nbclient`

## Cómo ejecutar los notebooks

### Opción A: Desde Jupyter Notebook (recomendado)
1. Abrir Jupyter Notebook en esta carpeta
2. Ejecutar cada notebook en orden (Bloque I → II → III → IV)
3. Ir a "Cell" → "Run All" en cada uno

### Opción B: Ejecución automática
```bash
pip install papermill
python EJECUTAR_NOTEBOOKS.py
```

## Notas importantes

- Los notebooks están **creados pero no ejecutados** (no tienen outputs)
- Ejecutarlos desde tu entorno local de Jupyter (que ya tienes configurado)
- El dataset Iris es clásico para aprendizaje automático y clasificación
- Se mantuvo el espíritu de los 4 bloques originales:
  - Bloque I: EDA (sí aplica, pero con Iris)
  - Bloque II: Regresión (se adaptó para predecir una medida floral)
  - Bloque III: Clasificación (donde Iris brilla - 3 especies)
  - Bloque IV: Clustering (sin usar etiquetas, agrupamiento natural)

## Cambios respecto a los originales

| Bloque Original | Adaptación para Iris |
|-----------------|----------------------|
| Usaba `ventas_mayo_2026.csv` | Usa `Iris.csv` |
| Bloque II predecía `importe` | Predice `PetalLengthCm` |
| Bloque III: abandono (binario) | Clasificación 3 especies (multiclase) |
| Bloque IV: segmentación clientes | Clustering de flores por similitud |

---
**Creado:** Abril 2026  
**Curso:** Big Data 2026  
**Dataset:** Iris (Fisher, 1936)
