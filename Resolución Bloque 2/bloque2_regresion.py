# =============================================================================
# BLOQUE II - REGRESIÓN Y COMPARACIÓN DE MODELOS
# Solución de ejercicios
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración
# Rutas absolutas para mayor compatibilidad
BASE_DIR = Path(r"C:\Users\juang\OneDrive\Desktop\Proyecto análisis de datos\Big_Data_2026-main\Big_Data_2026-main")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "Resolución Bloque 2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================
print("=" * 60)
print("BLOQUE II - REGRESIÓN Y COMPARACIÓN DE MODELOS")
print("=" * 60)

df = pd.read_csv(DATA_DIR / "ventas_mayo_2026.csv")
df = df.drop_duplicates()
df["fecha"] = pd.to_datetime(df["fecha"])
df["mes"] = df["fecha"].dt.month

print(f"\nDataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
print(f"\nColumnas disponibles: {list(df.columns)}")
print(f"\nValores nulos por columna:")
print(df.isnull().sum())

# =============================================================================
# 2. DEFINICIÓN DE FEATURES Y TARGET
# =============================================================================
target = "importe"

features_num = ["unidades", "precio_unitario", "descuento", "antiguedad_cliente_meses", "mes"]
features_cat = ["categoria", "region", "canal"]

X = df[features_num + features_cat]
y = df[target]

print(f"\nTarget: {target}")
print(f"Features numéricos: {features_num}")
print(f"Features categóricos: {features_cat}")

# =============================================================================
# 3. PREPROCESADOR
# =============================================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, features_num),
        ("cat", categorical_transformer, features_cat)
    ]
)

# =============================================================================
# 4. FUNCIÓN DE EVALUACIÓN
# =============================================================================
def evaluar_modelo(nombre, modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    return {
        "modelo": nombre,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }, pred

# =============================================================================
# EJERCICIO 1: COMPARACIÓN TEST SIZE 20% vs 30%
# =============================================================================
print("\n" + "=" * 60)
print("EJERCICIO 1: COMPARACIÓN TEST SIZE 20% vs 30%")
print("=" * 60)

# División con test_size=20%
X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# División con test_size=30%
X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTest size 20%: Train={X_train_20.shape[0]}, Test={X_test_20.shape[0]}")
print(f"Test size 30%: Train={X_train_30.shape[0]}, Test={X_test_30.shape[0]}")

# Evaluar Random Forest con ambos tamaños
modelo_rf_20 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42, max_depth=None))
])

modelo_rf_30 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42, max_depth=None))
])

res_20, pred_20 = evaluar_modelo("RF_20%", modelo_rf_20, X_train_20, X_test_20, y_train_20, y_test_20)
res_30, pred_30 = evaluar_modelo("RF_30%", modelo_rf_30, X_train_30, X_test_30, y_train_30, y_test_30)

print(f"\nResultados con test_size=20%:")
print(f"  MAE:  {res_20['MAE']:,.2f}")
print(f"  RMSE: {res_20['RMSE']:,.2f}")
print(f"  R²:   {res_20['R2']:.3f}")

print(f"\nResultados con test_size=30%:")
print(f"  MAE:  {res_30['MAE']:,.2f}")
print(f"  RMSE: {res_30['RMSE']:,.2f}")
print(f"  R²:   {res_30['R2']:.3f}")

print("\n-> Con test_size=30% tenemos más datos de entrenamiento, lo que suele")
print("   mejorar el rendimiento del modelo.")

# =============================================================================
# EJERCICIO 2: MODIFICAR max_depth EN RANDOM FOREST
# =============================================================================
print("\n" + "=" * 60)
print("EJERCICIO 2: MODIFICAR max_depth EN RANDOM FOREST")
print("=" * 60)

# Probar diferentes profundidades
profundidades = [5, 10, 15, 20, None]
resultados_depth = []

for depth in profundidades:
    modelo = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, max_depth=depth))
    ])
    
    modelo.fit(X_train_20, y_train_20)
    pred = modelo.predict(X_test_20)
    
    resultados_depth.append({
        "max_depth": str(depth) if depth is not None else "None (ilimitado)",
        "MAE": mean_absolute_error(y_test_20, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_20, pred)),
        "R2": r2_score(y_test_20, pred)
    })

df_depth = pd.DataFrame(resultados_depth)
print("\nComparación de profundidades:")
print(df_depth.to_string(index=False))

# Mejor profundidad
mejor_depth = df_depth.loc[df_depth["RMSE"].idxmin(), "max_depth"]
print(f"\n-> Mejor max_depth según RMSE: {mejor_depth}")

# =============================================================================
# EJERCICIO 3: AÑADIR O ELIMINAR VARIABLES
# =============================================================================
print("\n" + "=" * 60)
print("EJERCICIO 3: AÑADIR/ELIMINAR VARIABLES PREDICTORAS")
print("=" * 60)

# Caso A: Solo variables numéricas
features_num_only = ["unidades", "precio_unitario", "descuento"]
X_num = df[features_num_only]

X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(
    X_num, y, test_size=0.2, random_state=42
)

num_transformer_only = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

modelo_num_only = Pipeline(steps=[
    ("preprocessor", num_transformer_only),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

res_num_only, _ = evaluar_modelo("Solo numéricas", modelo_num_only, 
                                  X_train_num, X_test_num, y_train_num, y_test_num)

# Caso B: Todas las variables (incluyendo más features)
features_extended = ["unidades", "precio_unitario", "descuento", 
                     "antiguedad_cliente_meses", "mes", "categoria", 
                     "region", "canal", "cliente_id"]

# Convertir cliente_id a número (contando ocurrencias)
df["cliente_frecuencia"] = df.groupby("cliente_id")["cliente_id"].transform("count")
features_extended = ["unidades", "precio_unitario", "descuento", 
                     "antiguedad_cliente_meses", "mes", "cliente_frecuencia"]
features_cat_ext = ["categoria", "region", "canal"]

X_ext = df[features_extended + features_cat_ext]

X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
    X_ext, y, test_size=0.2, random_state=42
)

preprocessor_ext = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, features_extended),
        ("cat", categorical_transformer, features_cat_ext)
    ])

modelo_ext = Pipeline(steps=[
    ("preprocessor", preprocessor_ext),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

res_ext, _ = evaluar_modelo("Variables extendidas", modelo_ext,
                            X_train_ext, X_test_ext, y_train_ext, y_test_ext)

# Caso C: Variables originales (baseline)
res_baseline, _ = evaluar_modelo("Baseline (original)", modelo_rf_20,
                                 X_train_20, X_test_20, y_train_20, y_test_20)

print("\nComparación de features:")
print(f"{'Modelo':<25} {'MAE':>10} {'RMSE':>12} {'R2':>8}")
print("-" * 55)
print(f"{'Solo numéricas':<25} {res_num_only['MAE']:>10,.2f} {res_num_only['RMSE']:>12,.2f} {res_num_only['R2']:>8.3f}")
print(f"{'Original (baseline)':<25} {res_baseline['MAE']:>10,.2f} {res_baseline['RMSE']:>12,.2f} {res_baseline['R2']:>8.3f}")
print(f"{'Variables extendidas':<25} {res_ext['MAE']:>10,.2f} {res_ext['RMSE']:>12,.2f} {res_ext['R2']:>8.3f}")

# =============================================================================
# EJERCICIO 4: COMPARACIÓN FINAL DE MODELOS
# =============================================================================
print("\n" + "=" * 60)
print("EJERCICIO 4: COMPARACIÓN DE MODELOS")
print("=" * 60)

# Definir modelos
modelos = {
    "Linear Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),
    "Ridge (alpha=1.0)": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]),
    "Lasso (alpha=0.05)": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Lasso(alpha=0.05, max_iter=10000))
    ]),
    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15))
    ])
}

resultados = []
predicciones = {}

for nombre, modelo in modelos.items():
    res, pred = evaluar_modelo(nombre, modelo, X_train_20, X_test_20, y_train_20, y_test_20)
    resultados.append(res)
    predicciones[nombre] = pred

df_resultados = pd.DataFrame(resultados).sort_values("RMSE")
print("\nTabla comparativa de modelos:")
print(df_resultados.to_string(index=False))

# Gráfico de comparación
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# MAE
axes[0].bar(df_resultados["modelo"], df_resultados["MAE"], color="steelblue")
axes[0].set_title("MAE (Error Absoluto Medio)")
axes[0].set_ylabel("MAE")
axes[0].tick_params(axis="x", rotation=30)

# RMSE
axes[1].bar(df_resultados["modelo"], df_resultados["RMSE"], color="coral")
axes[1].set_title("RMSE (Raíz Error Cuadrático)")
axes[1].set_ylabel("RMSE")
axes[1].tick_params(axis="x", rotation=30)

# R²
axes[2].bar(df_resultados["modelo"], df_resultados["R2"], color="seagreen")
axes[2].set_title("R² (Varianza Explicada)")
axes[2].set_ylabel("R²")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparacion_modelos.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n-> Gráfico guardado: {OUTPUT_DIR / 'comparacion_modelos.png'}")

# =============================================================================
# EJERCICIO 5: GRÁFICO REAL VS PREDICHO
# =============================================================================
print("\n" + "=" * 60)
print("GRÁFICO: REAL VS PREDICHO")
print("=" * 60)

mejor_modelo = "Random Forest"
mejor_pred = predicciones[mejor_modelo]

plt.figure(figsize=(6, 6))
plt.scatter(y_test_20, mejor_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
plt.plot([y_test_20.min(), y_test_20.max()], [y_test_20.min(), y_test_20.max()], 
         "r--", lw=2, label="Línea perfecta")
plt.xlabel("Valor real (importe)", fontsize=12)
plt.ylabel("Valor predicho", fontsize=12)
plt.title(f"Real vs Predicho - {mejor_modelo}", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "real_vs_predicho.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"-> Gráfico guardado: {OUTPUT_DIR / 'real_vs_predicho.png'}")

# =============================================================================
# ANÁLISIS DE ERRORES
# =============================================================================
print("\n" + "=" * 60)
print("ANÁLISIS DE ERRORES")
print("=" * 60)

errores_df = pd.DataFrame({
    "real": y_test_20.values,
    "predicho": mejor_pred
})
errores_df["error"] = errores_df["real"] - errores_df["predicho"]
errores_df["error_abs"] = abs(errores_df["error"])
errores_df["error_pct"] = (errores_df["error_abs"] / errores_df["real"]) * 100

print("\nPeores predicciones (mayor error absoluto):")
peores = errores_df.sort_values("error_abs", ascending=False).head(10)
print(peores[["real", "predicho", "error", "error_pct"]].to_string(index=False))

print(f"\nEstadísticas de error:")
print(f"  Error medio: {errores_df['error'].mean():,.2f}")
print(f"  Error medio absoluto: {errores_df['error_abs'].mean():,.2f}")
print(f"  Error medio porcentual: {errores_df['error_pct'].mean():.1f}%")

# =============================================================================
# CONCLUSIÓN DE NEGOCIO
# =============================================================================
print("\n" + "=" * 60)
print("CONCLUSIÓN DE NEGOCIO")
print("=" * 60)

mejor_fila = df_resultados.iloc[0]
peor_fila = df_resultados.iloc[-1]

conclusion = f"""
ANÁLISIS DE MODELOS DE REGRESIÓN - BLOQUE II
=============================================

1. MODELO SELECCIONADO: {mejor_fila['modelo']}
   - MAE:  {mejor_fila['MAE']:,.2f} euros
   - RMSE: {mejor_fila['RMSE']:,.2f} euros
   - R²:   {mejor_fila['R2']:.3f} ({mejor_fila['R2']*100:.1f}% de varianza explicada)

2. JUSTIFICACIÓN DE SELECCIÓN:
   - Random Forest ofrece el mejor equilibrio entre precisión (menor RMSE)
     y capacidad de capturar relaciones no lineales.
   - El R² de {mejor_fila['R2']:.3f} indica que el modelo explica casi 
     el {mejor_fila['R2']*100:.0f}% de la variabilidad en los importes.
   - La regularización (Ridge/Lasso) no mejora significativamente el 
     rendimiento en este dataset.

3. COMPARACIÓN CON OTROS MODELOS:
   - Peor modelo: {peor_fila['modelo']} (RMSE: {peor_fila['RMSE']:,.2f})
   - La diferencia de RMSE es de {peor_fila['RMSE'] - mejor_fila['RMSE']:,.2f} euros

4. IMPLICACIONES DE NEGOCIO:
   - Con un MAE de {mejor_fila['MAE']:,.2f} euros, el modelo comete 
     errores promedio de aproximadamente ese valor.
   - Esto representa un { (mejor_fila['MAE'] / errores_df['real'].mean()) * 100:.1f}% 
     de error relativo sobre el importe promedio.
   - El modelo es útil para estimaciones rápidas pero no substitute 
     cotizaciones precisas en casos de alto valor.

5. RECOMENDACIONES:
   - Usar el modelo para segmentar operaciones de bajo/alto valor.
   - Para importes superiores a 3000 euros, considerar revisión manual.
   - Mejorar con más datos históricos o features externos (temporada, promociones).
"""

print(conclusion)

# Guardar conclusión a archivo
with open(OUTPUT_DIR / "conclusion_negocio.txt", "w", encoding="utf-8") as f:
    f.write(conclusion)
print(f"-> Conclusión guardada: {OUTPUT_DIR / 'conclusion_negocio.txt'}")

# =============================================================================
# GUARDAR RESULTADOS A CSV
# =============================================================================
df_resultados.to_csv(OUTPUT_DIR / "resultados_modelos.csv", index=False)
print(f"-> Resultados guardados: {OUTPUT_DIR / 'resultados_modelos.csv'}")

print("\n" + "=" * 60)
print("EJERCICIOS COMPLETADOS")
print("=" * 60)
print(f"\nArchivos generados en {OUTPUT_DIR}:")
print("  - comparacion_modelos.png")
print("  - real_vs_predicho.png")
print("  - conclusion_negocio.txt")
print("  - resultados_modelos.csv")