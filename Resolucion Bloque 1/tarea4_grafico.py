# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Cargar datos - ruta relativa desde la raiz del proyecto
import os
proyecto_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(proyecto_dir, 'data', 'ventas_mayo_2026.csv')
df = pd.read_csv(data_path)

# Limpieza basica
df_limpio = df.copy()
df_limpio = df_limpio.drop_duplicates()
df_limpio['fecha'] = pd.to_datetime(df_limpio['fecha'])
df_limpio['precio_unitario'] = df_limpio['precio_unitario'].fillna(df_limpio['precio_unitario'].median())
df_limpio['region'] = df_limpio['region'].fillna('Sin informar')

# TAREA 4: Grafico de barras con ventas por canal
ventas_por_canal = df_limpio.groupby('canal')['importe'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
colores = ['#2ecc71', '#3498db', '#9b59b6']
bars = plt.bar(ventas_por_canal.index, ventas_por_canal.values, color=colores, edgecolor='black', linewidth=1.2)

plt.title('Ventas totales por canal', fontsize=14, fontweight='bold')
plt.xlabel('Canal', fontsize=12)
plt.ylabel('Importe total (euros)', fontsize=12)
plt.xticks(rotation=0, fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Anadir valores en las barras
for bar, valor in zip(bars, ventas_por_canal.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
             f'{valor:,.0f} euros', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('ventas_por_canal.png', dpi=150)
print('Grafico guardado: ventas_por_canal.png')