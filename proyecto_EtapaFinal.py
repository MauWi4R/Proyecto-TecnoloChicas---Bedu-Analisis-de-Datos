import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import trim_mean

file_path = "datos_limpiadosII.csv"
data = pd.read_csv(file_path)
print(data.head())

numeric_columns = [col for col in data.select_dtypes(include=['float64', 'int64']).columns if col not in ["Year", "Population"]]
print("Columnas relevantes:", numeric_columns)

#Estimados de locación y variabilidad
def calculate_statistics(df, columns):
    stats = {}
    for col in columns:
        col_data = df[col].dropna()
        stats[col] = {
            "Promedio": col_data.mean(),
            "Mediana": col_data.median(),
            "Media Truncada (10%)": trim_mean(col_data, 0.1),
            "Desviación Estándar": col_data.std(),
            "Rango": col_data.max() - col_data.min(),
            "Percentil 25": np.percentile(col_data, 25),
            "Percentil 75": np.percentile(col_data, 75),
            "Rango Intercuartil": np.percentile(col_data, 75) - np.percentile(col_data, 25),
        }
    return stats

statistics = calculate_statistics(data, numeric_columns)

# Resultados
for col, stats in statistics.items():
    print(f"\nEstadísticas para {col}:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")

# Análisis de la distribución de las variables numéricas utilizando Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numeric_columns], orient="h", palette="Set2")
plt.title("Distribución de las Variables Numéricas")
plt.xlabel("Valor")
plt.show()


# Score de Rango Intercuartílico
def filter_outliers(df, columns):
    filtered_data = df.copy()
    outliers = {}
    for col in columns:
        Q1 = np.percentile(df[col].dropna(), 25)
        Q3 = np.percentile(df[col].dropna(), 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_data = filtered_data[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

    return filtered_data, outliers

# Filtrar valores atípicos
filtered_data, outliers = filter_outliers(data, numeric_columns)

# Comparación antes y después
statistics_filtered = calculate_statistics(filtered_data, numeric_columns)

print("\nComparación de estadísticas antes y después de filtrar:")
for col in numeric_columns:
    print(f"\nColumna: {col}")
    print(f"Media antes: {statistics[col]['Promedio']:.2f}, después: {statistics_filtered[col]['Promedio']:.2f}")
    print(f"Mediana antes: {statistics[col]['Mediana']:.2f}, después: {statistics_filtered[col]['Mediana']:.2f}")
    print(f"Desviación estándar antes: {statistics[col]['Desviación Estándar']:.2f}, después: {statistics_filtered[col]['Desviación Estándar']:.2f}")


# Histogramas
filtered_data[numeric_columns].hist(bins=15, figsize=(15, 10), color="skyblue", edgecolor="black")
plt.suptitle("Histogramas")
plt.show()

# Tablas de frecuencia
for col in numeric_columns:
    print(f"\nTabla de frecuencia para {col}:")
    print(filtered_data[col].value_counts(bins=10).sort_index())


from scipy.stats import skew, kurtosis

# Calcular asimetría y curtosis
for col in numeric_columns:
    col_data = filtered_data[col].dropna()
    print(f"\nColumna: {col}")
    print(f"Asimetría: {skew(col_data):.2f}")
    print(f"Curtosis: {kurtosis(col_data):.2f}")


# Gráfica de densidad segmentada por género
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=filtered_data, x="Prevalence in males (%)", label="Hombres", fill=True, color="blue")
sns.kdeplot(data=filtered_data, x="Prevalence in females (%)", label="Mujeres", fill=True, color="pink")
plt.title("Distribución de la Prevalencia por Género")
plt.xlabel("Prevalencia (%)")
plt.ylabel("Densidad")
plt.legend()
plt.show()
