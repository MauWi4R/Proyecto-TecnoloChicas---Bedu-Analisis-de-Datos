import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import trim_mean

file_path = "datos_limpiadosII.csv"
data = pd.read_csv(file_path)
print(data.head())

from google.colab import drive
drive.mount('/content/drive')

numeric_columns = [col for col in data.select_dtypes(include=['float64', 'int64']).columns if col not in ["Year"]]
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

# Análisis de la distribución de variables utilizando Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['Prevalence in males (%)', 'Prevalence in females (%)', 'Total (%)']], orient="h", palette="Set2")
plt.title("Distribución de las Variables Numéricas")
plt.xlabel("Valor")
plt.show()

# Análisis de la distribución de "Depressive disorders (Number)" utilizando un boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=data['Depressive disorders (Number)'], orient="h")
plt.title("Distribución de 'Depressive disorders (Number)'")
plt.xlabel("Valor")
plt.show()

# Análisis de la distribución de "Population" utilizando un boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=data['Population'], orient="h")
plt.title("Distribución de 'Population'")
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

data["Year"] = data["Year"].astype(int)

sns.set_style("whitegrid")

# Gráfica de barras de prevalencia media por País 'Entity'
plt.figure(figsize=(12, 6))
country_mean = data.groupby("Entity")["Total (%)"].mean().sort_values()  # Ordena de menor a mayor
sns.barplot(x=country_mean.values, y=country_mean.index, hue=country_mean.index, palette="viridis", legend=False)
plt.title("Prevalencia Media de Depresión por País")
plt.xlabel("Prevalencia Media (%)")
plt.ylabel("País")
plt.show()

# Gráfica de línea de prevalencia media por año 'Year'
plt.figure(figsize=(10, 5))
year_mean = data.groupby("Year")["Total (%)"].mean()
sns.lineplot(x=year_mean.index, y=year_mean.values, hue= year_mean.index, marker="o", color="b", legend=False)
plt.title("Evolución de la Prevalencia Media de Depresión a lo Largo del Tiempo")
plt.xlabel("Año")
plt.ylabel("Prevalencia media (%)")
plt.xticks(rotation=45)
plt.show()

# Gráfica combinada para la prevalencia por género a lo largo del tiempo
plt.figure(figsize=(10, 6))
sns.lineplot(x="Year", y="Prevalence in males (%)", data=data, label="Hombres", marker="o")
sns.lineplot(x="Year", y="Prevalence in females (%)", data=data, label="Mujeres", marker="s")
plt.title("Prevalencia de Depresión por Género a lo Largo del Tiempo")
plt.xlabel("Año")
plt.ylabel("Prevalencia (%)")
plt.legend()
plt.show()

# Creación de Tabla de Contingencia
contingency_gender = data.pivot_table(index="Entity",
                                    values=["Prevalence in males (%)", "Prevalence in females (%)"],
                                    aggfunc="mean")
print(contingency_gender.head())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=True)

# Gráfico de barras para hombres
sns.barplot(x=contingency_gender["Prevalence in males (%)"],
            y=contingency_gender.index,
            ax=axes[0],
            hue=contingency_gender.index,
            palette="Blues_r",
            legend=False)

axes[0].set_title("Prevalencia de Depresión en Hombres")
axes[0].set_xlabel("Prevalencia (%)")
axes[0].set_ylabel("País")

# Gráfico de barras para mujeres
sns.barplot(x=contingency_gender["Prevalence in females (%)"],
            y=contingency_gender.index,
            ax=axes[1],
            hue=contingency_gender.index,
            palette="Reds_r",
            legend=False)
axes[1].set_title("Prevalencia de Depresión en Mujeres")
axes[1].set_xlabel("Prevalencia (%)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

# Boxplots y violinplots para análisis numérico-categórico
# Boxplot de "Total (%)" por país 'Entity'
plt.figure(figsize=(12, 6))
sns.boxplot(x="Total (%)", y="Entity", data=data, hue="Entity", palette="coolwarm", legend=False)
plt.title("Distribución de la prevalencia de depresión por país")
plt.xlabel("Total (%)")
plt.ylabel("País")
plt.show()

# Violinplot de "Total (%)" por año
plt.figure(figsize=(10, 6))
sns.violinplot(x="Year", y="Total (%)", data=data, hue="Year", palette="coolwarm", legend=False)
plt.title("Distribución de la Prevalencia de Depresión Por Año")
plt.xlabel("Año")
plt.ylabel("Total (%)")
plt.xticks(rotation=40)
plt.show()

# Matriz de Correlaciones
correlation_matrix = data[numeric_columns].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlaciones")
plt.show()

# Pairplot
sns.pairplot(correlation_matrix)
plt.show()

# Gráficas de dispersión
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=data.index, y=data[col])
    plt.title(f"Dispersión de {col}")
plt.tight_layout()
plt.show()

