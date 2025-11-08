# Importación de librerías y supresión de advertencias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualizar_datos(fuente: str = "data/raw/bank-additional-full.csv",
                    salida: str = "docs/figures/"):
    """Genera una serie de gráficos sobre los datos y los exporta

    Args:
        fuente (str, optional): ruta al archivo de datos. Defaults to "data/raw/bank-additional-full.csv".
        salida (str, optional): ruta al directorio de salida para los gráficos. Defaults to "docs/figures/".
    """

    # Crear el directorio de salida si no existe
    Path(salida).mkdir(parents=True, exist_ok=True)

    # Leer los datos
    df = pd.read_csv(fuente, sep=';')

    # Gráfico 1: Distribución de la variable objetivo
    plt.figure(figsize=(6, 4))
    sns.countplot(x="y", data=df)
    plt.title("Distribución de la variable objetivo (suscripción al depósito)")
    plt.xlabel("¿Suscribió un depósito a plazo?")
    plt.ylabel("Cantidad de clientes")
    plt.savefig(f"{salida}/distribucion_target.png")
    plt.close()

    # Gráfico 2: Distribución del nivel educativo
    plt.figure(figsize=(6, 4))
    col = "education"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(f"{salida}/distribucion_educacion.png")
    plt.close()

    # Gráfico 3: Matriz de correlación de variables numéricas
    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de correlaciones entre variables numéricas")
    plt.savefig(f"{salida}/matriz_correlaciones.png")
    plt.close()

    # Gráfico 4: Distribución de la variable 'job'
    plt.figure(figsize=(8, 4))
    order = df["job"].value_counts().index
    sns.countplot(y="job", data=df, order=order)
    plt.title("Distribución por tipo de empleo")
    plt.xlabel("Cantidad")
    plt.ylabel("Ocupación")
    plt.savefig(f"{salida}/distribucion_job.png")
    plt.close()


if __name__ == "__main__":
    visualizar_datos()