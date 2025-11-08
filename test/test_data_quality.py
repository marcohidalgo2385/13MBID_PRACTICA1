import pandas as pd
from pandera.pandas import DataFrameSchema, Column
from pandera import Column, DataFrameSchema, Check
import pytest

@pytest.fixture
def datos_banco():
    """Fixture para cargar y limpiar los datos del banco desde un archivo CSV."""
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=';')

    # 🔹 Limpieza de datos
    df = df.drop_duplicates()        # Eliminar filas duplicadas
    df = df.dropna()                 # Eliminar valores nulos
    df = df[df["age"] > 17]          # Filtrar registros con edad válida (>17)

    return df

def test_esquema(datos_banco):
    """Test de esquema para el DataFrame de datos_banco.

    Args:
        datos_banco (pd.DataFrame): DataFrame que contiene los datos del banco.
    """
    df = datos_banco
    esquema = DataFrameSchema({
        "age": Column(int, Check.ge(18), nullable=False),
    "job": Column(str, nullable=False),
    "marital": Column(str, nullable=False),
    "education": Column(str, nullable=False),
    "default": Column(str, nullable=True),
    "housing": Column(str, nullable=False),
    "loan": Column(str, nullable=False),
    "contact": Column(str, nullable=False),
    "month": Column(str, nullable=False),
    "day_of_week": Column(str, nullable=False),
    "duration": Column(int, Check.ge(0), nullable=False),
    "campaign": Column(int, Check.ge(1), nullable=False),
    "pdays": Column(int, Check.ge(-1), nullable=False),
    "previous": Column(int, Check.ge(0), nullable=False),
    "poutcome": Column(str, nullable=False),
    "emp.var.rate": Column(float, nullable=False),
    "cons.price.idx": Column(float, nullable=False),
    "cons.conf.idx": Column(float, nullable=False),
    "euribor3m": Column(float, nullable=False),
    "nr.employed": Column(float, nullable=False),
    "y": Column(str, Check.isin(["yes", "no"]), nullable=False)
    })

    esquema.validate(df)


def test_basico(datos_banco):
    """Test básico para verificar que el DataFrame de datos_banco no está vacío
    y contiene las columnas esperadas.

    Args:
        datos_banco (pd.DataFrame): DataFrame que contiene los datos del banco.
    """
    df = datos_banco
    # Verificar que el DataFrame no está vacío
    assert not df.empty, "El DataFrame está vacío." 
    # Verificar nulos
    assert df.isnull().sum().sum() == 0, "El DataFrame contiene valores nulos."
    # Verificar cantidad de columnas
    assert df.shape[1] == 21, f"El DataFrame debería tener 21 columnas, pero tiene {df.shape[1]}."

    def test_validaciones_extra(datos_banco):
     """Validaciones adicionales sobre el DataFrame de datos_banco."""

    df = datos_banco

    # Verificar que no existan filas duplicadas
    assert df.duplicated().sum() == 0, "El DataFrame contiene filas duplicadas."

    # Verificar que todas las edades sean mayores a 17 años
    assert (df["age"] > 17).all(), "Existen registros con edad menor o igual a 17 años."