import pandas as pd

def test_great_expectations():
    """Test para verificar que los datos cumplen con las expectativas definidas
    en un archivo de Great Expectations.
    """

    # Cargar los datos
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=';')

    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}
    }

    def add_expectation(expectation_name, condition, message=""):
        results["statistics"]["total_count"] += 1
        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": expectation_name,
                "success": True
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })

    # 1. Edad entre 18 y 100 (tolerancia: al menos 99% válidas)
    add_expectation(
        "age_range",
        df["age"].between(18, 100).mean() > 0.99,
        "Más del 1% de las edades están fuera del rango esperado (18–100)."
    )

    # 2. Target con valores válidos
    add_expectation(
        "target_values",
        df["y"].isin(["yes", "no"]).all(),
        "La columna 'y' contiene valores no válidos."
    )

    # 3. Sin valores nulos en columnas críticas
    critical_cols = ["age", "job", "marital", "education", "y"]
    add_expectation(
        "no_nulls_in_critical_cols",
        df[critical_cols].isnull().sum().sum() == 0,
        "Existen valores nulos en columnas críticas."
    )

    # 4. Duración positiva (tolerancia: al menos 99% válidas)
    add_expectation(
        "duration_positive",
        (df["duration"] > 0).mean() > 0.99,
        "Más del 1% de los registros tienen duración de llamada menor o igual a 0."
    )

    # Resultado final
    success_rate = results["statistics"]["success_count"] / results["statistics"]["total_count"]
    print(f"Éxito: {results['success']} ({success_rate*100:.1f}% de expectativas cumplidas)")
    for exp in results["expectations"]:
        print(f"- {exp['expectation']}: {'OK' if exp['success'] else 'FALLA'}")

    assert results["success"], "Algunas expectativas no se cumplieron."
