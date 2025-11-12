"""
Script para entrenar un modelo de clasificación utilizando la técnica con mejor rendimiento
que fuera seleccionada durante la experimentación.
"""

# Importaciones generales
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import json

# Importaciones para el preprocesamiento y modelado
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample

# Importaciones para la evaluación - experimentación
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import argparse


def load_data(path):
    """Cargar y limpiar datos, detectando automáticamente la columna objetivo."""
    df = pd.read_csv(path)

    # Detectar columna objetivo
    target_col = None
    for col in df.columns:
        if col.lower() in ['y', 'target', 'deposit', 'subscribed']:
            target_col = col
            break
    if not target_col:
        raise ValueError("No se encontró una columna objetivo (y/target/deposit/subscribed).")

    # Limpiar columna objetivo
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    df = df[df[target_col].isin(['yes', 'no', '1', '0', 'true', 'false'])]

    # Mapear a binario
    df[target_col] = df[target_col].replace({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})

    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if len(df) == 0:
        raise ValueError(f"No se encontraron registros válidos en el dataset {path} después de limpiar la columna {target_col}.")

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def create_preprocessor(X_train):
    """Crea el preprocesador de datos."""
    numerical_columns = X_train.select_dtypes(exclude='object').columns
    categorical_columns = X_train.select_dtypes(include='object').columns

    X_train = X_train.copy()
    int_columns = X_train.select_dtypes(include='int').columns
    for col in int_columns:
        X_train[col] = X_train[col].astype('float')

    numerical_columns = X_train.select_dtypes(exclude='object').columns

    num_pipeline = Pipeline([('RobustScaler', RobustScaler())])
    # Allow the encoder to ignore unknown categories at transform time (robust for new/rare values)
    cat_pipeline = Pipeline([('OneHotEncoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])

    preprocessor_full = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_columns),
        ('cat_pipeline', cat_pipeline, categorical_columns)
    ]).set_output(transform='pandas')

    return preprocessor_full, X_train


def balance_data(X, y, random_state=42):
    """Balancea las clases mediante submuestreo."""
    train_data = X.copy()
    train_data['target'] = y.reset_index(drop=True)
    class_0 = train_data[train_data['target'] == 0]
    class_1 = train_data[train_data['target'] == 1]
    min_count = min(len(class_0), len(class_1))
    class_0_balanced = resample(class_0, n_samples=min_count, random_state=random_state)
    class_1_balanced = resample(class_1, n_samples=min_count, random_state=random_state)
    balanced_data = pd.concat([class_0_balanced, class_1_balanced])
    x_train_resampled = balanced_data.drop('target', axis=1)
    y_train_resampled = balanced_data['target']
    return x_train_resampled, y_train_resampled


def train_model(
    data_path: str = 'data/processed/bank-processed.csv',
    model_output_path: str = 'models/decision_tree_model.pkl',
    preprocessor_output_path: str = 'models/preprocessor.pkl',
    metrics_output_path: str = 'metrics/model_metrics.json'
):
    """Entrena el modelo de clasificación y guarda los artefactos."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Proyecto 13MBID-ABR2526 - Producción")

    with mlflow.start_run(run_name="DecisionTree_Production"):
        print("Cargando datos...")
        X_train, X_test, y_train, y_test = load_data(data_path)

        print("Creando preprocesador...")
        preprocessor, X_train_converted = create_preprocessor(X_train)
        X_test = X_test.copy()
        for col in X_test.select_dtypes(include=['int64', 'int32']).columns:
            X_test[col] = X_test[col].astype('float64')

        print("Preprocesando datos...")
        X_train_prep = preprocessor.fit_transform(X_train_converted)
        X_test_prep = preprocessor.transform(X_test)

        print("Balanceando datos...")
        X_train_balanced, y_train_balanced = balance_data(X_train_prep, y_train)

        print("\nEntrenando modelo Decision Tree...")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_balanced, y_train_balanced)

        print("Evaluando modelo...")
        y_pred = model.predict(X_test_prep)

        # Métricas
        metrics = {
            "f1_score": float(f1_score(y_test, y_pred)),
            "recall_score": float(recall_score(y_test, y_pred)),
            "precision_score": float(precision_score(y_test, y_pred)),
            "accuracy_score": float(accuracy_score(y_test, y_pred))
        }

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Registrar parámetros
        mlflow.log_params({
            "model_type": "DecisionTreeClassifier",
            "criterion": model.criterion,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "balancing_method": "undersampling",
            "train_samples": len(X_train_balanced),
            "test_samples": len(X_test),
            "random_state": 42
        })

        # Registrar métricas
        mlflow.log_metrics(metrics)

        # Matriz de confusión
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes']).plot(ax=ax)
        plt.title('Confusion Matrix - Production Model')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Firmas
        pipeline_signature = infer_signature(X_train, y_pred)
        preprocessor_signature = infer_signature(X_train, X_train_prep)
        model_signature = infer_signature(X_train_prep, y_pred)

        # Registrar modelos con nuevo parámetro `name`
        mlflow.sklearn.log_model(sk_model=Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ]), name="model", signature=pipeline_signature)

        mlflow.sklearn.log_model(sk_model=preprocessor, name="preprocessor", signature=preprocessor_signature)
        mlflow.sklearn.log_model(sk_model=model, name="classifier", signature=model_signature)

        # Guardar artefactos locales
        print("\nGuardando modelos...")
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_output_path)
        joblib.dump(preprocessor, preprocessor_output_path)

        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    return model, preprocessor, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo de producción")
    parser.add_argument("--data-path", type=str, default="data/processed/bank-processed.csv",
                        help="Ruta al archivo de datos procesados")
    parser.add_argument("--model-output", type=str, default="models/decision_tree_model.pkl",
                        help="Ruta donde guardar el modelo")
    parser.add_argument("--preprocessor-output", type=str, default="models/preprocessor.pkl",
                        help="Ruta donde guardar el preprocesador")
    parser.add_argument("--metrics-output", type=str, default="metrics/model_metrics.json",
                        help="Ruta donde guardar las métricas")

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        model_output_path=args.model_output,
        preprocessor_output_path=args.preprocessor_output,
        metrics_output_path=args.metrics_output
    )
