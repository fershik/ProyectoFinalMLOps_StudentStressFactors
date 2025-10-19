import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='src/config.yaml'):
    """Cargar configuración desde archivo YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_and_preprocess_data(config):
    """Cargar y preprocesar los datos"""
    logger.info("Cargando datos...")
    
    # Leer CSV
    df = pd.read_csv(config['data']['path'])
    
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Renombrar columnas directamente por índice (evita problemas con emojis)
    df.columns = [
        'sleep_quality',
        'headaches_frequency',
        'academic_performance',
        'study_load',
        'extracurricular_freq',
        'stress_level'
    ]
    
    logger.info("Columnas renombradas exitosamente")
    logger.info(f"Nuevas columnas: {list(df.columns)}")
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Valores nulos encontrados:\n{null_counts[null_counts > 0]}")
        df = df.dropna()
        logger.info(f"Filas después de eliminar nulos: {len(df)}")
    else:
        logger.info("No se encontraron valores nulos")
    
    # Separar features y target
    X = df.drop(columns=['stress_level'])
    y = df['stress_level']
    
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target: stress_level")
    logger.info(f"Distribución de clases:\n{y.value_counts().sort_index()}")
    
    return X, y, df


def split_and_scale_data(X, y, config):
    """Dividir datos en train/test y escalar"""
    logger.info("Dividiendo datos en train/test...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} muestras")
    logger.info(f"Test set: {X_test.shape[0]} muestras")
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Features escaladas correctamente")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, config):
    """Entrenar el modelo"""
    logger.info("Entrenando modelo...")
    
    model_params = config['model']['params']
    logger.info(f"Parámetros del modelo: {model_params}")
    
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    logger.info("Modelo entrenado exitosamente")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluar el modelo"""
    logger.info("Evaluando modelo...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score (weighted): {f1:.4f}")
    
    logger.info("\nReporte de clasificación:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    return accuracy, f1, y_pred


def log_mlflow(model, scaler, X_train, X_test, y_test, y_pred, config, accuracy, f1):
    """Registrar experimento en MLflow"""
    logger.info("Registrando en MLflow...")
    
    # Configurar MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=config['mlflow']['run_name']):
        # Log parámetros
        mlflow.log_params(config['model']['params'])
        mlflow.log_param("test_size", config['data']['test_size'])
        mlflow.log_param("random_state", config['data']['random_state'])
        
        # Log métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", f1)
        
        # Log modelo con signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_test[:5]
        )
        
        # Log scaler como artefacto adicional
        import joblib
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            joblib.dump(scaler, f)
            mlflow.log_artifact(f.name, "preprocessor")
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
    
    logger.info("Experimento registrado exitosamente en MLflow")


def main():
    """Pipeline principal"""
    try:
        logger.info("="*70)
        logger.info("INICIANDO PIPELINE DE MACHINE LEARNING")
        logger.info("="*70)
        
        # Cargar configuración
        config = load_config()
        logger.info("Configuración cargada exitosamente")
        
        # Cargar y preprocesar datos
        X, y, df = load_and_preprocess_data(config)
        
        # Dividir y escalar
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, config)
        
        # Entrenar modelo
        model = train_model(X_train, y_train, config)
        
        # Evaluar modelo
        accuracy, f1, y_pred = evaluate_model(model, X_test, y_test)
        
        # Registrar en MLflow
        log_mlflow(model, scaler, X_train, X_test, y_test, y_pred, config, accuracy, f1)
        
        logger.info("="*70)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE!")
        logger.info("="*70)
        logger.info(f"Accuracy final: {accuracy:.4f}")
        logger.info(f"F1-Score final: {f1:.4f}")
        logger.info("Ejecuta 'mlflow ui' para ver los resultados en http://localhost:5000")
        
    except Exception as e:
        logger.error("="*70)
        logger.error("ERROR EN EL PIPELINE")
        logger.error("="*70)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()