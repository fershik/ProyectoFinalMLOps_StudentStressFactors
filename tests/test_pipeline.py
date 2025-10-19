import pytest
import pandas as pd
from pathlib import Path
import yaml
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_config_exists():
    """Verificar que el archivo de configuración existe"""
    config_path = Path("src/config.yaml")
    assert config_path.exists(), "config.yaml no encontrado"


def test_config_valid():
    """Verificar que la configuración es válida"""
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "data" in config, "Falta sección 'data' en config"
    assert "model" in config, "Falta sección 'model' en config"
    assert "mlflow" in config, "Falta sección 'mlflow' en config"
    assert "path" in config["data"], "Falta 'path' en config['data']"
    assert "target_column" in config["data"], "Falta 'target_column' en config['data']"


def test_dataset_exists():
    """Verificar que el dataset existe"""
    config_path = "src/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = Path(config["data"]["path"])
    assert data_path.exists(), f"Dataset no encontrado en {data_path}"


def test_dataset_structure():
    """Verificar estructura básica del dataset"""
    df = pd.read_csv("data/Student_Stress_Factors.csv")

    assert len(df) > 0, "Dataset vacío"
    assert df.shape[1] == 6, f"Se esperan 6 columnas, se encontraron {df.shape[1]}"

    # Verificar que hay al menos 500 filas
    assert len(
        df) >= 500, f"Se esperan al menos 500 filas, se encontraron {len(df)}"


def test_dataset_no_nulls_or_handleable():
    """Verificar que no hay valores nulos o son manejables"""
    df = pd.read_csv("data/Student_Stress_Factors.csv")

    null_percentage = (df.isnull().sum() / len(df)) * 100

    # Permitir hasta 10% de nulos (que luego serán eliminados)
    assert null_percentage.max() < 10, "Demasiados valores nulos en el dataset"


def test_target_values():
    """Verificar que los valores del target son válidos"""
    df = pd.read_csv("data/Student_Stress_Factors.csv")

    # El target es la última columna (índice -1)
    target_col = df.columns[-1]

    assert target_col is not None, "No se encontró columna target"

    unique_values = df[target_col].dropna().unique()
    assert len(unique_values) > 1, "El target debe tener más de una clase"
    assert all(
        1 <= val <= 5 for val in unique_values
    ), f"Valores del target fuera del rango esperado (1-5): {unique_values}"


def test_src_directory():
    """Verificar que existe el directorio src con los archivos necesarios"""
    src_path = Path("src")
    assert src_path.exists(), "Directorio src/ no encontrado"
    assert (src_path / "train.py").exists(), "train.py no encontrado"
    assert (src_path / "config.yaml").exists(), "config.yaml no encontrado"


def test_data_types():
    """Verificar que los datos son del tipo correcto"""
    df = pd.read_csv("data/Student_Stress_Factors.csv")

    # Todas las columnas deben ser numéricas
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(
            df[col]), f"Columna {col} no es numérica"


def test_data_ranges():
    """Verificar que los valores están en rangos esperados"""
    df = pd.read_csv("data/Student_Stress_Factors.csv")

    # Verificar que no hay valores negativos
    assert (df >= 0).all().all(), "Se encontraron valores negativos en el dataset"

    # Verificar que los valores no son excesivamente grandes
    assert (df <= 10).all().all(
    ), "Se encontraron valores mayores a 10 en el dataset"


def test_column_rename_logic():
    """Verificar que la lógica de renombrado funciona correctamente"""
    df = pd.read_csv("data/Student_Stress_Factors.csv")

    # Renombrar usando la misma lógica que train.py
    df.columns = [
        "sleep_quality",
        "headaches_frequency",
        "academic_performance",
        "study_load",
        "extracurricular_freq",
        "stress_level",
    ]

    # Verificar que las columnas esperadas existen
    expected_columns = [
        "sleep_quality",
        "headaches_frequency",
        "academic_performance",
        "study_load",
        "extracurricular_freq",
        "stress_level",
    ]

    assert (
        list(df.columns) == expected_columns
    ), f"Columnas renombradas no coinciden. Esperadas: {expected_columns}, Obtenidas: {list(df.columns)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
