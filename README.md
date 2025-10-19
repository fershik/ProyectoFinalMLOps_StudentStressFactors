# 🎓 Pipeline MLOps - Predicción de Estrés Estudiantil

Sistema automatizado de Machine Learning para predecir niveles de estrés en estudiantes basado en factores como calidad del sueño, carga académica y actividades extracurriculares.

## 🚀 Características

- ✅ Pipeline completo de ML con preprocesamiento, entrenamiento y evaluación
- ✅ Tracking de experimentos con MLflow
- ✅ Automatización CI/CD con GitHub Actions
- ✅ Tests automatizados
- ✅ Makefile para gestión de tareas
- ✅ Configuración modular con YAML

## 📊 Dataset

**Student Stress Factors 2.csv**
- 520 registros de estudiantes
- 5 variables predictoras:
  - Calidad del sueño (1-5)
  - Frecuencia de dolores de cabeza
  - Rendimiento académico (1-5)
  - Carga de estudio (1-5)
  - Frecuencia de actividades extracurriculares
- Variable objetivo: Nivel de estrés (1-5)

## 🛠️ Instalación

### Requisitos Previos
- Python 3.10+
- Git

### Pasos

1. **Clonar el repositorio**
```bash
git clone 
cd student-stress-ml
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
make install
```

## 🎯 Uso

### Entrenar el modelo localmente

```bash
make train
```

### Ejecutar tests

```bash
make test
```

### Verificar estilo de código

```bash
make lint
```

### Limpiar archivos generados

```bash
make clean
```

## 📈 MLflow

### Visualizar experimentos

```bash
mlflow ui
```

Luego abrir en el navegador: http://localhost:5000

### Estructura de tracking

- **Parámetros**: hiperparámetros del modelo
- **Métricas**: accuracy, F1-score
- **Artefactos**: modelo entrenado, scaler
- **Signature**: firma del modelo con tipos de entrada/salida

## 🔄 CI/CD con GitHub Actions

El pipeline se ejecuta automáticamente en cada push o pull request:

1. **Instalación**: Instala todas las dependencias
2. **Lint**: Verifica el estilo del código con flake8
3. **Tests**: Ejecuta pruebas unitarias
4. **Entrenamiento**: Entrena el modelo completo
5. **Artefactos**: Guarda el modelo entrenado

### Ver resultados

1. Ve a la pestaña **Actions** en tu repositorio
2. Selecciona el workflow más reciente
3. Descarga los artefactos del modelo

## 📁 Estructura del Proyecto

```
student-stress-ml/
├── .github/workflows/    # Configuración CI/CD
├── data/                 # Dataset
├── src/                  # Código fuente
│   ├── train.py         # Pipeline principal
│   └── config.yaml      # Configuración
├── tests/               # Pruebas unitarias
├── mlruns/             # Tracking de MLflow
├── Makefile            # Automatización de tareas
├── requirements.txt    # Dependencias
└── README.md          # Este archivo
```

## 🧪 Modelo

- **Algoritmo**: Random Forest Classifier
- **Hiperparámetros**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2

## 📊 Métricas de Evaluación

- **Accuracy**: Precisión global del modelo
- **F1-Score (weighted)**: Media ponderada del F1 por clase

## 🔧 Configuración

Modifica `src/config.yaml` para ajustar:
- Rutas de datos
- Hiperparámetros del modelo
- Configuración de MLflow
- Parámetros de train/test split

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👥 Autor

Tu Nombre - [Tu GitHub](https://github.com/tu-usuario)

## 🙏 Agradecimientos

- Dataset obtenido de fuentes abiertas
- MLflow para tracking de experimentos
- GitHub Actions para CI/CD
```

---

## 🚀 Pasos para Implementar

### 1. Crear el repositorio en GitHub

```bash
git init
git add .
git commit -m "Initial commit: ML pipeline with CI/CD"
git branch -M main
git remote add origin <tu-repo-url>
git push -u origin main
```

### 2. Colocar el dataset

Asegúrate de que `Student Stress Factors 2.csv` esté en la carpeta `data/`

### 3. Ejecutar localmente

```bash
make install
make test
make train
mlflow ui
```

### 4. Activar GitHub Actions

El workflow se ejecutará automáticamente en cada push. Puedes verlo en la pestaña **Actions** de tu repositorio.

---

## ✅ Checklist del Proyecto

- [x] Dataset externo (no sklearn.datasets)
- [x] Preprocesamiento completo
- [x] Entrenamiento con scikit-learn
- [x] Métricas de evaluación (accuracy + F1)
- [x] Tracking con MLflow
- [x] Registro de modelo como artefacto
- [x] Estructura organizada (src/)
- [x] config.yaml con hiperparámetros
- [x] Makefile con tareas automáticas
- [x] Tests con pytest
- [x] CI/CD con GitHub Actions
- [x] README completo
- [x] .gitignore apropiado

---

## 💡 Mejoras Opcionales

1. **Agregar más modelos**: XGBoost, LightGBM, CatBoost
2. **Hyperparameter tuning**: GridSearchCV o Optuna
3. **Feature engineering**: Crear nuevas características
4. **Deployment**: Crear API con FastAPI
5. **Docker**: Containerizar la aplicación
6. **DVC**: Versionado de datos
7. **Monitoring**: Detectar data drift

¡Tu proyecto está listo para ser implementado! 🎉
