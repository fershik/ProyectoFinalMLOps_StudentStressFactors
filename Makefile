.PHONY: install train test lint clean help

help:
	@echo "Comandos disponibles:"
	@echo "  make install    - Instalar dependencias"
	@echo "  make train      - Entrenar el modelo"
	@echo "  make test       - Ejecutar pruebas"
	@echo "  make lint       - Verificar estilo de código"
	@echo "  make clean      - Limpiar archivos generados"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503

clean:
	rm -rf mlruns/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete