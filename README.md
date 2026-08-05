# TechMind API 🧠

![TechMind Banner](https://img.shields.io/badge/Status-Active-brightgreen) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal) ![Machine Learning](https://img.shields.io/badge/AI-NLP-orange)

TechMind es una plataforma avanzada basada en **Procesamiento de Lenguaje Natural (NLP)** diseñada para analizar, clasificar y extraer metadatos de documentos y contenidos técnicos. Combina la eficiencia de **FastAPI** con un pipeline de IA robusto respaldado por **spaCy, BERTopic y SentenceTransformers**.

> **Disclaimer de Ciberseguridad**  
> This project is for educational and ethical cybersecurity purposes only.

---

## 🎯 Características Principales

- **Detección de Idioma Automatizada:** Asegura que solo se procese contenido soportado.
- **Extracción de Keywords Inteligente:** Utiliza KeyBERT para obtener las palabras clave más relevantes.
- **Clasificación por Tópicos (Topic Modeling):** Agrupa el texto en categorías pre-entrenadas usando BERTopic.
- **Arquitectura Modular (src/):** Código escalable con estándares de la industria, separando estrictamente las capas de Data Science y Backend.

---

## 🏗️ Arquitectura del Proyecto

El repositorio adopta el patrón de diseño `src/` (Source Layout) recomendado en Python:

```
TechMind/
├── src/                # Código fuente principal
│   └── app/
│       ├── api/        # Endpoints y Rutas (FastAPI)
│       ├── ml/         # Lógica de Inteligencia Artificial
│       └── services/   # Orquestación e Inferencia
├── data/               # Datasets crudos y preprocesados
├── models/             # Artefactos binarios de IA (.joblib)
├── scripts/            # Utilidades y automatización de entrenamiento
├── tests/              # Pruebas unitarias automatizadas (Mocks)
├── configs/            # Configuración de despliegue
└── legacy_java_backend/# [Legado] API original en Spring Boot
```

---

## 🚀 Instalación y Uso Local

Este proyecto puede ejecutarse localmente de forma independiente sin conexión externa a bases de datos en la nube.

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/TechMind.git
cd TechMind
```

### 2. Configurar entorno virtual
```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Generar Modelos de IA
*Si descargas el repo por primera vez, los archivos `.joblib` en `models/` no existirán debido a que pesan gigabytes y están protegidos por el `.gitignore`.*

Para entrenar y generarlos localmente usando el dataset integrado:
```bash
python scripts/extracted_notebook.py
```

### 5. Iniciar la API
```bash
# Exportar PYTHONPATH para que Python detecte la carpeta src/
# En Windows (Powershell):
$env:PYTHONPATH="src"
# En Linux/macOS:
export PYTHONPATH="src"

python -m uvicorn app.main:app --reload
```

La API estará disponible en `http://localhost:8000/docs` (Swagger UI).

---

## 🧪 Pruebas Automatizadas (Tests)

Las pruebas están diseñadas con **Mocks** (simulando los modelos de IA), lo que permite que se ejecuten en milisegundos sin requerir los archivos `.joblib`.

```bash
$env:PYTHONPATH="src"  # O export PYTHONPATH="src"
pytest tests/ -v
```

---

## 🤝 Cómo Contribuir

¡Las contribuciones son bienvenidas! Sigue este flujo de trabajo:

1. Realiza un **Fork** de este repositorio.
2. Crea una rama para tu característica: `git checkout -b feature/nueva-caracteristica`
3. Haz tus cambios y escribe pruebas si es necesario.
4. Confirma tus cambios usando *Conventional Commits*: `git commit -m "feat: agrega nueva clasificación"`
5. Sube los cambios a tu Fork: `git push origin feature/nueva-caracteristica`
6. Abre un **Pull Request (PR)** hacia la rama `main`. El pipeline de CI/CD ejecutará automáticamente las pruebas.

---
*Mantenido por el equipo de TechMind.*
