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
├── datasets/           # Datasets crudos y preprocesados
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

Los artefactos `.joblib` no se versionan porque pesan cientos de MB. Para
generarlos localmente:

```powershell
python scripts/build_fallback.py      # corpus de respaldo, solo la primera vez
python scripts/entrenar.py --offline  # entrena y serializa en models/
```

El modo `--offline` usa un corpus curado de 80 documentos en 8 categorías,
sin depender de Wikipedia. Es rápido y reproducible.

Para construir el corpus real scrapeando las 98 fuentes de
`configs/semillas_documentacion.csv`:

### 5. Iniciar la API
```
Antes de iniciar el servidor, es necesario configurar las variables de entorno básicas para habilitar las peticiones CORS y definir el comportamiento de la carga de modelos:

**En Windows (PowerShell):**
```powershell
$env:PYTHONPATH="src"
$env:http://localhost:5173
if os.getenv("PRECARGAR_MODELOS", "1") == "1":

python -m uvicorn app.main:app --reload
```
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
