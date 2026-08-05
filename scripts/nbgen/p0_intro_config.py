"""Sección 0 — Portada, objetivos, entorno, configuración, semillas y logging."""

from .core import md, code

# ============================== PORTADA ==============================
md(r'''
# TechMind — Notebook de Ciencia de Datos
### Organización Inteligente del Conocimiento Técnico

**Entregable:** Notebook del equipo de Ciencia de Datos (Google Colab / Jupyter)
**Stack:** Python · Pandas · spaCy · Scikit-Learn · Sentence-Transformers · KeyBERT · BERTopic · ChromaDB
**Referencia de arquitectura:** `Technology_Architecture.md`
**Referencia de pipeline:** diagrama *Data Ingestion and Processing*
**Versión del pipeline:** 2.0.0
''')

# ============================== INTRODUCCIÓN ==============================
md(r'''
---
## Introducción

Una organización técnica acumula conocimiento más rápido de lo que puede organizarlo. Artículos,
documentación interna, tutoriales y anotaciones de estudio se depositan en carpetas, wikis y chats
sin taxonomía común, hasta que encontrar algo depende de recordar dónde se guardó. El costo no es el
almacenamiento: es que el conocimiento existente se vuelve inaccesible y se reescribe.

**TechMind** ataca ese problema con Ciencia de Datos. El sistema recibe contenido técnico en texto
libre y devuelve, de forma automática, cuatro cosas que un humano tendría que producir a mano:

1. **Una categoría temática** con su probabilidad asociada — clasificación supervisada.
2. **Un conjunto de palabras clave** representativas — extracción híbrida semántica + estadística + reglas.
3. **Un tópico emergente** descubierto sin supervisión — el sistema encuentra estructura que la
   taxonomía impuesta no contempla.
4. **Contenido relacionado** — búsqueda por significado sobre un índice vectorial.

Este notebook es el entregable de Ciencia de Datos: construye el corpus, entrena y evalúa los
modelos, y produce los artefactos serializados que la API REST en FastAPI carga en producción.

## Objetivos del notebook

| # | Objetivo | Criterio de cumplimiento | Sección |
|---|---|---|---|
| 1 | Construir un corpus técnico propio a partir de fuentes públicas | ≥ 200 documentos etiquetados en ≥ 6 categorías | §2 |
| 2 | Garantizar que ningún documento inválido entre al pipeline | Validación explícita con errores tipificados y trazables | §2.1 |
| 3 | Procesar solo documentos en el idioma soportado | Detección automática de idioma, con arquitectura abierta a inglés | §2.2 |
| 4 | Transformar texto crudo en representaciones aptas para modelado | Matriz TF-IDF + matriz de embeddings densos de 384 dimensiones | §5.1 |
| 5 | Entrenar y **comparar** dos representaciones bajo el mismo clasificador | Léxica (TF-IDF) vs. semántica (SBERT), decisión por F1-macro | §5.2 |
| 6 | Evaluar con métricas apropiadas a un problema desbalanceado | Accuracy, precision, recall, F1 macro/weighted, matriz de confusión, CV 5-fold | §5.3 |
| 7 | Explicar por qué el modelo clasifica como clasifica | Explicabilidad global (coeficientes) y local (ablación por término) | §5.5 |
| 8 | Descubrir organización temática sin supervisión | BERTopic con etiquetas legibles, contrastado contra KMeans | §5.6 |
| 9 | Serializar artefactos versionados y reproducibles | `.joblib` + `metadata.json` con hiperparámetros, métricas y hash del dataset | §5.7 |
| 10 | Entregar al backend una capa de inferencia lista para importar | Clase `TechMindInference` sin estado global + `techmind_core.py` exportable | §7 |

## Alcance y no-alcance

**Dentro de alcance:** recolección, validación, preprocesamiento, entrenamiento, evaluación,
clustering, indexación vectorial, inferencia y serialización.

**Fuera de alcance (deliberadamente):** el servidor HTTP —que corresponde a `backend/` en FastAPI—,
el frontend, la orquestación con Docker y el despliegue en OCI más allá de la subida de artefactos a
Object Storage (§8.1). Este notebook produce *los modelos*, no *el servicio*.

## Cómo leer este notebook

Cada sección abre con una explicación en prosa de **qué problema resuelve** y **por qué se resuelve
así**, seguida del código. Las decisiones tecnológicas no se justifican aquí en detalle: viven en
`Technology_Architecture.md` y se resumen en §3. Las celdas de código llevan el prefijo de su
sección (`§4.3`) para que la referencia cruzada desde el backend sea inequívoca.
''')

# ============================== DIAGRAMA 1 — PIPELINE COMPLETO ==============================
md(r'''
---
## Diagrama 1 — Flujo completo del pipeline

Vista de extremo a extremo. Las cuatro etapas del diagrama *Data Ingestion and Processing* del brief
se corresponden una a una con las secciones §2, §3–§4, §5 y §6 de este notebook.

```mermaid
flowchart TD
    subgraph E1["Etapa 1 · Ingesta y Normalización (§2)"]
        A1["Fuentes públicas<br/>Wikipedia ES / corpus fallback"]
        A2["Validación de entrada<br/>título · texto · UTF-8 · longitud"]
        A3["Detección de idioma<br/>langdetect + heurística"]
        A4["Limpieza y normalización<br/>HTML · Unicode NFKC · ruido"]
        A5["Deduplicación<br/>exacta + near-duplicate"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph E2["Etapa 2 · Preprocesamiento NLP (§4)"]
        B1["Tokenización"]
        B2["Stopwords"]
        B3["Lematización"]
        B4["Filtrado POS<br/>NOUN · PROPN · ADJ"]
        B5["EntityRuler<br/>entidades técnicas"]
        B1 --> B2 --> B3 --> B4
        B1 --> B5
    end

    subgraph E3["Etapa 3 · Modelado (§5)"]
        C1["TF-IDF<br/>representación léxica"]
        C2["Embeddings SBERT<br/>representación semántica"]
        C3["KeyBERT + YAKE<br/>keywords candidatas"]
        C4["Ranking híbrido RRF<br/>informacion_adicional"]
        C5["Clasificador<br/>categoria + probabilidad"]
        C6["BERTopic / KMeans<br/>tópicos emergentes"]
        C1 --> C5
        C2 --> C5
        C2 --> C3 --> C4
        C1 --> C3
        B5 --> C4
        C2 --> C6
    end

    subgraph E4["Etapa 4 · Base de conocimiento y consumo (§6-§7)"]
        D1[("ChromaDB<br/>índice vectorial")]
        D2[("PostgreSQL<br/>metadatos · vía backend")]
        D3["Búsqueda semántica"]
        D4["Recomendación de relacionados"]
        D5["TechMindInference<br/>contrato JSON del brief"]
        D1 --> D3
        D1 --> D4
        D3 --> D5
        D4 --> D5
    end

    A5 --> B1
    C2 --> D1
    C5 --> D2
    C4 --> D2
    C6 --> D2
    D5 --> API(["POST /contenido<br/>FastAPI"])
```

> **Nota sobre Mermaid.** GitHub y JupyterLab renderizan estos bloques de forma nativa. Google Colab
> no lo hace: si estás en Colab verás el código fuente del diagrama. Ejecuta la celda §0.8 para
> renderizarlos localmente vía la API pública de mermaid.ink, o consulta el notebook desde GitHub.
''')

# ============================== 0. ENTORNO ==============================
md(r'''
---
# 0. Configuración del entorno

Esta sección deja el entorno en un estado conocido y reproducible antes de tocar un solo dato. Son
seis pasos y ninguno es opcional: instalar dependencias compatibles entre sí, verificar que la
instalación no rompió la ABI de numpy, centralizar todos los parámetros del pipeline en un único
objeto de configuración, fijar las semillas de todas las fuentes de aleatoriedad, activar el
registro de eventos y dejar constancia de las versiones exactas usadas.

> **Colab:** ejecuta §0.1 una sola vez (~3-5 min). Si Colab ofrece **RESTART SESSION**, acéptalo y
> continúa desde §0.2 sin repetir §0.1.
''')

code(r'''
# @title 0.1 — Instalación de dependencias
#
# REGLA CRÍTICA EN COLAB: no fijar ni reinstalar numpy, pandas, scikit-learn,
# matplotlib ni seaborn. Colab ya los trae, y el resto de sus paquetes vienen
# compilados contra ESA versión de numpy. Forzar otra versión rompe la ABI y
# produce el error:
#     ValueError: numpy.dtype size changed, may indicate binary incompatibility
# Solo instalamos lo que Colab NO trae, y dejamos que pip resuelva contra numpy 2.

INSTALAR = True  # ponlo en False si ya instalaste las dependencias

PAQUETES = [
    "spacy>=3.8",                  # 3.8+ es la primera serie compatible con numpy 2
    "sentence-transformers>=3.0",
    "keybert>=0.8",
    "bertopic>=0.16",
    "yake",
    "chromadb>=0.5",
    "beautifulsoup4",
    "langdetect>=1.0.9",           # detección de idioma (§2.3)
]

if INSTALAR:
    especificacion = " ".join(f'"{p}"' for p in PAQUETES)
    get_ipython().system(f"pip install -q -U {especificacion}")
    get_ipython().system("python -m spacy download es_core_news_sm")
    print("\n>>> Instalación completa.")
    print(">>> Si Colab ofrece 'RESTART SESSION', acéptalo y continúa desde §0.2 "
          "(NO vuelvas a ejecutar esta celda).")
''')

code(r'''
# @title 0.2 — Verificación de compatibilidad binaria (ejecuta esto tras reiniciar)
# Diagnostica el error de ABI de numpy antes de que aparezca a mitad del pipeline.
try:
    import numpy, pandas, sklearn
    numpy.zeros(3).mean()
    pandas.DataFrame({"x": [1, 2]}).sum()
    print("OK — stack numérico coherente")
    print(f"   numpy {numpy.__version__} | pandas {pandas.__version__} | sklearn {sklearn.__version__}")

    # El fallo de ABI no siempre aparece en numpy/pandas: suele saltar al
    # importar las librerías COMPILADAS que instala §0.1. Comprobarlas aquí
    # cuesta segundos y evita descubrir el problema a mitad del pipeline.
    print("\n   Comprobando el stack pesado:")
    for _lib in ("spacy", "sentence_transformers", "keybert",
                 "bertopic", "chromadb", "langdetect"):
        try:
            _m = __import__(_lib)
            print(f"     OK    {_lib:<22} {getattr(_m, '__version__', 'instalada')}")
        except Exception as _e:
            print(f"     FALLO {_lib:<22} {type(_e).__name__}: {str(_e)[:60]}")
            print(f"           -> reinstala con §0.1 y reinicia el entorno.")

    # Los conflictos que pip reporta sobre paquetes que este notebook NO usa
    # (google-adk, opentelemetry y demás preinstalados de Colab) son ruido:
    # lo que importa es que los imports de arriba funcionen.
except ValueError as e:
    if "dtype size changed" in str(e) or "binary incompatibility" in str(e):
        print("FALLO DE ABI DE NUMPY\n")
        print("Causa: alguna librería quedó compilada contra una versión de numpy")
        print("distinta a la instalada (típico al fijar numpy o pandas en Colab).\n")
        print("Solución:")
        print("  1. Entorno de ejecución -> Desconectar y eliminar entorno de ejecución")
        print("  2. Ejecuta SOLO §0.1 (que ya no toca numpy/pandas)")
        print("  3. Acepta el reinicio si Colab lo ofrece")
        print("  4. Continúa desde esta celda, sin repetir §0.1")
        raise
    raise
''')

code(r'''
# @title 0.3 — Imports
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import unicodedata
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import sqlite3

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 120)
sns.set_theme(style="whitegrid")

print("Imports listos.")
''')

# ============================== 0.4 CONFIGURACIÓN CENTRALIZADA ==============================
md(r'''
### 0.4 Configuración centralizada

**El problema que resuelve.** Un parámetro escrito directamente donde se usa —`min_df=2`,
`C=5.0`, `test_size=0.25`— es invisible: para saber con qué configuración se entrenó un modelo hay
que leer todo el notebook, y para cambiarla hay que editar N celdas y confiar en no haber olvidado
ninguna. Peor aún, el backend en FastAPI necesita *exactamente* los mismos valores en inferencia que
los usados en entrenamiento (el `top_k` de keywords, el modelo de embeddings, los umbrales), y no
tiene forma de conocerlos si viven dispersos en el código.

**La solución.** Un único objeto `CFG` de tipo `dataclass` que concentra todos los parámetros del
pipeline, agrupados por etapa. Tres propiedades lo hacen útil más allá del orden:

- **Serializable.** `asdict(CFG)` produce el `config.json` que se guarda junto a los modelos (§5.7),
  de modo que cada artefacto queda acompañado de la configuración exacta con que se produjo.
- **Tipado.** Los type hints permiten que el IDE autocomplete y que un valor mal escrito falle en la
  celda de configuración, no tres horas después en medio del entrenamiento.
- **Hasheable.** `CFG.huella()` produce un hash corto de la configuración, usado para invalidar la
  caché de embeddings (§5.1) cuando cambia el modelo o el preprocesamiento.

**Regla de trabajo del equipo:** ninguna celda posterior debe contener un literal numérico o de
cadena que gobierne el comportamiento del pipeline. Si aparece uno, va a `CFG`.
''')

code(r'''
# @title 0.4 — Configuración centralizada del pipeline
@dataclass(frozen=True)
class ConfigRutas:
    """Rutas del proyecto, espejo de Technology_Architecture.md §18."""
    base: Path = Path("data_science")

    @property
    def raw(self) -> Path:        return self.base / "datasets" / "raw"
    @property
    def processed(self) -> Path:  return self.base / "datasets" / "processed"
    @property
    def models(self) -> Path:     return self.base / "models"
    @property
    def cache(self) -> Path:      return self.base / "cache"
    @property
    def logs(self) -> Path:       return self.base / "logs"
    @property
    def chroma(self) -> Path:     return self.base / "chroma_db"

    def crear(self) -> None:
        """Crea todos los directorios del proyecto si no existen."""
        for d in (self.raw, self.processed, self.models,
                  self.cache, self.logs, self.chroma):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ConfigValidacion:
    """Umbrales de aceptación de un documento de entrada (§2.2)."""
    titulo_min_chars: int = 3
    titulo_max_chars: int = 300
    texto_min_chars: int = 20        # mínimo para inferencia (documento suelto)
    texto_max_chars: int = 50_000    # techo defensivo: evita DoS por payload gigante
    corpus_min_chars: int = 250      # mínimo más estricto para entrar al corpus de entrenamiento
    max_ratio_no_alfabetico: float = 0.45   # por encima => probable binario o tabla
    max_ratio_mayusculas: float = 0.80      # por encima => probable grito / encabezado degenerado
    shingle_n: int = 3                      # ventana de palabras del shingling (calibrado en §2.4)
    umbral_near_duplicate: float = 0.60     # Jaccard mínimo para considerar dos docs casi iguales


@dataclass(frozen=True)
class ConfigIdioma:
    """Detección de idioma y arquitectura multilenguaje (§2.3)."""
    idioma_objetivo: str = "es"
    idiomas_soportados: tuple = ("es",)                 # ampliar a ("es", "en") en v2.1
    confianza_minima: float = 0.70
    modelos_spacy: dict = field(default_factory=lambda: {
        "es": "es_core_news_sm",
        "en": "en_core_web_sm",      # requiere: python -m spacy download en_core_web_sm
    })
    rechazar_idioma_no_soportado: bool = True


@dataclass(frozen=True)
class ConfigCorpus:
    """Recolección del corpus desde fuentes públicas (§2.4)."""
    archivo_semillas: str = "semillas_documentacion.csv"
    extractor_por_defecto: str = "html"    # ver EXTRACTORES en §2.3.2
    respetar_robots: bool = True           # consulta robots.txt antes de cada dominio
    eliminar_bloques_codigo: bool = True   # imprescindible en documentación técnica
    min_chars_pagina: int = 400            # por debajo, probablemente sea una SPA sin SSR
    archivo_fallback: str = "corpus_fallback.csv"
    usar_fallback: bool = False
    fallback_automatico: bool = True   # si el corpus scrapeado resulta inservible (§2.4.3)
    min_categorias: int = 3            # umbral de corpus apto para clasificar
    min_documentos: int = 30
    api_wikipedia: str = "https://es.wikipedia.org/w/api.php"
    # Wikimedia EXIGE un User-Agent con datos de contacto; sin él puede bloquear
    # la IP sin aviso (meta.wikimedia.org/wiki/User-Agent_policy). Sustituye el
    # correo por uno real del equipo antes de ejecutar.
    contacto: str = "equipo-techmind@ejemplo.org"
    user_agent_plantilla: str = "TechMind/2.0 (hackathon; {contacto}) python-requests"
    timeout_http: int = 20
    pausa_entre_peticiones: float = 0.5    # 0.15 s era agresivo para 76 peticiones
    max_reintentos: int = 3
    backoff_base: float = 1.5              # espera = backoff_base ** intento
    intercalar_semillas: bool = True       # round-robin entre categorías
    max_docs_por_semilla: int = 6

    @property
    def user_agent(self) -> str:
        """User-Agent conforme a la política de Wikimedia."""
        return self.user_agent_plantilla.format(contacto=self.contacto)
    secciones_excluidas: tuple = (
        "véase también", "referencias", "bibliografía", "enlaces externos",
        "notas", "fuentes", "lecturas adicionales",
    )


@dataclass(frozen=True)
class ConfigNLP:
    """Preprocesamiento lingüístico con spaCy (§4)."""
    # Gobierna SIMULTÁNEAMENTE el texto de entrenamiento (§2.4.2) y el de
    # inferencia (§7.2). Tenerlo en un solo sitio hace imposible que ambos
    # diverjan: un desajuste ahí entrena el modelo sobre una distribución y
    # lo evalúa sobre otra. Ver la nota de diseño en §4.
    incluir_titulo_en_texto: bool = False
    # El EntityRuler y KeyBERT NO se entrenan: son reglas y similitud coseno.
    # Darles el título no introduce desajuste alguno y recupera tecnologías que
    # solo aparecen ahí ("Clasificación de texto con Scikit-Learn").
    incluir_titulo_en_entidades: bool = True
    pos_relevantes: tuple = ("NOUN", "PROPN", "ADJ")
    long_minima_token: int = 3
    batch_size_spacy: int = 32
    stopwords_extra: tuple = (
        # Verbos y sustantivos genéricos que sobreviven al filtro de spaCy
        "ser", "estar", "haber", "tener", "hacer", "poder", "deber", "ejemplo",
        "caso", "forma", "manera", "parte", "tipo", "vez", "año", "también",
        "así", "cual", "mismo", "otro", "cada", "sobre",
        # Términos que aparecen en LAS OCHO categorías y por tanto no discriminan.
        # Medido en §4.4: 'servicio', 'aplicación', 'sistema' y 'software' eran
        # el vocabulario compartido entre Cloud, DevOps y Backend, y explicaban
        # buena parte de sus confusiones. Un término presente en todas partes
        # aporta ruido al clasificador léxico y contamina las keywords.
        "servicio", "aplicación", "sistema", "software", "dato", "usuario",
        "herramienta", "recurso", "proceso", "elemento", "conjunto", "nivel",
    )
    etiquetas_entidad: tuple = ("TECH", "ORG", "PRODUCT")


@dataclass(frozen=True)
class ConfigEmbeddings:
    """Representación semántica densa (§5.1)."""
    modelo: str = "paraphrase-multilingual-MiniLM-L12-v2"
    dimension_esperada: int = 384
    batch_size: int = 32
    normalizar: bool = True          # normalizados => producto punto == similitud coseno
    usar_cache: bool = True
    archivo_cache: str = "embeddings_cache.joblib"


@dataclass(frozen=True)
class ConfigKeywords:
    """Extracción y ranking de palabras clave (§5.1)."""
    top_k: int = 5
    ngram_min: int = 1
    ngram_max: int = 3
    keybert_diversidad: float = 0.75     # MMR: 0 = relevancia pura, 1 = diversidad pura
    filtrar_keywords_por_pos: bool = True   # descarta candidatas sin sustantivos
    keybert_candidatos: int = 10         # se extraen 2x top_k y luego se fusiona
    yake_dedup_limite: float = 0.7
    rrf_k: int = 60                      # constante de amortiguación de Reciprocal Rank Fusion
    peso_keybert: float = 1.0
    peso_yake: float = 0.8
    peso_entidades: float = 2.5          # precisión perfecta por construcción: manda


@dataclass(frozen=True)
class ConfigTFIDF:
    """Representación léxica dispersa (§5.1)."""
    ngram_min: int = 1
    ngram_max: int = 3
    min_df: int = 2
    max_df: float = 0.85
    sublinear_tf: bool = True
    max_features: int = 8000


@dataclass(frozen=True)
class ConfigClasificacion:
    """Entrenamiento y evaluación del clasificador temático (§5.2-§5.4)."""
    test_size: float = 0.25
    cv_folds: int = 5
    max_iter: int = 2000
    C: float = 5.0
    class_weight: str = "balanced"
    metrica_decision: str = "f1_macro"   # criterio de selección entre modelo A y B
    # Con 8 clases el azar es 1/8 = 0.125, así que un 0.40 es un listón muy
    # alto heredado de la intuición binaria. Se fija en ~2x el azar.
    umbral_confianza_baja: float = 0.25  # por debajo, la API marca la predicción
    calibrar_probabilidades: bool = True   # CalibratedClassifierCV (§5.3.3)
    metodo_calibracion: str = "sigmoid"    # "sigmoid" (Platt) o "isotonic"
    n_terminos_explicabilidad: int = 8


@dataclass(frozen=True)
class ConfigClustering:
    """Descubrimiento de tópicos y agrupamiento (§5.6)."""
    k_min: int = 2
    k_max: int = 16
    kmeans_n_init: int = 10
    bertopic_min_topic_size_divisor: int = 60   # min_topic_size = max(3, n_docs // divisor)
    bertopic_min_topic_size_piso: int = 3
    bertopic_ngram_max: int = 2
    bertopic_min_df: int = 2
    umap_n_neighbors: int = 8        # menor = preserva estructura local, menos ruido
    asignar_outliers_con_kmeans: bool = True
    n_palabras_por_topico: int = 7


@dataclass(frozen=True)
class ConfigVectorial:
    """Base de datos vectorial ChromaDB (§6)."""
    nombre_coleccion: str = "techmind"
    metrica: str = "cosine"
    n_resultados_busqueda: int = 5
    n_relacionados: int = 3


@dataclass(frozen=True)
class ConfigPersistencia:
    """Persistencia relacional y auditoría (§6.6).

    SQLite es la base relacional **del notebook**, no de producción: en el
    despliegue, PostgreSQL ocupa este rol (Technology_Architecture.md §9). El
    esquema es deliberadamente el mismo para que la migración sea un cambio de
    cadena de conexión, no un rediseño.
    """
    archivo_sqlite: str = "techmind.db"
    archivo_historial: str = "historial_versiones.jsonl"
    persistir_predicciones: bool = True   # registra cada inferencia en predicciones_api


@dataclass(frozen=True)
class Config:
    """Configuración raíz del pipeline de TechMind.

    Agrupa toda la parametrización del proyecto. Se serializa a `config.json`
    junto con los artefactos del modelo (§5.7) para que el backend en FastAPI
    consuma exactamente los mismos valores usados en entrenamiento.

    Attributes:
        version: Versión semántica del pipeline, escrita en `metadata.json`.
        random_state: Semilla única propagada a todas las fuentes de aleatoriedad.
        nivel_log: Nivel del logger raíz de TechMind.

    Example:
        >>> CFG.keywords.top_k
        5
        >>> CFG.huella()[:8]
        'a3f19c02'
    """
    version: str = "2.0.0"
    random_state: int = 42
    nivel_log: str = "INFO"

    rutas: ConfigRutas = field(default_factory=ConfigRutas)
    validacion: ConfigValidacion = field(default_factory=ConfigValidacion)
    idioma: ConfigIdioma = field(default_factory=ConfigIdioma)
    corpus: ConfigCorpus = field(default_factory=ConfigCorpus)
    nlp: ConfigNLP = field(default_factory=ConfigNLP)
    embeddings: ConfigEmbeddings = field(default_factory=ConfigEmbeddings)
    keywords: ConfigKeywords = field(default_factory=ConfigKeywords)
    tfidf: ConfigTFIDF = field(default_factory=ConfigTFIDF)
    clasificacion: ConfigClasificacion = field(default_factory=ConfigClasificacion)
    clustering: ConfigClustering = field(default_factory=ConfigClustering)
    vectorial: ConfigVectorial = field(default_factory=ConfigVectorial)
    persistencia: ConfigPersistencia = field(default_factory=ConfigPersistencia)

    def a_dict(self) -> dict:
        """Devuelve la configuración como diccionario JSON-serializable."""
        crudo = asdict(self)
        crudo["rutas"] = {"base": str(self.rutas.base)}
        return crudo

    def huella(self) -> str:
        """Hash SHA-256 de la configuración completa.

        Se usa para invalidar la caché de embeddings cuando cambia cualquier
        parámetro que afecte a su cálculo.

        Returns:
            Hash hexadecimal de 64 caracteres.
        """
        payload = json.dumps(self.a_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def guardar(self, ruta: Path) -> Path:
        """Persiste la configuración en disco como JSON."""
        ruta.write_text(
            json.dumps(self.a_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ruta


CFG = Config()
CFG.rutas.crear()

print(f"Configuración v{CFG.version} cargada | huella: {CFG.huella()[:12]}")
print(f"Directorios creados bajo: {CFG.rutas.base.resolve()}")
print(f"\nParámetros clave:")
print(f"  embeddings        : {CFG.embeddings.modelo} ({CFG.embeddings.dimension_esperada} dims)")
print(f"  idioma objetivo   : {CFG.idioma.idioma_objetivo} (soportados: {CFG.idioma.idiomas_soportados})")
print(f"  keywords top-k    : {CFG.keywords.top_k}")
print(f"  texto válido      : {CFG.validacion.texto_min_chars}-{CFG.validacion.texto_max_chars:,} caracteres")
print(f"  split / CV        : test={CFG.clasificacion.test_size} / {CFG.clasificacion.cv_folds}-fold")
''')

# ============================== 0.5 REPRODUCIBILIDAD ==============================
md(r'''
### 0.5 Reproducibilidad

**El problema que resuelve.** Un pipeline que produce métricas distintas en cada ejecución no es
evaluable: no se puede saber si una mejora de 2 puntos de F1 viene del cambio que se hizo o del
azar. Y ante un jurado, un resultado que no se reproduce no es un resultado.

**Las fuentes de aleatoriedad de este pipeline** son cinco y hay que fijarlas todas; fijar solo
`numpy` —el error habitual— deja tres sin controlar:

| Fuente | Qué afecta aquí | Cómo se fija |
|---|---|---|
| `random` (stdlib) | Barajado interno de scikit-learn y de HDBSCAN | `random.seed()` |
| `numpy` | Inicialización de KMeans, splits, UMAP | `np.random.seed()` |
| `torch` | Forward pass de Sentence-Transformers (dropout desactivado, pero el orden de reducción en GPU no es determinista) | `torch.manual_seed()` + `use_deterministic_algorithms` |
| `PYTHONHASHSEED` | Orden de iteración de sets y dicts, que se propaga al orden de tópicos de BERTopic | Variable de entorno |
| `langdetect` | El algoritmo es estocástico por diseño (muestreo de n-gramas) | `DetectorFactory.seed = 0` |

`PYTHONHASHSEED` merece una advertencia: solo tiene efecto si se fija **antes** de arrancar el
intérprete. Dentro de un notebook ya en ejecución la asignación no cambia el hash de las cadenas de
esta sesión; la dejamos igualmente porque afecta a los subprocesos que se lancen y porque documenta
el requisito para quien ejecute el pipeline como script.
''')

code(r'''
# @title 0.5 — Fijado de semillas para reproducibilidad
def fijar_semillas(semilla: int = CFG.random_state, verboso: bool = True) -> dict:
    """Fija todas las fuentes de aleatoriedad del pipeline.

    Args:
        semilla: Valor de la semilla a propagar.
        verboso: Si es True, imprime el estado de cada fuente.

    Returns:
        Diccionario {fuente: estado} donde estado es "fijada", "no disponible"
        o un mensaje explicativo.

    Example:
        >>> fijar_semillas(42, verboso=False)["numpy"]
        'fijada'
    """
    estado: dict = {}

    # Comprobar ANTES de asignar: si no venía fijada al arrancar el kernel, el
    # hash de las cadenas de ESTA sesión ya está decidido y asignarla ahora no
    # lo cambia. Sin este aviso, la línea siguiente da una falsa sensación de
    # reproducibilidad total.
    _previa = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = str(semilla)
    if _previa == str(semilla):
        estado["PYTHONHASHSEED"] = "fijada antes de arrancar el kernel (efectiva)"
    else:
        estado["PYTHONHASHSEED"] = (
            f"asignada ahora (venía como {_previa!r}) — NO afecta a esta sesión")
        # Aquí todavía no existe `log`: el logger se configura en §0.6.
        if verboso:
            print(f"\n  AVISO: PYTHONHASHSEED no estaba fijada al arrancar el kernel.")
            print(f"  El orden de iteración de sets y dicts de esta sesión ya está")
            print(f"  decidido, así que el orden de los tópicos de BERTopic puede")
            print(f"  variar entre reinicios aunque el resto sí sea reproducible.")
            print(f"  Para reproducibilidad estricta, arranca el proceso con")
            print(f"  PYTHONHASHSEED={semilla} en el entorno.\n")

    random.seed(semilla)
    estado["random"] = "fijada"

    np.random.seed(semilla)
    estado["numpy"] = "fijada"

    try:
        import torch
        torch.manual_seed(semilla)
        torch.cuda.manual_seed_all(semilla)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        estado["torch"] = f"fijada (cuda disponible: {torch.cuda.is_available()})"
    except ImportError:
        estado["torch"] = "no disponible"

    try:
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        estado["langdetect"] = "fijada"
    except ImportError:
        estado["langdetect"] = "no disponible"

    # scikit-learn no tiene semilla global: se pasa random_state en cada estimador.
    # CFG.random_state es la única fuente de ese valor en todo el notebook.
    estado["scikit-learn"] = f"vía random_state={semilla} en cada estimador"

    if verboso:
        print(f"SEMILLAS FIJADAS (valor = {semilla})\n")
        for fuente, st in estado.items():
            print(f"  {fuente:<16} {st}")

    return estado


ESTADO_SEMILLAS = fijar_semillas()
''')

# ============================== 0.6 LOGGING ==============================
md(r'''
### 0.6 Registro de eventos (logging)

**El problema que resuelve.** `print()` sirve para mostrar un resultado, no para diagnosticar un
pipeline. Cuando el entrenamiento tarda catorce minutos y algo sale mal, se necesita saber qué etapa
consumió el tiempo, qué documentos se descartaron y por qué, y en qué punto exacto se lanzó la
excepción. Un `print` no lleva marca de tiempo, no distingue severidad, no se puede silenciar por
módulo y no queda escrito en ningún lado cuando la sesión de Colab se cierra.

**La solución.** Un logger `techmind` con dos destinos: la salida estándar del notebook (formato
compacto, legible durante la ejecución) y un archivo en `data_science/logs/pipeline.log` (formato
extendido con timestamp, nivel, módulo y línea) que sobrevive al cierre de la sesión y se puede
adjuntar como evidencia de la corrida.

Sobre eso se montan dos utilidades de instrumentación:

- **`@cronometrar`** — decorador que registra duración y resultado de una función. Se aplica a las
  operaciones costosas: scraping, preprocesamiento, cálculo de embeddings, entrenamiento.
- **`etapa()`** — context manager que delimita un bloque del pipeline, registra su duración y, si
  algo falla dentro, lo deja registrado como `ERROR` con traza antes de re-lanzar la excepción.

Los eventos que el brief pide registrar —inicio de entrenamiento, inicio de procesamiento, tiempo de
ejecución, errores, advertencias, generación de embeddings, clasificación y clustering— quedan
cubiertos por estas dos primitivas aplicadas en sus respectivas secciones.
''')

code(r'''
# @title 0.6 — Configuración del logger y utilidades de instrumentación
FORMATO_CONSOLA = "%(asctime)s │ %(levelname)-7s │ %(message)s"
FORMATO_ARCHIVO = "%(asctime)s │ %(levelname)-7s │ %(name)s:%(lineno)d │ %(message)s"


def configurar_logging(nivel: str = CFG.nivel_log,
                       archivo: Path | None = None) -> logging.Logger:
    """Configura el logger del pipeline con salida a consola y a archivo.

    Args:
        nivel: Nivel mínimo a registrar ("DEBUG", "INFO", "WARNING", "ERROR").
        archivo: Ruta del archivo de log. Por defecto `logs/pipeline.log`.

    Returns:
        El logger configurado, listo para usar.

    Example:
        >>> log = configurar_logging("INFO")
        >>> log.info("pipeline iniciado")
    """
    archivo = archivo or (CFG.rutas.logs / "pipeline.log")

    logger = logging.getLogger("techmind")
    logger.setLevel(getattr(logging, nivel.upper()))
    logger.handlers.clear()      # idempotente: re-ejecutar la celda no duplica salidas
    logger.propagate = False

    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(logging.Formatter(FORMATO_CONSOLA, datefmt="%H:%M:%S"))
    logger.addHandler(consola)

    disco = logging.FileHandler(archivo, mode="a", encoding="utf-8")
    disco.setLevel(logging.DEBUG)
    disco.setFormatter(logging.Formatter(FORMATO_ARCHIVO, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(disco)

    return logger


log = configurar_logging()


def cronometrar(etiqueta: str | None = None) -> Callable:
    """Decorador que registra la duración de una función en el log.

    Args:
        etiqueta: Nombre legible de la operación. Por defecto usa `func.__name__`.

    Returns:
        El decorador propiamente dicho.

    Example:
        >>> @cronometrar("cálculo de embeddings")
        ... def calcular(xs): return [x * 2 for x in xs]
        >>> _ = calcular([1, 2, 3])
    """
    def decorador(func: Callable) -> Callable:
        nombre = etiqueta or func.__name__

        @wraps(func)
        def envoltorio(*args, **kwargs):
            log.info(f"▶ inicio · {nombre}")
            t0 = time.perf_counter()
            try:
                resultado = func(*args, **kwargs)
            except Exception as exc:
                dt = time.perf_counter() - t0
                log.error(f"✖ fallo  · {nombre} tras {dt:.2f}s — {type(exc).__name__}: {exc}")
                raise
            dt = time.perf_counter() - t0
            log.info(f"✔ fin    · {nombre} en {dt:.2f}s")
            TIEMPOS.append({"operacion": nombre, "segundos": round(dt, 3)})
            return resultado

        return envoltorio
    return decorador


TIEMPOS: list = []   # acumulador para el reporte de rendimiento de §8.3


@contextmanager
def etapa(nombre: str) -> Iterator[None]:
    """Context manager que delimita y cronometra una etapa del pipeline.

    Args:
        nombre: Nombre de la etapa, tal como aparecerá en el log.

    Yields:
        None. El bloque se ejecuta dentro del contexto instrumentado.

    Example:
        >>> with etapa("limpieza de texto"):
        ...     resultado = 1 + 1
    """
    log.info(f"┌─ etapa · {nombre}")
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        log.exception(f"└─ ERROR en etapa '{nombre}': {type(exc).__name__}: {exc}")
        raise
    dt = time.perf_counter() - t0
    log.info(f"└─ etapa · {nombre} completada en {dt:.2f}s")
    TIEMPOS.append({"operacion": f"[etapa] {nombre}", "segundos": round(dt, 3)})


log.info(f"TechMind v{CFG.version} — logging activo")
log.info(f"Log persistente en: {CFG.rutas.logs / 'pipeline.log'}")
log.debug(f"Huella de configuración: {CFG.huella()}")
''')

# ============================== 0.7 VERSIONES ==============================
code(r'''
# @title 0.7 — Registro de versiones (exigido por el brief: "dependencias y versiones utilizadas")
def registrar_versiones() -> dict:
    """Captura la versión de cada dependencia relevante del pipeline.

    Returns:
        Diccionario {librería: versión}. Las librerías ausentes se marcan
        como "no disponible" en lugar de lanzar excepción.
    """
    import sklearn
    versiones = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    for lib in ("spacy", "sentence_transformers", "keybert", "bertopic",
                "yake", "chromadb", "langdetect", "torch", "umap", "hdbscan"):
        try:
            modulo = __import__(lib)
            versiones[lib] = getattr(modulo, "__version__", "instalada")
        except Exception:
            versiones[lib] = "no disponible"
    return versiones


VERSIONES = registrar_versiones()
log.info(f"Entorno: Python {VERSIONES['python']}, {len(VERSIONES)} dependencias registradas")
pd.DataFrame(VERSIONES.items(), columns=["librería", "versión"])
''')

# ============================== 0.8 RENDER MERMAID ==============================
md(r'''
### 0.8 Renderizado de diagramas (opcional, solo para Colab)

Colab no renderiza bloques Mermaid dentro de celdas markdown. Esta utilidad los envía a la API
pública de [mermaid.ink](https://mermaid.ink) y muestra la imagen resultante dentro del notebook.
Es puramente cosmética: si prefieres no depender de un servicio externo, deja `RENDERIZAR_MERMAID`
en `False` y lee los diagramas desde GitHub, que sí los renderiza de forma nativa.
''')

code(r'''
# @title 0.8 — Utilidad para renderizar Mermaid en Colab (opcional)
import base64
from IPython.display import Image, display, Markdown

RENDERIZAR_MERMAID = False   # ponlo en True si estás en Colab y quieres ver los diagramas


def mostrar_mermaid(codigo: str, titulo: str = "") -> None:
    """Renderiza un diagrama Mermaid vía mermaid.ink, con degradación elegante.

    Args:
        codigo: Cuerpo del diagrama en sintaxis Mermaid.
        titulo: Encabezado opcional a mostrar sobre el diagrama.

    Returns:
        None. Muestra la imagen (o el código fuente si falla la red).
    """
    if titulo:
        display(Markdown(f"**{titulo}**"))
    if not RENDERIZAR_MERMAID:
        display(Markdown(f"```mermaid\n{codigo.strip()}\n```"))
        return
    try:
        payload = base64.urlsafe_b64encode(codigo.strip().encode("utf-8")).decode("ascii")
        display(Image(url=f"https://mermaid.ink/img/{payload}"))
    except Exception as exc:
        log.warning(f"No se pudo renderizar el diagrama ({type(exc).__name__}); se muestra el código.")
        display(Markdown(f"```mermaid\n{codigo.strip()}\n```"))


def diagramas_del_notebook() -> list:
    """Extrae los bloques Mermaid escritos en las celdas markdown del notebook.

    Sin esto, `mostrar_mermaid` era una función muerta: los siete diagramas viven
    en celdas markdown y nadie la llamaba nunca, así que activar
    `RENDERIZAR_MERMAID` no producía ningún efecto visible.

    Returns:
        Lista de (título, código) por cada bloque ```mermaid encontrado.

    Example:
        >>> len(diagramas_del_notebook()) >= 1
        True
    """
    try:
        celdas = get_ipython().user_ns.get("In", [])
    except Exception:
        return []

    # Las celdas markdown no están en `In`; se leen del .ipynb si está a mano.
    for nombre in ("techmind_eda_modelado.ipynb",):
        ruta = Path(nombre)
        if not ruta.exists():
            continue
        contenido = json.loads(ruta.read_text(encoding="utf-8"))
        salida, titulo = [], ""
        for celda in contenido.get("cells", []):
            if celda.get("cell_type") != "markdown":
                continue
            texto = "".join(celda.get("source", []))
            for linea in texto.split("\n"):
                if linea.startswith("## Diagrama"):
                    titulo = linea.lstrip("# ").strip()
            for bloque in re.findall(r"```mermaid\n(.*?)```", texto, re.S):
                salida.append((titulo or "Diagrama", bloque))
        return salida
    return []


print("Utilidad de diagramas lista. RENDERIZAR_MERMAID =", RENDERIZAR_MERMAID)
if RENDERIZAR_MERMAID:
    _diagramas = diagramas_del_notebook()
    if _diagramas:
        print(f"Renderizando {len(_diagramas)} diagrama(s) vía mermaid.ink...\n")
        for _titulo, _codigo in _diagramas:
            mostrar_mermaid(_codigo, _titulo)
    else:
        print("No se encontró el .ipynb en el directorio actual; los diagramas se")
        print("siguen viendo como código en sus celdas markdown.")
else:
    print("Ponlo en True si estás en Colab y quieres ver los 7 diagramas renderizados.")
''')
