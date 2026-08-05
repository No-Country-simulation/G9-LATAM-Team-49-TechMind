from __future__ import annotations
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
#     get_ipython().system(f"pip install -q -U {especificacion}")
#     get_ipython().system("python -m spacy download es_core_news_sm")
    print("\n>>> Instalación completa.")
    print(">>> Si Colab ofrece 'RESTART SESSION', acéptalo y continúa desde §0.2 "
          "(NO vuelvas a ejecutar esta celda).")

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

# @title 0.3 — Imports

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

# @title 0.4 — Configuración centralizada del pipeline
@dataclass(frozen=True)
class ConfigRutas:
    """Rutas del proyecto, espejo de Technology_Architecture.md §18."""
    base: Path = Path(".")

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
    archivo_semillas: str = "data/raw/semillas_documentacion.csv"
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

# @title 0.8 — Utilidad para renderizar Mermaid en Colab (opcional)
import base64
# from IPython.display import Image, display, Markdown

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
#         display(Markdown(f"**{titulo}**"))
        pass
    if not RENDERIZAR_MERMAID:
#         display(Markdown(f"```mermaid\n{codigo.strip()}\n```"))
        return
    try:
        payload = base64.urlsafe_b64encode(codigo.strip().encode("utf-8")).decode("ascii")
#         display(Image(url=f"https://mermaid.ink/img/{payload}"))
    except Exception as exc:
        log.warning(f"No se pudo renderizar el diagrama ({type(exc).__name__}); se muestra el código.")
#         display(Markdown(f"```mermaid\n{codigo.strip()}\n```"))


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
#         celdas = get_ipython().user_ns.get("In", [])
        pass
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

# @title 2.1 — Validador de entrada
class CodigoError:
    """Códigos de error de validación. El backend los mapea a respuestas HTTP 422."""
    CAMPO_FALTANTE       = "campo_faltante"
    TIPO_INVALIDO        = "tipo_invalido"
    CAMPO_VACIO          = "campo_vacio"
    CODIFICACION         = "codificacion_invalida"
    TEXTO_CORRUPTO       = "texto_corrupto"
    MUY_CORTO            = "longitud_insuficiente"
    MUY_LARGO            = "longitud_excesiva"
    CONTENIDO_DEGENERADO = "contenido_degenerado"
    IDIOMA_NO_SOPORTADO  = "idioma_no_soportado"


@dataclass
class ResultadoValidacion:
    """Resultado de validar un documento de entrada.

    Attributes:
        valido: True si no se detectó ningún error bloqueante.
        errores: Lista de (codigo, mensaje) que impiden procesar el documento.
        advertencias: Lista de (codigo, mensaje) que no impiden el procesamiento.
        titulo: Título normalizado (espacios colapsados, sin caracteres de control).
        texto: Texto normalizado.
        metricas: Estadísticos calculados durante la validación.

    Example:
        >>> r = validar_entrada("Spring Boot", "Framework de Java para APIs REST." * 3)
        >>> r.valido
        True
    """
    valido: bool = True
    errores: list = field(default_factory=list)
    advertencias: list = field(default_factory=list)
    titulo: str = ""
    texto: str = ""
    metricas: dict = field(default_factory=dict)

    def agregar_error(self, codigo: str, mensaje: str) -> None:
        """Registra un error bloqueante y marca el resultado como inválido."""
        self.errores.append((codigo, mensaje))
        self.valido = False

    def agregar_advertencia(self, codigo: str, mensaje: str) -> None:
        """Registra una advertencia no bloqueante."""
        self.advertencias.append((codigo, mensaje))

    def mensaje(self) -> str:
        """Devuelve todos los errores concatenados en una sola línea legible."""
        return " | ".join(f"[{c}] {m}" for c, m in self.errores) or "sin errores"


class ErrorValidacion(ValueError):
    """Excepción lanzada cuando se exige un documento válido y no lo es.

    Lleva adjunto el `ResultadoValidacion` completo para que el backend pueda
    construir una respuesta HTTP 422 detallada sin re-validar.
    """

    def __init__(self, resultado: ResultadoValidacion):
        self.resultado = resultado
        super().__init__(resultado.mensaje())


# --- Patrones de detección de corrupción -----------------------------------
# Mojibake: secuencias que aparecen al leer UTF-8 como Latin-1/CP1252.
RE_MOJIBAKE = re.compile(
    r"Ã[\x80-\xbf¡-ÿ]|â€[\x9c\x9d\x99\x93\x94]|Â[\xa0-\xbf]|ï»¿|Ã"
)
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CARACTER_REEMPLAZO = "�"


def _normalizar_espacios(texto: str) -> str:
    """Colapsa espacios en blanco consecutivos y recorta los extremos."""
    return re.sub(r"\s+", " ", texto).strip()


def _es_utf8_valido(texto: str) -> bool:
    """Verifica que la cadena sobreviva un round-trip UTF-8 sin surrogates sueltos."""
    try:
        texto.encode("utf-8").decode("utf-8")
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def _detectar_corrupcion(texto: str) -> list:
    """Devuelve la lista de síntomas de corrupción encontrados en el texto.

    Args:
        texto: Cadena a inspeccionar.

    Returns:
        Lista de descripciones legibles. Vacía si el texto está limpio.

    Example:
        >>> _detectar_corrupcion("configuraciÃ³n del servidor")
        ['secuencias mojibake (UTF-8 leído como Latin-1)']
    """
    sintomas: list = []
    if RE_MOJIBAKE.search(texto):
        sintomas.append("secuencias mojibake (UTF-8 leído como Latin-1)")
    if CARACTER_REEMPLAZO in texto:
        n = texto.count(CARACTER_REEMPLAZO)
        sintomas.append(f"{n} carácter(es) de reemplazo U+FFFD (pérdida previa de información)")
    controles = RE_CONTROL.findall(texto)
    if controles:
        sintomas.append(f"{len(controles)} carácter(es) de control no imprimibles")
    return sintomas


def validar_entrada(titulo: Any,
                    texto: Any,
                    modo: str = "inferencia",
                    cfg: Config = CFG) -> ResultadoValidacion:
    """Valida un documento de entrada contra el contrato de datos de TechMind.

    Aplica siete controles: presencia y tipo, codificación UTF-8, corrupción,
    longitud mínima, longitud máxima, ratio de caracteres alfabéticos y ratio
    de mayúsculas. No lanza excepciones: devuelve el diagnóstico completo.

    Args:
        titulo: Título del contenido. Obligatorio, debe ser `str`.
        texto: Cuerpo del contenido. Obligatorio, debe ser `str`.
        modo: "inferencia" usa `texto_min_chars`; "corpus" usa el umbral más
            estricto `corpus_min_chars`, aplicable al material de entrenamiento.
        cfg: Configuración con los umbrales. Por defecto la global `CFG`.

    Returns:
        Un `ResultadoValidacion` con el veredicto, los errores tipificados,
        las advertencias, los campos normalizados y las métricas calculadas.

    Example:
        >>> r = validar_entrada("API REST", "x")
        >>> r.valido, r.errores[0][0]
        (False, 'longitud_insuficiente')
    """
    v = cfg.validacion
    res = ResultadoValidacion()
    minimo = v.corpus_min_chars if modo == "corpus" else v.texto_min_chars

    # --- Control 1: presencia y tipo -------------------------------------
    for nombre, valor in (("titulo", titulo), ("texto", texto)):
        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            res.agregar_error(CodigoError.CAMPO_FALTANTE,
                              f"El campo '{nombre}' es obligatorio y no fue proporcionado.")
        elif not isinstance(valor, str):
            res.agregar_error(CodigoError.TIPO_INVALIDO,
                              f"El campo '{nombre}' debe ser una cadena de texto, "
                              f"se recibió {type(valor).__name__}.")
    if not res.valido:
        return res

    titulo_n = _normalizar_espacios(RE_CONTROL.sub(" ", titulo))
    texto_n = texto.strip()

    if not titulo_n:
        res.agregar_error(CodigoError.CAMPO_VACIO, "El campo 'titulo' está vacío.")
    if not texto_n:
        res.agregar_error(CodigoError.CAMPO_VACIO, "El campo 'texto' está vacío.")
    if not res.valido:
        return res

    # --- Control 2: codificación UTF-8 -----------------------------------
    for nombre, valor in (("titulo", titulo_n), ("texto", texto_n)):
        if not _es_utf8_valido(valor):
            res.agregar_error(CodigoError.CODIFICACION,
                              f"El campo '{nombre}' contiene surrogates sin emparejar "
                              f"y no es serializable a UTF-8.")
    if not res.valido:
        return res

    # --- Control 3: corrupción -------------------------------------------
    for nombre, valor in (("titulo", titulo_n), ("texto", texto_n)):
        sintomas = _detectar_corrupcion(valor)
        if sintomas:
            res.agregar_error(CodigoError.TEXTO_CORRUPTO,
                              f"El campo '{nombre}' presenta: {'; '.join(sintomas)}.")

    # --- Controles 4 y 5: longitudes -------------------------------------
    if len(titulo_n) < v.titulo_min_chars:
        res.agregar_error(CodigoError.MUY_CORTO,
                          f"El 'titulo' tiene {len(titulo_n)} caracteres; "
                          f"el mínimo es {v.titulo_min_chars}.")
    if len(titulo_n) > v.titulo_max_chars:
        res.agregar_error(CodigoError.MUY_LARGO,
                          f"El 'titulo' tiene {len(titulo_n)} caracteres; "
                          f"el máximo es {v.titulo_max_chars}.")
    if len(texto_n) < minimo:
        res.agregar_error(CodigoError.MUY_CORTO,
                          f"El 'texto' tiene {len(texto_n)} caracteres; el mínimo "
                          f"en modo '{modo}' es {minimo}.")
    if len(texto_n) > v.texto_max_chars:
        res.agregar_error(CodigoError.MUY_LARGO,
                          f"El 'texto' tiene {len(texto_n):,} caracteres; "
                          f"el máximo es {v.texto_max_chars:,}.")

    # --- Controles 6 y 7: forma del contenido ----------------------------
    n = len(texto_n)
    n_alfabeticos = sum(1 for c in texto_n if c.isalpha() or c.isspace())
    ratio_no_alfa = 1 - (n_alfabeticos / n) if n else 1.0

    letras = [c for c in texto_n if c.isalpha()]
    ratio_mayus = (sum(1 for c in letras if c.isupper()) / len(letras)) if letras else 0.0

    res.metricas = {
        "n_chars_titulo": len(titulo_n),
        "n_chars_texto": n,
        "n_palabras": len(texto_n.split()),
        "ratio_no_alfabetico": round(ratio_no_alfa, 4),
        "ratio_mayusculas": round(ratio_mayus, 4),
    }

    if ratio_no_alfa > v.max_ratio_no_alfabetico:
        res.agregar_error(CodigoError.CONTENIDO_DEGENERADO,
                          f"El {ratio_no_alfa:.0%} de los caracteres no son alfabéticos "
                          f"(máximo {v.max_ratio_no_alfabetico:.0%}): probable tabla, "
                          f"volcado de log o binario mal decodificado.")
    if ratio_mayus > v.max_ratio_mayusculas:
        res.agregar_advertencia(CodigoError.CONTENIDO_DEGENERADO,
                                f"El {ratio_mayus:.0%} de las letras están en mayúscula; "
                                f"la lematización puede degradarse.")

    res.titulo, res.texto = titulo_n, texto_n
    return res


def exigir_valido(titulo: Any, texto: Any, modo: str = "inferencia") -> ResultadoValidacion:
    """Valida y lanza `ErrorValidacion` si el documento no es válido.

    Es la variante que usa la capa de inferencia (§7), donde un documento
    inválido debe interrumpir el procesamiento.

    Args:
        titulo: Título del contenido.
        texto: Cuerpo del contenido.
        modo: "inferencia" o "corpus".

    Returns:
        El `ResultadoValidacion`, garantizado válido.

    Raises:
        ErrorValidacion: Si algún control falla.
    """
    res = validar_entrada(titulo, texto, modo=modo)
    if not res.valido:
        log.warning(f"Documento rechazado: {res.mensaje()}")
        raise ErrorValidacion(res)
    for codigo, mensaje in res.advertencias:
        log.warning(f"Advertencia [{codigo}]: {mensaje}")
    return res


log.info("Validador de entrada listo (7 controles activos)")

# @title 2.1b — Batería de casos de validación (evidencia de que los controles funcionan)
CASOS_VALIDACION = [
    ("caso válido",
     "Introducción a Spring Boot",
     "En este contenido se presentan los conceptos básicos para la creación de APIs REST utilizando Java y Spring Boot."),
    ("título ausente",            None, "Un texto perfectamente razonable sobre bases de datos relacionales y sus índices."),
    ("texto ausente",             "Título válido", None),
    ("título vacío",              "   ", "Un texto perfectamente razonable sobre bases de datos relacionales y sus índices."),
    ("texto vacío",               "Título válido", ""),
    ("tipo inválido en texto",    "Título válido", 12345),
    ("texto demasiado corto",     "Título válido", "muy corto"),
    ("texto demasiado largo",     "Título válido", "palabra " * 10_000),
    ("título demasiado largo",    "T" * 400, "Un texto perfectamente razonable sobre bases de datos relacionales."),
    ("mojibake",                  "ConfiguraciÃ³n", "La configuraciÃ³n del servidor se define en el archivo de propiedades Ã±ade."),
    ("carácter de reemplazo",     "Título válido", "El servidor devuelve un error � al procesar la solicitud entrante del cliente."),
    ("caracteres de control",     "Título válido", "Texto con bytes\x00binarios\x07incrustados en medio de la prosa del documento."),
    ("contenido degenerado",      "Log dump", "2024-01-01|##|>>>|{{{|@@@|%%%|&&&|***|+++|===|///|\\\\\\|~~~|^^^|[[[|]]]|"),
    ("mayúsculas excesivas",      "AVISO", "ESTE DOCUMENTO ESTA COMPLETAMENTE EN MAYUSCULAS Y NO APORTA SENAL LIMPIA."),
]

filas = []
for descripcion, t, x in CASOS_VALIDACION:
    r = validar_entrada(t, x)
    filas.append({
        "caso": descripcion,
        "válido": "sí" if r.valido else "no",
        "código": r.errores[0][0] if r.errores else ("—" if not r.advertencias else r.advertencias[0][0]),
        "detalle": (r.errores[0][1] if r.errores
                    else (r.advertencias[0][1] if r.advertencias else "aceptado sin observaciones"))[:88],
    })

log.info(f"Batería de validación ejecutada: {len(CASOS_VALIDACION)} casos")
# display(pd.DataFrame(filas))

# @title 2.2 — Detección de idioma y registro multilenguaje
@dataclass
class Idioma:
    """Resultado de la detección de idioma de un documento.

    Attributes:
        codigo: Código ISO 639-1 detectado ("es", "en", "und" si no se pudo determinar).
        confianza: Probabilidad asignada por el detector, en [0, 1].
        metodo: "langdetect", "heuristica_stopwords" o "no_disponible".
        soportado: True si el idioma está en `CFG.idioma.idiomas_soportados`.
    """
    codigo: str = "und"
    confianza: float = 0.0
    metodo: str = "no_disponible"
    soportado: bool = False

    def a_dict(self) -> dict:
        """Serializa el resultado para incluirlo en la respuesta JSON de la API."""
        return {"codigo": self.codigo, "confianza": round(self.confianza, 4),
                "metodo": self.metodo, "soportado": self.soportado}


# --- Nivel 2: stopwords por idioma para la heurística de respaldo ---------
_STOPWORDS_REFERENCIA = {
    "es": {"de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
           "un", "para", "con", "no", "una", "su", "al", "es", "lo", "como", "más",
           "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre"},
    "en": {"the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "as", "was",
           "with", "be", "by", "on", "not", "he", "this", "are", "or", "his", "from",
           "at", "which", "but", "have", "an", "they", "you", "were", "their", "one"},
}


def _detectar_por_stopwords(texto: str) -> Idioma:
    """Detecta idioma por proporción de stopwords conocidas (respaldo sin dependencias).

    Args:
        texto: Texto a analizar.

    Returns:
        Un `Idioma` con método "heuristica_stopwords".
    """
    tokens = re.findall(r"[a-záéíóúüñ]+", texto.lower())
    if len(tokens) < 5:
        return Idioma(metodo="heuristica_stopwords")

    ratios = {
        idioma: sum(1 for t in tokens if t in palabras) / len(tokens)
        for idioma, palabras in _STOPWORDS_REFERENCIA.items()
    }
    mejor = max(ratios, key=ratios.get)
    total = sum(ratios.values())
    confianza = ratios[mejor] / total if total > 0 else 0.0
    return Idioma(codigo=mejor if ratios[mejor] > 0.05 else "und",
                  confianza=round(confianza, 4),
                  metodo="heuristica_stopwords",
                  soportado=mejor in CFG.idioma.idiomas_soportados)


def detectar_idioma(texto: str, cfg: Config = CFG) -> Idioma:
    """Detecta el idioma de un texto con langdetect y respaldo heurístico.

    Args:
        texto: Texto a analizar. Debe haber pasado la validación de longitud.
        cfg: Configuración con los idiomas soportados y el umbral de confianza.

    Returns:
        Un `Idioma` con código ISO 639-1, confianza, método y si está soportado.

    Example:
        >>> detectar_idioma("Este es un texto técnico sobre bases de datos.").codigo
        'es'
    """
    try:
        from langdetect import detect_langs
        candidatos = detect_langs(texto)
        if candidatos:
            mejor = candidatos[0]
            codigo = mejor.lang.split("-")[0]
            return Idioma(codigo=codigo,
                          confianza=float(mejor.prob),
                          metodo="langdetect",
                          soportado=codigo in cfg.idioma.idiomas_soportados)
    except ImportError:
        log.debug("langdetect no instalado; usando heurística de stopwords.")
    except Exception as exc:
        log.debug(f"langdetect falló ({type(exc).__name__}); usando heurística.")

    return _detectar_por_stopwords(texto)


class RegistroIdiomas:
    """Registro de pipelines de spaCy por idioma, con carga perezosa.

    Permite que el sistema procese cada documento con el modelo lingüístico de
    su propio idioma. Los modelos se cargan la primera vez que se piden y se
    cachean, de modo que activar el inglés en el futuro no implique cargar dos
    modelos en memoria si nunca llega un documento en inglés.

    Example:
        >>> registro = RegistroIdiomas()
        >>> nlp_es = registro.obtener("es")
        >>> nlp_es.lang
        'es'
    """

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._cache: dict = {}

    def obtener(self, codigo_idioma: str):
        """Devuelve el pipeline de spaCy para un idioma, cargándolo si hace falta.

        Args:
            codigo_idioma: Código ISO 639-1 ("es", "en").

        Returns:
            El objeto `Language` de spaCy correspondiente.

        Raises:
            ValueError: Si el idioma no tiene modelo registrado en la configuración.
        """
        if codigo_idioma in self._cache:
            return self._cache[codigo_idioma]

        nombre_modelo = self.cfg.idioma.modelos_spacy.get(codigo_idioma)
        if nombre_modelo is None:
            raise ValueError(
                f"No hay modelo de spaCy registrado para el idioma '{codigo_idioma}'. "
                f"Registrados: {list(self.cfg.idioma.modelos_spacy)}"
            )

        import spacy
        try:
            modelo = spacy.load(nombre_modelo)
        except OSError:
            log.warning(f"Modelo '{nombre_modelo}' ausente; descargando...")
#             get_ipython().system(f"python -m spacy download {nombre_modelo}")
            modelo = spacy.load(nombre_modelo)

        log.info(f"Pipeline spaCy cargado para '{codigo_idioma}': {nombre_modelo}")
        self._cache[codigo_idioma] = modelo
        return modelo

    @property
    def cargados(self) -> list:
        """Lista de códigos de idioma actualmente en memoria."""
        return list(self._cache)


REGISTRO_IDIOMAS = RegistroIdiomas()

# --- Demostración sobre textos de control ---------------------------------
MUESTRAS_IDIOMA = [
    ("español", "Este contenido explica los conceptos básicos para construir APIs REST con Java y Spring Boot."),
    ("inglés",  "This content explains the basic concepts for building REST APIs using Java and Spring Boot."),
    ("portugués", "Este conteúdo explica os conceitos básicos para construir APIs REST usando Java e Spring Boot."),
    ("mixto",   "El framework Spring Boot provides autoconfiguration para simplificar el desarrollo backend."),
]

filas = []
for etiqueta, muestra in MUESTRAS_IDIOMA:
    d = detectar_idioma(muestra)
    filas.append({"muestra": etiqueta, "detectado": d.codigo,
                  "confianza": round(d.confianza, 3), "método": d.metodo,
                  "soportado": "sí" if d.soportado else "no"})
def run_eda_and_training():

    log.info(f"Idiomas soportados en esta versión: {CFG.idioma.idiomas_soportados}")
    # display(pd.DataFrame(filas))

    # @title 2.3.1 — Carga del archivo de semillas
    RUTA_SEMILLAS = Path(CFG.corpus.archivo_semillas)

    if not RUTA_SEMILLAS.exists():
        try:
            from google.colab import files
            print(f"Sube el archivo {RUTA_SEMILLAS.name}:")
            files.upload()
        except ImportError:
            raise FileNotFoundError(
                f"No se encontró {RUTA_SEMILLAS.name}. Colócalo junto al notebook."
            )

    semillas = pd.read_csv(RUTA_SEMILLAS)
    log.info(f"Semillas cargadas: {len(semillas)} artículos en "
             f"{semillas['categoria'].nunique()} categorías")

    # --- Validación del esquema ---------------------------------------------
    # Sin esto, una fila mal formada no falla al cargarse: falla tres minutos más
    # tarde, dentro del scraping, y el `except` genérico la contabiliza como error
    # de red. Ocurrió: 25 filas con fuente 'wikipedia' pero sin la columna
    # `titulo_wikipedia` se reportaron como 25 fallos de conectividad inexistentes.
    def validar_esquema_semillas(semillas: pd.DataFrame) -> list:
        """Comprueba que cada fila tiene lo que su extractor necesita.

        Args:
            semillas: DataFrame recién cargado.

        Returns:
            Lista de mensajes de problema. Vacía si el archivo es válido.
        """
        problemas = []
        if "categoria" not in semillas.columns:
            problemas.append("Falta la columna obligatoria 'categoria'.")
            return problemas

        tiene_url = "url" in semillas.columns
        tiene_titulo_wiki = "titulo_wikipedia" in semillas.columns
        if not (tiene_url or tiene_titulo_wiki):
            problemas.append("El archivo necesita una columna 'url' o 'titulo_wikipedia'.")
            return problemas

        for idx, fila in semillas.iterrows():
            fuente = str(fila.get("fuente", "") or "").strip().lower()
            url = str(fila.get("url", "") or "").strip()
            titulo_wiki = str(fila.get("titulo_wikipedia", "") or "").strip()
            if url.lower() == "nan":
                url = ""
            if titulo_wiki.lower() == "nan":
                titulo_wiki = ""

            if fuente == "wikipedia":
                if not titulo_wiki and "/wiki/" not in url:
                    problemas.append(
                        f"Fila {idx}: fuente 'wikipedia' sin 'titulo_wikipedia' ni URL "
                        f"de Wikipedia de la que deducirlo.")
            elif fuente in ("html", ""):
                if not url:
                    problemas.append(f"Fila {idx}: extractor HTML sin columna 'url'.")

        return problemas


    _problemas = validar_esquema_semillas(semillas)
    if _problemas:
        print("PROBLEMAS EN EL ARCHIVO DE SEMILLAS")
        print("=" * 68)
        for p in _problemas[:12]:
            print(f"  {p}")
            log.error(f"Esquema de semillas: {p}")
        if len(_problemas) > 12:
            print(f"  ... y {len(_problemas) - 12} más")
        print("\n  Corrígelos antes de scrapear: cada fila inválida se traduce en un")
        print("  documento menos, y el fallo aparece disfrazado de error de red.")
    else:
        print(f"Esquema válido: las {len(semillas)} semillas tienen lo que su extractor necesita.")

    # display(semillas.groupby("categoria").size().rename("artículos").to_frame())
    semillas.head()

    # @title 2.3.2 — Registro de extractores, robots.txt y limitación por dominio
    from urllib.parse import urlparse
    from urllib.robotparser import RobotFileParser

    from bs4 import BeautifulSoup

    HEADERS = {"User-Agent": CFG.corpus.user_agent}
    SESION = requests.Session()
    SESION.headers.update(HEADERS)

    # Contadores de diagnóstico del scraping, consultados en §2.3.4.
    ESTADO_SCRAPING = {"ok": 0, "vacios": 0, "inexistentes": 0, "bloqueados": 0,
                       "limitados": 0, "errores": 0, "robots": 0, "sin_ssr": 0,
                       "extractor": 0}

    def _con_jitter(segundos: float, cfg: Config = CFG) -> float:
        """Añade una perturbación aleatoria a un tiempo de espera.

        Sin jitter, varios reintentos que fallaron a la vez vuelven a golpear el
        servidor exactamente en el mismo instante. Con un solo cliente el efecto es
        menor, pero es gratis y es la práctica correcta. Usa el `random` ya sembrado
        en §0.5, así que sigue siendo reproducible.

        Args:
            segundos: Espera base calculada por el backoff.
            cfg: Configuración (no usada hoy; deja la puerta abierta a parametrizarlo).

        Returns:
            La espera con una perturbación de hasta el 30 % hacia arriba.
        """
        return segundos * (1.0 + random.random() * 0.3)


    _ROBOTS_CACHE: dict = {}
    _ULTIMA_PETICION: dict = {}


    def permitido_por_robots(url: str, cfg: Config = CFG) -> bool:
        """Comprueba si `robots.txt` del dominio autoriza a descargar esa URL.

        El veredicto se cachea por dominio: consultar robots.txt en cada petición
        duplicaría el tráfico. Si el archivo no existe o no se puede leer, se
        autoriza —es la interpretación estándar— pero se deja constancia en el log.

        Args:
            url: URL completa que se pretende descargar.
            cfg: Configuración; si `respetar_robots` es False, siempre autoriza.

        Returns:
            True si se puede descargar.

        Example:
            >>> permitido_por_robots("https://es.react.dev/learn")
            True
        """
        if not cfg.corpus.respetar_robots:
            return True

        dominio = urlparse(url).netloc
        if dominio not in _ROBOTS_CACHE:
            parser = RobotFileParser()
            parser.set_url(f"{urlparse(url).scheme}://{dominio}/robots.txt")
            try:
                parser.read()
                _ROBOTS_CACHE[dominio] = parser
                log.debug(f"robots.txt leído para {dominio}")
            except Exception as exc:
                log.warning(f"No se pudo leer robots.txt de {dominio} "
                            f"({type(exc).__name__}); se asume permitido.")
                _ROBOTS_CACHE[dominio] = None

        parser = _ROBOTS_CACHE[dominio]
        if parser is None:
            return True
        return parser.can_fetch(cfg.corpus.user_agent, url)


    def esperar_turno(url: str, cfg: Config = CFG) -> None:
        """Aplica la pausa entre peticiones **por dominio**, no global.

        Con una sola fuente da igual, pero al combinar varias sería absurdo penalizar
        a un dominio por peticiones hechas a otro.
        """
        dominio = urlparse(url).netloc
        ultima = _ULTIMA_PETICION.get(dominio, 0.0)
        espera = cfg.corpus.pausa_entre_peticiones - (time.time() - ultima)
        if espera > 0:
            time.sleep(espera)
        _ULTIMA_PETICION[dominio] = time.time()


    def _descargar(url: str, cfg: Config = CFG) -> str:
        """Descarga una URL con reintentos, backoff y control de robots.txt.

        Args:
            url: Dirección a descargar.
            cfg: Configuración de red.

        Returns:
            El HTML crudo, o cadena vacía si no se pudo obtener.
        """
        if not permitido_por_robots(url, cfg):
            ESTADO_SCRAPING["robots"] += 1
            log.warning(f"robots.txt prohíbe descargar {url}; se omite.")
            return ""

        for intento in range(cfg.corpus.max_reintentos):
            try:
                esperar_turno(url, cfg)
                r = SESION.get(url, timeout=cfg.corpus.timeout_http)

                if r.status_code == 403:
                    ESTADO_SCRAPING["bloqueados"] += 1
                    log.error(f"HTTP 403 en {url}. Revisa CFG.corpus.contacto.")
                    return ""
                if r.status_code == 404:
                    ESTADO_SCRAPING["inexistentes"] += 1
                    log.warning(f"HTTP 404: la URL ya no existe -> {url}")
                    return ""
                if r.status_code in (429, 503):
                    ESTADO_SCRAPING["limitados"] += 1
                    espera = _con_jitter(float(r.headers.get(
                        "Retry-After", cfg.corpus.backoff_base ** (intento + 1))))
                    log.warning(f"HTTP {r.status_code} en {url}; espera {espera:.1f}s.")
                    time.sleep(espera)
                    continue

                r.raise_for_status()
                r.encoding = r.apparent_encoding or "utf-8"
                return r.text

            except requests.exceptions.RequestException as exc:
                espera = _con_jitter(cfg.corpus.backoff_base ** (intento + 1))
                log.warning(f"Error de red en {url} ({type(exc).__name__}); "
                            f"reintento {intento + 1}/{cfg.corpus.max_reintentos}.")
                time.sleep(espera)

        ESTADO_SCRAPING["errores"] += 1
        return ""


    def extraer_texto_wikipedia(titulo: str, idioma: str = "es",
                                cfg: Config = CFG) -> str:
        """Descarga el texto plano de un artículo de Wikipedia, con reintentos.

        Implementa backoff exponencial y respeta la cabecera `Retry-After`. Es
        necesario porque Wikimedia limita la tasa de peticiones y **bloquea la IP
        sin aviso** si el User-Agent no incluye datos de contacto
        (meta.wikimedia.org/wiki/User-Agent_policy). Sin reintentos, un bloqueo
        temporal a mitad del recorrido se traduce en categorías enteras vacías.

        Args:
            titulo: Título exacto del artículo.
            idioma: Subdominio de idioma de Wikipedia.
            cfg: Configuración con timeout, reintentos y backoff.

        Returns:
            El texto plano del artículo, o cadena vacía si no existe o si todos
            los intentos fallan.

        Example:
            >>> texto = extraer_texto_wikipedia("Spring Framework")
            >>> isinstance(texto, str)
            True
        """
        url = f"https://{idioma}.wikipedia.org/w/api.php"
        params = {"action": "query", "format": "json", "prop": "extracts",
                  "explaintext": 1, "redirects": 1, "titles": titulo}

        for intento in range(cfg.corpus.max_reintentos):
            try:
                r = SESION.get(url, params=params, timeout=cfg.corpus.timeout_http)

                if r.status_code == 403:
                    ESTADO_SCRAPING["bloqueados"] += 1
                    log.error(
                        f"HTTP 403 al pedir '{titulo}'. Wikimedia bloquea peticiones cuyo "
                        f"User-Agent no identifica al responsable. Revisa "
                        f"CFG.corpus.contacto (ahora: '{cfg.corpus.contacto}').")
                    return ""

                if r.status_code in (429, 503):
                    ESTADO_SCRAPING["limitados"] += 1
                    espera = _con_jitter(float(r.headers.get(
                        "Retry-After", cfg.corpus.backoff_base ** (intento + 1))))
                    log.warning(f"HTTP {r.status_code} (límite de tasa) en '{titulo}'. "
                                f"Esperando {espera:.1f}s antes de reintentar "
                                f"({intento + 1}/{cfg.corpus.max_reintentos}).")
                    time.sleep(espera)
                    continue

                r.raise_for_status()
                paginas = r.json().get("query", {}).get("pages", {})
                for _, pagina in paginas.items():
                    if "missing" in pagina:
                        ESTADO_SCRAPING["inexistentes"] += 1
                        log.warning(f"Artículo inexistente en Wikipedia: '{titulo}'")
                        return ""
                    extracto = pagina.get("extract", "") or ""
                    if extracto:
                        ESTADO_SCRAPING["ok"] += 1
                    else:
                        ESTADO_SCRAPING["vacios"] += 1
                    return extracto

            except requests.exceptions.RequestException as exc:
                espera = _con_jitter(cfg.corpus.backoff_base ** (intento + 1))
                log.warning(f"Error de red en '{titulo}' ({type(exc).__name__}); "
                            f"reintento {intento + 1}/{cfg.corpus.max_reintentos} "
                            f"en {espera:.1f}s.")
                time.sleep(espera)

        ESTADO_SCRAPING["errores"] += 1
        log.error(f"'{titulo}' abandonado tras {cfg.corpus.max_reintentos} intentos.")
        return ""


    def _texto_de_celda(valor: Any) -> str:
        """Convierte una celda de CSV a cadena, tratando los nulos como vacío.

        Existe por un fallo real: pandas convierte las celdas vacías en `NaN`, y
        `float("nan")` es **truthy**. Un `str(fila.get("x", "") or "")` devuelve
        entonces la cadena `"nan"`, que parece un valor legítimo. En el extractor
        HTML eso se traducía en `select_one("nan")` y un aviso por cada fila.

        Args:
            valor: Contenido de la celda.

        Returns:
            El texto recortado, o cadena vacía si era nulo.

        Example:
            >>> _texto_de_celda(float("nan")), _texto_de_celda("  main  ")
            ('', 'main')
        """
        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            return ""
        return str(valor).strip()


    def extraer_html_generico(fila: pd.Series, cfg: Config = CFG) -> str:
        """Extrae el contenido principal de una página de documentación técnica.

        Localiza el contenedor principal por heurísticas —`<article>`, `<main>`,
        `[role=main]`, o el `<div>` con más texto— y lo recorre emitiendo los
        encabezados con la convención `== Título ==` que usa `partir_en_documentos`.

        Elimina navegación, pies y barras laterales. Con el código distingue dos
        casos: los **bloques** `<pre>` se eliminan enteros, pero el código **en
        línea** se desenvuelve conservando su texto, porque ahí es donde aparecen los
        nombres de tecnologías dentro de la prosa.

        Args:
            fila: Fila de semillas. Usa `url` y, opcionalmente, `selector` para
                forzar un contenedor CSS cuando las heurísticas fallen.
            cfg: Configuración de corpus.

        Returns:
            Texto plano con marcadores de sección, o cadena vacía si falla.

        Example:
            >>> extraer_html_generico(pd.Series({"url": "https://es.react.dev/learn"}))[:20]
            'Inicio rápido'
        """
        url = _texto_de_celda(fila.get("url"))
        if not url:
            log.warning("Fila de semillas sin columna 'url'; se omite.")
            return ""

        html = _descargar(url, cfg)
        if not html:
            return ""

        sopa = BeautifulSoup(html, "html.parser")

        # Ruido estructural: nunca es contenido.
        for etiqueta in sopa(["script", "style", "nav", "footer", "aside", "header",
                              "form", "button", "svg", "noscript", "iframe"]):
            etiqueta.decompose()

        # Código: hay que distinguir dos casos que parecen uno solo.
        #
        #  · BLOQUES (<pre>): ejemplos de varias líneas. Se eliminan enteros.
        #  · EN LÍNEA (<code> dentro de un párrafo): "usa el hook `useState` para...".
        #    Aquí se DESENVUELVE la etiqueta conservando el texto. Eliminarla borraría
        #    los nombres de tecnologías de la prosa —exactamente el vocabulario del que
        #    viven el clasificador léxico y el EntityRuler—, que es lo contrario de lo
        #    que se pretende.
        if cfg.corpus.eliminar_bloques_codigo:
            for etiqueta in sopa(["pre"]):
                etiqueta.decompose()
            for etiqueta in sopa(["code", "samp", "kbd", "var", "tt"]):
                etiqueta.unwrap()

        # Contenedor principal: selector explícito > semántica HTML > mayor densidad.
        # OJO con el `or ""`: una celda vacía del CSV llega como float('nan'), que es
        # TRUTHY, así que sobrevive al `or` y acaba en `select_one("nan")`. Hay que
        # comprobar el nulo explícitamente.
        selector = _texto_de_celda(fila.get("selector"))
        principal = None
        if selector:
            principal = sopa.select_one(selector)
            if principal is None:
                log.warning(f"El selector '{selector}' no encontró nada en {url}.")
        if principal is None:
            for busqueda in ("article", "main", "[role=main]", "div.content", "div.body"):
                principal = sopa.select_one(busqueda)
                if principal is not None:
                    break
        if principal is None:
            divs = sopa.find_all("div")
            principal = max(divs, key=lambda d: len(d.get_text(" ", strip=True)),
                            default=sopa)

        # Recorrido: encabezados a marcadores de sección, párrafos y listas a texto.
        partes = []
        for elemento in principal.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            texto = elemento.get_text(" ", strip=True)
            if not texto:
                continue
            if elemento.name.startswith("h"):
                partes.append(f"\n== {texto} ==\n")
            else:
                partes.append(texto)

        resultado = "\n".join(partes).strip()

        if len(resultado) < cfg.corpus.min_chars_pagina:
            ESTADO_SCRAPING["sin_ssr"] += 1
            log.warning(
                f"Solo {len(resultado)} caracteres útiles en {url}. Probablemente sea "
                f"una web renderizada en el cliente: `requests` no ejecuta JavaScript. "
                f"Esa fuente necesitaría un navegador headless.")
            return resultado

        ESTADO_SCRAPING["ok"] += 1
        return resultado


    def titulo_wikipedia_de_url(url: str) -> str:
        """Deduce el título de un artículo a partir de su URL de Wikipedia.

        Args:
            url: Dirección del tipo `https://es.wikipedia.org/wiki/Aprendizaje_autom%C3%A1tico`.

        Returns:
            El título decodificado y con espacios, o cadena vacía si no aplica.

        Example:
            >>> titulo_wikipedia_de_url("https://es.wikipedia.org/wiki/Base_de_datos")
            'Base de datos'
        """
        from urllib.parse import unquote
        if "/wiki/" not in url:
            return ""
        return unquote(url.split("/wiki/", 1)[1]).replace("_", " ").strip()


    def extraer_wikipedia_desde_fila(fila: pd.Series, cfg: Config = CFG) -> str:
        """Adaptador que conecta el extractor de Wikipedia con el registro.

        Acepta las dos formas de identificar el artículo: la columna
        `titulo_wikipedia` del esquema original, o una `url` de Wikipedia de la que
        se deduce el título. Sin esta segunda vía, un archivo de semillas que use
        `url` para todas las fuentes hace fallar todas las filas de Wikipedia con
        `KeyError` — que es exactamente lo que ocurrió y se contabilizó como 25
        errores de red que no lo eran.

        Args:
            fila: Fila de semillas.
            cfg: Configuración de corpus.

        Returns:
            El texto plano del artículo, o cadena vacía si no se pudo determinar.
        """
        titulo = _texto_de_celda(fila.get("titulo_wikipedia"))
        if not titulo:
            titulo = titulo_wikipedia_de_url(_texto_de_celda(fila.get("url")))
        if not titulo:
            log.error("Fila con fuente 'wikipedia' sin 'titulo_wikipedia' ni URL de "
                      "Wikipedia utilizable; se omite.")
            return ""
        return extraer_texto_wikipedia(titulo, _texto_de_celda(fila.get("idioma")) or "es", cfg)


    # --- Registro de extractores -------------------------------------------
    # Añadir una fuente = escribir una función (fila, cfg) -> str y registrarla aquí.
    EXTRACTORES = {
        "wikipedia": extraer_wikipedia_desde_fila,
        "html": extraer_html_generico,
    }


    def obtener_extractor(fila: pd.Series, cfg: Config = CFG):
        """Selecciona el extractor de una fila de semillas.

        La columna `fuente` manda; si no existe, se deduce del esquema: una fila con
        `titulo_wikipedia` usa Wikipedia y una con `url` usa el extractor HTML. Esa
        deducción mantiene compatible el archivo de semillas antiguo.

        Args:
            fila: Fila del DataFrame de semillas.
            cfg: Configuración con el extractor por defecto.

        Returns:
            La función extractora correspondiente.
        """
        nombre = _texto_de_celda(fila.get("fuente")).lower()
        if not nombre:
            nombre = ("wikipedia" if _texto_de_celda(fila.get("titulo_wikipedia"))
                      else cfg.corpus.extractor_por_defecto)
        if nombre not in EXTRACTORES:
            log.warning(f"Extractor '{nombre}' no registrado; se usa "
                        f"'{cfg.corpus.extractor_por_defecto}'. "
                        f"Registrados: {list(EXTRACTORES)}")
            nombre = cfg.corpus.extractor_por_defecto
        return EXTRACTORES[nombre]


    def titulo_de_fila(fila: pd.Series) -> str:
        """Título legible de una semilla, sea cual sea su esquema."""
        for columna in ("titulo", "titulo_wikipedia"):
            valor = _texto_de_celda(fila.get(columna))
            if valor:
                return valor
        url = _texto_de_celda(fila.get("url"))
        return url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ") or url


    def intercalar_por_categoria(semillas: pd.DataFrame) -> pd.DataFrame:
        """Reordena las semillas alternando categorías (round-robin).

        El archivo de semillas viene agrupado por categoría: diez de Backend, diez
        de Frontend, y así. Si el scraping se interrumpe a mitad —por límite de
        tasa, bloqueo o pérdida de red— ese orden hace que se pierdan **categorías
        enteras**, y el corpus resultante es inservible aunque tenga cientos de
        documentos.

        Intercalar convierte un fallo parcial en una degradación proporcional: con
        el 40 % de las peticiones exitosas se obtiene el 40 % de *cada* categoría,
        en vez del 100 % de tres y el 0 % de las otras cinco.

        Args:
            semillas: DataFrame con la columna `categoria`.

        Returns:
            El DataFrame reordenado en round-robin entre categorías.

        Example:
            >>> intercalar_por_categoria(semillas)["categoria"].head(3).tolist()
            ['Backend', 'Bases de Datos', 'Cloud']
        """
        s = semillas.copy()
        s["_turno"] = s.groupby("categoria").cumcount()
        return (s.sort_values(["_turno", "categoria"])
                 .drop(columns="_turno")
                 .reset_index(drop=True))


    def partir_en_documentos(texto: str, titulo: str, cfg: Config = CFG) -> list:
        """Trocea un artículo en párrafos-documento, descartando secciones no informativas.

        Args:
            texto: Texto plano completo del artículo.
            titulo: Título del artículo, usado como prefijo del título de cada documento.
            cfg: Configuración con el máximo de documentos y las secciones excluidas.

        Returns:
            Lista de diccionarios {titulo, texto}, uno por párrafo aceptado.
        """
        docs, seccion_actual = [], titulo
        for bloque in texto.split("\n"):
            bloque = bloque.strip()
            if not bloque:
                continue
            if bloque.startswith("=="):
                seccion_actual = bloque.strip("= ").strip()
                continue
            if seccion_actual.lower() in cfg.corpus.secciones_excluidas:
                continue
            if len(bloque) >= cfg.validacion.corpus_min_chars:
                docs.append({
                    "titulo": f"{titulo} — {seccion_actual}" if seccion_actual != titulo else titulo,
                    "texto": bloque,
                })
            if len(docs) >= cfg.corpus.max_docs_por_semilla:
                break
        return docs


    @cronometrar("recolección del corpus")
    def recolectar_corpus(semillas: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
        """Construye el corpus bruto desde Wikipedia o desde el archivo de respaldo.

        Args:
            semillas: DataFrame con columnas `titulo_wikipedia`, `categoria` e `idioma`.
            cfg: Configuración de corpus.

        Returns:
            DataFrame con columnas `titulo`, `texto`, `categoria` y `fuente`.

        Raises:
            RuntimeError: Si el scraping no devuelve ningún documento.
        """
        if cfg.corpus.usar_fallback:
            df = pd.read_csv(cfg.corpus.archivo_fallback)
            log.info(f"Corpus de respaldo cargado: {len(df)} documentos (sin red)")
            return df

        orden = intercalar_por_categoria(semillas) if cfg.corpus.intercalar_semillas else semillas
        if cfg.corpus.intercalar_semillas:
            log.info("Semillas intercaladas por categoría: un fallo parcial degradará "
                     "todas las categorías por igual en vez de vaciar unas pocas.")

        registros = []
        for i, fila in orden.iterrows():
            extractor = obtener_extractor(fila, cfg)
            titulo = titulo_de_fila(fila)
            try:
                bruto = extractor(fila, cfg)
            except Exception as exc:
                # Se cuenta aparte de los errores de red: un fallo aquí es casi
                # siempre una fila mal formada, no un problema de conectividad, y
                # confundirlos manda a depurar el sitio equivocado.
                ESTADO_SCRAPING["extractor"] += 1
                log.error(f"El extractor '{extractor.__name__}' falló en '{titulo}': "
                          f"{type(exc).__name__}: {exc}. Revisa el esquema de esa fila.")
                bruto = ""

            docs = partir_en_documentos(bruto, titulo, cfg)
            procedencia = (_texto_de_celda(fila.get("url"))
                           or "https://es.wikipedia.org/wiki/"
                           + _texto_de_celda(fila.get("titulo_wikipedia")).replace(" ", "_"))
            for d in docs:
                d["categoria"] = fila["categoria"]
                d["fuente"] = procedencia
                registros.append(d)
            log.debug(f"[{i+1}/{len(orden)}] {fila['categoria']:<16} "
                      f"{titulo[:40]} -> {len(docs)} docs")

        df = pd.DataFrame(registros)
        if len(df) == 0:
            raise RuntimeError(
                f"El scraping no devolvió documentos. Diagnóstico: {ESTADO_SCRAPING}.\n"
                f"Si hay bloqueos (403), revisa CFG.corpus.contacto. "
                f"Si no, pon CFG.corpus.usar_fallback = True para trabajar sin red."
            )
        log.info(f"Corpus bruto recolectado: {len(df)} documentos de {len(orden)} semillas")
        return df


    print(f"Extractores registrados: {list(EXTRACTORES)}")
    print(f"Extractor por defecto  : {CFG.corpus.extractor_por_defecto}")
    print(f"robots.txt             : {'se respeta' if CFG.corpus.respetar_robots else 'IGNORADO'}")
    print("\nDefiniciones listas. El sondeo (§2.3.3) va antes de la recolección (§2.3.4).")

    # @title 2.3.3 — Sondeo previo de las semillas (rápido, antes de scrapear)
    SONDEAR = True          # ponlo en False para saltarte esta comprobación
    SONDEAR_TODAS = False   # True = verifica todas las filas; False = una por categoría


    def sondear_semillas(semillas: pd.DataFrame, todas: bool = False,
                         cfg: Config = CFG) -> pd.DataFrame:
        """Comprueba accesibilidad, contenido e idioma de las semillas.

        Args:
            semillas: DataFrame de semillas.
            todas: Si es False, sondea solo la primera fila de cada categoría.
            cfg: Configuración de red.

        Returns:
            DataFrame con `categoria`, `titulo`, `chars`, `idioma` y `veredicto`.
        """
        muestra = semillas if todas else semillas.groupby("categoria").head(1)
        filas = []
        for _, fila in muestra.iterrows():
            extractor = obtener_extractor(fila, cfg)
            titulo = titulo_de_fila(fila)
            try:
                texto = extractor(fila, cfg)
            except Exception as exc:
                texto = ""
                log.warning(f"Sondeo fallido en '{titulo}': {type(exc).__name__}")

            idioma = detectar_idioma(texto, cfg).codigo if len(texto) > 50 else "—"
            if not texto:
                veredicto = "SIN CONTENIDO"
            elif len(texto) < cfg.corpus.min_chars_pagina:
                veredicto = "MUY POCO (¿SPA?)"
            elif idioma not in cfg.idioma.idiomas_soportados:
                veredicto = f"IDIOMA {idioma.upper()}"
            else:
                veredicto = "OK"

            filas.append({"categoria": fila["categoria"], "titulo": titulo[:38],
                          "chars": len(texto), "idioma": idioma, "veredicto": veredicto})
        return pd.DataFrame(filas)


    if SONDEAR:
        with etapa("sondeo de semillas"):
            sondeo = sondear_semillas(semillas, todas=SONDEAR_TODAS)

        print(f"SONDEO DE {len(sondeo)} SEMILLA(S)"
              f"{'' if SONDEAR_TODAS else ' — una por categoría'}")
        print("=" * 74)
    #     display(sondeo)

        problemas = sondeo[sondeo["veredicto"] != "OK"]
        if len(problemas):
            print(f"\n  {len(problemas)} semilla(s) con problemas:")
            for v, n in problemas["veredicto"].value_counts().items():
                print(f"    {v:<20} {n}")
            print("\n  Correcciones según el veredicto:")
            print("    SIN CONTENIDO    -> URL muerta o bloqueada: revísala en el navegador.")
            print("    MUY POCO (¿SPA?) -> la web necesita JavaScript; usa otra fuente")
            print("                        o añade la columna 'selector' con el contenedor real.")
            print("    IDIOMA XX        -> la traducción al español no existe para esa página;")
            print("                        busca el equivalente en /es/ o cambia de fuente.")
        else:
            print("\n  Todas las semillas sondeadas responden, tienen contenido y están en español.")
        # Las peticiones del sondeo no deben contarse como parte de la recolección.
        ESTADO_SCRAPING.update({k: 0 for k in ESTADO_SCRAPING})
    else:
        print("Sondeo omitido (SONDEAR = False).")

    # @title 2.3.4 — Recolección del corpus
    with etapa("ingesta de datos"):
        df_bruto = recolectar_corpus(semillas)

    print(f"\n>>> Corpus bruto: {len(df_bruto)} documentos, "
          f"{df_bruto['categoria'].nunique()} categorías")

    # @title 2.3.5 — Cobertura del scraping por categoría
    if not CFG.corpus.usar_fallback:
        esperadas = semillas["categoria"].value_counts()
        obtenidas = df_bruto["categoria"].value_counts()
        cobertura = pd.DataFrame({
            "semillas": esperadas,
            "documentos": obtenidas,
        }).fillna(0).astype(int)
        cobertura["docs_por_semilla"] = (cobertura["documentos"] /
                                         cobertura["semillas"]).round(2)
        cobertura = cobertura.sort_values("documentos")

        print("COBERTURA POR CATEGORÍA")
        print("=" * 62)
    #     display(cobertura)

        vacias = cobertura[cobertura["documentos"] == 0].index.tolist()
        total_peticiones = sum(ESTADO_SCRAPING.values())

        print("\nRESULTADO DE LAS PETICIONES")
        print("-" * 62)
        for clave, n in ESTADO_SCRAPING.items():
            if n:
                print(f"  {clave:<16} {n:>4}")
        if total_peticiones:
            tasa = ESTADO_SCRAPING["ok"] / total_peticiones
            print(f"  {'tasa de éxito':<16} {tasa:>7.1%}")

        if vacias:
            log.error(f"Categorías sin ningún documento: {vacias}")
            print(f"\n  ATENCIÓN: {len(vacias)} categoría(s) sin documentos: {vacias}")
            if ESTADO_SCRAPING["extractor"]:
                print(f"  Causa: {ESTADO_SCRAPING['extractor']} fila(s) hicieron fallar al")
                print("  extractor. NO es un problema de red: revisa el esquema del CSV")
                print("  (§2.3.1 lo valida al cargarlo) y el log para ver qué falta.")
            elif ESTADO_SCRAPING["bloqueados"]:
                print("  Causa: Wikimedia devolvió 403 (bloqueo por User-Agent).")
                print(f"  Corrección: pon un correo real en CFG.corpus.contacto (§0.4).")
            elif ESTADO_SCRAPING["limitados"]:
                print("  Causa: límite de tasa (HTTP 429/503).")
                print(f"  Corrección: sube CFG.corpus.pausa_entre_peticiones "
                      f"(ahora {CFG.corpus.pausa_entre_peticiones}s) y reejecuta §2.3.2.")
            elif ESTADO_SCRAPING["inexistentes"]:
                print("  Causa: títulos de artículo que ya no existen en Wikipedia ES.")
                print("  Corrección: revisa esas filas de semillas_wikipedia.csv.")
        elif len(cobertura) >= CFG.corpus.min_categorias:
            print(f"\n  Todas las categorías tienen documentos. Cobertura equilibrada.")
    else:
        print("Scraping omitido: CFG.corpus.usar_fallback = True")
        print(f"Corpus cargado desde {CFG.corpus.archivo_fallback}")

    # @title 2.4.1 — Funciones de limpieza y normalización
    from bs4 import BeautifulSoup

    RE_HTML       = re.compile(r"<[^>]+>")
    RE_URL        = re.compile(r"https?://\S+|www\.\S+")
    RE_REFS       = re.compile(r"\[\d+\]|\[cita requerida\]|\[nota \d+\]", re.IGNORECASE)
    RE_PARENTESIS = re.compile(r"\([^)]{0,3}\)")
    RE_ESPACIOS   = re.compile(r"\s+")
    RE_RUIDO      = re.compile(r"[^\wáéíóúüñÁÉÍÓÚÜÑ\s\.\,\;\:\-\+\#/]")


    def componer_entrada(titulo: str, texto: str, cfg: Config = CFG) -> str:
        """Compone el texto que ve el modelo, a partir del título y el cuerpo.

        **Única fuente de verdad** para esa composición. Existe porque el
        entrenamiento y la inferencia deben construir la entrada de forma idéntica:
        si el corpus se entrena solo con el cuerpo y la API antepone el título, el
        modelo predice sobre una distribución distinta de aquella con la que
        aprendió. Es un desajuste silencioso —no rompe nada— que desplaza las
        predicciones fronterizas.

        Centralizarlo en una función gobernada por `CFG.nlp.incluir_titulo_en_texto`
        hace que la divergencia sea imposible por construcción.

        Args:
            titulo: Título del contenido.
            texto: Cuerpo del contenido.
            cfg: Configuración que decide si el título se incorpora.

        Returns:
            El texto compuesto, sin limpiar todavía.

        Example:
            >>> componer_entrada("Spring Boot", "Framework de Java.")
            'Framework de Java.'
        """
        if cfg.nlp.incluir_titulo_en_texto and titulo:
            return f"{titulo}. {texto}"
        return texto


    def limpiar_texto(texto: Any) -> str:
        """Normaliza texto crudo a prosa plana apta para el pipeline de NLP.

        Aplica, en orden: manejo de nulos, remoción de HTML, URLs y referencias,
        normalización Unicode NFKC, remoción de paréntesis triviales y caracteres
        especiales, y colapso de espacios. No baja a minúsculas: eso ocurre en la
        lematización (§4.2), para no privar a spaCy de la señal de capitalización.

        Args:
            texto: Texto de entrada. Acepta None y NaN sin fallar.

        Returns:
            Texto normalizado. Cadena vacía si la entrada no era texto utilizable.

        Example:
            >>> limpiar_texto("<p>El <b>framework</b> Spring [1] — ver https://x.com</p>")
            'El framework Spring — ver'
        """
        if texto is None or not isinstance(texto, str):
            return ""
        if isinstance(texto, float) and pd.isna(texto):
            return ""

        texto = BeautifulSoup(texto, "html.parser").get_text(" ")
        texto = RE_HTML.sub(" ", texto)
        texto = RE_URL.sub(" ", texto)
        texto = RE_REFS.sub(" ", texto)
        texto = unicodedata.normalize("NFKC", texto)
        texto = RE_PARENTESIS.sub(" ", texto)
        texto = RE_RUIDO.sub(" ", texto)
        return RE_ESPACIOS.sub(" ", texto).strip()


    def hash_documento(texto: str) -> str:
        """Huella SHA-256 de un texto normalizado, usada para deduplicación exacta.

        Args:
            texto: Texto ya limpio.

        Returns:
            Hash hexadecimal de 64 caracteres, insensible a mayúsculas y espacios.
        """
        canonico = RE_ESPACIOS.sub(" ", texto.lower()).strip()
        return hashlib.sha256(canonico.encode("utf-8")).hexdigest()


    def _shingles(texto: str, n: int = None) -> set:
        """Conjunto de n-gramas de palabras de un texto, para similitud de Jaccard.

        Args:
            texto: Texto normalizado.
            n: Tamaño de la ventana. Por defecto, `CFG.validacion.shingle_n`.

        Returns:
            Conjunto de shingles. Si el texto es más corto que la ventana,
            devuelve el texto completo como shingle único.

        Example:
            >>> sorted(_shingles("a b c d", n=3))
            ['a b c', 'b c d']
        """
        n = n or CFG.validacion.shingle_n
        palabras = texto.lower().split()
        if len(palabras) < n:
            return {" ".join(palabras)}
        return {" ".join(palabras[i:i + n]) for i in range(len(palabras) - n + 1)}


    def jaccard(a: set, b: set) -> float:
        """Similitud de Jaccard entre dos conjuntos: |A∩B| / |A∪B|."""
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


    @cronometrar("deduplicación")
    def deduplicar(df: pd.DataFrame, columna: str = "texto_limpio",
                   cfg: Config = CFG) -> tuple:
        """Elimina duplicados exactos y near-duplicates del corpus.

        Args:
            df: DataFrame con la columna de texto ya limpio.
            columna: Nombre de la columna a comparar.
            cfg: Configuración con el umbral de Jaccard.

        Returns:
            Tupla (DataFrame sin duplicados, reporte con los conteos por tipo).

        Example:
            >>> limpio, reporte = deduplicar(df_raw)
            >>> reporte["duplicados_exactos"] >= 0
            True
        """
        n_inicial = len(df)

        df = df.copy()
        df["_hash"] = df[columna].map(hash_documento)
        df = df.drop_duplicates(subset="_hash", keep="first")
        n_exactos = n_inicial - len(df)

        # Near-duplicates: se conserva el documento más largo de cada par similar.
        df = df.sort_values(columna, key=lambda s: s.str.len(), ascending=False)
        shingles = [_shingles(t) for t in df[columna]]
        descartar: set = set()
        for i in range(len(df)):
            if i in descartar:
                continue
            for j in range(i + 1, len(df)):
                if j in descartar:
                    continue
                if jaccard(shingles[i], shingles[j]) > cfg.validacion.umbral_near_duplicate:
                    descartar.add(j)

        df = df.iloc[[i for i in range(len(df)) if i not in descartar]]
        df = df.drop(columns="_hash").reset_index(drop=True)

        reporte = {
            "documentos_iniciales": n_inicial,
            "duplicados_exactos": n_exactos,
            "near_duplicates": len(descartar),
            "documentos_finales": len(df),
        }
        log.info(f"Deduplicación: {n_inicial} → {len(df)} "
                 f"({n_exactos} exactos, {len(descartar)} near-duplicates)")
        return df, reporte


    log.info("Funciones de limpieza y deduplicación listas")

    # @title 2.4.2 — Aplicación del pipeline de ingesta: limpieza + validación + idioma + dedup
    @cronometrar("normalización y validación del corpus")
    def normalizar_corpus(df: pd.DataFrame, cfg: Config = CFG) -> tuple:
        """Aplica limpieza, validación en modo corpus y detección de idioma a todo el DataFrame.

        Args:
            df: Corpus bruto con columnas `titulo` y `texto`.
            cfg: Configuración del pipeline.

        Returns:
            Tupla (DataFrame de documentos aceptados, DataFrame de rechazos con su motivo).
        """
        aceptados, rechazados = [], []

        for _, fila in df.iterrows():
            titulo_limpio = limpiar_texto(fila["titulo"])
            # Misma composición que usará la inferencia (§7.2), vía `componer_entrada`.
            texto_limpio = limpiar_texto(
                componer_entrada(titulo_limpio, fila["texto"], cfg))

            res = validar_entrada(titulo_limpio, texto_limpio, modo="corpus", cfg=cfg)
            if not res.valido:
                rechazados.append({"titulo": str(fila["titulo"])[:70],
                                   "motivo": res.errores[0][0],
                                   "detalle": res.errores[0][1][:100]})
                continue

            idioma = detectar_idioma(res.texto, cfg=cfg)
            if cfg.idioma.rechazar_idioma_no_soportado and not idioma.soportado:
                rechazados.append({"titulo": str(fila["titulo"])[:70],
                                   "motivo": CodigoError.IDIOMA_NO_SOPORTADO,
                                   "detalle": f"idioma detectado: {idioma.codigo} "
                                              f"(confianza {idioma.confianza:.2f})"})
                continue
            if idioma.confianza < cfg.idioma.confianza_minima:
                log.debug(f"Confianza de idioma baja ({idioma.confianza:.2f}) "
                          f"en '{str(fila['titulo'])[:40]}'; se procesa igualmente.")

            registro = fila.to_dict()
            registro.update({
                "titulo": res.titulo,
                "texto_limpio": res.texto,
                "idioma": idioma.codigo,
                "idioma_confianza": round(idioma.confianza, 4),
                "idioma_metodo": idioma.metodo,
                "n_chars": res.metricas["n_chars_texto"],
                "n_palabras": res.metricas["n_palabras"],
                "ratio_no_alfabetico": res.metricas["ratio_no_alfabetico"],
            })
            aceptados.append(registro)

        df_ok = pd.DataFrame(aceptados)
        df_rech = pd.DataFrame(rechazados)
        log.info(f"Validación del corpus: {len(df_ok)} aceptados, {len(df_rech)} rechazados")
        if len(df_rech):
            for motivo, n in df_rech["motivo"].value_counts().items():
                log.warning(f"  rechazados por '{motivo}': {n}")
        return df_ok, df_rech


    with etapa("normalización y validación"):
        df_raw, df_rechazados = normalizar_corpus(df_bruto)
        df_raw, REPORTE_DEDUP = deduplicar(df_raw)
        df_raw["doc_id"] = [f"DOC-{i:04d}" for i in range(len(df_raw))]
        df_raw.to_csv(CFG.rutas.raw / "corpus_raw.csv", index=False)
        if len(df_rechazados):
            df_rechazados.to_csv(CFG.rutas.raw / "rechazos.csv", index=False)

    print("\nREPORTE DE INGESTA")
    print("-" * 62)
    print(f"  documentos brutos           : {len(df_bruto)}")
    print(f"  rechazados por validación   : {len(df_rechazados)}")
    print(f"  duplicados exactos          : {REPORTE_DEDUP['duplicados_exactos']}")
    print(f"  near-duplicates             : {REPORTE_DEDUP['near_duplicates']}")
    print(f"  ACEPTADOS                   : {len(df_raw)}")
    print(f"  idiomas presentes           : {dict(df_raw['idioma'].value_counts())}")

    if len(df_rechazados):
        print("\nMotivos de rechazo:")
    #     display(df_rechazados["motivo"].value_counts().rename("documentos").to_frame())

    df_raw[["doc_id", "categoria", "titulo", "idioma", "n_chars"]].head(8)

    # @title 2.4.3 — Diagnóstico de salud del corpus
    def diagnosticar_corpus(df: pd.DataFrame, df_rechazados: pd.DataFrame,
                            cfg: Config = CFG) -> dict:
        """Verifica que el corpus sea apto para entrenar y explica el fallo si no lo es.

        Comprueba número de categorías, documentos por categoría y tamaño total.
        No lanza excepción: informa y devuelve el diagnóstico, para que el equipo
        decida si continuar con un corpus reducido o corregir la ingesta.

        Args:
            df: Corpus ya validado y deduplicado.
            df_rechazados: Documentos descartados, con su motivo.
            cfg: Configuración con `cv_folds`.

        Returns:
            Diccionario con `apto`, `n_categorias`, `min_por_categoria`, `problemas`
            y `causas_probables`.

        Example:
            >>> diagnosticar_corpus(df_raw, df_rechazados)["apto"]
            True
        """
        conteos = df["categoria"].value_counts()
        n_cat = len(conteos)
        minimo = int(conteos.min()) if n_cat else 0
        problemas, causas = [], []

        if n_cat < 2:
            problemas.append(
                f"Solo {n_cat} categoría(s): imposible entrenar un clasificador. "
                f"`LogisticRegression.fit` exige al menos 2 clases.")
        elif n_cat == 2:
            problemas.append(
                "Solo 2 categorías: el problema es binario. El top-2 accuracy pierde sentido "
                "(siempre acierta) y se omitirá automáticamente en §5.4.1.")
        if n_cat and minimo < 2:
            escasas = conteos[conteos < 2].index.tolist()
            problemas.append(
                f"Categorías con un solo documento: {escasas}. El split estratificado de §5.3.1 "
                f"no puede repartirlas entre train y test.")
        if n_cat and minimo < cfg.clasificacion.cv_folds:
            problemas.append(
                f"La categoría menos poblada tiene {minimo} documento(s), menos que los "
                f"{cfg.clasificacion.cv_folds} folds de validación cruzada: §5.4.3 reducirá "
                f"el número de folds automáticamente.")
        if len(df) < cfg.corpus.min_documentos:
            problemas.append(
                f"Corpus de {len(df)} documentos: por debajo de {cfg.corpus.min_documentos} "
                f"las métricas tienen una varianza tan alta que no son interpretables.")

        # Desbalance: no impide entrenar, pero distorsiona el modelo de un modo que
        # las métricas macro disimulan. Con `class_weight="balanced"`, una categoría
        # escasa se sobre-predice —su recall sube y su precisión se hunde— y eso no
        # se ve en el F1-macro, que promedia ambas.
        if n_cat >= 2:
            ratio = conteos.max() / max(conteos.min(), 1)
            if ratio > 2.0:
                problemas.append(
                    f"Desbalance de {ratio:.1f}:1 entre '{conteos.index[0]}' "
                    f"({conteos.max()} docs) y '{conteos.index[-1]}' ({conteos.min()}). "
                    f"Las categorías escasas se sobre-predicen: revisa su precisión "
                    f"por clase en §5.4.2, no solo el F1-macro.")

        # --- Causas probables, en orden de frecuencia observada ---
        if problemas:
            if len(df_rechazados):
                motivos = df_rechazados["motivo"].value_counts()
                principal = motivos.index[0]
                causas.append(
                    f"{len(df_rechazados)} documento(s) rechazados en validación; el motivo más "
                    f"frecuente es '{principal}' ({motivos.iloc[0]} casos).")
                if principal == CodigoError.MUY_CORTO:
                    causas.append(
                        f"→ `CFG.validacion.corpus_min_chars` está en "
                        f"{cfg.validacion.corpus_min_chars}. Los párrafos de Wikipedia obtenidos son "
                        f"más cortos: baja el umbral a 150 y vuelve a ejecutar desde §2.3.2.")
                elif principal == CodigoError.IDIOMA_NO_SOPORTADO:
                    causas.append(
                        "→ El rechazo por idioma está descartando documentos. `langdetect` falla en "
                        "fragmentos cortos o muy técnicos: pon "
                        "`CFG.idioma.rechazar_idioma_no_soportado = False` para solo marcarlos.")
            if len(df) < 50:
                causas.append(
                    "Si el scraping devolvió pocos documentos, revisa la conectividad a "
                    "es.wikipedia.org, o activa `CFG.corpus.usar_fallback = True` para trabajar con "
                    "`corpus_fallback.csv` (80 documentos, 8 categorías, sin red).")

        apto = (n_cat >= cfg.corpus.min_categorias and minimo >= 2
                and len(df) >= cfg.corpus.min_documentos)
        return {"apto": apto, "n_categorias": n_cat, "min_por_categoria": minimo,
                "n_documentos": len(df), "problemas": problemas, "causas_probables": causas,
                "conteos": conteos.to_dict()}


    class CorpusInsuficienteError(RuntimeError):
        """El corpus no permite entrenar un clasificador ni siquiera con el respaldo."""


    def _informar(salud: dict, titulo: str) -> None:
        """Imprime el diagnóstico de salud de forma legible."""
        print(titulo)
        print("=" * 74)
        print(f"  documentos          : {salud['n_documentos']}")
        print(f"  categorías          : {salud['n_categorias']}")
        print(f"  mínimo por categoría: {salud['min_por_categoria']}")
        print(f"  veredicto           : "
              f"{'APTO PARA ENTRENAR' if salud['apto'] else 'CORPUS INSUFICIENTE'}")
        if salud["problemas"]:
            print("\n  PROBLEMAS DETECTADOS")
            print("  " + "-" * 70)
            for i, prob in enumerate(salud["problemas"], 1):
                print(f"   {i}. {prob}")
                log.warning(f"Salud del corpus: {prob}")
        if salud["causas_probables"]:
            print("\n  CAUSAS PROBABLES Y CÓMO CORREGIRLAS")
            print("  " + "-" * 70)
            for causa in salud["causas_probables"]:
                print(f"   {causa}")


    FUENTE_CORPUS = ("corpus_fallback.csv" if CFG.corpus.usar_fallback
                     else "Wikipedia ES (MediaWiki Action API)")
    RESPALDO_ACTIVADO = False

    SALUD = diagnosticar_corpus(df_raw, df_rechazados)
    _informar(SALUD, "DIAGNÓSTICO DE SALUD DEL CORPUS")

    # --- Recuperación automática -------------------------------------------
    _ruta_respaldo = Path(CFG.corpus.archivo_fallback)

    # Caso frecuente en Colab: el respaldo existe en el repositorio pero nadie lo
    # subió a la sesión. Sin este aviso, la recuperación no se activa y el usuario
    # solo ve la parada dura, sin entender por qué no funcionó la red de seguridad.
    if (not SALUD["apto"] and CFG.corpus.fallback_automatico
            and not CFG.corpus.usar_fallback and not _ruta_respaldo.exists()):
        print("\n" + "!" * 74)
        print("  NO SE PUEDE APLICAR LA RECUPERACIÓN AUTOMÁTICA")
        print("!" * 74)
        print(f"  El corpus es insuficiente y '{_ruta_respaldo.name}' NO está en la sesión.")
        print(f"  Directorio actual: {Path.cwd()}")
        print(f"\n  Súbelo con el panel de archivos de Colab (icono de carpeta) o ejecuta:")
        print(f"      from google.colab import files; files.upload()")
        print(f"\n  Está en el repositorio, junto a semillas_wikipedia.csv.")
        print("!" * 74)
        log.error(f"Recuperación imposible: falta {_ruta_respaldo}")

    if (not SALUD["apto"] and CFG.corpus.fallback_automatico
            and not CFG.corpus.usar_fallback and _ruta_respaldo.exists()):

        print("\n" + "=" * 74)
        print("  ACTIVANDO CORPUS DE RESPALDO")
        print("=" * 74)
        print(f"  El corpus recolectado no permite entrenar. Se recarga "
              f"'{_ruta_respaldo.name}'\n  y se reejecuta la normalización sobre él.")
        log.warning(f"Corpus recolectado inservible ({SALUD['n_categorias']} categorías, "
                    f"{SALUD['n_documentos']} documentos): activando respaldo.")

        with etapa("recuperación con corpus de respaldo"):
            df_bruto = pd.read_csv(_ruta_respaldo)
            df_raw, df_rechazados = normalizar_corpus(df_bruto)
            df_raw, REPORTE_DEDUP = deduplicar(df_raw)
            df_raw["doc_id"] = [f"DOC-{i:04d}" for i in range(len(df_raw))]
            df_raw.to_csv(CFG.rutas.raw / "corpus_raw.csv", index=False)

        RESPALDO_ACTIVADO = True
        FUENTE_CORPUS = "corpus_fallback.csv (respaldo automático)"
        SALUD = diagnosticar_corpus(df_raw, df_rechazados)
        print()
        _informar(SALUD, "DIAGNÓSTICO TRAS ACTIVAR EL RESPALDO")

    # --- Parada dura: sin dos clases no hay clasificación posible ----------
    if SALUD["n_categorias"] < 2:
        _pistas = []
        if not _ruta_respaldo.exists():
            _pistas.append(f"  1. Sube '{_ruta_respaldo.name}' a la sesión de Colab y reejecuta "
                           f"esta celda: la recuperación automática se encargará del resto.")
        else:
            _pistas.append("  1. Pon CFG.corpus.usar_fallback = True en §0.4 y reejecuta "
                           "desde §2.3.1.")
        if ESTADO_SCRAPING.get("bloqueados"):
            _pistas.append(f"  2. Wikimedia devolvió {ESTADO_SCRAPING['bloqueados']} bloqueo(s) 403: "
                           f"pon un correo real en CFG.corpus.contacto (§0.4).")
        elif ESTADO_SCRAPING.get("limitados"):
            _pistas.append(f"  2. Hubo {ESTADO_SCRAPING['limitados']} respuesta(s) de límite de tasa: "
                           f"sube CFG.corpus.pausa_entre_peticiones y reejecuta §2.3.2.")
        else:
            _pistas.append("  2. Revisa la cobertura por categoría de §2.3.5 para ver "
                           "en qué punto falló la recolección.")

        raise CorpusInsuficienteError(
            f"El corpus tiene {SALUD['n_categorias']} categoría(s) y "
            f"{SALUD['n_documentos']} documento(s): es imposible entrenar un clasificador.\n\n"
            f"Cómo desbloquearte:\n" + "\n".join(_pistas)
        )

    print("\n" + "=" * 74)
    if RESPALDO_ACTIVADO:
        print("  El pipeline continúa con el CORPUS DE RESPALDO.")
        print("  Indícalo al presentar resultados: no proceden del scraping.")
    elif SALUD["apto"]:
        print("  El corpus cumple las condiciones para entrenar y evaluar con garantías.")
    else:
        print("  El pipeline continúa, pero §5.3 a §5.6 se adaptarán al corpus disponible")
        print("  y las métricas no serán representativas. Corrige la ingesta antes de")
        print("  presentar resultados.")
    print("=" * 74)

    # display(pd.Series(SALUD["conteos"], name="documentos").to_frame())

    # @title 2.5 — EDA del corpus recolectado
    fig, axes = plt.subplots(2, 2, figsize=(17, 9))

    orden = df_raw["categoria"].value_counts().index
    sns.countplot(data=df_raw, y="categoria", order=orden, ax=axes[0, 0],
                  hue="categoria", palette="viridis", legend=False)
    axes[0, 0].set_title("Documentos por categoría", fontweight="bold")
    axes[0, 0].set_xlabel("nº documentos"); axes[0, 0].set_ylabel("")

    sns.histplot(df_raw["n_chars"], bins=30, ax=axes[0, 1], color="#2a6f97")
    axes[0, 1].axvline(df_raw["n_chars"].median(), color="crimson", ls="--",
                       label=f"mediana = {df_raw['n_chars'].median():.0f}")
    axes[0, 1].set_title("Distribución de longitud (caracteres)", fontweight="bold")
    axes[0, 1].legend()

    sns.boxplot(data=df_raw, x="n_chars", y="categoria", order=orden, ax=axes[1, 0],
                hue="categoria", palette="viridis", legend=False)
    axes[1, 0].set_title("Longitud por categoría", fontweight="bold"); axes[1, 0].set_ylabel("")

    sns.histplot(df_raw["idioma_confianza"], bins=25, ax=axes[1, 1], color="#588157")
    axes[1, 1].axvline(CFG.idioma.confianza_minima, color="crimson", ls="--",
                       label=f"umbral = {CFG.idioma.confianza_minima}")
    axes[1, 1].set_title("Confianza de la detección de idioma", fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout(); plt.show()

    desbalance = df_raw["categoria"].value_counts()
    n_baja_confianza = int((df_raw["idioma_confianza"] < CFG.idioma.confianza_minima).sum())

    print(f"Ratio de desbalance (mayor/menor clase) : {desbalance.max() / desbalance.min():.2f}")
    print(f"Longitud media                          : {df_raw['n_chars'].mean():.0f} caracteres")
    print(f"Total del corpus                        : {df_raw['n_chars'].sum():,} caracteres")
    print(f"Documentos con confianza de idioma baja : {n_baja_confianza} "
          f"({n_baja_confianza / len(df_raw):.1%})")
    if desbalance.max() / desbalance.min() > 2:
        log.warning("Desbalance > 2:1 — se usará F1-macro y split estratificado (§5.2).")

    # @title 4.1 — Pipeline de spaCy + EntityRuler de tecnologías
    import spacy

    # Diccionario de tecnologías. El modelo base en español no reconoce estos nombres como
    # entidades (Technology_Architecture.md §5, "Desventajas"): el EntityRuler lo corrige
    # por reglas, con precisión de 100% sobre los términos listados.
    TECNOLOGIAS = [
        # Lenguajes
        "Java", "Python", "JavaScript", "TypeScript", "Kotlin", "Swift", "Go", "Rust",
        "C#", "C++", "PHP", "Ruby", "Scala", "Perl", "R", "Dart", "Elixir",
        # Frameworks backend
        "Spring Boot", "Spring", "Django", "Flask", "FastAPI", "Node.js", "Express",
        "Laravel", ".NET", "ASP.NET", "Rails", "NestJS", "Quarkus", "Micronaut",
        # Frontend
        "React", "React Native", "Angular", "Vue.js", "Next.js", "Svelte", "Nuxt",
        "TailwindCSS", "Bootstrap", "HTML", "CSS", "Sass", "Webpack", "Vite", "DOM", "jQuery",
        # Datos
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Oracle", "SQL", "NoSQL",
        "ChromaDB", "Elasticsearch", "Cassandra", "DynamoDB", "Neo4j", "pgvector", "SQLAlchemy",
        # DevOps e infraestructura
        "Docker", "Kubernetes", "Jenkins", "GitLab", "GitHub", "Git", "Terraform",
        "Ansible", "Nginx", "Apache", "Helm", "Prometheus", "Grafana", "CI/CD",
        # Cloud
        "AWS", "Azure", "Google Cloud", "OCI", "Oracle Cloud", "Object Storage",
        "Lambda", "Kafka", "RabbitMQ", "S3", "EC2", "Cloud Functions",
        # Ciencia de datos y ML
        "TensorFlow", "PyTorch", "Scikit-Learn", "Pandas", "NumPy", "spaCy", "BERT",
        "Transformers", "Keras", "XGBoost", "Hugging Face", "Jupyter", "Matplotlib", "NLTK",
        # Protocolos y formatos
        "API REST", "REST", "GraphQL", "gRPC", "JSON", "XML", "YAML", "OAuth", "JWT",
        "HTTPS", "TLS", "SSL", "WebSocket", "HTTP", "SOAP", "OpenAPI", "Swagger",
        # Móvil
        "Android", "iOS", "Flutter", "Xamarin", "Ionic",
        # Conceptos
        "machine learning", "deep learning", "microservicios", "aprendizaje automático",
        "redes neuronales", "computación en la nube", "arquitectura hexagonal",
        "inyección de dependencias", "programación orientada a objetos", "test unitario",
    ]


    def construir_pipeline_nlp(codigo_idioma: str = CFG.idioma.idioma_objetivo,
                               tecnologias: list = TECNOLOGIAS):
        """Carga el pipeline de spaCy del idioma indicado y le añade el EntityRuler.

        Args:
            codigo_idioma: Código ISO 639-1 del idioma.
            tecnologias: Lista de términos técnicos a reconocer como entidades TECH.

        Returns:
            El objeto `Language` de spaCy con el componente `entity_ruler` activo.

        Example:
            >>> nlp = construir_pipeline_nlp("es")
            >>> [e.text for e in nlp("APIs REST con Spring Boot").ents if e.label_ == "TECH"]
            ['REST', 'Spring Boot']
        """
        modelo = REGISTRO_IDIOMAS.obtener(codigo_idioma)
        if "entity_ruler" not in modelo.pipe_names:
            # before="ner" da prioridad a las reglas sobre el NER estadístico; si el
            # modelo no trae 'ner', spaCy lanzaría ValueError, así que se añade al final.
            if "ner" in modelo.pipe_names:
                ruler = modelo.add_pipe("entity_ruler", before="ner")
            else:
                ruler = modelo.add_pipe("entity_ruler")
                log.warning("El modelo no tiene componente 'ner'; EntityRuler al final.")
            ruler.add_patterns([{"label": "TECH", "pattern": t} for t in tecnologias])
            log.info(f"EntityRuler añadido con {len(tecnologias)} patrones de tecnologías")
        return modelo


    nlp = construir_pipeline_nlp()

    log.info(f"Pipeline spaCy activo: {nlp.pipe_names}")
    doc_demo = nlp("En este contenido se presentan los conceptos básicos para la creación de "
                   "APIs REST utilizando Java y Spring Boot sobre Docker.")
    print("Pipeline :", nlp.pipe_names)
    print("Patrones :", len(TECNOLOGIAS), "tecnologías registradas")
    print("Entidades TECH detectadas:", [e.text for e in doc_demo.ents if e.label_ == "TECH"])

    # @title 4.2 — Tokenización, stopwords, lematización y filtrado POS en una pasada
    STOPWORDS_EXTRA = set(CFG.nlp.stopwords_extra)


    def preprocesar(doc, cfg: Config = CFG) -> dict:
        """Extrae tokens, lemas, lemas filtrados por POS y entidades de un Doc de spaCy.

        Resuelve en una sola iteración las cuatro cajas de la etapa 2 del diagrama:
        tokenización, eliminación de stopwords, lematización y filtrado por categoría
        gramatical. La conversión a minúsculas ocurre aquí, sobre el lema, para no
        privar al POS tagger y al NER de la señal de capitalización.

        Args:
            doc: Objeto `Doc` producido por el pipeline de spaCy.
            cfg: Configuración con las POS relevantes y la longitud mínima de token.

        Returns:
            Diccionario con las claves `tokens`, `lemas`, `lemas_pos`, `texto_lemas`,
            `texto_pos`, `entidades_tech` y `n_tokens`.

        Example:
            >>> p = preprocesar(nlp("Las APIs REST usan el protocolo HTTP para comunicarse."))
            >>> "api" in p["texto_pos"] or "rest" in p["texto_pos"]
            True
        """
        tokens_validos, lemas, lemas_pos = [], [], []
        pos_relevantes = set(cfg.nlp.pos_relevantes)

        for token in doc:
            # --- Tokenización: descarte de ruido no léxico ---
            if token.is_space or token.is_punct or token.like_num or token.is_digit:
                continue
            # --- Stopwords y tokens demasiado cortos ---
            if token.is_stop or len(token.text) < cfg.nlp.long_minima_token:
                continue
            # --- Lematización (y aquí sí, minúsculas) ---
            lema = token.lemma_.lower().strip()
            if not lema or lema in STOPWORDS_EXTRA:
                continue
            tokens_validos.append(token.text.lower())
            lemas.append(lema)
            # --- Filtrado por POS tagging ---
            if token.pos_ in pos_relevantes:
                lemas_pos.append(lema)

        entidades = sorted({e.text for e in doc.ents if e.label_ in cfg.nlp.etiquetas_entidad})

        return {
            "tokens": tokens_validos,
            "lemas": lemas,
            "lemas_pos": lemas_pos,
            "texto_lemas": " ".join(lemas),
            "texto_pos": " ".join(lemas_pos),
            "entidades_tech": entidades,
            "n_tokens": len(tokens_validos),
        }


    @cronometrar("preprocesamiento NLP del corpus")
    def preprocesar_corpus(df: pd.DataFrame, modelo_nlp, cfg: Config = CFG) -> pd.DataFrame:
        """Aplica `preprocesar` a todo el corpus usando procesamiento por lotes.

        `nlp.pipe` amortiza el overhead de construcción del Doc y es sustancialmente
        más rápido que llamar `nlp(texto)` documento a documento.

        Args:
            df: DataFrame con la columna `texto_limpio`.
            modelo_nlp: Pipeline de spaCy a aplicar.
            cfg: Configuración con el batch size.

        Returns:
            El DataFrame original con las columnas de preprocesamiento añadidas.
        """
        resultados = [
            preprocesar(d, cfg)
            for d in modelo_nlp.pipe(df["texto_limpio"].tolist(),
                                     batch_size=cfg.nlp.batch_size_spacy)
        ]
        salida = pd.concat([df.reset_index(drop=True), pd.DataFrame(resultados)], axis=1)
        log.info(f"Preprocesamiento completado: {len(salida)} documentos, "
                 f"{salida['n_tokens'].sum():,} tokens conservados")
        return salida


    with etapa("preprocesamiento NLP"):
        df = preprocesar_corpus(df_raw, nlp)

    df[["doc_id", "categoria", "n_tokens", "texto_pos"]].head(5)

    # @title 4.3 — Trazabilidad: el pipeline paso a paso sobre el ejemplo del brief
    EJEMPLO = ("En este contenido se presentan los conceptos básicos para la creación de "
               "APIs REST utilizando Java y Spring Boot.")

    d = nlp(limpiar_texto(EJEMPLO))
    p = preprocesar(d)

    print("TEXTO ORIGINAL\n ", EJEMPLO)
    print("\n0) VALIDACIÓN      ->", "aceptado" if validar_entrada("Ejemplo del brief", EJEMPLO).valido else "rechazado")
    print("1) IDIOMA          ->", detectar_idioma(EJEMPLO).a_dict())
    print("2) LIMPIEZA        ->", limpiar_texto(EJEMPLO))
    print("3) TOKENIZACIÓN + STOPWORDS ->", p["tokens"])
    print("4) LEMATIZACIÓN    ->", p["lemas"])
    print(f"5) FILTRADO POS {sorted(CFG.nlp.pos_relevantes)} ->", p["lemas_pos"])
    print("6) ENTIDADES TÉCNICAS ->", p["entidades_tech"])

    print("\n" + "-" * 80)
    print(f"{'TOKEN':<18}{'LEMA':<18}{'POS':<10}{'STOP':<8}{'CONSERVADO'}")
    print("-" * 80)
    for t in d[:18]:
        conservado = ("sí" if (t.pos_ in CFG.nlp.pos_relevantes and not t.is_stop
                               and not t.is_punct) else "no")
        print(f"{t.text:<18}{t.lemma_:<18}{t.pos_:<10}{str(t.is_stop):<8}{conservado}")

    # @title 4.4 — EDA sobre el texto ya procesado
    df["n_lemas_pos"] = df["lemas_pos"].apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    reduccion = pd.DataFrame({
        "etapa": ["1. tokens brutos", "2. sin stopwords", "3. filtrado POS"],
        "promedio": [
            df["texto_limpio"].str.split().apply(len).mean(),
            df["n_tokens"].mean(),
            df["n_lemas_pos"].mean(),
        ],
    })
    sns.barplot(data=reduccion, x="etapa", y="promedio", ax=axes[0],
                hue="etapa", palette="rocket", legend=False)
    axes[0].set_title("Reducción de dimensionalidad léxica por etapa", fontweight="bold")
    axes[0].set_ylabel("tokens promedio por documento")
    for i, v in enumerate(reduccion["promedio"]):
        axes[0].text(i, v + 1, f"{v:.0f}", ha="center", fontweight="bold")

    top = Counter([l for lista in df["lemas_pos"] for l in lista]).most_common(20)
    _pal = [w for w, _ in top]
    sns.barplot(x=[c for _, c in top], y=_pal, ax=axes[1],
                hue=_pal, palette="mako", legend=False)
    axes[1].set_title("Top 20 lemas (sustantivos, nombres propios y adjetivos)", fontweight="bold")
    axes[1].set_xlabel("frecuencia")

    plt.tight_layout(); plt.show()

    tasa = 1 - reduccion["promedio"].iloc[2] / reduccion["promedio"].iloc[0]
    print(f"Reducción total de tokens: {tasa:.1%}\n")

    print("Términos más frecuentes por categoría (señal cualitativa de separabilidad):\n")
    for cat in sorted(df["categoria"].unique()):
        lemas_cat = [l for lista in df.loc[df["categoria"] == cat, "lemas_pos"] for l in lista]
        tops = ", ".join(w for w, _ in Counter(lemas_cat).most_common(8))
        print(f"  {cat:<18} {tops}")

    n_con_entidades = int((df["entidades_tech"].apply(len) > 0).sum())
    print(f"\nDocumentos con al menos una entidad técnica detectada: "
          f"{n_con_entidades}/{len(df)} ({n_con_entidades / len(df):.1%})")


    # --- Solapamiento léxico entre categorías -------------------------------
    def solapamiento_categorias(df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
        """Mide cuánto vocabulario comparten las categorías dos a dos.

        Predice las confusiones del clasificador **antes** de entrenarlo: dos
        categorías cuyos términos más frecuentes coinciden no son separables por
        léxico, y el modelo A las confundirá haga lo que haga.

        Casi siempre el origen no es el modelo sino el corpus: dos semillas que
        tratan el mismo tema archivadas en categorías distintas. Ocurrió dos veces
        en este proyecto — 'Aplicaciones web progresivas' en Mobile cuando es
        Frontend, y 'Virtualización' en DevOps cuando Cloud ya tenía 'Máquina
        virtual'—, y en ambos casos la confusión apareció en la matriz de §5.4.2
        cuando ya era caro corregirla.

        Args:
            df: Corpus preprocesado con `categoria` y `lemas_pos`.
            top_n: Número de términos principales a comparar por categoría.

        Returns:
            DataFrame con los pares ordenados por solapamiento descendente.

        Example:
            >>> solapamiento_categorias(df).iloc[0]["solapamiento"] < 0.5
            True
        """
        vocab = {
            cat: {t for t, _ in Counter(
                [l for lista in df.loc[df["categoria"] == cat, "lemas_pos"] for l in lista]
            ).most_common(top_n)}
            for cat in sorted(df["categoria"].unique())
        }
        filas = []
        cats = list(vocab)
        for i, a in enumerate(cats):
            for b in cats[i + 1:]:
                comunes = vocab[a] & vocab[b]
                filas.append({
                    "categoria_a": a, "categoria_b": b,
                    "solapamiento": round(len(comunes) / top_n, 3),
                    "terminos_comunes": ", ".join(sorted(comunes)[:6]),
                })
        return pd.DataFrame(filas).sort_values("solapamiento", ascending=False)


    SOLAPAMIENTO = solapamiento_categorias(df)
    UMBRAL_SOLAPE = 0.45

    print("\n\nSOLAPAMIENTO LÉXICO ENTRE CATEGORÍAS")
    print("=" * 74)
    # display(SOLAPAMIENTO.head(6).reset_index(drop=True))

    _riesgo = SOLAPAMIENTO[SOLAPAMIENTO["solapamiento"] >= UMBRAL_SOLAPE]
    if len(_riesgo):
        print(f"\n  {len(_riesgo)} par(es) comparten más del {UMBRAL_SOLAPE:.0%} de su vocabulario:")
        for _, r in _riesgo.iterrows():
            log.warning(f"Solapamiento alto {r['categoria_a']}/{r['categoria_b']}: "
                        f"{r['solapamiento']:.0%}")
            print(f"    {r['categoria_a']} ↔ {r['categoria_b']}  ({r['solapamiento']:.0%})"
                  f"  — comparten: {r['terminos_comunes']}")
        print("\n  Espera verlos confundidos en la matriz de §5.4.2. La corrección NO es")
        print("  tocar el modelo: revisa si alguna semilla trata un tema que pertenece a")
        print("  la otra categoría, o si las dos categorías se solapan por definición.")
    else:
        print(f"\n  Ningún par supera el {UMBRAL_SOLAPE:.0%}: las categorías son "
              f"léxicamente separables.")

    df.to_csv(CFG.rutas.processed / "corpus_processed.csv", index=False)
    log.info(f"Corpus preprocesado guardado en {CFG.rutas.processed / 'corpus_processed.csv'}")

    # @title 5.1.1 — Vectorización TF-IDF (representación léxica)
    from sklearn.feature_extraction.text import TfidfVectorizer

    with etapa("vectorización TF-IDF"):
        tfidf = TfidfVectorizer(
            ngram_range=(CFG.tfidf.ngram_min, CFG.tfidf.ngram_max),
            min_df=CFG.tfidf.min_df,
            max_df=CFG.tfidf.max_df,
            sublinear_tf=CFG.tfidf.sublinear_tf,
            max_features=CFG.tfidf.max_features,
        )
        X_tfidf = tfidf.fit_transform(df["texto_pos"])
        vocabulario = np.array(tfidf.get_feature_names_out())

    log.info(f"Matriz TF-IDF: {X_tfidf.shape[0]}x{X_tfidf.shape[1]} "
             f"(densidad {X_tfidf.nnz / np.prod(X_tfidf.shape):.4%})")


    def keywords_tfidf(idx: int, top_k: int = CFG.keywords.top_k) -> list:
        """Extrae las keywords de un documento del corpus por peso TF-IDF.

        Solo aplicable a documentos que formaron parte del ajuste del vectorizador.
        Para texto nuevo se usa YAKE o KeyBERT, que no dependen del corpus.

        Args:
            idx: Índice posicional del documento en el DataFrame.
            top_k: Número de keywords a devolver.

        Returns:
            Lista de tuplas (keyword, peso) ordenada de mayor a menor peso.

        Example:
            >>> keywords_tfidf(0, top_k=3)
            [('spring boot', 0.4812), ('java', 0.3155), ('framework', 0.2903)]
        """
        fila = X_tfidf[idx].toarray().ravel()
        mejores = fila.argsort()[::-1][:top_k]
        return [(vocabulario[i], round(float(fila[i]), 4)) for i in mejores if fila[i] > 0]


    print(f"Matriz TF-IDF: {X_tfidf.shape[0]} documentos x {X_tfidf.shape[1]} términos")
    print(f"\nEjemplo — keywords TF-IDF de '{df.loc[0, 'titulo'][:60]}':")
    for kw, score in keywords_tfidf(0):
        print(f"   {score:.4f}  {kw}")

    # @title 5.1.2 — Caché de embeddings
    class CacheEmbeddings:
        """Caché persistente de embeddings, indexada por hash del texto y del modelo.

        Evita recalcular vectores ya conocidos. La clave incluye el nombre del modelo
        y el flag de normalización, de modo que cambiar de modelo invalida las
        entradas afectadas sin necesidad de purgar la caché manualmente.

        Attributes:
            aciertos: Número de vectores recuperados de la caché en esta sesión.
            fallos: Número de vectores que hubo que calcular.

        Example:
            >>> cache = CacheEmbeddings(CFG)
            >>> vectores = cache.codificar(["texto uno", "texto dos"], modelo_embeddings)
            >>> vectores.shape[1] == 384
            True
        """

        def __init__(self, cfg: Config = CFG):
            self.cfg = cfg
            self.ruta = cfg.rutas.cache / cfg.embeddings.archivo_cache
            self.aciertos = 0
            self.fallos = 0
            self._store: dict = {}
            if cfg.embeddings.usar_cache and self.ruta.exists():
                try:
                    self._store = joblib.load(self.ruta)
                    log.info(f"Caché de embeddings cargada: {len(self._store)} vectores")
                except Exception as exc:
                    log.warning(f"Caché ilegible ({type(exc).__name__}); se reconstruye desde cero.")
                    self._store = {}

        def _clave(self, texto: str) -> str:
            """Genera la clave de caché para un texto bajo la configuración actual."""
            semilla = f"{self.cfg.embeddings.modelo}|{self.cfg.embeddings.normalizar}|{texto}"
            return hashlib.sha256(semilla.encode("utf-8")).hexdigest()

        def codificar(self, textos: Sequence, modelo, mostrar_progreso: bool = False) -> np.ndarray:
            """Devuelve la matriz de embeddings de una lista de textos, usando la caché.

            Args:
                textos: Secuencia de textos a codificar.
                modelo: Instancia de `SentenceTransformer`.
                mostrar_progreso: Muestra la barra de progreso de sentence-transformers.

            Returns:
                Array de forma (len(textos), dimension) en el orden de entrada.
            """
            textos = list(textos)
            claves = [self._clave(t) for t in textos]

            pendientes_idx = [i for i, k in enumerate(claves) if k not in self._store]
            self.aciertos += len(textos) - len(pendientes_idx)
            self.fallos += len(pendientes_idx)

            if pendientes_idx:
                log.info(f"Embeddings: {len(pendientes_idx)} por calcular, "
                         f"{len(textos) - len(pendientes_idx)} desde caché")
                nuevos = modelo.encode(
                    [textos[i] for i in pendientes_idx],
                    batch_size=self.cfg.embeddings.batch_size,
                    show_progress_bar=mostrar_progreso,
                    normalize_embeddings=self.cfg.embeddings.normalizar,
                )
                for i, vector in zip(pendientes_idx, nuevos):
                    self._store[claves[i]] = np.asarray(vector, dtype=np.float32)
            else:
                log.info(f"Embeddings: {len(textos)} recuperados íntegramente de caché")

            return np.vstack([self._store[k] for k in claves])

        def guardar(self) -> None:
            """Persiste la caché en disco."""
            if not self.cfg.embeddings.usar_cache:
                return
            joblib.dump(self._store, self.ruta, compress=3)
            log.info(f"Caché persistida: {len(self._store)} vectores en {self.ruta}")

        def estadisticas(self) -> dict:
            """Devuelve métricas de uso de la caché en esta sesión."""
            total = self.aciertos + self.fallos
            return {
                "vectores_en_cache": len(self._store),
                "aciertos": self.aciertos,
                "fallos": self.fallos,
                "tasa_acierto": round(self.aciertos / total, 4) if total else 0.0,
                "tamano_mb": round(self.ruta.stat().st_size / 1e6, 2) if self.ruta.exists() else 0.0,
            }


    CACHE = CacheEmbeddings()
    print(f"Caché inicializada — vectores previos: {len(CACHE._store)}")

    # @title 5.1.3 — Cálculo de embeddings (insumo compartido de todo el sistema)
    from sentence_transformers import SentenceTransformer


    @cronometrar("carga del modelo de embeddings")
    def cargar_modelo_embeddings(cfg: Config = CFG) -> SentenceTransformer:
        """Carga el modelo de Sentence-Transformers y verifica su dimensionalidad.

        Args:
            cfg: Configuración con el nombre del modelo y la dimensión esperada.

        Returns:
            La instancia de `SentenceTransformer` lista para usar.

        Raises:
            ValueError: Si la dimensión real no coincide con `dimension_esperada`,
                lo que indicaría que el nombre del modelo cambió sin actualizar la
                configuración y rompería la compatibilidad con la colección de ChromaDB.
        """
        modelo = SentenceTransformer(cfg.embeddings.modelo)
        dim = modelo.get_sentence_embedding_dimension()
        if dim != cfg.embeddings.dimension_esperada:
            raise ValueError(
                f"El modelo '{cfg.embeddings.modelo}' produce vectores de {dim} dimensiones, "
                f"pero la configuración declara {cfg.embeddings.dimension_esperada}. "
                f"Actualiza CFG.embeddings.dimension_esperada y recrea la colección de ChromaDB."
            )
        log.info(f"Modelo de embeddings: {cfg.embeddings.modelo} ({dim} dimensiones)")
        return modelo


    with etapa("generación de embeddings"):
        modelo_embeddings = cargar_modelo_embeddings()
        embeddings = CACHE.codificar(df["texto_limpio"].tolist(), modelo_embeddings,
                                     mostrar_progreso=True)
        CACHE.guardar()
        np.save(CFG.rutas.processed / "embeddings.npy", embeddings)

    print(f"\nMatriz de embeddings: {embeddings.shape} ({embeddings.dtype})")
    print(f"Norma media de los vectores: {np.linalg.norm(embeddings, axis=1).mean():.4f} "
          f"(≈1.0 confirma normalización L2)")
    print(f"\nEstadísticas de caché: {CACHE.estadisticas()}")

    # @title 5.2.1 — KeyBERT (señal semántica) y YAKE (señal estadística)
    from keybert import KeyBERT
    import yake

    kw_model = KeyBERT(model=modelo_embeddings)   # reutiliza el modelo ya cargado en memoria

    extractor_yake = yake.KeywordExtractor(
        lan=CFG.idioma.idioma_objetivo,
        n=CFG.keywords.ngram_max,
        dedupLim=CFG.keywords.yake_dedup_limite,
        top=CFG.keywords.keybert_candidatos,
        features=None,
    )

    _STOPWORDS_SPACY = list(nlp.Defaults.stop_words)


    def keywords_keybert(texto: str, top_k: int = CFG.keywords.top_k,
                         diversidad: float = CFG.keywords.keybert_diversidad) -> list:
        """Extrae keywords por similitud semántica entre n-gramas y el documento completo.

        Args:
            texto: Texto limpio del documento.
            top_k: Número de keywords a devolver.
            diversidad: Parámetro de MMR. 0 = relevancia pura (riesgo de redundancia),
                1 = diversidad pura (riesgo de irrelevancia).

        Returns:
            Lista de tuplas (keyword, similitud). Mayor similitud es mejor.

        Example:
            >>> keywords_keybert("APIs REST con Java y Spring Boot", top_k=2)
            [('spring boot', 0.6821), ('apis rest', 0.5934)]
        """
        pares = kw_model.extract_keywords(
            texto,
            keyphrase_ngram_range=(CFG.keywords.ngram_min, CFG.keywords.ngram_max),
            stop_words=_STOPWORDS_SPACY,
            use_mmr=True,
            diversity=diversidad,
            top_n=top_k,
        )
        return [(kw, round(float(score), 4)) for kw, score in pares]


    def keywords_yake(texto: str, top_k: int = CFG.keywords.top_k) -> list:
        """Extrae keywords por heurísticas estadísticas, sin necesitar corpus de referencia.

        Args:
            texto: Texto limpio del documento.
            top_k: Número de keywords a devolver.

        Returns:
            Lista de tuplas (keyword, score). YAKE puntúa a la INVERSA:
            menor score significa keyword más relevante.

        Example:
            >>> keywords_yake("APIs REST con Java y Spring Boot", top_k=2)
            [('spring boot', 0.0184), ('apis rest', 0.0291)]
        """
        return [(kw, round(score, 4))
                for kw, score in extractor_yake.extract_keywords(texto)][:top_k]


    print("Ejemplo comparado sobre el documento 0:\n")
    print("  KeyBERT (mayor = mejor):")
    for kw, s in keywords_keybert(df.loc[0, "texto_limpio"]):
        print(f"     {s:.4f}  {kw}")
    print("\n  YAKE (menor = mejor):")
    for kw, s in keywords_yake(df.loc[0, "texto_limpio"]):
        print(f"     {s:.4f}  {kw}")

    # @title 5.2.2 — Comparación cualitativa de las tres vías (nodo "Enfoque de Modelado")
    MUESTRA = [0, len(df) // 3, 2 * len(df) // 3]

    for idx in MUESTRA:
        print("=" * 104)
        print(f"[{df.loc[idx, 'categoria']}] {df.loc[idx, 'titulo'][:80]}")
        print(f"  {df.loc[idx, 'texto_limpio'][:180]}...")
        print("-" * 104)
        n = CFG.keywords.top_k
        def _pad(lista):
            return (lista + [""] * n)[:n]
        comparacion = pd.DataFrame({
            "TF-IDF (estadístico de corpus)": _pad([k for k, _ in keywords_tfidf(idx)]),
            "YAKE (estadístico local)":       _pad([k for k, _ in keywords_yake(df.loc[idx, "texto_limpio"])]),
            "KeyBERT (semántico)":            _pad([k for k, _ in keywords_keybert(df.loc[idx, "texto_limpio"])]),
            "EntityRuler (reglas)":           _pad(list(df.loc[idx, "entidades_tech"])),
        })
    #     display(comparacion)

    # @title 5.2.3 — Ranking híbrido por Reciprocal Rank Fusion
    _MAPA_TECNOLOGIAS = {t.lower(): t for t in TECNOLOGIAS}


    def es_keyword_valida(frase: str, pipeline_nlp=None, cfg: Config = CFG) -> bool:
        """Decide si una frase candidata parece un término técnico y no prosa suelta.

        KeyBERT y YAKE puntúan n-gramas por relevancia semántica o estadística, pero
        ninguno comprueba la **categoría gramatical**. De ahí salían candidatas como
        *"presentan conceptos básicos"* o *"teóricos desalentaron tipo"*: frases
        verbales con buena similitud coseno que no son términos y quedan mal en el
        campo `informacion_adicional` de la respuesta.

        El criterio es sencillo y funciona: una keyword técnica se apoya en
        sustantivos o nombres propios. Se exige que la frase contenga al menos uno y
        que no empiece ni acabe en verbo, adverbio o preposición.

        Args:
            frase: Candidata a keyword.
            pipeline_nlp: Pipeline de spaCy. Por defecto, el global `nlp`.
            cfg: Configuración; si `filtrar_keywords_por_pos` es False, acepta todo.

        Returns:
            True si la frase se conserva.

        Example:
            >>> es_keyword_valida("spring boot"), es_keyword_valida("presentan conceptos")
            (True, False)
        """
        if not cfg.keywords.filtrar_keywords_por_pos:
            return True
        pipeline_nlp = pipeline_nlp if pipeline_nlp is not None else nlp

        doc = pipeline_nlp(frase)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if not tokens:
            return False

        # Debe apoyarse en al menos un sustantivo o nombre propio.
        if not any(t.pos_ in ("NOUN", "PROPN") for t in tokens):
            return False
        # Ni empezar ni terminar en verbo, adverbio o preposición: señal de frase
        # recortada de una oración, no de término.
        malos_extremos = {"VERB", "AUX", "ADV", "ADP", "SCONJ", "CCONJ", "DET", "PRON"}
        if tokens[0].pos_ in malos_extremos or tokens[-1].pos_ in malos_extremos:
            return False
        return True


    def rankear_keywords(texto: str,
                         doc_spacy=None,
                         top_k: int = CFG.keywords.top_k,
                         cfg: Config = CFG,
                         *,
                         modelo_keybert=None,
                         extractor=None,
                         pipeline_nlp=None,
                         mapa_tecnologias: dict = None) -> list:
        """Fusiona las señales de keywords mediante Reciprocal Rank Fusion.

        RRF calcula score(t) = Σ_r peso_r / (K + rango_r(t)), usando únicamente la
        POSICIÓN de cada término en el ranking de cada método. Esto lo hace robusto
        frente a puntajes no comparables entre sí (KeyBERT en [0,1], YAKE inverso,
        TF-IDF sin escala fija).

        Los cuatro parámetros de sólo-palabra clave permiten **inyectar las
        dependencias** en lugar de leerlas del ámbito global. El notebook las omite y
        usa los objetos ya cargados; la capa de inferencia (§7.2) las pasa
        explícitamente, de modo que la función no lea ninguna global cuando se
        importa desde el backend.

        Args:
            texto: Texto limpio del documento.
            doc_spacy: `Doc` de spaCy ya calculado. Si se proporciona, evita volver a
                ejecutar el pipeline lingüístico — optimización relevante en el camino
                crítico de la API, donde el `Doc` ya se calculó en el preprocesamiento.
            top_k: Número de keywords a devolver.
            cfg: Configuración con los pesos y la constante K de RRF.
            modelo_keybert: Instancia de `KeyBERT`. Por defecto, la global `kw_model`.
            extractor: Extractor de YAKE. Por defecto, el global `extractor_yake`.
            pipeline_nlp: Pipeline de spaCy. Por defecto, el global `nlp`.
            mapa_tecnologias: Mapa minúscula → capitalización canónica.

        Returns:
            Lista de keywords en orden de relevancia, con la capitalización canónica
            restaurada para las tecnologías conocidas.

        Example:
            >>> rankear_keywords("Conceptos básicos de APIs REST con Java y Spring Boot")
            ['Spring Boot', 'API REST', 'Java', 'concepto', 'creación']
        """
        modelo_keybert = modelo_keybert if modelo_keybert is not None else kw_model
        extractor = extractor if extractor is not None else extractor_yake
        pipeline_nlp = pipeline_nlp if pipeline_nlp is not None else nlp
        mapa_tecnologias = mapa_tecnologias if mapa_tecnologias is not None else _MAPA_TECNOLOGIAS

        k_rrf = cfg.keywords.rrf_k
        puntajes: dict = {}

        def acumular(lista: Iterable, peso: float) -> None:
            for rango, kw in enumerate(lista, start=1):
                clave = str(kw).lower().strip()
                if len(clave) < 3:
                    continue
                puntajes[clave] = puntajes.get(clave, 0.0) + peso / (k_rrf + rango)

        # Señal 1 — semántica
        pares_kb = modelo_keybert.extract_keywords(
            texto,
            keyphrase_ngram_range=(cfg.keywords.ngram_min, cfg.keywords.ngram_max),
            stop_words=_STOPWORDS_SPACY,
            use_mmr=True,
            diversity=cfg.keywords.keybert_diversidad,
            top_n=cfg.keywords.keybert_candidatos,
        )
        acumular([k for k, _ in pares_kb], cfg.keywords.peso_keybert)
        # Señal 2 — estadística local
        acumular([k for k, _ in extractor.extract_keywords(texto)][:cfg.keywords.keybert_candidatos],
                 cfg.keywords.peso_yake)
        # Señal 3 — reglas (reutiliza el Doc si ya existe: evita una pasada de spaCy)
        doc = doc_spacy if doc_spacy is not None else pipeline_nlp(texto)
        entidades = [e.text for e in doc.ents if e.label_ in cfg.nlp.etiquetas_entidad]
        acumular(entidades, cfg.keywords.peso_entidades)

        ordenadas = sorted(puntajes.items(), key=lambda par: -par[1])

        # Deduplicación por solapamiento + filtro gramatical. Las entidades del
        # EntityRuler se aceptan siempre: vienen del diccionario y son términos por
        # construcción, aunque spaCy etiquete raro alguna sigla.
        entidades_norm = {e.lower().strip() for e in entidades}
        seleccion: list = []
        for kw, _ in ordenadas:
            if any(kw in ya or ya in kw for ya in seleccion):
                continue
            if kw not in entidades_norm and not es_keyword_valida(kw, pipeline_nlp, cfg):
                continue
            seleccion.append(kw)
            if len(seleccion) >= top_k:
                break

        return [mapa_tecnologias.get(k, k) for k in seleccion]


    print("Ejemplo del brief:")
    print("  entrada :", EJEMPLO)
    print("  keywords:", rankear_keywords(limpiar_texto(EJEMPLO)))

    print("\nComprobación de la optimización (reutilizar el Doc de spaCy):")
    t0 = time.perf_counter(); _ = rankear_keywords(limpiar_texto(EJEMPLO)); t_sin = time.perf_counter() - t0
    doc_pre = nlp(limpiar_texto(EJEMPLO))
    t0 = time.perf_counter(); _ = rankear_keywords(limpiar_texto(EJEMPLO), doc_spacy=doc_pre); t_con = time.perf_counter() - t0
    print(f"  sin reutilizar Doc: {t_sin*1000:.1f} ms")
    print(f"  reutilizando Doc  : {t_con*1000:.1f} ms  ({(1 - t_con/t_sin):.0%} más rápido)")

    # @title 5.2.4 — Keywords de todo el corpus (persistencia, punto 14)
    @cronometrar("extracción de keywords del corpus")
    def extraer_keywords_corpus(df: pd.DataFrame) -> pd.DataFrame:
        """Calcula las keywords finales de cada documento del corpus.

        Reutiliza las entidades ya extraídas en §4.2 en lugar de reejecutar spaCy,
        aplicando RRF sobre las señales disponibles.

        Args:
            df: DataFrame preprocesado, con `texto_limpio` y `entidades_tech`.

        Returns:
            El DataFrame con la columna `keywords` añadida.
        """
        salidas = []
        for i, fila in df.iterrows():
            puntajes: dict = {}
            k = CFG.keywords.rrf_k

            def acumular(lista, peso):
                for rango, kw in enumerate(lista, start=1):
                    clave = str(kw).lower().strip()
                    if len(clave) >= 3:
                        puntajes[clave] = puntajes.get(clave, 0.0) + peso / (k + rango)

            acumular([w for w, _ in keywords_keybert(fila["texto_limpio"],
                                                     top_k=CFG.keywords.keybert_candidatos)],
                     CFG.keywords.peso_keybert)
            acumular([w for w, _ in keywords_tfidf(i, top_k=CFG.keywords.keybert_candidatos)],
                     CFG.keywords.peso_yake)
            acumular(fila["entidades_tech"], CFG.keywords.peso_entidades)

            entidades_norm = {e.lower().strip() for e in fila["entidades_tech"]}
            seleccion: list = []
            for kw, _ in sorted(puntajes.items(), key=lambda par: -par[1]):
                if any(kw in ya or ya in kw for ya in seleccion):
                    continue
                if kw not in entidades_norm and not es_keyword_valida(kw):
                    continue
                seleccion.append(kw)
                if len(seleccion) >= CFG.keywords.top_k:
                    break
            salidas.append([_MAPA_TECNOLOGIAS.get(x, x) for x in seleccion])

        df = df.copy()
        df["keywords"] = salidas
        return df


    with etapa("keywords del corpus"):
        df = extraer_keywords_corpus(df)

    # display(df[["doc_id", "categoria", "titulo", "keywords"]].head(8))

    # @title 5.3.1 — Preparación de datos y split estratificado
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline

    le = LabelEncoder()
    y = le.fit_transform(df["categoria"])
    CATEGORIAS = list(le.classes_)
    N_CLASES = len(CATEGORIAS)
    CONTEO_CLASES = pd.Series(y).value_counts()
    MIN_POR_CLASE = int(CONTEO_CLASES.min())


    def partir_dataset(y: np.ndarray, cfg: Config = CFG) -> tuple:
        """Divide en train/test estratificando solo si el corpus lo permite.

        `stratify` exige al menos 2 documentos por clase. Con un corpus degenerado
        (§2.4.3) esa condición no se cumple y scikit-learn lanza `ValueError`. En
        ese caso se degrada a un split aleatorio, avisando: es peor que estratificar,
        pero permite que el pipeline llegue hasta el final y el equipo vea el
        resultado completo antes de decidir.

        Args:
            y: Vector de etiquetas codificadas.
            cfg: Configuración con `test_size` y `random_state`.

        Returns:
            Tupla (índices de train, índices de test, estratificado: bool).
        """
        puede = MIN_POR_CLASE >= 2
        if not puede:
            escasas = [CATEGORIAS[i] for i in CONTEO_CLASES[CONTEO_CLASES < 2].index]
            log.warning(f"Split SIN estratificar: las categorías {escasas} tienen un solo "
                        f"documento. Las métricas por clase serán poco fiables.")
        tr, te = train_test_split(
            np.arange(len(y)),
            test_size=cfg.clasificacion.test_size,
            random_state=cfg.random_state,
            stratify=y if puede else None,
        )

        # Un split aleatorio puede dejar TODAS las muestras de una clase en test, y
        # entonces el entrenamiento recibe una sola clase y `LogisticRegression.fit`
        # falla. Reparamos moviendo un ejemplar de cada clase ausente desde test.
        if not puede:
            tr, te = list(tr), list(te)
            faltantes = set(np.unique(y)) - set(y[tr])
            for clase in faltantes:
                candidatos = [i for i in te if y[i] == clase]
                if candidatos and len(te) > 1:
                    movido = candidatos[0]
                    te.remove(movido)
                    tr.append(movido)
                    log.warning(f"Movido un documento de '{CATEGORIAS[clase]}' de test a train: "
                                f"sin él, el entrenamiento no vería esa clase.")
            tr, te = np.array(sorted(tr)), np.array(sorted(te))

        return tr, te, puede


    idx_train, idx_test, ESTRATIFICADO = partir_dataset(y)
    y_train, y_test = y[idx_train], y[idx_test]

    log.info(f"Split {'estratificado' if ESTRATIFICADO else 'ALEATORIO'}: "
             f"{len(idx_train)} train / {len(idx_test)} test / {N_CLASES} clases")
    print(f"Train: {len(idx_train)} docs | Test: {len(idx_test)} docs | Clases: {N_CLASES}")
    print(f"Estratificado: {'sí' if ESTRATIFICADO else 'NO (alguna clase con un solo documento)'}")
    print(f"Documentos en la categoría menos poblada: {MIN_POR_CLASE}")
    print("Categorías:", CATEGORIAS)
    print("\nDistribución por partición:")
    display(pd.DataFrame({
        "train": pd.Series(le.inverse_transform(y_train)).value_counts(),
        "test": pd.Series(le.inverse_transform(y_test)).value_counts(),
    }).fillna(0).astype(int))

    # @title 5.3.2 — Entrenamiento de los dos modelos
    @cronometrar("entrenamiento del clasificador")
    def entrenar_modelos(cfg: Config = CFG) -> tuple:
        """Entrena los dos modelos de clasificación sobre sus respectivas representaciones.

        Args:
            cfg: Configuración con los hiperparámetros del clasificador.

        Returns:
            Tupla (modelo_a, modelo_b, hiperparametros) donde `hiperparametros` es el
            diccionario que se persiste en `metadata.json` (§5.7).

        Raises:
            ValueError: Si el conjunto de entrenamiento no contiene al menos dos
                clases. El mensaje de scikit-learn en ese caso ("the data contains
                only one class") no dice de dónde viene el problema; este sí.
        """
        clases_train = np.unique(y_train)
        if len(clases_train) < 2:
            raise ValueError(
                f"El conjunto de entrenamiento contiene una sola clase "
                f"('{CATEGORIAS[int(clases_train[0])]}') y no se puede ajustar un clasificador.\n"
                f"Corpus: {len(df)} documentos en {N_CLASES} categoría(s), "
                f"la menos poblada con {MIN_POR_CLASE}.\n"
                f"Causa habitual: el corpus quedó degenerado tras la ingesta. Revisa el "
                f"diagnóstico de §2.4.3, o pon CFG.corpus.usar_fallback = True en §0.4 y "
                f"reejecuta desde §2.3.1."
            )

        c = cfg.clasificacion
        hiper = {
            "algoritmo": "LogisticRegression",
            "C": c.C,
            "max_iter": c.max_iter,
            "class_weight": c.class_weight,
            "solver": "lbfgs",
            "random_state": cfg.random_state,
            "test_size": c.test_size,
            "cv_folds": c.cv_folds,
            "tfidf": asdict(cfg.tfidf),
            "embeddings": asdict(cfg.embeddings),
        }

        log.info("Iniciando entrenamiento del modelo A (TF-IDF + LogReg)")
        modelo_a = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(cfg.tfidf.ngram_min, 2),
                min_df=cfg.tfidf.min_df,
                max_df=cfg.tfidf.max_df,
                sublinear_tf=cfg.tfidf.sublinear_tf,
                max_features=cfg.tfidf.max_features)),
            ("clf", LogisticRegression(
                max_iter=c.max_iter, C=c.C, class_weight=c.class_weight,
                random_state=cfg.random_state)),
        ])
        modelo_a.fit(df["texto_pos"].values[idx_train], y_train)

        log.info("Iniciando entrenamiento del modelo B (SBERT + LogReg)")
        modelo_b = LogisticRegression(
            max_iter=c.max_iter, C=c.C, class_weight=c.class_weight,
            random_state=cfg.random_state)
        modelo_b.fit(embeddings[idx_train], y_train)

        log.info("Entrenamiento completado para ambos modelos")
        return modelo_a, modelo_b, hiper


    with etapa("entrenamiento"):
        modelo_a, modelo_b, HIPERPARAMETROS = entrenar_modelos()

    textos_pos = df["texto_pos"].values
    print("Ambos modelos entrenados.")

    # @title 5.4.1 — Métricas de desempeño comparadas
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 classification_report, confusion_matrix, top_k_accuracy_score)

    pred_a = modelo_a.predict(textos_pos[idx_test])
    proba_a = modelo_a.predict_proba(textos_pos[idx_test])
    pred_b = modelo_b.predict(embeddings[idx_test])
    proba_b = modelo_b.predict_proba(embeddings[idx_test])


    def top_k_seguro(y_verdadero: np.ndarray, probabilidades: np.ndarray,
                     k: int = 2, n_clases: int = None) -> float:
        """Calcula top-k accuracy, devolviendo NaN cuando la métrica no tiene sentido.

        `top_k_accuracy_score` falla con `ValueError` si el problema es binario y las
        probabilidades vienen en dos columnas — que es exactamente lo que produce un
        corpus con solo dos categorías. Pasar `labels` no lo evita: scikit-learn solo
        reinterpreta el problema como multiclase si `len(labels) > 2`.

        Y aunque no fallara, la métrica sería inútil: con `k` mayor o igual al número
        de clases, el top-k acierta siempre por construcción y valdría 1.0.

        Args:
            y_verdadero: Etiquetas reales codificadas.
            probabilidades: Matriz (n_muestras, n_clases) de `predict_proba`.
            k: Número de predicciones principales a considerar.
            n_clases: Total de clases del problema. Por defecto, `N_CLASES`.

        Returns:
            El top-k accuracy, o `float("nan")` si `k >= n_clases`.

        Example:
            >>> top_k_seguro(np.array([0, 1]), np.array([[.6, .4], [.3, .7]]), k=2, n_clases=2)
            nan
        """
        n_clases = n_clases if n_clases is not None else N_CLASES
        if n_clases <= k:
            return float("nan")
        try:
            return top_k_accuracy_score(y_verdadero, probabilidades, k=k,
                                        labels=np.arange(n_clases))
        except ValueError as exc:
            log.warning(f"top-{k} accuracy omitido ({exc}).")
            return float("nan")


    def calcular_metricas(nombre: str, y_verdadero: np.ndarray,
                          y_predicho: np.ndarray, probabilidades: np.ndarray) -> dict:
        """Calcula el conjunto completo de métricas de clasificación de un modelo.

        Args:
            nombre: Etiqueta del modelo, usada como índice de la tabla.
            y_verdadero: Etiquetas reales codificadas.
            y_predicho: Etiquetas predichas codificadas.
            probabilidades: Matriz (n_muestras, n_clases) de `predict_proba`.

        Returns:
            Diccionario con accuracy, precision/recall/F1 en macro y weighted, y
            top-2 accuracy (NaN si el problema es binario).

        Example:
            >>> m = calcular_metricas("A", y_test, pred_a, proba_a)
            >>> 0.0 <= m["f1_macro"] <= 1.0
            True
        """
        comun = dict(zero_division=0)
        return {
            "modelo": nombre,
            "accuracy": accuracy_score(y_verdadero, y_predicho),
            "precision_macro": precision_score(y_verdadero, y_predicho, average="macro", **comun),
            "recall_macro": recall_score(y_verdadero, y_predicho, average="macro", **comun),
            "f1_macro": f1_score(y_verdadero, y_predicho, average="macro", **comun),
            "precision_weighted": precision_score(y_verdadero, y_predicho, average="weighted", **comun),
            "recall_weighted": recall_score(y_verdadero, y_predicho, average="weighted", **comun),
            "f1_weighted": f1_score(y_verdadero, y_predicho, average="weighted", **comun),
            "top2_accuracy": top_k_seguro(y_verdadero, probabilidades, k=2),
        }


    resumen = pd.DataFrame([
        calcular_metricas("A · TF-IDF + LogReg", y_test, pred_a, proba_a),
        calcular_metricas("B · SBERT + LogReg", y_test, pred_b, proba_b),
    ]).set_index("modelo").round(4)

    # Una columna íntegramente NaN estorba más de lo que informa.
    if resumen["top2_accuracy"].isna().all():
        resumen = resumen.drop(columns="top2_accuracy")
        print(f"Nota: top-2 accuracy omitido — el corpus tiene {N_CLASES} categorías y con "
              f"k=2 la métrica valdría 1.0 por construcción.\n")

    # display(resumen.style.background_gradient(cmap="Greens", axis=0))

    metrica = CFG.clasificacion.metrica_decision
    GANADOR = "B" if resumen.iloc[1][metrica] >= resumen.iloc[0][metrica] else "A"
    NOMBRE_GANADOR = resumen.index[1] if GANADOR == "B" else resumen.index[0]

    log.info(f"Modelo seleccionado: {NOMBRE_GANADOR} "
             f"({metrica} = {resumen.loc[NOMBRE_GANADOR, metrica]:.4f})")
    print(f"\n>>> Modelo seleccionado para producción: {NOMBRE_GANADOR}")
    print(f"    Criterio: mayor {metrica}")

    # @title 5.4.1b — Calibración de probabilidades (medida, no supuesta)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.base import clone


    def brier_multiclase(y_verdadero: np.ndarray, probabilidades: np.ndarray,
                         n_clases: int) -> float:
        """Brier score multiclase: error cuadrático medio sobre el vector one-hot.

        Mide calidad de las PROBABILIDADES, no de la decisión. Menor es mejor.
        """
        onehot = np.zeros_like(probabilidades)
        onehot[np.arange(len(y_verdadero)), y_verdadero] = 1.0
        return float(((probabilidades - onehot) ** 2).sum(axis=1).mean())


    MODELO_CALIBRADO = None
    if CFG.clasificacion.calibrar_probabilidades and MIN_POR_CLASE >= 3:
        with etapa("calibración de probabilidades"):
            # Se calibra el modelo GANADOR, sea cual sea, sobre su representación.
            X_train = (embeddings[idx_train] if GANADOR == "B" else textos_pos[idx_train])
            X_test = (embeddings[idx_test] if GANADOR == "B" else textos_pos[idx_test])
            base = clone(modelo_b if GANADOR == "B" else modelo_a)

            n_folds_cal = min(CFG.clasificacion.cv_folds, MIN_POR_CLASE)
            calibrado = CalibratedClassifierCV(
                base, method=CFG.clasificacion.metodo_calibracion, cv=n_folds_cal)
            calibrado.fit(X_train, y_train)

        proba_sin = proba_b if GANADOR == "B" else proba_a
        proba_con = calibrado.predict_proba(X_test)
        n_cl = len(CATEGORIAS)

        comparacion = pd.DataFrame([
            {"modelo": "sin calibrar",
             "brier": brier_multiclase(y_test, proba_sin, n_cl),
             "log_loss": log_loss(y_test, proba_sin, labels=np.arange(n_cl)),
             "f1_macro": f1_score(y_test, proba_sin.argmax(axis=1), average="macro", zero_division=0),
             "prob_media": float(proba_sin.max(axis=1).mean())},
            {"modelo": f"calibrado ({CFG.clasificacion.metodo_calibracion})",
             "brier": brier_multiclase(y_test, proba_con, n_cl),
             "log_loss": log_loss(y_test, proba_con, labels=np.arange(n_cl)),
             "f1_macro": f1_score(y_test, proba_con.argmax(axis=1), average="macro", zero_division=0),
             "prob_media": float(proba_con.max(axis=1).mean())},
        ]).set_index("modelo").round(4)

    #     display(comparacion)

        mejora_brier = comparacion.iloc[0]["brier"] - comparacion.iloc[1]["brier"]
        if mejora_brier > 0:
            MODELO_CALIBRADO = calibrado
            log.info(f"Calibración adoptada: Brier mejora {mejora_brier:+.4f}")
            print(f"\n  Calibración ADOPTADA — el Brier score mejora {mejora_brier:+.4f}.")
            print(f"  Confianza media: {comparacion.iloc[0]['prob_media']:.3f} -> "
                  f"{comparacion.iloc[1]['prob_media']:.3f}")
            print(f"  El F1 apenas se mueve ({comparacion.iloc[0]['f1_macro']:.4f} -> "
                  f"{comparacion.iloc[1]['f1_macro']:.4f}): la calibración reescala las")
            print(f"  probabilidades sin alterar el orden de las predicciones. Es lo esperado.")
        else:
            log.warning(f"Calibración descartada: el Brier empeora {mejora_brier:+.4f}")
            print(f"\n  Calibración DESCARTADA — empeora el Brier score en {-mejora_brier:.4f}.")
            print(f"  Con este tamaño de corpus el ajuste de calibración no tiene datos")
            print(f"  suficientes. Se conserva el modelo original.")

        print(f"\n  Recordatorio: con {n_cl} categorías el azar está en {1/n_cl:.3f}.")
        print(f"  El umbral de confianza baja está en {CFG.clasificacion.umbral_confianza_baja}, "
              f"≈{CFG.clasificacion.umbral_confianza_baja * n_cl:.1f}x el azar.")
    else:
        print("Calibración omitida (desactivada o categorías con menos de 3 documentos).")

    # @title 5.4.2 — Classification report y matrices de confusión
    pred_ganador = pred_b if GANADOR == "B" else pred_a

    print(f"CLASSIFICATION REPORT — {NOMBRE_GANADOR}\n")
    print(classification_report(y_test, pred_ganador, target_names=CATEGORIAS,
                                digits=3, zero_division=0))

    reporte_dict = classification_report(y_test, pred_ganador, target_names=CATEGORIAS,
                                         output_dict=True, zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(19, 7))
    for ax, pred, nombre in zip(axes, [pred_a, pred_b], ["A · TF-IDF", "B · SBERT"]):
        cm = confusion_matrix(y_test, pred, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=CATEGORIAS, yticklabels=CATEGORIAS, vmin=0, vmax=1)
        ax.set_title(f"Matriz de confusión normalizada — {nombre}", fontweight="bold")
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=45); ax.tick_params(axis="y", rotation=0)
    plt.tight_layout(); plt.show()

    # Lectura dirigida: los pares que más se confunden entre sí
    cm_ganador = confusion_matrix(y_test, pred_ganador, normalize="true")
    confusiones = [
        (CATEGORIAS[i], CATEGORIAS[j], cm_ganador[i, j])
        for i in range(len(CATEGORIAS)) for j in range(len(CATEGORIAS))
        if i != j and cm_ganador[i, j] > 0
    ]
    confusiones.sort(key=lambda t: -t[2])
    if confusiones:
        print("Pares más confundidos (real → predicho):")
        for real, predicho, tasa in confusiones[:5]:
            print(f"  {real:<18} → {predicho:<18} {tasa:.1%}")

    # @title 5.4.3 — Validación cruzada estratificada
    @cronometrar("validación cruzada")
    def validar_cruzado(cfg: Config = CFG) -> dict:
        """Ejecuta validación cruzada estratificada, adaptando los folds al corpus.

        `StratifiedKFold` exige al menos un documento por clase en cada fold: con
        `n_splits=5` y una categoría de 3 documentos, falla. En vez de abortar,
        reducimos los folds al máximo viable y lo registramos — un 3-fold sobre un
        corpus pequeño sigue siendo más informativo que un único split.

        Args:
            cfg: Configuración con el número de folds y la métrica de decisión.

        Returns:
            Diccionario con las claves "A", "B" (arrays de puntajes por fold) y
            "n_folds" (los efectivamente usados).
        """
        n_folds = min(cfg.clasificacion.cv_folds, MIN_POR_CLASE)
        if n_folds < cfg.clasificacion.cv_folds:
            log.warning(f"Validación cruzada reducida a {n_folds} folds: la categoría menos "
                        f"poblada tiene {MIN_POR_CLASE} documento(s).")
        if n_folds < 2:
            log.error("Imposible validar de forma cruzada: alguna categoría tiene un solo "
                      "documento. Se omite §5.4.3.")
            return {"A": np.array([]), "B": np.array([]), "n_folds": 0}

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.random_state)
        puntajes_a = cross_val_score(modelo_a, textos_pos, y, cv=cv,
                                     scoring=cfg.clasificacion.metrica_decision)
        puntajes_b = cross_val_score(
            LogisticRegression(max_iter=cfg.clasificacion.max_iter, C=cfg.clasificacion.C,
                               class_weight=cfg.clasificacion.class_weight,
                               random_state=cfg.random_state),
            embeddings, y, cv=cv, scoring=cfg.clasificacion.metrica_decision)
        return {"A": puntajes_a, "B": puntajes_b, "n_folds": n_folds}


    with etapa("validación cruzada"):
        CV = validar_cruzado()

    N_FOLDS = CV["n_folds"]
    if N_FOLDS == 0:
        print("Validación cruzada omitida: el corpus no tiene documentos suficientes por categoría.")
        print("Corrige la ingesta (§2.4.3) antes de interpretar las métricas de §5.4.1.")

    print(f"{CFG.clasificacion.metrica_decision} en validación cruzada "
          f"({N_FOLDS}-fold estratificado):\n")
    if N_FOLDS >= 2:
        for clave, nombre in [("A", "A · TF-IDF + LogReg"), ("B", "B · SBERT  + LogReg")]:
            p = CV[clave]
            print(f"  {nombre} : {p.mean():.4f} ± {p.std():.4f}   {np.round(p, 3)}")

        estabilidad = float(CV[GANADOR].std())
        if estabilidad > 0.10:
            log.warning(f"Desviación entre folds alta ({estabilidad:.3f}): "
                        f"el resultado no es estable con este tamaño de corpus.")
            print(f"\n  AVISO: desviación de {estabilidad:.3f} entre folds. La media de arriba "
                  f"no debe presentarse como un valor puntual.")

        # --- Contraste entre el conjunto de prueba y la validación cruzada ---
        # Es la comprobación más importante de esta celda. Un único split de test
        # sobre un corpus pequeño puede salir afortunado: el modelo parece mucho
        # mejor de lo que es. La validación cruzada usa todos los datos como test
        # exactamente una vez, así que su media es la estimación fiable.
        media_cv = float(CV[GANADOR].mean())
        metrica_test = float(resumen.loc[NOMBRE_GANADOR, CFG.clasificacion.metrica_decision])
        brecha = metrica_test - media_cv

        # Umbral: hay DOS fuentes de variación y compararse solo con una da falsos
        # positivos. La desviación entre folds mide cuánto varía el modelo según qué
        # datos se entrenan; el error estándar del test mide cuánto varía la
        # medición por tener solo `len(idx_test)` documentos. Con folds muy estables
        # —una desviación de 0.008 se ha visto— el criterio antiguo saltaba ante
        # diferencias perfectamente normales de muestreo. Se toma la mayor de las dos.
        error_estandar_test = float(np.sqrt(
            max(metrica_test * (1 - metrica_test), 1e-6) / max(len(idx_test), 1)))
        umbral = 2 * max(estabilidad, error_estandar_test)

        print(f"\nCONTRASTE TEST ↔ VALIDACIÓN CRUZADA")
        print("-" * 62)
        print(f"  {CFG.clasificacion.metrica_decision} en test : {metrica_test:.4f} "
              f"(error estándar ≈ {error_estandar_test:.4f} con {len(idx_test)} documentos)")
        print(f"  {CFG.clasificacion.metrica_decision} en CV   : {media_cv:.4f} ± {estabilidad:.4f}")
        print(f"  diferencia          : {brecha:+.4f}   (umbral de alerta: ±{umbral:.4f})")

        if abs(brecha) > umbral:
            log.warning(f"El test ({metrica_test:.4f}) se aparta de la CV ({media_cv:.4f}) "
                        f"más de dos desviaciones: no es representativo.")
            print(f"\n  ATENCIÓN: la diferencia supera dos desviaciones típicas.")
            if brecha > 0:
                print(f"  El split de prueba salió FAVORABLE. Con {len(idx_test)} documentos de")
                print(f"  test, ese {metrica_test:.4f} no es reproducible.")
                print(f"  → Presenta {media_cv:.4f} ± {estabilidad:.4f} (validación cruzada),")
                print(f"    que es la estimación honesta. Reportar el test sería inflar el")
                print(f"    resultado, y un jurado técnico lo detecta comparando ambas cifras.")
            else:
                print(f"  El split de prueba salió DESFAVORABLE: el modelo es mejor de lo que")
                print(f"  sugiere el test. Usa igualmente la cifra de validación cruzada.")
            print(f"\n  Causa de fondo: con {len(df)} documentos el conjunto de prueba es")
            print(f"  demasiado pequeño para estimar de forma estable. Ampliar el corpus")
            print(f"  reduce esta brecha más que cualquier ajuste del modelo.")
        else:
            print(f"\n  Ambas medidas concuerdan: el resultado es estable y {metrica_test:.4f}")
            print(f"  se puede presentar como el desempeño real del modelo.")

        plt.figure(figsize=(7, 4))
        _cv_largo = pd.DataFrame({"A · TF-IDF": CV["A"], "B · SBERT": CV["B"]}).melt(
            var_name="modelo", value_name=CFG.clasificacion.metrica_decision)
        sns.boxplot(data=_cv_largo, x="modelo", y=CFG.clasificacion.metrica_decision,
                    hue="modelo", palette="Set2", legend=False)
        plt.title(f"Distribución de {CFG.clasificacion.metrica_decision} por fold "
                  f"({N_FOLDS} folds)", fontweight="bold")
        plt.ylabel(CFG.clasificacion.metrica_decision)
        plt.tight_layout(); plt.show()

    # @title 5.5.1 — Explicabilidad global: qué términos usa el modelo TF-IDF
    vect_a = modelo_a.named_steps["tfidf"]
    clf_a = modelo_a.named_steps["clf"]
    vocab_a = np.array(vect_a.get_feature_names_out())

    n_cols = 3
    n_filas = int(np.ceil(len(CATEGORIAS) / n_cols))
    fig, axes = plt.subplots(n_filas, n_cols, figsize=(6 * n_cols, 3.2 * n_filas))
    ejes = np.array(axes).ravel()

    for ax, (i, cat) in zip(ejes, enumerate(CATEGORIAS)):
        coef = clf_a.coef_[i]
        top = coef.argsort()[::-1][:CFG.clasificacion.n_terminos_explicabilidad]
        sns.barplot(x=coef[top], y=vocab_a[top], ax=ax,
                    hue=vocab_a[top], palette="crest", legend=False)
        ax.set_title(cat, fontweight="bold"); ax.set_xlabel("coeficiente")
    for ax in ejes[len(CATEGORIAS):]:
        ax.axis("off")

    plt.suptitle("Explicabilidad global — términos con mayor peso positivo por categoría",
                 y=1.005, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()

    # @title 5.5.2 — Explicabilidad local: ablación por término y centroides de clase
    CENTROIDES = np.vstack([
        embeddings[y == i].mean(axis=0) / np.linalg.norm(embeddings[y == i].mean(axis=0))
        for i in range(len(CATEGORIAS))
    ])


    def _predecir_probabilidades(texto_limpio: str, texto_pos: str) -> np.ndarray:
        """Devuelve la distribución de probabilidad sobre categorías del modelo ganador."""
        if GANADOR == "B":
            vector = CACHE.codificar([texto_limpio], modelo_embeddings)
            return modelo_b.predict_proba(vector)[0]
        return modelo_a.predict_proba([texto_pos])[0]


    def explicar_prediccion(texto_limpio: str,
                            texto_pos: str,
                            candidatos: Sequence,
                            max_terminos: int = 6,
                            *,
                            predictor: Callable = None,
                            centroides: np.ndarray = None,
                            categorias: Sequence = None,
                            codificador: Callable = None) -> dict:
        """Explica una predicción por ablación de términos y similitud a centroides.

        Para cada término candidato, elimina sus ocurrencias del texto, recalcula la
        probabilidad de la categoría originalmente predicha y reporta la diferencia.
        Una diferencia positiva significa que el término sostenía la decisión; una
        negativa, que el modelo decidió a pesar de él.

        Los parámetros de sólo-palabra clave permiten inyectar las dependencias en vez
        de leerlas del ámbito global, para que la capa de inferencia (§7.2) sea
        autocontenida al importarse desde el backend.

        Args:
            texto_limpio: Texto normalizado, entrada del modelo semántico.
            texto_pos: Lemas filtrados por POS, entrada del modelo léxico.
            candidatos: Términos a evaluar (típicamente las keywords y entidades).
            max_terminos: Tope de términos a ablacionar, para acotar el costo.
            predictor: Callable (texto_limpio, texto_pos) -> distribución de probabilidad.
            centroides: Matriz (n_categorias, dim) de centroides normalizados.
            categorias: Nombres de las categorías, alineados con `centroides`.
            codificador: Callable (list[str]) -> matriz de embeddings.

        Returns:
            Diccionario con `categoria`, `probabilidad`, `metodo`, `terminos_a_favor`,
            `terminos_en_contra` y `similitud_centroides`.

        Example:
            >>> exp = explicar_prediccion(texto, texto_pos, ["Java", "Spring Boot"])
            >>> exp["terminos_a_favor"][0]["termino"]
            'Spring Boot'
        """
        predictor = predictor or _predecir_probabilidades
        centroides = centroides if centroides is not None else CENTROIDES
        categorias = list(categorias) if categorias is not None else CATEGORIAS
        codificador = codificador or (lambda ts: CACHE.codificar(ts, modelo_embeddings))

        probas = predictor(texto_limpio, texto_pos)
        idx_pred = int(np.argmax(probas))
        p0 = float(probas[idx_pred])

        contribuciones = []
        for termino in list(candidatos)[:max_terminos]:
            patron = re.compile(re.escape(str(termino)), re.IGNORECASE)
            limpio_sin = patron.sub(" ", texto_limpio)
            pos_sin = patron.sub(" ", texto_pos)
            if limpio_sin.strip() == texto_limpio.strip():
                continue   # el término no aparecía literalmente; ablacionarlo no informa
            p_sin = float(predictor(limpio_sin, pos_sin)[idx_pred])
            contribuciones.append({"termino": str(termino),
                                   "contribucion": round(p0 - p_sin, 4)})

        contribuciones.sort(key=lambda d: -d["contribucion"])

        vector = codificador([texto_limpio])[0]
        similitudes = centroides @ vector
        ranking = np.argsort(similitudes)[::-1][:3]

        return {
            "categoria": categorias[idx_pred],
            "probabilidad": round(p0, 4),
            "metodo": "ablación de términos sobre el modelo en producción + centroides de clase",
            "terminos_a_favor": [c for c in contribuciones if c["contribucion"] > 0],
            "terminos_en_contra": [c for c in contribuciones if c["contribucion"] < 0][::-1],
            "similitud_centroides": [
                {"categoria": categorias[i], "similitud": round(float(similitudes[i]), 4)}
                for i in ranking
            ],
        }


    # --- Demostración sobre el ejemplo del brief ------------------------------
    _texto_ej = limpiar_texto(EJEMPLO)
    _doc_ej = nlp(_texto_ej)
    _proc_ej = preprocesar(_doc_ej)
    _kws_ej = rankear_keywords(_texto_ej, doc_spacy=_doc_ej)

    explicacion = explicar_prediccion(_texto_ej, _proc_ej["texto_pos"], _kws_ej)

    print(f"DOCUMENTO : {EJEMPLO[:88]}...")
    print(f"PREDICCIÓN: {explicacion['categoria']} (probabilidad {explicacion['probabilidad']})")
    print(f"MÉTODO    : {explicacion['metodo']}\n")

    print("Términos que SOSTIENEN la categoría (la probabilidad cae al eliminarlos):")
    for c in explicacion["terminos_a_favor"]:
        print(f"  +{c['contribucion']:.4f}  {c['termino']:<24} "
              f"{'█' * max(1, int(c['contribucion'] * 120))}")
    if not explicacion["terminos_a_favor"]:
        print("  (ninguno: la decisión no depende de un término aislado)")

    if explicacion["terminos_en_contra"]:
        print("\nTérminos que RESTAN a la categoría (la probabilidad sube al eliminarlos):")
        for c in explicacion["terminos_en_contra"]:
            print(f"  {c['contribucion']:.4f}  {c['termino']:<24} "
                  f"{'▒' * max(1, int(abs(c['contribucion']) * 120))}")
        print("  → El modelo eligió esta categoría A PESAR de estos términos. Si alguno es")
        print("    técnicamente central al tema, indica corpus pobre en ese concepto.")

    print("\nSimilitud contra los centroides de cada categoría:")
    for s in explicacion["similitud_centroides"]:
        print(f"  {s['categoria']:<22} {s['similitud']:.4f}")
    if explicacion["similitud_centroides"][0]["categoria"] != explicacion["categoria"]:
        print("\n  AVISO: el centroide más cercano NO coincide con la categoría predicha.")
        print("  Las dos señales discrepan — documento probablemente ambiguo o fronterizo.")

    # @title 5.6.1 — KMeans: selección de k por codo y coeficiente de silueta
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    with etapa("clustering KMeans"):
        # El barrido de k debe caber en el corpus: KMeans exige k <= n_documentos, y
        # el coeficiente de silueta necesita al menos 2 grupos. Con un corpus pequeño
        # el rango original quedaba vacío y `np.argmax([])` fallaba.
        k_tope = min(CFG.clustering.k_max, max(2, len(df) // 5), len(df) - 1)
        rango_k = list(range(CFG.clustering.k_min, k_tope + 1))

        if len(rango_k) < 1:
            log.warning(f"Corpus de {len(df)} documentos: insuficiente para barrer k. "
                        f"Se usa k=2 sin selección.")
            rango_k = [2]

        inercias, siluetas = [], []
        for k in rango_k:
            km = KMeans(n_clusters=k, random_state=CFG.random_state,
                        n_init=CFG.clustering.kmeans_n_init).fit(embeddings)
            inercias.append(km.inertia_)
            siluetas.append(silhouette_score(embeddings, km.labels_)
                            if k < len(df) else float("nan"))

        K_OPTIMO = rango_k[int(np.nanargmax(siluetas))] if len(rango_k) > 1 else rango_k[0]
        kmeans = KMeans(n_clusters=K_OPTIMO, random_state=CFG.random_state,
                        n_init=CFG.clustering.kmeans_n_init).fit(embeddings)
        df["cluster_kmeans"] = kmeans.labels_

    if len(rango_k) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
        axes[0].plot(rango_k, inercias, "o-", color="#2a6f97")
        axes[0].set_title("Método del codo", fontweight="bold")
        axes[0].set_xlabel("k"); axes[0].set_ylabel("inercia")
        axes[1].plot(rango_k, siluetas, "o-", color="#e07a5f")
        axes[1].axvline(K_OPTIMO, ls="--", color="crimson", label=f"k óptimo = {K_OPTIMO}")
        axes[1].set_title("Coeficiente de silueta", fontweight="bold")
        axes[1].set_xlabel("k"); axes[1].legend()
        plt.tight_layout(); plt.show()
        print("Obsérvese que el codo no es nítido: es exactamente la ambigüedad que BERTopic evita.")
    else:
        print(f"Barrido de k omitido: el corpus ({len(df)} documentos) no admite explorar "
              f"un rango de valores.")

    log.info(f"KMeans: k={K_OPTIMO}, silueta={np.nanmax(siluetas):.4f}")
    print(f"KMeans con k={K_OPTIMO} | silueta máxima = {np.nanmax(siluetas):.4f}")

    # @title 5.6.2 — BERTopic: descubrimiento automático de tópicos
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer


    @cronometrar("clustering BERTopic")
    def entrenar_bertopic(textos: list, vectores: np.ndarray, cfg: Config = CFG) -> tuple:
        """Ajusta BERTopic sobre embeddings ya calculados.

        `min_topic_size` se escala con el tamaño del corpus: con el valor por defecto,
        HDBSCAN marcaría como ruido una fracción excesiva de un corpus pequeño
        (Technology_Architecture.md §8).

        Args:
            textos: Lista de documentos limpios.
            vectores: Matriz de embeddings correspondiente, para no recalcularla.
            cfg: Configuración de clustering.

        Returns:
            Tupla (modelo entrenado, lista de tópicos por documento, matriz de probabilidades).
        """
        vectorizador = CountVectorizer(
            stop_words=_STOPWORDS_SPACY,
            ngram_range=(1, cfg.clustering.bertopic_ngram_max),
            min_df=cfg.clustering.bertopic_min_df,
        )
        # UMAP explícito: con `n_neighbors` bajo se preserva la estructura LOCAL,
        # que es lo que HDBSCAN necesita para no marcar como ruido los nichos
        # pequeños. Con el valor por defecto (15) el 22 % del corpus quedaba fuera.
        from umap import UMAP
        reductor = UMAP(n_neighbors=cfg.clustering.umap_n_neighbors,
                        n_components=5, min_dist=0.0, metric="cosine",
                        random_state=cfg.random_state)

        modelo = BERTopic(
            embedding_model=modelo_embeddings,
            umap_model=reductor,
            vectorizer_model=vectorizador,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            min_topic_size=max(cfg.clustering.bertopic_min_topic_size_piso,
                               len(textos) // cfg.clustering.bertopic_min_topic_size_divisor),
            calculate_probabilities=True,
            verbose=False,
            language="multilingual",
        )
        topicos, probabilidades = modelo.fit_transform(textos, vectores)
        return modelo, topicos, probabilidades


    def rescatar_outliers(topicos: list, etiquetas_kmeans: np.ndarray,
                          cfg: Config = CFG) -> tuple:
        """Asigna un tópico aproximado a los documentos marcados como ruido.

        HDBSCAN deja fuera los documentos que no encajan en ningún grupo denso
        —hasta un 22 % del corpus en una corrida—. Para un sistema de organización
        del conocimiento eso es un agujero: esos documentos se quedan sin campo
        `tema`. Se les asigna el tópico dominante de su cluster de KMeans, que sí
        particiona el espacio completo.

        La asignación se marca como aproximada: no es lo mismo que HDBSCAN haya
        encontrado densidad ahí.

        Args:
            topicos: Lista de tópicos por documento, con -1 para los outliers.
            etiquetas_kmeans: Cluster de KMeans de cada documento.
            cfg: Configuración; si el rescate está desactivado, no toca nada.

        Returns:
            Tupla (tópicos rescatados, número de documentos rescatados).
        """
        if not cfg.clustering.asignar_outliers_con_kmeans:
            return list(topicos), 0

        topicos = list(topicos)
        # Tópico mayoritario dentro de cada cluster de KMeans, ignorando el ruido.
        dominante = {}
        for cluster in np.unique(etiquetas_kmeans):
            votos = [t for t, c in zip(topicos, etiquetas_kmeans) if c == cluster and t != -1]
            if votos:
                dominante[cluster] = Counter(votos).most_common(1)[0][0]

        rescatados = 0
        for i, t in enumerate(topicos):
            if t == -1 and etiquetas_kmeans[i] in dominante:
                topicos[i] = dominante[etiquetas_kmeans[i]]
                rescatados += 1
        return topicos, rescatados


    with etapa("descubrimiento de tópicos"):
        topic_model, topicos, probs_topicos = entrenar_bertopic(
            df["texto_limpio"].tolist(), embeddings)
        n_outliers_hdbscan = int(sum(1 for t in topicos if t == -1))
        topicos, N_RESCATADOS = rescatar_outliers(topicos, df["cluster_kmeans"].values)
        df["topico_bertopic"] = topicos
        df["tema_aproximado"] = [
            o == -1 and n != -1 for o, n in zip(
                [-1 if i < n_outliers_hdbscan else 0 for i in range(len(df))], topicos)]

    info_topicos = topic_model.get_topic_info()
    n_outliers = int((df["topico_bertopic"] == -1).sum())

    log.info(f"BERTopic: {len(info_topicos) - 1} tópicos, {n_outliers} outliers "
             f"({n_outliers / len(df):.1%})")
    if n_outliers / len(df) > 0.35:
        log.warning("Más del 35% del corpus quedó como outlier: considera KMeans como respaldo.")

    print(f"Tópicos descubiertos: {len(info_topicos) - 1} (excluyendo el tópico de ruido)")
    print(f"Outliers de HDBSCAN  : {n_outliers_hdbscan} "
          f"({n_outliers_hdbscan / len(df):.1%} del corpus)")
    if N_RESCATADOS:
        print(f"Rescatados con KMeans: {N_RESCATADOS} -> quedan {n_outliers} sin tema "
              f"({n_outliers / len(df):.1%})")
        print(f"  Los rescatados llevan un tema APROXIMADO: se les asignó el tópico")
        print(f"  dominante de su cluster de KMeans, no una densidad hallada por HDBSCAN.\n")
    else:
        print()
    # display(info_topicos.head(12)[["Topic", "Count", "Name"]])

    # @title 5.6.3 — Etiquetas legibles por tópico (c-TF-IDF) y contraste con las categorías
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ETIQUETAS_TOPICO = {
        t: ", ".join(w for w, _ in topic_model.get_topic(t)[:4])
        for t in sorted(set(topicos)) if t != -1
    }
    ETIQUETAS_TOPICO[-1] = "(sin tema definido / outlier)"

    print("PALABRAS REPRESENTATIVAS POR TÓPICO (generadas automáticamente vía c-TF-IDF)\n")
    for t in sorted(set(topicos)):
        if t == -1:
            continue
        palabras = ", ".join(w for w, _ in
                             topic_model.get_topic(t)[:CFG.clustering.n_palabras_por_topico])
        sub = df[df["topico_bertopic"] == t]
        modo = sub["categoria"].mode()
        cat_dominante = modo.iloc[0] if len(modo) else "—"
        pureza = (sub["categoria"] == cat_dominante).mean() if len(sub) else 0
        print(f"Tópico {t:>2} ({len(sub):>3} docs) │ {palabras}")
        print(f"{'':>17}└─ categoría dominante: {cat_dominante} (pureza {pureza:.0%})\n")

    mask = df["topico_bertopic"] != -1
    ari_bt = adjusted_rand_score(df.loc[mask, "categoria"], df.loc[mask, "topico_bertopic"])
    nmi_bt = normalized_mutual_info_score(df.loc[mask, "categoria"], df.loc[mask, "topico_bertopic"])
    ari_km = adjusted_rand_score(df["categoria"], df["cluster_kmeans"])
    nmi_km = normalized_mutual_info_score(df["categoria"], df["cluster_kmeans"])

    print("Concordancia entre el clustering no supervisado y las etiquetas de la taxonomía:")
    print(f"  BERTopic  ARI = {ari_bt:.4f} | NMI = {nmi_bt:.4f}")
    print(f"  KMeans    ARI = {ari_km:.4f} | NMI = {nmi_km:.4f}")
    print("\nARI/NMI cercanos a 0 = agrupación independiente de la taxonomía impuesta;")
    print("cercanos a 1 = el clustering reconstruye por sí solo las categorías etiquetadas.")

    # @title 5.6.4 — Visualización del espacio semántico (proyección 2D)
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=CFG.random_state).fit_transform(embeddings)
    df["x"], df["y"] = coords[:, 0], coords[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, col, titulo in zip(
        axes,
        ["categoria", "cluster_kmeans", "topico_bertopic"],
        ["Etiquetas reales (supervisado)", f"KMeans (k={K_OPTIMO})", "BERTopic (automático)"],
    ):
        sns.scatterplot(data=df, x="x", y="y", hue=col, palette="tab10", s=45,
                        alpha=0.8, ax=ax, legend="brief")
        ax.set_title(titulo, fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("")
        ax.legend(fontsize=7, loc="best", ncol=2)

    plt.suptitle("Espacio de embeddings proyectado a 2D (PCA)", y=1.02, fontsize=14,
                 fontweight="bold")
    plt.tight_layout(); plt.show()

    varianza = PCA(n_components=2, random_state=CFG.random_state).fit(embeddings)
    print(f"Varianza explicada por las 2 componentes: "
          f"{varianza.explained_variance_ratio_.sum():.1%} de 384 dimensiones originales.")
    print("Una fracción baja es esperable: la proyección es orientativa, no una medida de separabilidad.")

    # @title 5.7 — Guardado de artefactos con versionado completo
    def hash_dataset(df: pd.DataFrame, columnas: Sequence = ("doc_id", "texto_limpio", "categoria")) -> str:
        """Calcula un hash reproducible del contenido del corpus.

        Args:
            df: DataFrame del corpus.
            columnas: Columnas que definen la identidad del dataset.

        Returns:
            Hash SHA-256 hexadecimal del contenido ordenado.
        """
        sub = df[list(columnas)].sort_values("doc_id")
        payload = sub.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


    @cronometrar("serialización de artefactos")
    def serializar_artefactos(cfg: Config = CFG) -> dict:
        """Guarda todos los artefactos del pipeline con sus metadatos de versionado.

        Args:
            cfg: Configuración del pipeline.

        Returns:
            Diccionario {nombre_artefacto: descripción}.
        """
        artefactos: dict = {}
        destino = cfg.rutas.models

        # 1 — Clasificador ganador
        modelo_final = modelo_b if GANADOR == "B" else modelo_a
        tipo_clf = "sbert+logreg" if GANADOR == "B" else "tfidf+logreg"
        joblib.dump(modelo_final, destino / "modelo_clasificacion.joblib")
        artefactos["modelo_clasificacion.joblib"] = f"Clasificador temático ({tipo_clf})"

        # 2 — Codificador de etiquetas
        joblib.dump(le, destino / "label_encoder.joblib")
        artefactos["label_encoder.joblib"] = "Mapeo índice ↔ nombre de categoría"

        # 3 — Vectorizador TF-IDF del corpus (keywords estadísticas y modelo A)
        joblib.dump(tfidf, destino / "vectorizador_tfidf.joblib")
        artefactos["vectorizador_tfidf.joblib"] = "TfidfVectorizer ajustado al corpus"

        # 4 — KMeans (respaldo de clustering documentado)
        joblib.dump(kmeans, destino / "modelo_kmeans.joblib")
        artefactos["modelo_kmeans.joblib"] = f"KMeans k={K_OPTIMO} sobre embeddings"

        # 5 — Centroides de clase (explicabilidad local sin recalcular el corpus)
        joblib.dump({"centroides": CENTROIDES, "categorias": CATEGORIAS},
                    destino / "centroides_clase.joblib")
        artefactos["centroides_clase.joblib"] = "Centroides por categoría (explicabilidad §5.5)"

        # 6 — BERTopic en formato nativo (safetensors, sin duplicar el modelo de embeddings)
        topic_model.save(str(destino / "modelo_bertopic"), serialization="safetensors",
                         save_ctfidf=True, save_embedding_model=cfg.embeddings.modelo)
        artefactos["modelo_bertopic/"] = "Modelo de tópicos + c-TF-IDF"

        # 7 — Configuración completa
        cfg.guardar(destino / "config.json")
        artefactos["config.json"] = "Configuración exacta usada en el entrenamiento"

        # 8 — Metadatos: el contrato Data Science ↔ Backend
        fila_ganadora = resumen.loc[NOMBRE_GANADOR]
        metadatos = {
            "version": cfg.version,
            "huella_configuracion": cfg.huella(),
            "fecha_entrenamiento": pd.Timestamp.now().isoformat(),

            "modelo": {
                "tipo_clasificador": tipo_clf,
                "representacion": "embeddings SBERT" if GANADOR == "B" else "TF-IDF sobre lemas POS",
                "modelo_embeddings": cfg.embeddings.modelo,
                "dimension_embeddings": int(embeddings.shape[1]),
                "modelo_spacy": cfg.idioma.modelos_spacy[cfg.idioma.idioma_objetivo],
                "idiomas_soportados": list(cfg.idioma.idiomas_soportados),
                "categorias": CATEGORIAS,
                "n_topicos_bertopic": int(len(info_topicos) - 1),
                "k_kmeans": int(K_OPTIMO),
            },

            "hiperparametros": HIPERPARAMETROS,

            "metricas": {
                "conjunto_prueba": {k: float(fila_ganadora[k]) for k in resumen.columns},
                "validacion_cruzada": {
                    "metrica": cfg.clasificacion.metrica_decision,
                    "folds_configurados": cfg.clasificacion.cv_folds,
                    "folds_efectivos": N_FOLDS,
                    "media": float(CV[GANADOR].mean()) if N_FOLDS >= 2 else None,
                    "desviacion": float(CV[GANADOR].std()) if N_FOLDS >= 2 else None,
                    "por_fold": [round(float(v), 4) for v in CV[GANADOR]],
                },
                "por_categoria": {
                    cat: {m: round(float(v), 4) for m, v in vals.items()}
                    for cat, vals in reporte_dict.items()
                    if cat in CATEGORIAS
                },
                "clustering": {
                    "ari_bertopic": round(float(ari_bt), 4),
                    "nmi_bertopic": round(float(nmi_bt), 4),
                    "ari_kmeans": round(float(ari_km), 4),
                    "nmi_kmeans": round(float(nmi_km), 4),
                    "silueta_kmeans": round(float(max(siluetas)), 4),
                    "ratio_outliers": round(n_outliers / len(df), 4),
                },
            },

            "dataset": {
                "nombre": "TechMind corpus técnico ES",
                "fuente": FUENTE_CORPUS,
                "respaldo_automatico_activado": RESPALDO_ACTIVADO,
                "hash_sha256": hash_dataset(df),
                "n_documentos": int(len(df)),
                "n_documentos_train": int(len(idx_train)),
                "n_documentos_test": int(len(idx_test)),
                "n_categorias": len(CATEGORIAS),
                "distribucion_categorias": df["categoria"].value_counts().to_dict(),
                "idiomas": df["idioma"].value_counts().to_dict(),
                "n_caracteres_total": int(df["n_chars"].sum()),
                "documentos_rechazados": int(len(df_rechazados)),
                "reporte_deduplicacion": REPORTE_DEDUP,
            },

            "entorno": {"versiones": VERSIONES, "semillas": ESTADO_SEMILLAS},
        }

        ruta_meta = destino / "metadata.json"
        ruta_meta.write_text(json.dumps(metadatos, ensure_ascii=False, indent=2,
                                        default=str), encoding="utf-8")
        artefactos["metadata.json"] = "Contrato de versionado Data Science ↔ Backend"

        return artefactos, metadatos


    with etapa("serialización"):
        ARTEFACTOS, METADATOS = serializar_artefactos()

    print(f"ARTEFACTOS SERIALIZADOS EN {CFG.rutas.models}\n")
    for nombre, descripcion in ARTEFACTOS.items():
        ruta = CFG.rutas.models / nombre
        tam = (sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file())
               if ruta.is_dir() else ruta.stat().st_size)
        print(f"  {nombre:<32} {tam / 1024:>9,.1f} KB   {descripcion}")

    print(f"\nVERSIONADO DEL MODELO")
    print("-" * 62)
    print(f"  versión              : {METADATOS['version']}")
    print(f"  fecha entrenamiento  : {METADATOS['fecha_entrenamiento'][:19]}")
    print(f"  modelo               : {METADATOS['modelo']['tipo_clasificador']}")
    print(f"  f1_macro (test)      : {METADATOS['metricas']['conjunto_prueba']['f1_macro']:.4f}")
    _cv_meta = METADATOS["metricas"]["validacion_cruzada"]
    if _cv_meta["media"] is not None:
        print(f"  f1_macro (CV)        : {_cv_meta['media']:.4f} ± {_cv_meta['desviacion']:.4f} "
              f"({_cv_meta['folds_efectivos']} folds)")
    else:
        print(f"  f1_macro (CV)        : no disponible (corpus insuficiente)")
    print(f"  dataset (hash)       : {METADATOS['dataset']['hash_sha256'][:24]}...")
    print(f"  documentos           : {METADATOS['dataset']['n_documentos']}")
    print(f"  huella configuración : {METADATOS['huella_configuracion'][:24]}...")

    # @title 6.1 — Poblado de la base de conocimiento vectorial (ChromaDB)
    import chromadb


    @cronometrar("indexación en ChromaDB")
    def poblar_chromadb(df: pd.DataFrame, vectores: np.ndarray, cfg: Config = CFG):
        """Crea la colección vectorial y la puebla con el corpus completo.

        Args:
            df: Corpus con `doc_id`, `titulo`, `categoria`, `topico_bertopic` y `cluster_kmeans`.
            vectores: Matriz de embeddings alineada con las filas de `df`.
            cfg: Configuración de la base vectorial.

        Returns:
            La colección de ChromaDB poblada.
        """
        cliente = chromadb.PersistentClient(path=str(cfg.rutas.chroma))
        try:
            cliente.delete_collection(cfg.vectorial.nombre_coleccion)
            log.debug("Colección previa eliminada para reindexar desde cero.")
        except Exception:
            pass

        coleccion = cliente.create_collection(
            name=cfg.vectorial.nombre_coleccion,
            metadata={"hnsw:space": cfg.vectorial.metrica},
        )
        coleccion.add(
            ids=df["doc_id"].tolist(),
            embeddings=vectores.tolist(),
            documents=df["texto_limpio"].tolist(),
            metadatas=[
                {
                    "titulo": str(r["titulo"]),
                    "categoria": str(r["categoria"]),
                    "idioma": str(r["idioma"]),
                    "topico": int(r["topico_bertopic"]),
                    "cluster": int(r["cluster_kmeans"]),
                    "fuente": str(r.get("fuente", "")),
                }
                for _, r in df.iterrows()
            ],
        )
        log.info(f"ChromaDB poblada: {coleccion.count()} documentos "
                 f"(métrica: {cfg.vectorial.metrica})")
        return coleccion


    with etapa("indexación vectorial"):
        coleccion = poblar_chromadb(df, embeddings)

    print(f">>> ChromaDB: {coleccion.count()} documentos indexados en "
          f"'{CFG.vectorial.nombre_coleccion}' (métrica {CFG.vectorial.metrica})")

    # @title 6.2 — Organización automática: tópicos y tags del corpus
    tabla_organizacion = (
        df.groupby(["categoria", "topico_bertopic"])
          .agg(documentos=("doc_id", "count"))
          .reset_index()
          .sort_values(["categoria", "documentos"], ascending=[True, False])
    )
    tabla_organizacion["etiqueta_topico"] = tabla_organizacion["topico_bertopic"].map(ETIQUETAS_TOPICO)

    plt.figure(figsize=(12, 6))
    pivote = df.pivot_table(index="categoria", columns="topico_bertopic",
                            values="doc_id", aggfunc="count").fillna(0)
    sns.heatmap(pivote, annot=True, fmt=".0f", cmap="YlGnBu",
                cbar_kws={"label": "documentos"})
    plt.title("Organización automática: categorías etiquetadas × tópicos descubiertos",
              fontweight="bold")
    plt.xlabel("tópico BERTopic"); plt.ylabel("")
    plt.tight_layout(); plt.show()

    print("Lectura del mapa de calor: una fila concentrada en una columna significa que la")
    print("categoría etiquetada coincide con un tema emergente; una fila dispersa significa")
    print("que la categoría agrupa contenidos que el modelo considera temáticamente distintos.\n")
    # display(tabla_organizacion.head(15))

    # @title 6.3 — Motor de búsqueda semántica
    def buscar_semantica(consulta: str,
                         n: int = CFG.vectorial.n_resultados_busqueda,
                         categoria: str = None) -> pd.DataFrame:
        """Busca documentos por significado, no por coincidencia léxica.

        Args:
            consulta: Texto libre de la consulta.
            n: Número de resultados a devolver.
            categoria: Si se especifica, restringe la búsqueda a esa categoría
                aprovechando el filtrado combinado vector+metadatos de ChromaDB.

        Returns:
            DataFrame con `doc_id`, `similitud`, `categoria`, `titulo` y `extracto`,
            ordenado de mayor a menor similitud.

        Example:
            >>> buscar_semantica("desplegar aplicaciones en contenedores", n=3)
        """
        vector = CACHE.codificar([consulta], modelo_embeddings).tolist()
        res = coleccion.query(
            query_embeddings=vector,
            n_results=n,
            where={"categoria": categoria} if categoria else None,
        )
        return pd.DataFrame({
            "doc_id": res["ids"][0],
            "similitud": [round(1 - d, 4) for d in res["distances"][0]],
            "categoria": [m["categoria"] for m in res["metadatas"][0]],
            "titulo": [m["titulo"][:70] for m in res["metadatas"][0]],
            "extracto": [d[:120] + "..." for d in res["documents"][0]],
        })


    CONSULTAS_DEMO = [
        "¿cómo construir servicios web escalables?",
        "modelos que aprenden a partir de datos",
        "desplegar aplicaciones en contenedores",
    ]

    for consulta in CONSULTAS_DEMO:
        print("=" * 106)
        print(f"CONSULTA: {consulta}")
    #     display(buscar_semantica(consulta, n=4))

    print("=" * 106)
    print("Nótese que ninguna consulta comparte vocabulario literal con los documentos que recupera:")
    print("es exactamente la propiedad que TF-IDF no puede ofrecer y que justifica los embeddings (§3.2).")

    # @title 6.4 — Recomendación de contenido relacionado
    def recomendar_relacionados(doc_id: str, n: int = CFG.vectorial.n_relacionados) -> pd.DataFrame:
        """Recupera los documentos más similares a uno dado, excluyéndolo de sí mismo.

        Args:
            doc_id: Identificador del documento de referencia.
            n: Número de recomendaciones a devolver.

        Returns:
            DataFrame con `doc_id`, `similitud`, `categoria` y `titulo`.

        Raises:
            ValueError: Si el `doc_id` no existe en el corpus.

        Example:
            >>> recomendar_relacionados("DOC-0000", n=3)
        """
        posiciones = df.index[df["doc_id"] == doc_id]
        if len(posiciones) == 0:
            raise ValueError(f"El doc_id '{doc_id}' no existe en el corpus.")

        res = coleccion.query(
            query_embeddings=[embeddings[posiciones[0]].tolist()],
            n_results=n + 1,   # +1 porque el propio documento será el vecino más cercano
        )
        filas = [
            {"doc_id": _id, "similitud": round(1 - dist, 4),
             "categoria": meta["categoria"], "titulo": meta["titulo"][:70]}
            for _id, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])
            if _id != doc_id
        ]
        return pd.DataFrame(filas[:n])


    ref = df.loc[0]
    print(f"DOCUMENTO DE REFERENCIA: [{ref['categoria']}] {ref['titulo'][:70]}")
    print(f"  {ref['texto_limpio'][:200]}...\n")
    print("CONTENIDOS RELACIONADOS:")
    # display(recomendar_relacionados(ref["doc_id"], n=5))

    # @title 6.5 — Persistencia completa de resultados
    @cronometrar("persistencia de resultados")
    def persistir_resultados(cfg: Config = CFG) -> dict:
        """Guarda todos los productos del pipeline en sus formatos correspondientes.

        Args:
            cfg: Configuración con las rutas de salida.

        Returns:
            Diccionario {ruta: descripción} de los archivos escritos.
        """
        escritos: dict = {}

        # --- Predicciones del modelo sobre todo el corpus ---
        if GANADOR == "B":
            probabilidades = modelo_b.predict_proba(embeddings)
        else:
            probabilidades = modelo_a.predict_proba(textos_pos)
        predicciones = probabilidades.argmax(axis=1)

        resultados = pd.DataFrame({
            "doc_id": df["doc_id"],
            "titulo": df["titulo"],
            "categoria_real": df["categoria"],
            "categoria_predicha": le.inverse_transform(predicciones),
            "probabilidad": probabilidades.max(axis=1).round(4),
            "confianza_baja": probabilidades.max(axis=1) < cfg.clasificacion.umbral_confianza_baja,
            "particion": ["test" if i in set(idx_test) else "train" for i in range(len(df))],
            "topico": df["topico_bertopic"],
            "etiqueta_topico": df["topico_bertopic"].map(ETIQUETAS_TOPICO),
            "cluster_kmeans": df["cluster_kmeans"],
            "keywords": df["keywords"].apply(lambda ks: ", ".join(ks)),
            "entidades_tecnicas": df["entidades_tech"].apply(lambda es: ", ".join(es)),
        })
        resultados["acierto"] = resultados["categoria_real"] == resultados["categoria_predicha"]

        ruta = cfg.rutas.processed / "resultados_clasificacion.csv"
        resultados.to_csv(ruta, index=False)
        escritos[str(ruta)] = "Predicción, probabilidad, tópico y keywords por documento"

        # --- Corpus final enriquecido ---
        columnas = ["doc_id", "titulo", "categoria", "idioma", "idioma_confianza",
                    "texto_limpio", "texto_pos", "keywords", "entidades_tech",
                    "topico_bertopic", "cluster_kmeans", "n_tokens", "n_chars", "fuente"]
        columnas = [c for c in columnas if c in df.columns]
        ruta = cfg.rutas.processed / "corpus_final.csv"
        df[columnas].to_csv(ruta, index=False)
        escritos[str(ruta)] = "Corpus con todas las anotaciones del pipeline"

        # --- Embeddings ---
        ruta = cfg.rutas.processed / "embeddings.npy"
        np.save(ruta, embeddings)
        escritos[str(ruta)] = f"Matriz de embeddings {embeddings.shape} float32"

        # --- Índice de keywords: qué documentos mencionan cada término ---
        indice: dict = {}
        for _, fila in df.iterrows():
            for kw in fila["keywords"]:
                indice.setdefault(kw.lower(), []).append(fila["doc_id"])
        ruta = cfg.rutas.processed / "indice_keywords.json"
        ruta.write_text(json.dumps(indice, ensure_ascii=False, indent=2), encoding="utf-8")
        escritos[str(ruta)] = f"Índice invertido de {len(indice)} keywords → documentos"

        # --- Reporte de tiempos de ejecución ---
        ruta = cfg.rutas.logs / "tiempos_ejecucion.csv"
        pd.DataFrame(TIEMPOS).to_csv(ruta, index=False)
        escritos[str(ruta)] = f"{len(TIEMPOS)} operaciones cronometradas"

        CACHE.guardar()
        return escritos, resultados


    with etapa("persistencia"):
        ARCHIVOS_ESCRITOS, RESULTADOS = persistir_resultados()

    print("ARCHIVOS PERSISTIDOS\n")
    for ruta, descripcion in ARCHIVOS_ESCRITOS.items():
        tam = Path(ruta).stat().st_size / 1024
        print(f"  {Path(ruta).name:<34} {tam:>9,.1f} KB   {descripcion}")

    print(f"\nResumen de clasificación sobre el corpus completo:")
    print(f"  aciertos                : {RESULTADOS['acierto'].sum()}/{len(RESULTADOS)} "
          f"({RESULTADOS['acierto'].mean():.1%})")
    print(f"  confianza baja (<{CFG.clasificacion.umbral_confianza_baja}) : "
          f"{RESULTADOS['confianza_baja'].sum()} documentos")
    # display(RESULTADOS.head(6)[["doc_id", "categoria_real", "categoria_predicha",
    #                             "probabilidad", "acierto", "keywords"]])

    # @title 6.6.1 — Esquema relacional y poblado
    ESQUEMA_SQL = """
    CREATE TABLE IF NOT EXISTS documentos (
        doc_id      TEXT PRIMARY KEY,
        titulo      TEXT,
        categoria   TEXT,
        idioma      TEXT,
        n_tokens    INTEGER,
        n_chars     INTEGER,
        fuente      TEXT
    );
    CREATE TABLE IF NOT EXISTS keywords_documento (
        doc_id      TEXT,
        keyword     TEXT,
        rango       INTEGER,
        FOREIGN KEY (doc_id) REFERENCES documentos(doc_id)
    );
    CREATE TABLE IF NOT EXISTS resultados_clustering (
        doc_id            TEXT PRIMARY KEY,
        cluster_kmeans    INTEGER,
        topico_bertopic   INTEGER,
        etiqueta_topico   TEXT,
        FOREIGN KEY (doc_id) REFERENCES documentos(doc_id)
    );
    CREATE TABLE IF NOT EXISTS resultados_clasificacion (
        doc_id          TEXT,
        categoria_real  TEXT,
        categoria_pred  TEXT,
        probabilidad    REAL,
        acierto         INTEGER,
        particion       TEXT,
        modelo          TEXT,
        FOREIGN KEY (doc_id) REFERENCES documentos(doc_id)
    );
    CREATE TABLE IF NOT EXISTS predicciones_api (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha           TEXT,
        titulo          TEXT,
        categoria       TEXT,
        probabilidad    REAL,
        confianza_baja  INTEGER,
        idioma          TEXT,
        keywords        TEXT,
        latencia_ms     REAL,
        version_modelo  TEXT
    );
    CREATE TABLE IF NOT EXISTS versiones_modelo (
        version              TEXT,
        fecha                TEXT,
        huella_configuracion TEXT,
        tipo_clasificador    TEXT,
        dataset              TEXT,
        hash_dataset         TEXT,
        n_documentos         INTEGER,
        parametros_json      TEXT,
        metricas_json        TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_kw ON keywords_documento(keyword);
    CREATE INDEX IF NOT EXISTS idx_cat ON documentos(categoria);
    CREATE INDEX IF NOT EXISTS idx_pred_fecha ON predicciones_api(fecha);
    """


    def inicializar_sqlite(cfg: Config = CFG) -> sqlite3.Connection:
        """Crea el esquema relacional si no existe y devuelve una conexión abierta.

        Args:
            cfg: Configuración con la ruta del archivo de base de datos.

        Returns:
            Conexión SQLite lista para usar.

        Example:
            >>> con = inicializar_sqlite()
            >>> con.execute("SELECT COUNT(*) FROM documentos").fetchone()[0] >= 0
            True
        """
        ruta = cfg.rutas.base / cfg.persistencia.archivo_sqlite
        con = sqlite3.connect(str(ruta))
        con.executescript(ESQUEMA_SQL)
        con.commit()
        log.info(f"Esquema SQLite inicializado en {ruta}")
        return con


    @cronometrar("poblado de SQLite")
    def poblar_sqlite(con: sqlite3.Connection, df: pd.DataFrame,
                      resultados: pd.DataFrame, cfg: Config = CFG) -> dict:
        """Puebla las tablas relacionales desde el corpus procesado.

        Args:
            con: Conexión SQLite abierta.
            df: Corpus con anotaciones del pipeline.
            resultados: DataFrame de §6.5 con las predicciones por documento.
            cfg: Configuración del pipeline.

        Returns:
            Diccionario {tabla: número de filas}.
        """
        # Tablas sin clave primaria natural: se vacían para que re-ejecutar no duplique.
        con.execute("DELETE FROM keywords_documento")
        con.execute("DELETE FROM resultados_clasificacion")

        con.executemany(
            "INSERT OR REPLACE INTO documentos VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r["doc_id"], r["titulo"], r["categoria"], r["idioma"],
              int(r["n_tokens"]), int(r["n_chars"]), str(r.get("fuente", "")))
             for _, r in df.iterrows()],
        )

        con.executemany(
            "INSERT INTO keywords_documento VALUES (?, ?, ?)",
            [(r["doc_id"], kw, rango)
             for _, r in df.iterrows()
             for rango, kw in enumerate(r["keywords"], start=1)],
        )

        con.executemany(
            "INSERT OR REPLACE INTO resultados_clustering VALUES (?, ?, ?, ?)",
            [(r["doc_id"], int(r["cluster_kmeans"]), int(r["topico_bertopic"]),
              ETIQUETAS_TOPICO.get(int(r["topico_bertopic"]), ""))
             for _, r in df.iterrows()],
        )

        tipo_clf = "sbert+logreg" if GANADOR == "B" else "tfidf+logreg"
        con.executemany(
            "INSERT INTO resultados_clasificacion VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r["doc_id"], r["categoria_real"], r["categoria_predicha"],
              float(r["probabilidad"]), int(bool(r["acierto"])), r["particion"], tipo_clf)
             for _, r in resultados.iterrows()],
        )

        con.commit()

        conteos = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                   for t in ("documentos", "keywords_documento",
                             "resultados_clustering", "resultados_clasificacion")}
        log.info(f"SQLite poblada: {conteos}")
        return conteos


    with etapa("persistencia relacional"):
        con_sqlite = inicializar_sqlite()
        CONTEOS_SQLITE = poblar_sqlite(con_sqlite, df, RESULTADOS)

    print("TABLAS POBLADAS\n")
    for tabla, n in CONTEOS_SQLITE.items():
        print(f"  {tabla:<28} {n:>6} filas")
    print(f"\nBase de datos: {CFG.rutas.base / CFG.persistencia.archivo_sqlite}")

    # @title 6.6.2 — Consultas relacionales de demostración
    CONSULTAS_DEMO = {
        "Documentos por categoría":
            """SELECT categoria, COUNT(*) AS documentos, ROUND(AVG(n_tokens), 1) AS tokens_medios
               FROM documentos GROUP BY categoria ORDER BY documentos DESC""",

        "Keywords más frecuentes del corpus":
            """SELECT keyword, COUNT(*) AS documentos
               FROM keywords_documento GROUP BY keyword
               ORDER BY documentos DESC LIMIT 10""",

        "Precisión por categoría (desde SQL, sin recalcular)":
            """SELECT categoria_real,
                      COUNT(*) AS total,
                      SUM(acierto) AS aciertos,
                      ROUND(100.0 * SUM(acierto) / COUNT(*), 1) AS pct_acierto
               FROM resultados_clasificacion
               GROUP BY categoria_real ORDER BY pct_acierto ASC""",

        "Documentos donde el modelo falla con alta confianza":
            """SELECT c.doc_id, d.titulo, c.categoria_real, c.categoria_pred, c.probabilidad
               FROM resultados_clasificacion c JOIN documentos d ON d.doc_id = c.doc_id
               WHERE c.acierto = 0 AND c.probabilidad > 0.6
               ORDER BY c.probabilidad DESC LIMIT 5""",
    }

    for titulo, sql in CONSULTAS_DEMO.items():
        print("=" * 92)
        print(titulo)
        print("-" * 92)
    #     display(pd.read_sql(sql, con_sqlite))

    print("=" * 92)
    print("Estas cuatro consultas son la justificación de la tabla: ninguna se puede responder")
    print("con ChromaDB, y todas requerirían cargar el CSV completo en memoria sin SQL.")

    # @title 6.7 — Registro append-only de la versión entrenada
    def registrar_version_modelo(metadatos: dict, con: sqlite3.Connection = None,
                                 cfg: Config = CFG) -> dict:
        """Añade la versión actual al historial, sin sobrescribir entrenamientos previos.

        Escribe en dos destinos de solo-anexado: la tabla `versiones_modelo` y un
        archivo JSONL. `metadata.json` no se toca: sigue describiendo la versión
        vigente, que es lo que el backend espera encontrar.

        Args:
            metadatos: Diccionario producido por `serializar_artefactos` (§5.7).
            con: Conexión SQLite. Si es None, usa la global `con_sqlite`.
            cfg: Configuración del pipeline.

        Returns:
            Diccionario con el número total de versiones registradas y la ruta del JSONL.

        Example:
            >>> registrar_version_modelo(METADATOS)["n_versiones"] >= 1
            True
        """
        con = con if con is not None else con_sqlite
        ruta_jsonl = cfg.rutas.models / cfg.persistencia.archivo_historial

        prueba = metadatos["metricas"]["conjunto_prueba"]
        con.execute(
            "INSERT INTO versiones_modelo VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                metadatos["version"],
                metadatos["fecha_entrenamiento"],
                metadatos["huella_configuracion"][:16],
                metadatos["modelo"]["tipo_clasificador"],
                metadatos["dataset"]["fuente"],
                metadatos["dataset"]["hash_sha256"][:16],
                metadatos["dataset"]["n_documentos"],
                json.dumps(metadatos["hiperparametros"], ensure_ascii=False, default=str),
                json.dumps(metadatos["metricas"], ensure_ascii=False, default=str),
            ),
        )
        con.commit()

        with open(ruta_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadatos, ensure_ascii=False, default=str) + "\n")

        n = con.execute("SELECT COUNT(*) FROM versiones_modelo").fetchone()[0]
        log.info(f"Versión {metadatos['version']} registrada en el historial "
                 f"(total acumulado: {n})")
        return {"n_versiones": n, "jsonl": str(ruta_jsonl)}


    REGISTRO_VERSION = registrar_version_modelo(METADATOS)

    print(f"Versiones en el historial: {REGISTRO_VERSION['n_versiones']}")
    print(f"JSONL de solo-anexado    : {REGISTRO_VERSION['jsonl']}\n")

    historial = pd.read_sql(
        """SELECT version, substr(fecha, 1, 19) AS fecha, tipo_clasificador,
                  n_documentos, huella_configuracion
           FROM versiones_modelo ORDER BY fecha DESC LIMIT 10""",
        con_sqlite)
    # display(historial)

    if len(historial) > 1:
        print("\nComparación de F1-macro entre versiones registradas:")
        for _, fila in pd.read_sql(
                "SELECT version, fecha, metricas_json FROM versiones_modelo ORDER BY fecha",
                con_sqlite).iterrows():
            m = json.loads(fila["metricas_json"])
            print(f"  {fila['fecha'][:19]}  v{fila['version']:<8} "
                  f"f1_macro = {m['conjunto_prueba']['f1_macro']:.4f}")
    else:
        print("\nEsta es la primera versión registrada. Al reentrenar, esta celda mostrará")
        print("la comparación de métricas entre corridas sin necesidad de reejecutar nada.")

    # @title 7.1 — Contrato de salida
    @dataclass
    class RespuestaContenido:
        """Respuesta del endpoint POST /contenido.

        Los tres primeros campos son el contrato mínimo exigido por el brief y no
        cambian de nombre ni de tipo. El resto son extensiones propias del equipo,
        admitidas explícitamente por el enunciado.

        Attributes:
            categoria: Categoría temática asignada.
            probabilidad: Confianza de la predicción, en [0, 1].
            informacion_adicional: Palabras clave representativas del contenido.
            titulo: Título recibido, normalizado.
            idioma: Resultado de la detección de idioma.
            tema: Tópico emergente descubierto por BERTopic.
            entidades_tecnicas: Tecnologías reconocidas por el EntityRuler.
            distribucion_categorias: Tres categorías más probables con su probabilidad.
            metricas_texto: Estadísticos del documento procesado.
            explicacion: Justificación de la categoría asignada (opcional).
            relacionados: Contenidos semánticamente próximos (opcional).
            advertencias: Observaciones no bloqueantes de la validación.

        Example:
            >>> r = RespuestaContenido("Backend", 0.89, ["Java", "Spring Boot"])
            >>> r.a_dict()["categoria"]
            'Backend'
        """
        categoria: str
        probabilidad: float
        informacion_adicional: list

        titulo: str = ""
        idioma: dict = field(default_factory=dict)
        tema: dict = field(default_factory=dict)
        entidades_tecnicas: list = field(default_factory=list)
        distribucion_categorias: dict = field(default_factory=dict)
        metricas_texto: dict = field(default_factory=dict)
        explicacion: dict = field(default_factory=dict)
        relacionados: list = field(default_factory=list)
        advertencias: list = field(default_factory=list)

        def a_dict(self) -> dict:
            """Serializa la respuesta a un diccionario JSON-compatible."""
            return asdict(self)

        def a_json(self, indent: int = 2) -> str:
            """Serializa la respuesta a una cadena JSON."""
            return json.dumps(self.a_dict(), ensure_ascii=False, indent=indent)


    print("Contrato de salida definido.")
    print("Campos obligatorios del brief:", ["categoria", "probabilidad", "informacion_adicional"])

    # @title 7.2 — Clase de inferencia sin estado global
class TechMindInference:
    """Capa de inferencia de TechMind, lista para importar desde FastAPI.

    Encapsula el pipeline completo de predicción sin depender de variables
    globales del notebook. Todas las dependencias se inyectan en el constructor,
    lo que permite instanciarla desde otro proceso, sustituirlas por dobles en
    los tests, o mantener dos versiones cargadas durante un despliegue gradual.

    Uso previsto en el backend::

        # app/services/nlp_service.py
        servicio = TechMindInference.desde_artefactos(Path("models"))

        # app/api/routes/contenido.py
        @router.post("/contenido")
        async def procesar(payload: ContenidoRequest):
            return await run_in_threadpool(
                servicio.predecir, payload.titulo, payload.texto
            )

    Attributes:
        metadatos: Contenido de `metadata.json`, con versión, métricas y dataset.
        categorias: Lista ordenada de categorías que el modelo puede predecir.

    Example:
        >>> servicio = TechMindInference.desde_objetos(...)
        >>> r = servicio.predecir("Spring Boot", "Framework de Java para APIs REST.")
        >>> r.categoria in servicio.categorias
        True
    """

    def __init__(self, *, modelo_clasificacion, label_encoder, modelo_embeddings,
                 pipeline_nlp, cfg: Config, tipo_clasificador: str,
                 modelo_keybert=None, extractor_yake=None, mapa_tecnologias: dict = None,
                 metadatos: dict = None, topic_model=None, coleccion_vectorial=None,
                 etiquetas_topico: dict = None, centroides: np.ndarray = None,
                 cache=None, conexion_sqlite=None):
        self.clf = modelo_clasificacion
        self.le = label_encoder
        self.embedder = modelo_embeddings
        self.nlp = pipeline_nlp
        self.cfg = cfg
        self.tipo_clasificador = tipo_clasificador
        # --- Dependencias del ranking de keywords (antes leídas del ámbito global) ---
        self.keybert = modelo_keybert
        self.yake = extractor_yake
        self.mapa_tecnologias = mapa_tecnologias or {}
        # --- Resto ---
        self.metadatos = metadatos or {}
        self.topic_model = topic_model
        self.coleccion = coleccion_vectorial
        self.etiquetas_topico = etiquetas_topico or {}
        self.centroides = centroides
        self.cache = cache
        self.con = conexion_sqlite
        self.categorias = list(label_encoder.classes_)
        log.info(f"TechMindInference lista · clasificador={tipo_clasificador} · "
                 f"{len(self.categorias)} categorías · "
                 f"auditoría={'sí' if conexion_sqlite is not None else 'no'}")

    # ------------------------------------------------------------------ #
    # Carga del modelo — se ejecuta una vez al arrancar el proceso        #
    # ------------------------------------------------------------------ #
    @classmethod
    def desde_artefactos(cls, directorio: Path, cfg: Config = None) -> "TechMindInference":
        """Construye la capa de inferencia cargando los artefactos desde disco.

        Es el punto de entrada que usa el backend: lee `metadata.json`, carga el
        clasificador, el codificador de etiquetas, el modelo de embeddings y el
        pipeline de spaCy declarados allí, y devuelve un servicio listo.

        Args:
            directorio: Carpeta con los artefactos serializados en §5.7.
            cfg: Configuración. Si es None, se lee `config.json` del directorio.

        Returns:
            Una instancia de `TechMindInference`.

        Raises:
            FileNotFoundError: Si falta `metadata.json` o el clasificador.
        """
        directorio = Path(directorio)
        ruta_meta = directorio / "metadata.json"
        if not ruta_meta.exists():
            raise FileNotFoundError(f"No se encontró metadata.json en {directorio}")

        metadatos = json.loads(ruta_meta.read_text(encoding="utf-8"))
        cfg = cfg or Config()

        clf = joblib.load(directorio / "modelo_clasificacion.joblib")
        codificador = joblib.load(directorio / "label_encoder.joblib")

        from sentence_transformers import SentenceTransformer
        import spacy
        embedder = SentenceTransformer(metadatos["modelo"]["modelo_embeddings"])
        pipeline = spacy.load(metadatos["modelo"]["modelo_spacy"])

        # --- Diccionario de tecnologías ---
        ruta_tecnologias = directorio / "tecnologias.json"
        if ruta_tecnologias.exists():
            tecnologias = json.loads(ruta_tecnologias.read_text(encoding="utf-8"))
        else:
            tecnologias = metadatos.get("modelo", {}).get("tecnologias", [])
            log.warning(f"No se encontró {ruta_tecnologias.name}: el EntityRuler quedará "
                        f"con {len(tecnologias)} patrones y las entidades técnicas se "
                        f"detectarán peor.")
        mapa = {t.lower(): t for t in tecnologias}

        # --- EntityRuler: IMPRESCINDIBLE reconstruirlo aquí ---
        # `spacy.load()` devuelve el modelo BASE, que no conoce la etiqueta TECH:
        # es_core_news_sm solo produce ORG/LOC/PER/MISC. Sin volver a añadir el
        # ruler, `preprocesar()` devolvería `entidades_tech` vacío y —peor—
        # `rankear_keywords()` perdería su señal de mayor peso (peso_entidades),
        # degradando silenciosamente `informacion_adicional` en producción
        # respecto de lo que muestra el notebook.
        if "entity_ruler" not in pipeline.pipe_names and tecnologias:
            # `before="ner"` da prioridad a las reglas sobre el NER estadístico
            # (evita que "Java" se etiquete como isla). Pero falla con
            # ValueError si el pipeline no tiene componente `ner`, cosa que
            # ocurre con modelos recortados: se degrada a añadirlo al final.
            if "ner" in pipeline.pipe_names:
                ruler = pipeline.add_pipe("entity_ruler", before="ner")
            else:
                ruler = pipeline.add_pipe("entity_ruler")
                log.warning("El pipeline no tiene componente 'ner'; el EntityRuler "
                            "se añade al final.")
            ruler.add_patterns([{"label": "TECH", "pattern": t} for t in tecnologias])
            log.info(f"EntityRuler restaurado con {len(tecnologias)} patrones.")

        centroides = None
        ruta_centroides = directorio / "centroides_clase.joblib"
        if ruta_centroides.exists():
            centroides = joblib.load(ruta_centroides)["centroides"]

        # Dependencias del ranking de keywords, reconstruidas desde los artefactos.
        from keybert import KeyBERT
        import yake
        keybert = KeyBERT(model=embedder)
        extractor = yake.KeywordExtractor(
            lan=cfg.idioma.idioma_objetivo, n=cfg.keywords.ngram_max,
            dedupLim=cfg.keywords.yake_dedup_limite,
            top=cfg.keywords.keybert_candidatos, features=None)

        # --- BERTopic: sin él, el campo `tema` sale siempre vacío ---
        topic_model = None
        etiquetas = {}
        ruta_bertopic = directorio / "modelo_bertopic"
        if ruta_bertopic.exists():
            try:
                from bertopic import BERTopic
                topic_model = BERTopic.load(str(ruta_bertopic), embedding_model=embedder)
                etiquetas = {
                    t: ", ".join(w for w, _ in topic_model.get_topic(t)[:4])
                    for t in topic_model.get_topics() if t != -1
                }
                etiquetas[-1] = "(sin tema definido / outlier)"
                log.info(f"BERTopic cargado: {len(etiquetas) - 1} tópicos.")
            except Exception as exc:
                log.warning(f"No se pudo cargar BERTopic ({type(exc).__name__}): "
                            f"el campo 'tema' vendrá vacío.")

        # --- ChromaDB: sin él, `relacionados` sale siempre vacío ---
        coleccion = None
        if cfg.rutas.chroma.exists():
            try:
                import chromadb
                cliente = chromadb.PersistentClient(path=str(cfg.rutas.chroma))
                coleccion = cliente.get_collection(cfg.vectorial.nombre_coleccion)
                log.info(f"ChromaDB conectada: {coleccion.count()} documentos indexados.")
            except Exception as exc:
                log.warning(f"No se pudo conectar a ChromaDB ({type(exc).__name__}): "
                            f"el campo 'relacionados' vendrá vacío.")

        log.info(f"Artefactos cargados desde {directorio} "
                 f"(versión {metadatos.get('version')})")
        return cls(
            modelo_clasificacion=clf,
            label_encoder=codificador,
            modelo_embeddings=embedder,
            pipeline_nlp=pipeline,
            cfg=cfg,
            tipo_clasificador=metadatos["modelo"]["tipo_clasificador"],
            modelo_keybert=keybert,
            extractor_yake=extractor,
            mapa_tecnologias=mapa,
            metadatos=metadatos,
            topic_model=topic_model,
            etiquetas_topico=etiquetas,
            coleccion_vectorial=coleccion,
            centroides=centroides,
            cache=CacheEmbeddings(cfg),
        )

    @classmethod
    def desde_objetos(cls, **kwargs) -> "TechMindInference":
        """Construye la capa de inferencia con objetos ya en memoria.

        Es la vía que usa este notebook, donde los modelos acaban de entrenarse y
        no tiene sentido volver a leerlos del disco.
        """
        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    # Predicción — se ejecuta en cada petición                            #
    # ------------------------------------------------------------------ #
    def _codificar(self, textos: list) -> np.ndarray:
        """Codifica textos usando la caché si está disponible."""
        if self.cache is not None:
            return self.cache.codificar(textos, self.embedder)
        return self.embedder.encode(textos,
                                    normalize_embeddings=self.cfg.embeddings.normalizar)

    def _clasificar(self, texto_limpio: str, texto_pos: str) -> tuple:
        """Devuelve (vector, distribución de probabilidad) según el tipo de clasificador."""
        vector = self._codificar([texto_limpio])
        if self.tipo_clasificador == "sbert+logreg":
            probas = self.clf.predict_proba(vector)[0]
        else:
            probas = self.clf.predict_proba([texto_pos])[0]
        return vector, probas

    def _probabilidades(self, texto_limpio: str, texto_pos: str) -> np.ndarray:
        """Predictor inyectable para la explicabilidad: usa solo estado de la instancia."""
        return self._clasificar(texto_limpio, texto_pos)[1]

    def _keywords(self, texto_limpio: str, doc_spacy) -> list:
        """Ranking de keywords usando exclusivamente las dependencias inyectadas."""
        return rankear_keywords(
            texto_limpio,
            doc_spacy=doc_spacy,
            top_k=self.cfg.keywords.top_k,
            cfg=self.cfg,
            modelo_keybert=self.keybert,
            extractor=self.yake,
            pipeline_nlp=self.nlp,
            mapa_tecnologias=self.mapa_tecnologias,
        )

    def _explicar(self, texto_limpio: str, texto_pos: str, candidatos: Sequence) -> dict:
        """Explicabilidad local usando exclusivamente las dependencias inyectadas."""
        return explicar_prediccion(
            texto_limpio, texto_pos, candidatos,
            predictor=self._probabilidades,
            centroides=self.centroides,
            categorias=self.categorias,
            codificador=self._codificar,
        )

    def _auditar(self, respuesta: "RespuestaContenido", latencia_ms: float) -> None:
        """Registra la predicción en la tabla `predicciones_api` si hay conexión.

        Args:
            respuesta: Respuesta ya construida.
            latencia_ms: Duración total de la inferencia en milisegundos.

        Returns:
            None. Los fallos de auditoría se registran pero no interrumpen la
            respuesta: no servir una predicción correcta porque falló el log
            sería un modo de fallo peor que el propio fallo.
        """
        if self.con is None or not self.cfg.persistencia.persistir_predicciones:
            return
        try:
            self.con.execute(
                "INSERT INTO predicciones_api "
                "(fecha, titulo, categoria, probabilidad, confianza_baja, "
                " idioma, keywords, latencia_ms, version_modelo) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pd.Timestamp.now().isoformat(),
                 respuesta.titulo,
                 respuesta.categoria,
                 respuesta.probabilidad,
                 int(respuesta.probabilidad < self.cfg.clasificacion.umbral_confianza_baja),
                 respuesta.idioma.get("codigo", ""),
                 ", ".join(respuesta.informacion_adicional),
                 round(latencia_ms, 2),
                 self.metadatos.get("version", self.cfg.version)),
            )
            self.con.commit()
        except Exception as exc:
            log.warning(f"No se pudo auditar la predicción ({type(exc).__name__}): {exc}")

    def predecir(self, titulo: str, texto: str, *,
                 incluir_relacionados: bool = True,
                 incluir_explicacion: bool = True,
                 n_relacionados: int = None) -> RespuestaContenido:
        """Ejecuta el pipeline completo de inferencia sobre un documento.

        Args:
            titulo: Título del contenido. Obligatorio.
            texto: Cuerpo del contenido. Obligatorio.
            incluir_relacionados: Consulta ChromaDB por contenidos similares.
            incluir_explicacion: Calcula la explicabilidad local por ablación.
                Tiene costo: multiplica el número de predicciones por el número
                de términos evaluados.
            n_relacionados: Número de recomendaciones. Por defecto, el de la configuración.

        Returns:
            Un `RespuestaContenido` con el contrato del brief más las extensiones.

        Raises:
            ErrorValidacion: Si el documento no supera los controles de §2.1.

        Example:
            >>> servicio.predecir("Docker", "Guía de contenedores y orquestación.").categoria
            'DevOps'
        """
        t0 = time.perf_counter()

        # --- 1. Validación ---
        validacion = exigir_valido(titulo, texto, modo="inferencia")

        # --- 2. Idioma ---
        idioma = detectar_idioma(validacion.texto, cfg=self.cfg)
        if self.cfg.idioma.rechazar_idioma_no_soportado and not idioma.soportado:
            resultado = ResultadoValidacion()
            resultado.agregar_error(
                CodigoError.IDIOMA_NO_SOPORTADO,
                f"Idioma '{idioma.codigo}' no soportado. "
                f"Idiomas disponibles: {list(self.cfg.idioma.idiomas_soportados)}.")
            raise ErrorValidacion(resultado)

        # --- 3. Limpieza ---
        # Dos textos, con propósitos distintos:
        #
        #  · `texto_limpio` alimenta al CLASIFICADOR y se compone con la misma
        #    función que usó el entrenamiento (§2.4.2), para que el modelo
        #    prediga sobre la distribución con la que aprendió.
        #
        #  · `texto_analisis` alimenta al EntityRuler y a KeyBERT, que NO se
        #    entrenan —son reglas y similitud coseno— y por tanto no sufren
        #    desajuste. Darles el título recupera tecnologías que solo aparecen
        #    ahí: sin esto, "Clasificación de texto con Scikit-Learn" devuelve
        #    `entidades_tecnicas` vacío porque el cuerpo no menciona la librería.
        texto_limpio = limpiar_texto(
            componer_entrada(validacion.titulo, validacion.texto, self.cfg))

        usar_titulo_aparte = (self.cfg.nlp.incluir_titulo_en_entidades
                              and not self.cfg.nlp.incluir_titulo_en_texto)
        texto_analisis = (limpiar_texto(f"{validacion.titulo}. {validacion.texto}")
                          if usar_titulo_aparte else texto_limpio)

        # --- 4. Preprocesamiento NLP ---
        doc = self.nlp(texto_limpio)          # define `texto_pos` para el clasificador
        proc = preprocesar(doc, self.cfg)

        # Segunda pasada solo si los textos difieren. Cuesta ~10 ms sobre un
        # documento corto y es el precio de no degradar el contrato de salida.
        doc_analisis = self.nlp(texto_analisis) if usar_titulo_aparte else doc
        proc_analisis = preprocesar(doc_analisis, self.cfg) if usar_titulo_aparte else proc

        # --- 5. Clasificación ---
        vector, probas = self._clasificar(texto_limpio, proc["texto_pos"])
        idx = int(np.argmax(probas))
        categoria = str(self.le.inverse_transform([idx])[0])
        probabilidad = round(float(probas[idx]), 4)

        if probabilidad < self.cfg.clasificacion.umbral_confianza_baja:
            log.warning(f"Confianza baja ({probabilidad}) al clasificar "
                        f"'{validacion.titulo[:40]}' como '{categoria}'")
            validacion.agregar_advertencia(
                "confianza_baja",
                f"La probabilidad ({probabilidad}) está por debajo del umbral "
                f"{self.cfg.clasificacion.umbral_confianza_baja}: el contenido puede "
                f"ser ambiguo o pertenecer a una categoría no cubierta.")

        # --- 6. Keywords (sobre el texto de análisis, que sí incluye el título) ---
        keywords = self._keywords(texto_analisis, doc_analisis)

        # --- 7. Tópico ---
        tema = {"id": -1, "etiqueta": "(sin tema definido)"}
        if self.topic_model is not None:
            try:
                topicos_pred, _ = self.topic_model.transform([texto_limpio], vector)
                tid = int(topicos_pred[0])
                tema = {"id": tid, "etiqueta": self.etiquetas_topico.get(
                    tid, "(sin tema definido)")}
            except Exception as exc:
                log.debug(f"BERTopic.transform falló ({type(exc).__name__}); tema por defecto.")

        respuesta = RespuestaContenido(
            categoria=categoria,
            probabilidad=probabilidad,
            informacion_adicional=keywords,
            titulo=validacion.titulo,
            idioma=idioma.a_dict(),
            tema=tema,
            entidades_tecnicas=proc_analisis["entidades_tech"],
            distribucion_categorias={
                str(c): round(float(p), 4)
                for c, p in sorted(zip(self.categorias, probas), key=lambda x: -x[1])[:3]
            },
            metricas_texto={
                "n_tokens": proc["n_tokens"],
                "n_caracteres": len(texto_limpio),
                "n_palabras": validacion.metricas.get("n_palabras", 0),
            },
            advertencias=[{"codigo": c, "mensaje": m} for c, m in validacion.advertencias],
        )

        # --- 8. Explicabilidad ---
        if incluir_explicacion and self.centroides is not None:
            try:
                respuesta.explicacion = self._explicar(
                    texto_limpio, proc["texto_pos"], keywords)
            except Exception as exc:
                log.debug(f"Explicabilidad omitida ({type(exc).__name__}).")

        # --- 9. Recomendación ---
        if incluir_relacionados and self.coleccion is not None:
            n = n_relacionados or self.cfg.vectorial.n_relacionados
            res = self.coleccion.query(query_embeddings=vector.tolist(), n_results=n)
            respuesta.relacionados = [
                {"doc_id": _id, "titulo": m["titulo"][:70],
                 "categoria": m["categoria"], "similitud": round(1 - d, 4)}
                for _id, d, m in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])
            ]

        dt = (time.perf_counter() - t0) * 1000

        # --- 10. Auditoría (no bloqueante) ---
        self._auditar(respuesta, dt)

        log.info(f"Inferencia completada en {dt:.0f} ms → "
                 f"{categoria} ({probabilidad:.2f})")
        return respuesta

    def predecir_lote(self, documentos: Sequence, **kwargs) -> pd.DataFrame:
        """Procesa una lista de documentos, reportando errores sin abortar el lote.

        Args:
            documentos: Secuencia de diccionarios con claves `titulo` y `texto`.
            **kwargs: Argumentos que se pasan a `predecir`.

        Returns:
            DataFrame con una fila por documento. Los fallidos llevan `error`
            poblado y el resto de campos en None.
        """
        filas = []
        for entrada in documentos:
            try:
                r = self.predecir(entrada.get("titulo"), entrada.get("texto"), **kwargs)
                filas.append({
                    "titulo": r.titulo, "categoria": r.categoria,
                    "probabilidad": r.probabilidad,
                    "keywords": ", ".join(r.informacion_adicional),
                    "tema": r.tema.get("etiqueta"), "idioma": r.idioma.get("codigo"),
                    "error": None,
                })
            except ErrorValidacion as exc:
                filas.append({"titulo": entrada.get("titulo"), "categoria": None,
                              "probabilidad": None, "keywords": None, "tema": None,
                              "idioma": None,
                              "error": exc.resultado.errores[0][0]})
            except Exception as exc:
                log.exception(f"Fallo inesperado procesando '{entrada.get('titulo')}'")
                filas.append({"titulo": entrada.get("titulo"), "categoria": None,
                              "probabilidad": None, "keywords": None, "tema": None,
                              "idioma": None, "error": f"{type(exc).__name__}: {exc}"})
        return pd.DataFrame(filas)

    def salud(self) -> dict:
        """Estado del servicio, para un endpoint GET /health del backend."""
        return {
            'estado': 'operativo',
            'version_pipeline': self.metadatos.get('version', self.cfg.version),
            'clasificador': self.tipo_clasificador,
            'categorias': self.categorias,
            'idiomas_soportados': list(self.cfg.idioma.idiomas_soportados),
            'modelo_embeddings': self.cfg.embeddings.modelo,
            'indice_vectorial': self.coleccion.count() if self.coleccion is not None else None,
            'explicabilidad': self.centroides is not None
        }
