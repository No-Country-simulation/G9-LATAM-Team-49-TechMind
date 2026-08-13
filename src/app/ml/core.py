"""Nucleo de inferencia de TechMind — sin dependencias del notebook.

Contiene SOLO lo que la API necesita: configuracion, validacion, deteccion de
idioma, preprocesamiento, keywords, explicabilidad y TechMindInference.

Sin efectos secundarios al importar: no crea directorios, no fija semillas
globales, no reconfigura el logging root y no imprime nada.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

log = logging.getLogger("techmind")

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
    if pipeline_nlp is None:
        raise ValueError('pipeline_nlp es obligatorio')

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
    if modelo_keybert is None:
        raise ValueError('modelo_keybert es obligatorio')
    if extractor is None:
        raise ValueError('extractor es obligatorio')
    if pipeline_nlp is None:
        raise ValueError('pipeline_nlp es obligatorio')
    mapa_tecnologias = mapa_tecnologias or {t.lower(): t for t in TECNOLOGIAS}

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
        stop_words=list(pipeline_nlp.Defaults.stop_words),
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
    if predictor is None:
        raise ValueError('predictor es obligatorio')
    if centroides is None:
        raise ValueError('centroides es obligatorio')
    categorias = list(categorias)
    if codificador is None:
        raise ValueError('codificador es obligatorio')

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
    doc_id: str = ""
    tiempo_ms: float = 0.0
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
        if self.tipo_clasificador.startswith("sbert+logreg"):
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
                 n_relacionados: int = None,
                 n_keywords: int = None,
                 id_externo: str = None) -> RespuestaContenido:
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
        if n_keywords:
            keywords = keywords[:n_keywords]
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
        respuesta.tiempo_ms = round(dt, 2)
        respuesta.doc_id = id_externo or hashlib.sha1(
            texto_limpio.encode("utf-8")).hexdigest()[:16]
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
