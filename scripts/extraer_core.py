"""Genera src/app/ml/core.py extrayendo de pipeline.py solo lo que la API necesita.

Migracion de un solo uso. Tras ejecutarlo y commitear core.py, este script
puede borrarse: core.py pasa a ser la fuente de verdad de la capa de servicio.

Los numeros de linea corresponden al commit ed4f9cd.
"""
import re
import textwrap
from pathlib import Path

P = Path("src/app/ml/pipeline.py")
L = P.read_text(encoding="utf-8").splitlines(keepends=True)


def R(a, b, dedent=False):
    s = "".join(L[a - 1:b])
    return textwrap.dedent(s) if dedent else s


CAB = '''"""Nucleo de inferencia de TechMind — sin dependencias del notebook.

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

'''

# Rangos validos para ed4f9cd. Incluyen la linea del decorador @dataclass.
PARTES = [
    R(107, 381),           # Config dataclasses + CFG  (sin .crear() ni prints)
    R(394, 463),           # fijar_semillas — se define, NO se invoca
    R(693, 944),           # CodigoError .. exigir_valido (sin la bateria de tests)
    R(984, 1130),          # Idioma, detectar_idioma, RegistroIdiomas
    R(1926, 1994, True),   # RE_* + componer_entrada + limpiar_texto
    R(2429, 2490, True),   # TECNOLOGIAS + construir_pipeline_nlp
    R(2503, 2503, True),   # STOPWORDS_EXTRA
    R(2506, 2557, True),   # preprocesar
    R(2774, 2861, True),   # CacheEmbeddings
    R(3005, 3047, True),   # es_keyword_valida
    R(3050, 3141, True),   # rankear_keywords
    R(3716, 3791, True),   # explicar_prediccion
    R(4731, 4778, True),   # RespuestaContenido
    R(4785, len(L)),       # TechMindInference
]
txt = CAB + "\n\n".join(p.rstrip() + "\n" for p in PARTES)

# Eliminar los fallbacks a globales de entrenamiento.
SUST = [
    (r"modelo_keybert = modelo_keybert if modelo_keybert is not None else kw_model",
     "if modelo_keybert is None:\n        raise ValueError('modelo_keybert es obligatorio')"),
    (r"extractor = extractor if extractor is not None else extractor_yake",
     "if extractor is None:\n        raise ValueError('extractor es obligatorio')"),
    (r"pipeline_nlp = pipeline_nlp if pipeline_nlp is not None else nlp",
     "if pipeline_nlp is None:\n        raise ValueError('pipeline_nlp es obligatorio')"),
    (r"mapa_tecnologias = mapa_tecnologias if mapa_tecnologias is not None else _MAPA_TECNOLOGIAS",
     "mapa_tecnologias = mapa_tecnologias or {t.lower(): t for t in TECNOLOGIAS}"),
    (r"stop_words=_STOPWORDS_SPACY,",
     "stop_words=list(pipeline_nlp.Defaults.stop_words),"),
    (r"predictor = predictor or _predecir_probabilidades",
     "if predictor is None:\n        raise ValueError('predictor es obligatorio')"),
    (r"centroides = centroides if centroides is not None else CENTROIDES",
     "if centroides is None:\n        raise ValueError('centroides es obligatorio')"),
    (r"categorias = list\(categorias\) if categorias is not None else CATEGORIAS",
     "categorias = list(categorias)"),
    (r"codificador = codificador or \(lambda ts: CACHE\.codificar\(ts, modelo_embeddings\)\)",
     "if codificador is None:\n        raise ValueError('codificador es obligatorio')"),
]
for pat, rep in SUST:
    txt, n = re.subn(pat, rep, txt)
    if n == 0:
        print(f"  AVISO: no se sustituyo -> {pat[:60]}")

Path("src/app/ml/core.py").write_text(txt, encoding="utf-8")
print(f"OK core.py generado: {len(txt.splitlines())} lineas")