"""Sección 1-2 — Contrato de datos, validación, idioma, ingesta y limpieza."""

from .core import md, code

# ============================== 1. CONTRATO ==============================
md(r'''
---
# 1. El contrato de datos

Antes de escribir código de procesamiento conviene fijar qué entra y qué sale del sistema, porque
todo lo demás —validación, esquema de base de datos, modelos Pydantic del backend, tests— se deriva
de ahí.

## Entrada

```json
{
  "titulo": "Introducción a Spring Boot",
  "texto": "En este contenido se presentan los conceptos básicos para la creación de APIs REST utilizando Java y Spring Boot."
}
```

## Salida (contrato mínimo del brief)

```json
{
  "categoria": "Backend",
  "probabilidad": 0.89,
  "informacion_adicional": ["Java", "Spring Boot", "API REST"]
}
```

## Salida extendida (este proyecto)

El brief admite explícitamente que *"la estructura final de la respuesta podrá variar según el
enfoque elegido por el equipo"*. Añadimos cinco campos, cada uno respaldado por una capacidad real
del pipeline y no por decoración:

| Campo | Tipo | De dónde sale | Para qué sirve |
|---|---|---|---|
| `idioma` | objeto | §2.2 — detección automática | El cliente sabe si el documento se procesó en el idioma soportado |
| `tema` | objeto | §5.6 — BERTopic | Organización emergente, independiente de la taxonomía impuesta |
| `entidades_tecnicas` | lista | §4.1 — EntityRuler | Tecnologías nombradas explícitamente, con precisión de reglas |
| `explicacion` | objeto | §5.5 — ablación por término | Justifica la categoría asignada ante el usuario |
| `relacionados` | lista | §6.4 — ChromaDB | Recomendación de contenido, recurso opcional del brief |

Los tres campos del contrato mínimo **nunca** cambian de nombre ni de tipo: son la frontera estable
entre Ciencia de Datos y Backend, y §7.4 incluye un test que lo verifica.
''')

# ============================== 2. INGESTA ==============================
md(r'''
---
# 2. Ingesta, validación y normalización
### Etapa 1 del diagrama

> **Texto Técnico de Entrada → Validación → Detección de idioma → Extracción → Limpieza**

Esta etapa tiene una responsabilidad que conviene enunciar sin ambigüedad: **garantizar que ningún
dato inválido, corrupto o en un idioma no soportado llegue al modelo**. Es una función de guardia,
no de transformación.

La razón es económica. Un documento malo que pasa la puerta no se detiene: contamina la matriz
TF-IDF con tokens basura, desplaza el centroide de su categoría en el espacio de embeddings, aparece
como vecino en las recomendaciones y, si es lo bastante raro, HDBSCAN lo convierte en un tópico
propio. Detectarlo en la puerta cuesta microsegundos; detectarlo después de entrenar cuesta una
reejecución completa del pipeline.

El brief pide que el corpus lo construya el propio equipo a partir de **fuentes públicas**. Usamos la
**API pública de Wikipedia en español**: es estable, tiene licencia abierta (CC BY-SA), está en el
idioma de los ejemplos del brief y cubre las categorías técnicas del dominio.

**Estrategia de etiquetado.** Cada artículo semilla se asocia a priori a una categoría técnica
(*distant supervision*): el título del artículo actúa como etiqueta débil de todos los párrafos que
contiene. Esto produce un dataset supervisado sin anotación manual, a cambio de asumir ruido en las
etiquetas — limitación que documentamos explícitamente en §9.
''')

md(r'''
## Diagrama 2 — Flujo de ingesta y validación

```mermaid
flowchart TD
    IN["Documento crudo<br/>{titulo, texto}"] --> V0{"¿Tipos correctos<br/>y no nulos?"}
    V0 -->|no| R1["RECHAZO<br/>campo_faltante / tipo_invalido"]
    V0 -->|sí| V1{"¿Codificación<br/>UTF-8 válida?"}
    V1 -->|no| R2["RECHAZO<br/>codificacion_invalida"]
    V1 -->|sí| V2{"¿Contiene mojibake<br/>o caracteres de control?"}
    V2 -->|sí| R3["RECHAZO<br/>texto_corrupto"]
    V2 -->|no| V3{"¿Longitud dentro<br/>de rango?"}
    V3 -->|no| R4["RECHAZO<br/>muy_corto / muy_largo"]
    V3 -->|sí| V4{"¿Ratio de caracteres<br/>alfabéticos aceptable?"}
    V4 -->|no| R5["RECHAZO<br/>contenido_degenerado"]
    V4 -->|sí| L1["Detección de idioma"]

    L1 --> L2{"¿Idioma en<br/>idiomas_soportados?"}
    L2 -->|no| R6["RECHAZO<br/>idioma_no_soportado"]
    L2 -->|baja confianza| W1["ADVERTENCIA<br/>se procesa igual"]
    L2 -->|sí| C1["Limpieza de texto"]
    W1 --> C1

    C1 --> C2["Normalización Unicode NFKC"]
    C2 --> C3["Remoción HTML · URLs · refs"]
    C3 --> C4["Colapso de espacios"]
    C4 --> D1{"¿Duplicado exacto?"}
    D1 -->|sí| R7["DESCARTE<br/>duplicado"]
    D1 -->|no| D2{"¿Near-duplicate<br/>Jaccard > 0.60?"}
    D2 -->|sí| R8["DESCARTE<br/>near_duplicate"]
    D2 -->|no| OK(["Documento aceptado<br/>→ Etapa 2 NLP"])

    R1 --> LOG[("pipeline.log<br/>+ reporte de rechazos")]
    R2 --> LOG
    R3 --> LOG
    R4 --> LOG
    R5 --> LOG
    R6 --> LOG
    R7 --> LOG
    R8 --> LOG
```
''')

# ============================== 2.1 VALIDACIÓN ==============================
md(r'''
## 2.1 Validación de entrada

**El problema que resuelve.** El endpoint `POST /contenido` es una superficie pública: recibe lo que
el cliente mande. FastAPI + Pydantic validan *tipos* (que `titulo` sea `str`, que exista `texto`),
pero no validan *contenido*: para Pydantic, un `texto` de tres caracteres, uno de 40 MB, uno lleno de
bytes de control o uno que es la palabra `"aaaaaaaa"` repetida son todos `str` perfectamente válidos.
La validación semántica es responsabilidad de esta capa, y vive aquí —en Ciencia de Datos— porque
los umbrales dependen de propiedades del modelo, no del protocolo HTTP.

**Los siete controles y por qué cada uno.**

1. **Presencia y tipo.** `titulo` y `texto` son obligatorios y deben ser cadenas. Un `None` que llega
   al modelo produce un `AttributeError` cincuenta líneas más adelante, con un mensaje que no señala
   la causa real.

2. **Codificación UTF-8.** Python 3 maneja `str` como Unicode, así que el fallo típico no es un byte
   inválido sino un **surrogate no emparejado** (`\ud800`-`\udfff`), que se cuela al decodificar con
   `errors="surrogateescape"` y explota recién al serializar la respuesta JSON — es decir, después de
   haber gastado todo el cómputo del pipeline. Lo verificamos con un round-trip `encode/decode`.

3. **Detección de corrupción.** Tres síntomas distintos:
   - **Mojibake**: texto UTF-8 leído como Latin-1, que produce secuencias características
     (`Ã©` por `é`, `Ã±` por `ñ`, `â€œ` por `"`). Es frecuente al importar CSVs generados en Windows.
   - **Carácter de reemplazo** `�`: evidencia de que ya hubo una pérdida de información aguas
     arriba. El texto es irrecuperable.
   - **Caracteres de control** (categoría Unicode `Cc`, excepto tabulador y salto de línea): indican
     que el payload no es texto, o que se coló contenido binario.

4. **Longitud mínima.** Un texto de diez caracteres no tiene señal suficiente para clasificar: el
   embedding resultante queda cerca del centroide global del espacio y la probabilidad de la
   categoría predicha es esencialmente ruido. Distinguimos dos umbrales: `texto_min_chars` (20) para
   inferencia, y `corpus_min_chars` (250) —más estricto— para entrar al corpus de entrenamiento,
   porque un documento de entrenamiento pobre daña a *todo* el modelo, no solo a su propia predicción.

5. **Longitud máxima.** Control defensivo. Sin techo, un cliente puede enviar 40 MB de texto y forzar
   al servidor a cargar todo en memoria y ejecutar spaCy sobre ello, bloqueando el event loop de
   FastAPI. El techo de 50.000 caracteres cubre con holgura cualquier artículo técnico real.

6. **Ratio de caracteres alfabéticos.** Un texto donde más del 45 % de los caracteres no son letras
   ni espacios no es prosa: es una tabla, un volcado de log, un bloque de código o binario mal
   decodificado. spaCy lo tokenizará, pero el resultado no será información.

7. **Ratio de mayúsculas.** Más del 80 % en mayúsculas indica un encabezado degenerado o texto
   gritado; la lematización de spaCy pierde precisión porque el POS tagger usa la capitalización como
   señal.

**Contrato de error.** `validar_entrada()` **no lanza excepciones**: devuelve un `ResultadoValidacion`
con la lista de errores tipificados. La decisión de lanzar o no corresponde a quien llama —el
notebook filtra en silencio y reporta al final; el backend traduce el código de error a un HTTP 422
con mensaje legible. Separar *detección* de *reacción* es lo que permite reutilizar el mismo
validador en ambos contextos.
''')

code(r'''
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
''')

code(r'''
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
display(pd.DataFrame(filas))
''')

# ============================== 2.2 IDIOMA ==============================
md(r'''
## 2.2 Detección de idioma

**El problema que resuelve.** Todo el pipeline está calibrado para español: el modelo de spaCy es
`es_core_news_sm`, las stopwords son españolas, el corpus de entrenamiento es Wikipedia ES y las
categorías se aprendieron sobre vocabulario español. Un documento en inglés no falla ruidosamente —
falla en silencio, que es peor: spaCy lo tokeniza, no reconoce sus stopwords, lematiza mal, y el
clasificador emite una categoría con una probabilidad que parece confiable pero no lo es.

**El método: dos niveles con degradación explícita.**

*Nivel 1 — `langdetect`.* Puerto Python del detector de Nakagawa & Matsumoto, basado en perfiles de
n-gramas de caracteres sobre 55 idiomas. Devuelve una distribución de probabilidad, no solo una
etiqueta, lo cual permite aplicar el umbral `confianza_minima`. Es una dependencia pequeña y sin
modelos que descargar. Su debilidad conocida son los textos muy cortos (menos de ~30 caracteres),
donde los n-gramas no alcanzan a discriminar; por eso lo aplicamos después de la validación de
longitud mínima, nunca antes.

*Nivel 2 — heurística de stopwords.* Si `langdetect` no está instalado o falla, calculamos qué
fracción de los tokens del documento pertenece a la lista de stopwords de cada idioma soportado. El
español y el inglés tienen conjuntos de palabras funcionales casi disjuntos (`de/la/que/el` frente a
`the/of/and/to`), así que el ratio discrimina bien. Es menos preciso que `langdetect`, pero no tiene
dependencias y nunca deja al pipeline sin respuesta.

**Determinismo.** `langdetect` es estocástico por diseño: muestrea n-gramas, de modo que dos llamadas
sobre el mismo texto pueden diferir. `DetectorFactory.seed = 0` —fijado en §0.5— lo vuelve
determinista, requisito para que el pipeline sea reproducible.

**Arquitectura para multilenguaje.** El soporte de inglés no está activo, pero la arquitectura ya lo
contempla y añadirlo no requiere reescribir nada:

1. `CFG.idioma.idiomas_soportados` pasa de `("es",)` a `("es", "en")`.
2. `CFG.idioma.modelos_spacy` ya mapea `"en" → "en_core_web_sm"`; basta descargar el modelo.
3. `RegistroIdiomas` carga el pipeline de spaCy correspondiente **bajo demanda** y lo cachea, de modo
   que el documento se procesa con el modelo de *su* idioma, no con uno fijo.
4. El modelo de embeddings ya es multilingüe (`paraphrase-multilingual-MiniLM-L12-v2`): proyecta
   español e inglés al **mismo espacio vectorial**, así que un documento en inglés es directamente
   comparable con uno en español sin traducción intermedia. Esta es la razón concreta por la que
   `Technology_Architecture.md` §6 eligió un modelo multilingüe en vez de uno monolingüe más pequeño.

Lo único que faltaría es reentrenar el clasificador con documentos etiquetados en inglés — y por eso
`idiomas_soportados` sigue en `("es",)`: la arquitectura está lista, los datos no.
''')

code(r'''
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
            get_ipython().system(f"python -m spacy download {nombre_modelo}")
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

log.info(f"Idiomas soportados en esta versión: {CFG.idioma.idiomas_soportados}")
display(pd.DataFrame(filas))
''')

# ============================== 2.3 SEMILLAS Y SCRAPING ==============================
md(r'''
## 2.3 Recolección del corpus

Cargamos el archivo de semillas y consultamos la **MediaWiki Action API**
(`action=query&prop=extracts&explaintext=1`), que devuelve el texto plano del artículo sin marcado
wiki. Por artículo extraemos hasta `max_docs_por_semilla` párrafos que superen el umbral de longitud;
cada párrafo se convierte en un **documento independiente** del corpus, lo cual es coherente con el
caso de uso del brief (fragmentos técnicos de una o pocas oraciones, como el ejemplo de Spring Boot).

El interruptor `CFG.corpus.usar_fallback` permite ejecutar todo el pipeline sin red, cargando
`corpus_fallback.csv`. Es una red de seguridad para la demo, no la ruta principal.
''')

code(r'''
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

display(semillas.groupby("categoria").size().rename("artículos").to_frame())
semillas.head()
''')

md(r'''
### 2.3.2 Arquitectura de extractores

**El problema.** La primera versión tenía una única función, `extraer_texto_wikipedia`, atada a la
MediaWiki API. Cambiar de fuente significaba reescribir la ingesta. Y el brief no exige Wikipedia:
exige *fuentes públicas*, en plural.

**La solución: un registro de extractores.** Cada fuente es una función que recibe una fila de
semillas y devuelve texto plano. Se registra en el diccionario `EXTRACTORES` y el resto del pipeline
no se entera de cuál se usó. Añadir una fuente nueva es escribir una función y registrarla; no tocar
nada más.

**La convención que lo hace posible.** Todos los extractores devuelven texto plano con los títulos de
sección marcados como `== Título ==`. Es el formato que ya emitía la MediaWiki API, así que
`partir_en_documentos` —que trocea en párrafos y descarta secciones no informativas— sigue
funcionando sin cambios, sea cual sea la fuente.

| Extractor | Fuente | Columna de semillas |
|---|---|---|
| `wikipedia` | MediaWiki Action API | `titulo_wikipedia` |
| `html` | Cualquier web con HTML servido por el servidor | `url` |

### Tres decisiones que la documentación técnica impone

**1. El código hay que tratarlo en dos niveles, no en uno.** Una página de documentación es mitad
prosa, mitad ejemplos, y la distinción importa:

- Los **bloques** `<pre>` son ejemplos de varias líneas. Se eliminan enteros: un `const [x, setX] =
  useState('')` dispara el filtro de `contenido_degenerado` de §2.1, que rechaza documentos con más
  del 45 % de caracteres no alfabéticos.
- El código **en línea** —`usa el hook `useState` para gestionar el estado`— es distinto. Ahí la
  etiqueta se **desenvuelve** conservando el texto. Borrarla dejaría *"usa el hook para gestionar el
  estado"*, eliminando de la prosa justo los nombres de tecnologías de los que viven el clasificador
  léxico y el `EntityRuler`.

Tratar ambos igual —el error de la primera versión de esta celda— destruye la señal que se pretende
capturar. Se eliminan además `<nav>`, `<footer>`, `<aside>` y `<script>`.

**2. Hay que respetar `robots.txt`.** Wikipedia publica una API pensada para consumo automatizado;
un sitio de documentación cualquiera, no. Antes de la primera petición a cada dominio se consulta su
`robots.txt` y se cachea el veredicto. Es cortesía y es lo correcto, y con `CFG.corpus.contacto` bien
puesto además evita bloqueos.

**3. Sin JavaScript no hay contenido en algunas webs.** El extractor descarga HTML crudo: no ejecuta
JavaScript. Una web de documentación renderizada en el cliente devolvería un esqueleto vacío. Por eso
se avisa cuando una página produce menos de `min_chars_pagina` caracteres: casi siempre significa que
ese sitio necesita un navegador, no un `requests.get`.
''')

code(r'''
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
                   "extractor": 0, "irrecuperables": 0}

# Excepciones que NO tiene sentido reintentar: el resultado será idéntico.
NO_REINTENTABLES = (
    requests.exceptions.TooManyRedirects,   # bucle de redirección del servidor
    requests.exceptions.MissingSchema,      # URL sin http:// o https://
    requests.exceptions.InvalidURL,
    requests.exceptions.InvalidSchema,
    requests.exceptions.URLRequired,
)

_PISTA_IRRECUPERABLE = {
    "TooManyRedirects":
        "El servidor entra en bucle de redirección, normalmente por negociación "
        "de idioma o cookies. Abre la URL en el navegador, copia la dirección "
        "final a la que llega y ponla en el CSV.",
    "MissingSchema":  "Falta el esquema: la URL debe empezar por https://.",
    "InvalidURL":     "La URL está malformada; revisa esa fila del CSV.",
    "InvalidSchema":  "Esquema no soportado (¿ftp://, file://?).",
    "URLRequired":    "La fila no trae URL.",
}

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

        except NO_REINTENTABLES as exc:
            # Bucles de redirección, URLs malformadas o fallos de certificado son
            # DETERMINISTAS: dependen de la configuración del servidor o de la
            # propia URL, no del estado de la red. Reintentarlos gasta seis
            # peticiones y llena el log de ruido que oculta la causa real.
            ESTADO_SCRAPING["irrecuperables"] += 1
            log.error(f"{type(exc).__name__} en {url}: no se reintenta porque el "
                      f"fallo es determinista. {_PISTA_IRRECUPERABLE.get(type(exc).__name__, '')}")
            return ""

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
print("\nDefiniciones listas. El sondeo (§2.3.3) va antes de la recolección (§2.3.4).")''')

md(r'''
### 2.3.3 Sondeo previo de las semillas

**Por qué antes y no después.** Un scraping completo tarda minutos. Descubrir al final que la mitad
de las URLs devuelven 404, o que el sitio es una SPA sin contenido servido, es tirar ese tiempo. Esta
celda sondea una muestra —o todas, si el archivo es pequeño— y responde tres preguntas por fila:

| Comprobación | Qué detecta |
|---|---|
| Código HTTP | URLs muertas o movidas, bloqueos por User-Agent |
| Caracteres útiles tras extraer | Webs renderizadas en el cliente, que devuelven un esqueleto |
| Idioma detectado | Documentación que resultó estar en inglés pese a la URL `/es/` |

**La tercera importa más de lo que parece.** La documentación oficial en español tiene cobertura
desigual: MDN, React, Vue, Kubernetes, Django y Python mantienen traducciones amplias; Docker, Spring
y PostgreSQL no las tienen. Si una URL `/es/` cae de vuelta al inglés, esos documentos se rechazarán
en §2.4.2 por `idioma_no_soportado` y esa categoría se quedará sin material. Mejor saberlo aquí.

> Sondea con `SONDEAR_TODAS = False` una muestra por categoría (rápido). Ponlo en `True` para
> verificar el archivo entero antes de una corrida seria.
''')

code(r'''
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
    display(sondeo)

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
''')

md(r'''
### 2.3.4 Recolección del corpus

Con las semillas ya sondeadas, se lanza la recolección. Es la operación más lenta del notebook: a
`pausa_entre_peticiones` por dominio y con reintentos, un archivo de ~56 semillas tarda entre uno y
tres minutos.

Está en su propia celda por una razón práctica: **el sondeo tiene que poder ejecutarse sin
desencadenar el scraping**. Cuando ambas cosas vivían en la misma celda, el sondeo se ejecutaba
después de la recolección y no prevenía nada — que es exactamente lo que ocurrió en la primera
corrida con documentación oficial.
''')

code(r'''
# @title 2.3.4 — Recolección del corpus
with etapa("ingesta de datos"):
    df_bruto = recolectar_corpus(semillas)

print(f"\n>>> Corpus bruto: {len(df_bruto)} documentos, "
      f"{df_bruto['categoria'].nunique()} categorías")
''')

md(r'''
### 2.3.5 Cobertura del scraping

**Por qué esta celda existe.** Un scraping puede "funcionar" y aun así producir un corpus
inservible. El caso concreto que motivó añadirla: 49 documentos recolectados —cifra que parece
razonable— pero **todos de una única categoría**, porque las peticiones empezaron a fallar tras las
primeras diez semillas y el archivo de semillas venía agrupado por categoría.

El total de documentos no revela ese problema. La cobertura *por categoría*, sí. Esta celda la hace
visible de inmediato, junto con el desglose de por qué falló cada petición: artículo inexistente,
límite de tasa, bloqueo por User-Agent o error de red. Cada causa tiene una corrección distinta y
conviene saber cuál aplicar antes de reintentar.
''')

code(r'''
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
    display(cobertura)

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
        if ESTADO_SCRAPING["irrecuperables"]:
            print(f"  Causa: {ESTADO_SCRAPING['irrecuperables']} URL(s) con fallo")
            print("  DETERMINISTA (bucle de redirección, URL malformada). No se")
            print("  reintentan porque el resultado sería idéntico. Revisa el log:")
            print("  indica qué hacer con cada una.")
        elif ESTADO_SCRAPING["extractor"]:
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
''')

# ============================== 2.4 LIMPIEZA ==============================
md(r'''
## 2.4 Limpieza y normalización

Diez transformaciones en orden fijo. El orden importa: normalizar Unicode **antes** de aplicar los
patrones de ruido evita que una comilla tipográfica `’` sobreviva como carácter especial, y colapsar
espacios **al final** recoge los huecos que dejan las sustituciones anteriores.

| # | Transformación | Por qué |
|---|---|---|
| 1 | Manejo de nulos | Un `NaN` de pandas o un `None` no es `str`; devolvemos cadena vacía en vez de propagar el error |
| 2 | Remoción de HTML | BeautifulSoup extrae el texto; un `<div>` residual sería tokenizado como palabra |
| 3 | Remoción de URLs | No aportan señal temática y sí muchos tokens espurios |
| 4 | Remoción de referencias | `[1]`, `[cita requerida]` son artefactos de Wikipedia, no contenido |
| 5 | **Normalización Unicode NFKC** | Unifica formas equivalentes: `ﬁ`→`fi`, `＝`→`=`, comillas tipográficas→rectas. Sin esto, `café` (é precompuesta) y `café` (e + acento combinante) serían tokens distintos |
| 6 | Remoción de paréntesis triviales | `()`, `(a)` quedan tras quitar referencias |
| 7 | Remoción de caracteres especiales | Conserva letras acentuadas, `#`, `+`, `/`, `-` y `.` porque son parte de nombres técnicos (`C#`, `C++`, `Node.js`, `CI/CD`) |
| 8 | Colapso de espacios | Un único espacio entre tokens |
| 9 | **Minúsculas** | Aplicado en el paso de lematización (§4.2), **no aquí**: el POS tagger y el NER de spaCy usan la capitalización como señal, y `Java` en minúscula deja de ser reconocible como entidad. Bajar a minúsculas antes de spaCy degradaría ambos |
| 10 | Deduplicación | Exacta por hash, más near-duplicate por Jaccard sobre shingles |

**Sobre la deduplicación near-duplicate.** Wikipedia repite pasajes casi idénticos entre artículos
relacionados (la introducción de "Spring Framework" y la de "Spring Boot" comparten frases). Un
duplicado exacto lo atrapa un hash; uno que difiere en tres palabras, no. Comparamos conjuntos de
*shingles* de 5 palabras por similitud de Jaccard: si dos documentos comparten más del
`umbral_near_duplicate` de sus shingles, conservamos solo el más largo.

**Calibración del umbral: medida, no supuesta.** La intuición engaña aquí. Cambiar **una sola
palabra** no baja el Jaccard un poco: rompe `n` shingles de golpe, uno por cada ventana que contenía
esa palabra. Medimos el efecto sobre un documento real de 35 palabras, variando la ventana `n` y el
número de palabras alteradas:

| ventana `n` | 1 palabra | 2 palabras | 3 palabras | 5 palabras | 8 palabras | solapamiento del 50 % |
|---|---|---|---|---|---|---|
| **3** | 0.83 | 0.74 | **0.61** | 0.40 | 0.27 | **0.36** |
| 4 | 0.78 | 0.68 | 0.56 | 0.33 | 0.19 | 0.34 |
| 5 | 0.72 | 0.63 | 0.51 | 0.27 | 0.15 | 0.33 |

Un umbral útil tiene que quedar **por debajo** de la columna "pocas palabras cambiadas" —para
atrapar esos casos, que son near-duplicates genuinos— y **por encima** de la última columna, que
representa dos documentos distintos que comparten un pasaje y que no se deben borrar.

Con `n = 3` esas dos regiones están bien separadas: 0.61 frente a 0.36. Fijamos el umbral en
**0.60**, justo en medio. Con la configuración inicial (`n = 5`, umbral 0.92) el detector no atrapaba
absolutamente nada que el hash exacto no hubiera capturado ya — la etapa existía sin hacer trabajo.

El costo es cuadrático en el número de documentos: aceptable para el orden de magnitud de este corpus
(cientos), inviable a partir de decenas de miles. Queda anotado en §9.3 como migración a MinHash/LSH.
''')

code(r'''
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
''')

code(r'''
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
    display(df_rechazados["motivo"].value_counts().rename("documentos").to_frame())

df_raw[["doc_id", "categoria", "titulo", "idioma", "n_chars"]].head(8)
''')

md(r'''
### 2.4.3 Diagnóstico de salud del corpus

**Por qué existe esta celda.** Las etapas anteriores descartan documentos por seis motivos distintos.
Cada descarte es individualmente correcto, pero **el efecto acumulado puede vaciar el corpus sin que
nada falle**: el pipeline sigue adelante, entrena un modelo con dos categorías y tres documentos, y
el problema solo aparece cincuenta celdas más tarde, disfrazado de error críptico de scikit-learn.

Ese fallo silencioso es el peor modo posible: el error que se ve no señala la causa. Un
`ValueError: y_true is binary while y_score is 2d` en la celda de métricas no dice *"tu scraping
falló y solo sobrevivieron dos categorías"*, que es lo que realmente pasó.

Esta celda verifica cuatro condiciones que el resto del pipeline da por supuestas:

| Condición | Por qué importa | Qué rompe si falla |
|---|---|---|
| ≥ 2 categorías | Sin dos clases no hay nada que clasificar | `LogisticRegression.fit` |
| ≥ 3 categorías | Con 2 el problema es binario | `top_k_accuracy_score` con `k=2` |
| ≥ 2 documentos por categoría | El split estratificado necesita repartir | `train_test_split(stratify=y)` |
| ≥ `cv_folds` documentos por categoría | Cada fold necesita un ejemplar | `StratifiedKFold` |

**Y además recupera el pipeline, no solo avisa.** Diagnosticar sin actuar deja al equipo con un
notebook roto y un mensaje. Si el corpus recolectado resulta inservible y
`CFG.corpus.fallback_automatico` está activo, la celda **recarga `corpus_fallback.csv`** —80
documentos redactados por el equipo, 8 categorías equilibradas— y reejecuta la normalización sobre
él. El pipeline continúa con datos utilizables y el cambio queda registrado en el log, impreso en
pantalla y anotado en `metadata.json`, de modo que nadie presente resultados del respaldo creyendo
que son del scraping.

Las causas probables se enumeran **en orden de frecuencia observada**, porque casi siempre es una de
tres: el scraping no llegó a Wikipedia, el umbral `corpus_min_chars` es demasiado exigente para los
párrafos obtenidos, o el rechazo por idioma se llevó por delante documentos legítimos que
`langdetect` no supo clasificar en fragmentos cortos.

> **Honestidad en la entrega.** El brief pide construir el corpus desde fuentes públicas, y eso lo
> cumple la ruta de Wikipedia. El respaldo es una red de seguridad para que la demo no se caiga: si
> se activa, hay que decirlo. Con 80 documentos limpios y temáticamente separables, las métricas
> además saldrán optimistas respecto de un corpus recolectado real.
''')

code(r'''
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

display(pd.Series(SALUD["conteos"], name="documentos").to_frame())
''')

# ============================== 2.7 EDA ==============================
md(r'''
## 2.5 EDA — Exploración del corpus recolectado

Primer requisito explícito del brief: *"Exploración y limpieza de datos (EDA)"*. Revisamos tres cosas
porque las tres condicionan decisiones posteriores concretas:

- **Balance de clases** → determina si las métricas deben ser macro-promediadas (§5.3) y si el split
  necesita estratificación.
- **Distribución de longitudes** → un corpus con documentos muy cortos degrada tanto TF-IDF (pocas
  estadísticas) como los embeddings (poco contexto).
- **Confianza de la detección de idioma** → documentos con confianza baja son candidatos a ser texto
  mixto o muy técnico, donde el detector se apoya en poca señal léxica natural.
''')

code(r'''
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
''')
