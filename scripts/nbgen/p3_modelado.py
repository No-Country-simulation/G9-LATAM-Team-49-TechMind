"""Sección 5 — Representaciones, keywords, clasificación, explicabilidad, clustering y versionado."""

from .core import md, code

# ============================== 5. MODELADO ==============================
md(r'''
---
# 5. Extracción de keywords y modelado
### Etapa 3 del diagrama

El diagrama del brief plantea una **bifurcación** explícita en el nodo *Enfoque de Modelado*:

```
                    ┌─ Estadístico / Frecuencia ─► TF-IDF o YAKE ─┐
   Enfoque de ──────┤                                             ├─► Ranking & Selección
    Modelado        └─ Semántico / Embeddings ──► KeyBERT/SBERT ──┘         │
                                                                            ▼
                                                              Agrupamiento / Clustering
```

**Implementamos ambas ramas y las comparamos** — no elegimos una a ciegas. Esto responde al diagrama,
que las presenta como alternativas, y al criterio de rigor analítico de `Technology_Architecture.md`
§7: mostrar que se evaluó más de un enfoque antes de adoptar uno como método de producción.

La sección se organiza en siete bloques:

| Bloque | Qué produce | Sección |
|---|---|---|
| Representaciones | Matriz TF-IDF dispersa + matriz de embeddings densa | §5.1 |
| Keywords | `informacion_adicional` por fusión de cuatro señales | §5.2 |
| Clasificación | `categoria` + `probabilidad` | §5.3 |
| Evaluación | Accuracy, precision, recall, F1, matriz de confusión, CV | §5.4 |
| Explicabilidad | Por qué el modelo decidió lo que decidió | §5.5 |
| Clustering | Tópicos emergentes con etiquetas legibles | §5.6 |
| Serialización | Artefactos `.joblib` + `metadata.json` versionado | §5.7 |
''')

md(r'''
## Diagrama 4 — Flujo de generación de embeddings (con caché)

```mermaid
flowchart TD
    IN["Lote de textos limpios"] --> H["Para cada texto:<br/>clave = SHA256(modelo ‖ normalizar ‖ texto)"]
    H --> Q{"¿clave presente<br/>en la caché?"}

    Q -->|"sí · acierto"| HIT["Recuperar vector<br/>del diccionario en memoria"]
    Q -->|"no · fallo"| MISS["Acumular en lote pendiente"]

    MISS --> BATCH{"¿Lote pendiente<br/>no vacío?"}
    BATCH -->|no| ENS
    BATCH -->|sí| ENC["SentenceTransformer.encode<br/>batch_size = 32"]

    subgraph ENCODER["Encoder · paraphrase-multilingual-MiniLM-L12-v2"]
        T1["Tokenización WordPiece"]
        T2["12 capas Transformer"]
        T3["Mean pooling sobre tokens"]
        T4["Normalización L2"]
        T1 --> T2 --> T3 --> T4
    end

    ENC --> ENCODER
    ENCODER --> UPD["Escribir vectores nuevos<br/>en la caché"]
    UPD --> PERSIST[("embeddings_cache.joblib<br/>en disco")]
    UPD --> ENS

    HIT --> ENS["Ensamblar matriz<br/>en el orden de entrada"]
    ENS --> OUT["ndarray (n_docs, 384)<br/>float32, norma 1"]

    OUT --> U1["Clasificación §5.3"]
    OUT --> U2["KeyBERT §5.2"]
    OUT --> U3["BERTopic §5.6"]
    OUT --> U4["ChromaDB §6"]

    style ENCODER fill:#e8f4ea,stroke:#2d6a4f
    style PERSIST fill:#fdf4e3,stroke:#b07d2b
```

**Por qué la caché importa.** Codificar es la operación más costosa del pipeline: domina el tiempo de
ejecución del notebook y, en producción, la latencia de `POST /contenido`. Como los vectores son
deterministas dada la tupla (modelo, texto), recalcularlos es trabajo puro desperdiciado. La caché
convierte el ciclo de desarrollo —donde uno reejecuta el notebook decenas de veces cambiando solo el
clasificador— de minutos a segundos, y en producción evita recodificar contenido reenviado.

La clave incluye el **nombre del modelo** y el flag de **normalización**: cambiar de modelo invalida
automáticamente las entradas afectadas, sin necesidad de purgar la caché a mano.
''')

# ============================== 5.1 REPRESENTACIONES ==============================
md(r'''
## 5.1 Las dos representaciones del texto

Un mismo documento se representa de dos maneras que capturan cosas distintas, y esa diferencia es el
objeto del experimento de §5.3.

**TF-IDF (léxica, dispersa, ~8.000 dimensiones).** Pondera cada término por su frecuencia en el
documento penalizada por su frecuencia en el corpus: un término que aparece en todos los documentos
no discrimina. La alimentamos con los lemas ya filtrados por POS, de modo que la matriz contenga solo
sustantivos, nombres propios y adjetivos en forma canónica. Es interpretable —cada dimensión es una
palabra— y muy barata de calcular, pero ciega a los sinónimos.

**Embeddings SBERT (semántica, densa, 384 dimensiones).** Un vector por documento donde la cercanía
geométrica corresponde a cercanía de significado. Los calculamos sobre el **texto limpio completo**,
no sobre los lemas: el modelo fue entrenado con lenguaje natural íntegro, y la sintaxis y las
palabras funcionales aportan contexto que el filtrado POS destruiría.

Ese detalle —qué texto recibe cada representación— es fácil de equivocar y tiene consecuencias
medibles: alimentar SBERT con lemas sueltos degrada la calidad del vector, y alimentar TF-IDF con
prosa completa llena la matriz de stopwords.
''')

code(r'''
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
''')

code(r'''
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
''')

code(r'''
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
''')

# ============================== 5.2 KEYWORDS ==============================
md(r'''
## 5.2 Extracción y ranking de palabras clave

Cuatro señales, cada una con una debilidad que las otras compensan:

| Señal | Qué capta | Dónde falla |
|---|---|---|
| **KeyBERT** (semántica) | Frases cuyo significado resume el documento | Puede omitir un nombre propio técnico poco frecuente |
| **YAKE** (estadística) | Términos distintivos sin necesitar corpus | Sensible a la posición; ignora el significado |
| **TF-IDF** (estadística de corpus) | Lo que distingue este documento del resto | Solo aplicable a documentos del corpus de ajuste |
| **EntityRuler** (reglas) | Tecnologías nombradas, con precisión de 100 % | Cobertura finita: solo lo que está en el diccionario |

**Cómo se fusionan: Reciprocal Rank Fusion.** El obstáculo obvio es que los puntajes no son
comparables entre sí — KeyBERT devuelve similitudes coseno en [0, 1], YAKE devuelve un score
*inverso* (menor es mejor) y TF-IDF devuelve pesos sin escala fija. Normalizarlos exigiría suposiciones
sobre sus distribuciones que no tenemos motivo para hacer.

RRF esquiva el problema: **ignora los puntajes y usa solo el orden**. Cada método vota por sus
candidatos y el voto de la posición `r` vale `peso / (K + r)`. Un término que aparece razonablemente
alto en tres listas distintas supera a uno que encabeza una sola. `K = 60` es la constante estándar
de la literatura de fusión de rankings; amortigua las diferencias entre las primeras posiciones para
que un primer puesto no aplaste automáticamente a un segundo.

Las entidades del EntityRuler llevan peso `1.5`, superior al resto, por una razón concreta: su
precisión es perfecta por construcción. Si el diccionario dice que "Spring Boot" es una tecnología y
aparece en el texto, no hay incertidumbre que ponderar.

**Post-proceso.** Deduplicación por solapamiento —si ya seleccionamos "api rest", descartamos "api"—
y restauración de la capitalización canónica de las tecnologías conocidas, para que la salida diga
`"Spring Boot"` y no `"spring boot"`.
''')

code(r'''
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
''')

code(r'''
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
    display(comparacion)
''')

code(r'''
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
''')

code(r'''
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

display(df[["doc_id", "categoria", "titulo", "keywords"]].head(8))
''')

# ============================== 5.3 CLASIFICACIÓN ==============================
md(r'''
## 5.3 Clasificación temática

El brief pide `categoria` y `probabilidad`. Entrenamos **el mismo clasificador sobre dos
representaciones distintas** para aislar el efecto de la representación — que es la variable que
realmente queremos estudiar:

| Modelo | Representación | Hipótesis |
|---|---|---|
| **A — Baseline léxico** | TF-IDF sobre lemas filtrados por POS | Separa bien si las categorías usan vocabularios disjuntos |
| **B — Semántico** | Embeddings SBERT (384 dims) | Generaliza a sinónimos y paráfrasis no vistas en el entrenamiento |

Ambos usan **Regresión Logística** con idénticos hiperparámetros. Cualquier diferencia de desempeño
es atribuible a la representación, no al algoritmo — esa es la razón de mantener el clasificador fijo.

**Decisiones de diseño y su justificación:**

- **`stratify=y`** en el split: el corpus está desbalanceado (§2.5); sin estratificar, una categoría
  minoritaria puede quedar sin representación en test y su recall sería indefinido.
- **`class_weight="balanced"`**: penaliza más los errores en clases minoritarias, evitando que el
  modelo maximice accuracy simplemente prediciendo siempre la clase mayoritaria.
- **`C=5.0`**: regularización relativamente débil. Con muchas dimensiones y pocos documentos, una
  regularización fuerte aplana los coeficientes hasta volver al modelo indeciso.
- **`max_iter=2000`**: `lbfgs` no converge con el valor por defecto (100) en espacios de 8.000
  dimensiones, y una advertencia de no convergencia silenciada produce un modelo mal ajustado.
''')

md(r'''
## Diagrama 5 — Flujo de entrenamiento

```mermaid
flowchart TD
    CORPUS[("corpus_processed.csv<br/>n documentos etiquetados")] --> LE["LabelEncoder<br/>categoría → índice entero"]
    LE --> SPLIT{"train_test_split<br/>test_size=0.25<br/>stratify=y<br/>random_state=42"}

    SPLIT -->|75%| TR["Conjunto de entrenamiento"]
    SPLIT -->|25%| TE["Conjunto de prueba<br/>(intocado hasta §5.4)"]

    TR --> RA["Representación A<br/>TF-IDF sobre lemas POS"]
    TR --> RB["Representación B<br/>Embeddings SBERT"]

    RA --> MA["Modelo A<br/>LogisticRegression<br/>C=5.0 · balanced"]
    RB --> MB["Modelo B<br/>LogisticRegression<br/>C=5.0 · balanced"]

    MA --> EV
    MB --> EV
    TE --> EV["Evaluación §5.4<br/>accuracy · precision · recall<br/>F1 macro/weighted · top-2"]

    EV --> CV["Validación cruzada<br/>StratifiedKFold 5-fold<br/>sobre el corpus completo"]
    CV --> SEL{"¿Qué modelo tiene<br/>mayor F1-macro?"}

    SEL -->|"modelo A"| GA["Ganador: TF-IDF + LogReg"]
    SEL -->|"modelo B"| GB["Ganador: SBERT + LogReg"]

    GA --> SER
    GB --> SER["Serialización §5.7"]

    SER --> ART1["modelo_clasificacion.joblib"]
    SER --> ART2["label_encoder.joblib"]
    SER --> ART3["vectorizador_tfidf.joblib"]
    SER --> ART4["modelo_bertopic/"]
    SER --> ART5["metadata.json<br/>versión · fecha · hiperparámetros<br/>métricas · dataset · hash"]
    SER --> ART6["config.json"]

    ART1 --> OCI[("OCI Object Storage §8.1")]
    ART5 --> OCI

    style TE fill:#fdeaea,stroke:#a4342f
    style SER fill:#e8f4ea,stroke:#2d6a4f
```

**Por qué el conjunto de prueba aparece marcado.** No participa en ninguna decisión hasta §5.4: ni en
el ajuste del vectorizador, ni en la selección de hiperparámetros, ni en la elección del modelo
ganador salvo como métrica final. Ajustar TF-IDF sobre el corpus completo antes de partirlo —error
frecuente— filtra estadísticas del test al entrenamiento y produce métricas optimistas que no se
sostienen en producción. Aquí `modelo_a` es un `Pipeline` que incluye el vectorizador, de modo que
`fit` solo ve datos de entrenamiento.
''')

code(r'''
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
''')

code(r'''
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
''')

# ============================== 5.4 EVALUACIÓN ==============================
md(r'''
## 5.4 Evaluación

Siete instrumentos, cada uno respondiendo una pregunta distinta. El brief pide "métricas de desempeño
apropiadas"; con un corpus desbalanceado, "apropiadas" tiene un significado técnico preciso.

| Métrica | Pregunta que responde | Cuándo engaña |
|---|---|---|
| **Accuracy** | ¿Qué fracción del total acertó? | Con clases desbalanceadas: predecir siempre la mayoritaria puede dar 60 % |
| **Precision (macro)** | De lo que predijo como categoría X, ¿cuánto era X? | Ignora lo que dejó de encontrar |
| **Recall (macro)** | De todo lo que era X, ¿cuánto encontró? | Ignora los falsos positivos |
| **F1 (macro)** | Media armónica de ambas, promediada **por clase con igual peso** | — es nuestra métrica de decisión |
| **F1 (weighted)** | Igual, pero ponderando por frecuencia de clase | Las clases mayoritarias dominan el promedio |
| **Top-2 accuracy** | ¿La categoría correcta estuvo entre las dos más probables? | Sobreestima si hay pocas clases |
| **Matriz de confusión** | ¿*Qué* confunde con *qué*? | No es un escalar; hay que leerla |

**Por qué F1-macro es la métrica de decisión.** Macro promedia por clase con igual peso, sin importar
cuántos documentos tenga cada una. Un modelo que clasifica perfectamente las tres categorías grandes
y falla por completo en las cinco pequeñas tendría buen accuracy y mal F1-macro — y sería inútil,
porque las categorías minoritarias son precisamente las que un sistema de organización de
conocimiento debe distinguir.

**Por qué también reportamos top-2.** Ante contenido genuinamente interdisciplinario —un artículo
sobre desplegar modelos de ML en Kubernetes es simultáneamente DevOps y Data Science— la etiqueta
única es una simplificación del problema, no un error del modelo. El top-2 accuracy separa "el modelo
no entendió el documento" de "el documento no tiene una única categoría correcta".

**Por qué validación cruzada además del split.** Un único split de 25 % sobre un corpus de pocos
cientos de documentos tiene alta varianza: la métrica que reporta depende de qué documentos cayeron
en test. La CV estratificada de 5 folds usa todos los datos como test exactamente una vez, y la
**desviación estándar entre folds** es tan informativa como la media: una desviación alta indica que
el resultado no es estable y no debería presentarse como definitivo.
''')

code(r'''
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

display(resumen.style.background_gradient(cmap="Greens", axis=0))

metrica = CFG.clasificacion.metrica_decision
GANADOR = "B" if resumen.iloc[1][metrica] >= resumen.iloc[0][metrica] else "A"
NOMBRE_GANADOR = resumen.index[1] if GANADOR == "B" else resumen.index[0]

log.info(f"Modelo seleccionado: {NOMBRE_GANADOR} "
         f"({metrica} = {resumen.loc[NOMBRE_GANADOR, metrica]:.4f})")
print(f"\n>>> Modelo seleccionado para producción: {NOMBRE_GANADOR}")
print(f"    Criterio: mayor {metrica}")
''')

md(r'''
### 5.4.1b Calibración de probabilidades

**El síntoma.** El ejemplo del brief se clasifica bien pero con probabilidad 0.47, y en una demo eso
*parece* inseguridad. Conviene desmontar primero una intuición equivocada: con **ocho** categorías el
azar es 1/8 = 0.125, así que 0.47 es casi cuatro veces el azar. No es comparable a un 0.47 en un
problema binario, donde sí sería indecisión pura.

**El problema real, que sí existe.** La regresión logística multinomial no produce probabilidades
*calibradas*: el número no significa «acierto el 47 % de las veces que digo esto». Y el contrato de
salida del brief incluye un campo `probabilidad` que un consumidor va a interpretar literalmente.

**La solución y su límite.** `CalibratedClassifierCV` reajusta la función que convierte las
puntuaciones del modelo en probabilidades, usando validación cruzada interna. Con el método sigmoide
(Platt) ajusta dos parámetros por clase, que es lo apropiado con pocos datos; el isotónico es más
flexible pero necesita más muestras y sobreajusta con 400 documentos.

Un matiz que conviene tener claro: **la calibración no cambia el orden de las predicciones**, así que
el F1 y la accuracy se mantienen casi idénticos. No mejora el modelo — hace que su número de
confianza signifique algo. Por eso la celda **mide** el efecto con Brier score y log-loss y solo
adopta el modelo calibrado si mejora; adoptarlo a ciegas sería fe, no ingeniería.
''')

code(r'''
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

    display(comparacion)

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

# ── Modelo que realmente se sirve ────────────────────────────────────────────
# A partir de aquí NADIE vuelve a escribir `modelo_b if GANADOR == "B" else modelo_a`.
# Un solo nombre canónico evita que la calibración se quede en la tabla de métricas
# y nunca llegue ni al artefacto serializado ni al servicio de inferencia.
MODELO_SERVIDO = MODELO_CALIBRADO if MODELO_CALIBRADO is not None else (
    modelo_b if GANADOR == "B" else modelo_a)
CALIBRACION_APLICADA = MODELO_CALIBRADO is not None
TIPO_CLASIFICADOR = ("sbert+logreg" if GANADOR == "B" else "tfidf+logreg") + (
    f"+calibrado({CFG.clasificacion.metodo_calibracion})" if CALIBRACION_APLICADA else "")

print(f"\n  MODELO_SERVIDO -> {TIPO_CLASIFICADOR}")
print(f"  Este es el objeto que se serializa en 5.7 y el que responde en /contenido.")
''')

code(r'''
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
''')

code(r'''
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
''')

# ============================== 5.5 EXPLICABILIDAD ==============================
md(r'''
## 5.5 Explicabilidad

**El problema que resuelve.** El endpoint devuelve `{"categoria": "Backend", "probabilidad": 0.89}`.
Un usuario razonable pregunta: *¿por qué Backend?* Sin respuesta, el sistema es un oráculo: se acepta
o se rechaza, pero no se audita. Y un modelo que no se puede auditar no se puede depurar — cuando
clasifique mal de forma sistemática, no habrá manera de saber si el problema es el corpus, las
etiquetas o la representación.

Implementamos dos niveles, que responden preguntas distintas.

### Nivel global — ¿qué ha aprendido el modelo?

Los coeficientes de la regresión logística sobre TF-IDF son directamente legibles: cada uno mide
cuánto empuja un término hacia una categoría. Es un diagnóstico del **modelo en su conjunto**, no de
una predicción concreta, y sirve sobre todo para detectar aprendizaje espurio: si el término con
mayor peso para "DevOps" resultara ser un artefacto del scraping y no un concepto de DevOps, el
problema está en el corpus.

Solo aplica al modelo A. Los coeficientes del modelo B operan sobre 384 dimensiones latentes sin
significado individual: leerlos no informa nada.

### Nivel local — ¿por qué *este* documento?

Aquí usamos **ablación por término**, que es agnóstica al modelo y por tanto funciona igual sea cual
sea el ganador:

1. Se predice la categoría del documento completo → probabilidad `p₀`.
2. Para cada término candidato (keywords y entidades), se elimina del texto y se vuelve a predecir la
   probabilidad de *la misma categoría* → `pᵢ`.
3. La contribución del término es `p₀ − pᵢ`. Positiva significa que sostiene la decisión; negativa,
   que la contradice y el modelo decidió a pesar de él.

Es una aproximación local a un valor de Shapley con un solo orden de eliminación. No es SHAP —que
promedia sobre todas las coaliciones posibles y sería computacionalmente inviable en el camino
crítico de la API—, pero captura la intuición central y cuesta `n+1` predicciones en lugar de `2ⁿ`.

El resultado se separa en **términos a favor** (su eliminación baja la probabilidad: sostienen la
decisión) y **términos en contra** (su eliminación la sube: el modelo decidió *a pesar* de ellos).
Esa segunda lista es la más informativa de las dos para depurar — un término técnico que rema en
contra de la categoría correcta suele señalar que el corpus de esa categoría es pobre en ese
concepto.

**Y una segunda señal, gratuita:** la similitud coseno del documento contra el **centroide de cada
categoría** en el espacio de embeddings. Responde "¿a qué se parece este documento?" con independencia
de lo que el clasificador haya decidido, y cuando ambas señales discrepan, esa discrepancia es en sí
misma un diagnóstico útil.

### Nota sobre el enfoque alternativo: modelo sustituto (*surrogate*)

Existe una alternativa más barata: explicar siempre con el modelo TF-IDF, descomponiendo la
predicción en `peso_tfidf × coeficiente_de_clase` por término. Es una práctica estándar y su
justificación es sólida — las 384 dimensiones de SBERT no tienen significado léxico individual, así
que no se pueden leer.

No la adoptamos como método principal por una razón concreta: **si el modelo en producción es el B
(SBERT), un sustituto TF-IDF explica un modelo distinto del que tomó la decisión**. Las explicaciones
serían plausibles y a veces sencillamente falsas, que es el peor resultado posible en explicabilidad.
La ablación evita el problema porque interroga al modelo real, sea cual sea, a cambio de `n+1`
predicciones en vez de una multiplicación de vectores.
''')

code(r'''
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
''')

code(r'''
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
''')

# ============================== 5.6 CLUSTERING ==============================
md(r'''
## 5.6 Agrupamiento y descubrimiento de tópicos

Última caja de la etapa 3. El diagrama del brief sugiere *"Ej. LDA o K-Means"*;
`Technology_Architecture.md` §8 adopta **BERTopic** como método de producción y recomienda mantener
**KMeans como comparación metodológica**. Hacemos exactamente eso, y la comparación no es decorativa:
mostrar el codo y la silueta de KMeans documenta *por qué* fijar `k` a mano es un problema real y no
una objeción teórica.

El clustering es **no supervisado**: descubre estructura temática que puede no coincidir con las
categorías etiquetadas, y ese desacuerdo es su valor. Organiza el conocimiento por temas emergentes,
no por la taxonomía que impusimos al recolectar. Lo cuantificamos con ARI y NMI: cerca de 0 significa
que el clustering encontró una organización independiente de nuestras etiquetas; cerca de 1, que
reconstruyó la taxonomía por sí solo.
''')

code(r'''
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
''')

code(r'''
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
display(info_topicos.head(12)[["Topic", "Count", "Name"]])
''')

code(r'''
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
''')

code(r'''
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
''')

# ============================== 5.7 SERIALIZACIÓN ==============================
md(r'''
## 5.7 Serialización y versionado de modelos

Requisito explícito del brief. Pero guardar el `.joblib` es la parte fácil; la difícil es que dentro
de tres semanas alguien pueda responder **qué es exactamente ese archivo**.

Un artefacto sin metadatos es un binario opaco: no se sabe con qué datos se entrenó, con qué
hiperparámetros, qué desempeño tuvo ni si el código que intenta cargarlo es compatible. Por eso cada
corrida produce un `metadata.json` con seis bloques, que es el **contrato entre Ciencia de Datos y
Backend**:

| Bloque | Contenido | Para qué sirve |
|---|---|---|
| **Versión** | Versión semántica del pipeline + huella de configuración | El backend puede rechazar un artefacto de versión incompatible |
| **Fecha** | Timestamp ISO 8601 del entrenamiento | Trazabilidad y detección de modelos obsoletos |
| **Parámetros** | Hiperparámetros del clasificador + configuración completa | Reproducir el entrenamiento exactamente |
| **Métricas** | Todas las de §5.4 más la CV | Comparar contra la siguiente versión sin reejecutar |
| **Modelo** | Qué representación ganó, qué embeddings, qué spaCy | El backend carga el modelo correcto sin adivinar |
| **Dataset** | Nombre, tamaño, categorías, **hash SHA-256 del corpus** | Detectar si el corpus cambió desde el entrenamiento |

**El hash del dataset merece énfasis.** Es la única forma de responder con certeza *"¿este modelo se
entrenó con estos datos?"*. Sin él, un corpus modificado silenciosamente entre corridas produce un
modelo cuyo desempeño reportado ya no corresponde a nada verificable.

**Qué no se serializa con joblib, y por qué.** Los modelos de Sentence-Transformers y de spaCy se
cargan por nombre desde su propio caché (`SentenceTransformer(nombre)`), no se empaquetan:
serializarlos duplicaría cientos de megabytes sin beneficio alguno, y el nombre queda registrado en
`metadata.json` para que el backend cargue exactamente el mismo.
''')

code(r'''
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

    # 1 — Clasificador servido (ganador, ya calibrado si la calibración se adoptó)
    joblib.dump(MODELO_SERVIDO, destino / "modelo_clasificacion.joblib")
    artefactos["modelo_clasificacion.joblib"] = f"Clasificador temático ({TIPO_CLASIFICADOR})"

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
            "tipo_clasificador": TIPO_CLASIFICADOR,
            "calibracion_aplicada": CALIBRACION_APLICADA,
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
''')
