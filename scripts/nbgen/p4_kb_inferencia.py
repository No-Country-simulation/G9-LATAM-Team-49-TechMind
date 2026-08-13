"""Sección 6 — Base de conocimiento y persistencia. Sección 7 — Capa de inferencia API-ready."""

from .core import md, code

# ============================== 6. BASE DE CONOCIMIENTO ==============================
md(r'''
---
# 6. Base de conocimiento y consumo
### Etapa 4 del diagrama

> **Base de Datos Vectorial** → *Organización Automática* · *Búsqueda Semántica* · *Recomendación*

Con los modelos entrenados, el sistema necesita una capa que responda consultas. Aquí poblamos
ChromaDB con los embeddings del corpus y validamos las tres capacidades de consumo del diagrama, que
son simultáneamente tres "recursos opcionales" del brief.

**Por qué dos bases de datos y no una.** ChromaDB responde *"¿qué se parece a esto?"*; PostgreSQL
—gestionado por el backend, fuera del alcance de este notebook— responde *"¿qué documentos son de
categoría Backend, creados esta semana, con más de 500 caracteres?"*. Son preguntas de naturaleza
distinta: una es geométrica, la otra relacional. Forzar cualquiera de las dos en el motor equivocado
produce consultas lentas y código retorcido. El identificador del documento es el mismo en ambos
sistemas, y esa es toda la integración que hace falta.
''')

md(r'''
## Diagrama 6 — Flujo de búsqueda semántica y recomendación

```mermaid
flowchart TD
    subgraph INDEXACION["Indexación · ocurre una vez por documento"]
        I1["Documento procesado"] --> I2["Embedding 384-d<br/>normalizado L2"]
        I2 --> I3["collection.add()<br/>id · vector · documento · metadatos"]
        I3 --> I4[("ChromaDB<br/>índice HNSW · métrica coseno")]
    end

    subgraph CONSULTA["Consulta · cada petición"]
        Q1["Texto de consulta<br/>o doc_id de referencia"] --> Q2{"¿Qué tipo<br/>de consulta?"}
        Q2 -->|"texto libre"| Q3["encode(consulta)<br/>mismo modelo, mismo espacio"]
        Q2 -->|"doc_id"| Q4["Recuperar vector<br/>ya indexado"]
        Q3 --> Q5
        Q4 --> Q5["Vector de consulta"]
    end

    Q5 --> F{"¿Filtro por<br/>metadatos?"}
    F -->|"sí"| F1["where={'categoria': 'Backend'}<br/>filtrado + ANN en una sola pasada"]
    F -->|"no"| F2["Búsqueda ANN sin restricción"]

    F1 --> KNN
    F2 --> KNN["Búsqueda de k vecinos<br/>más cercanos sobre HNSW"]
    I4 -.->|"índice"| KNN

    KNN --> R["Resultados ordenados<br/>similitud = 1 − distancia coseno"]
    R --> EXCL{"¿Es recomendación<br/>de relacionados?"}
    EXCL -->|"sí"| EX["Excluir el documento<br/>de referencia de sí mismo"]
    EXCL -->|"no"| OUT
    EX --> OUT["Respuesta enriquecida<br/>doc_id · título · categoría · similitud"]

    OUT --> API1["GET /buscar?texto=..."]
    OUT --> API2["GET /similares/{id}"]
    OUT --> API3["campo 'relacionados'<br/>de POST /contenido"]

    style I4 fill:#fdf4e3,stroke:#b07d2b
```

**El detalle que hace que esto funcione:** la consulta y los documentos se codifican con **el mismo
modelo**, y por tanto viven en el mismo espacio vectorial. Es la razón por la que una consulta en
lenguaje natural —*"¿cómo construir servicios web escalables?"*— recupera documentos que no contienen
ninguna de esas palabras. Si el modelo de indexación y el de consulta divergieran, la búsqueda
devolvería ruido; por eso `metadata.json` registra el nombre exacto del modelo y §5.1.3 verifica la
dimensionalidad al cargarlo.
''')

md(r'''
## 6.1 Poblado del índice vectorial

Insertamos los embeddings ya calculados junto con los metadatos que después permitirán filtrar. La
colección se declara con `hnsw:space = "cosine"` porque los vectores están normalizados L2 (§5.1.3):
con norma unitaria, la distancia coseno y el producto punto coinciden, y ChromaDB puede usar el
índice HNSW sin normalizar en cada consulta.

**Se reindexa desde cero en cada corrida** (`delete_collection` seguido de `create_collection`). En
un notebook de entrenamiento es lo correcto: si cambió el modelo de embeddings, los vectores viejos
son incompatibles con los nuevos y mezclarlos produce una búsqueda silenciosamente rota. En
producción, en cambio, el backend hace `upsert` incremental por documento — son dos regímenes de
escritura distintos, y conviene no confundirlos.
''')

code(r'''
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
''')

md(r'''
## 6.2 Organización automática

Primera de las tres capacidades de consumo del diagrama. Cruzamos la taxonomía impuesta (categorías
etiquetadas) contra la estructura descubierta (tópicos de BERTopic) para responder una pregunta
concreta: **¿coincide nuestra taxonomía con la organización real del contenido?**

El mapa de calor es la respuesta. Una fila concentrada en una columna significa que la categoría se
corresponde con un tema coherente. Una fila dispersa significa que esa categoría agrupa contenidos
que el modelo considera temáticamente distintos — señal de que la etiqueta es demasiado amplia, o de
que el corpus de esa categoría es heterogéneo. En ambos casos, es información accionable sobre el
diseño de la taxonomía, no un fallo del clustering.
''')

code(r'''
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
display(tabla_organizacion.head(15))
''')

md(r'''
## 6.3 Motor de búsqueda semántica

Segunda capacidad. La consulta se codifica con **el mismo modelo** que indexó los documentos, de modo
que ambos viven en el mismo espacio vectorial y la comparación es directa.

Las tres consultas de demostración están elegidas para que **no compartan vocabulario literal** con
los documentos que deberían recuperar: *"¿cómo construir servicios web escalables?"* no contiene las
palabras "API", "REST" ni "backend". Si el sistema las recupera igualmente, es porque está operando
sobre significado — que es exactamente lo que TF-IDF no puede hacer y lo que justifica el costo de
los embeddings (§3.2).

El parámetro `categoria` aprovecha el filtrado combinado de ChromaDB: restringe la búsqueda por
metadatos **dentro** de la misma consulta ANN, sin una segunda pasada. Es lo que implementa el
recurso opcional *"consulta por categorías"* del brief.
''')

code(r'''
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
    display(buscar_semantica(consulta, n=4))

print("=" * 106)
print("Nótese que ninguna consulta comparte vocabulario literal con los documentos que recupera:")
print("es exactamente la propiedad que TF-IDF no puede ofrecer y que justifica los embeddings (§3.2).")
''')

md(r'''
## 6.4 Recomendación de contenido relacionado

Tercera capacidad, y la más barata de las tres: el vector del documento **ya está indexado**, así que
recomendar no requiere codificar nada — solo consultar vecinos.

El único detalle no obvio es pedir `n + 1` resultados y filtrar el propio documento: su vecino más
cercano es siempre él mismo, con similitud 1.0. Olvidarlo produce una recomendación que sugiere al
usuario el artículo que está leyendo.

**Limitación honesta:** esta recomendación es puramente semántica, así que siempre devuelve *lo más
parecido*. Para un sistema de aprendizaje eso no siempre es lo deseable — a veces el contenido más
útil es el complementario, no el redundante. Queda anotado en §9.3.
''')

code(r'''
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
display(recomendar_relacionados(ref["doc_id"], n=5))
''')

# ============================== 6.5 PERSISTENCIA ==============================
md(r'''
## 6.5 Persistencia de resultados

Todo lo que el pipeline produce se guarda, en el formato adecuado a cada tipo de dato. La regla es
simple: **si costó cómputo generarlo, se persiste**; recalcular es aceptable durante el desarrollo,
no en una demo ni en producción.

| Qué | Dónde | Formato | Por qué ese formato |
|---|---|---|---|
| Documentos procesados | `datasets/processed/corpus_final.csv` | CSV | Legible, inspeccionable, cargable desde cualquier herramienta |
| Embeddings | `datasets/processed/embeddings.npy` | NumPy binario | Preserva `float32` sin pérdida ni conversión de texto |
| Caché de embeddings | `cache/embeddings_cache.joblib` | joblib comprimido | Indexado por hash: reutilizable entre corridas |
| Keywords | columna `keywords` de `corpus_final.csv` | lista serializada | Va con su documento; el backend la lleva a `JSONB` en PostgreSQL |
| Categorías y probabilidades | `datasets/processed/resultados_clasificacion.csv` | CSV | Predicción, probabilidad y acierto por documento |
| Vectores + metadatos | `chroma_db/` | ChromaDB persistente | Índice HNSW ya construido: no hay que reindexar al arrancar |
| Modelos | `models/*.joblib` | joblib | Requisito explícito del brief |
| Metadatos y config | `models/metadata.json`, `models/config.json` | JSON | Legible por humanos y por el backend |
| Log de ejecución | `logs/pipeline.log` | texto | Evidencia de la corrida, con timestamps |

**Sobre `resultados_clasificacion.csv`.** Guarda la predicción del modelo para **todo** el corpus,
incluida la columna `acierto`. Sirve para tres cosas concretas: analizar errores sin reejecutar el
pipeline, alimentar directamente la tabla `contenidos` de PostgreSQL, y establecer la línea base
contra la que se compara la siguiente versión del modelo.
''')

code(r'''
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
display(RESULTADOS.head(6)[["doc_id", "categoria_real", "categoria_predicha",
                            "probabilidad", "acierto", "keywords"]])
''')

# ============================== 6.6 SQLITE ==============================
md(r'''
## 6.6 Persistencia relacional (SQLite)

**Por qué una tercera base de datos.** Hasta aquí el pipeline persiste en archivos planos (CSV, JSON,
`.npy`) y en ChromaDB. Falta el tipo de consulta más común de todas: la relacional.

Ninguna de las dos piezas actuales la resuelve. ChromaDB responde *"¿qué se parece a esto?"* pero no
*"dame las keywords del documento DOC-0042"* ni *"¿cuántas predicciones se hicieron hoy con confianza
por debajo de 0.4?"*. Los CSV responden ambas, pero solo cargándolos enteros en memoria y filtrando
con pandas — inviable desde un backend que atiende peticiones concurrentes.

**Por qué SQLite aquí y PostgreSQL allá.** `Technology_Architecture.md` §9 elige PostgreSQL para
producción y descarta SQLite explícitamente como base del servicio: archivo local, concurrencia de
escritura serializada, incompatible con contenedores efímeros. Ese mismo documento **recomienda
SQLite dentro del notebook** de Ciencia de Datos, que es un proceso único, local y desechable — el
escenario donde SQLite es la elección correcta y PostgreSQL sería una pieza de infraestructura
innecesaria.

El detalle que hace útil esta decisión: **el esquema es el mismo**. Las seis tablas de abajo se
traducen a PostgreSQL cambiando `TEXT` por `VARCHAR` y `AUTOINCREMENT` por `SERIAL`. Migrar es un
cambio de cadena de conexión, no un rediseño del modelo de datos.

| Tabla | Qué guarda | Pregunta que responde |
|---|---|---|
| `documentos` | Metadatos por documento | *¿Qué documentos hay de categoría Backend?* |
| `keywords_documento` | Keywords rankeadas, con su posición | *¿Qué documentos mencionan "Docker"?* |
| `resultados_clustering` | Cluster KMeans y tópico BERTopic | *¿Qué documentos forman el tópico 3?* |
| `resultados_clasificacion` | Predicción del modelo por documento | *¿Dónde se equivoca el modelo?* |
| `predicciones_api` | Registro de cada inferencia servida | *¿Cuántas peticiones tuvieron confianza baja esta semana?* |
| `versiones_modelo` | Historial de entrenamientos | *¿Mejoró el F1 respecto de la versión anterior?* |

**Sobre la idempotencia.** `documentos` y `resultados_clustering` tienen clave primaria natural
(`doc_id`), así que usan `INSERT OR REPLACE` y re-ejecutar la celda es inocuo. `keywords_documento` y
`resultados_clasificacion` no la tienen —un documento tiene varias keywords— así que se vacían antes
de reinsertar. Es correcto mientras el corpus se reconstruya completo en cada corrida, que es lo que
hace este notebook; si el corpus creciera de forma incremental habría que pasar a un upsert por lote,
y queda anotado en §9.3.
''')

code(r'''
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

    tipo_clf = TIPO_CLASIFICADOR
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
''')

code(r'''
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
    display(pd.read_sql(sql, con_sqlite))

print("=" * 92)
print("Estas cuatro consultas son la justificación de la tabla: ninguna se puede responder")
print("con ChromaDB, y todas requerirían cargar el CSV completo en memoria sin SQL.")
''')

md(r'''
## 6.7 Historial de versiones del modelo

`metadata.json` (§5.7) describe la versión **actual** con todo detalle — pero se sobrescribe en cada
entrenamiento. Después de tres corridas no queda rastro de las dos primeras, y la pregunta más
elemental del ciclo de vida de un modelo se vuelve incontestable: *¿esta versión es mejor que la
anterior?*

`registrar_version_modelo()` resuelve eso escribiendo en **dos destinos de solo-anexado**, sin tocar
`metadata.json` (que el backend sigue leyendo tal cual):

- **Tabla `versiones_modelo`** — consultable por SQL, para comparar métricas entre entrenamientos.
- **`historial_versiones.jsonl`** — una línea JSON por corrida. Formato deliberadamente humilde:
  sobrevive a que la base de datos se borre, se lee con `tail`, se versiona en Git sin conflictos de
  merge (cada corrida es una línea nueva) y no requiere ninguna herramienta para inspeccionarlo.

No es MLflow ni Weights & Biases, y no pretende serlo: para un MVP de hackathon, un JSONL de
solo-anexado da el 90 % del valor —trazabilidad y comparación entre versiones— con cero
infraestructura adicional que desplegar y explicar al jurado.
''')

code(r'''
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
display(historial)

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
''')

# ============================== 7. INFERENCIA ==============================
md(r'''
---
# 7. Capa de inferencia — lista para FastAPI

Esta sección es la **frontera entre Ciencia de Datos y Backend**, y su diseño responde a un problema
concreto del prototipo anterior: la función de inferencia dependía de una decena de variables
globales del notebook (`modelo_b`, `le`, `nlp`, `coleccion`, `topic_model`…). Eso funciona mientras
todo vive en el mismo kernel, y deja de funcionar en cuanto el backend intenta importarla — porque
esas globales no existen en `app/services/nlp_service.py`.

**La solución: separar explícitamente las cuatro responsabilidades** que el brief pide distinguir.

| Responsabilidad | Dónde vive | Cuándo se ejecuta |
|---|---|---|
| **Entrenamiento** | §5 de este notebook | Offline, una vez por versión del modelo |
| **Carga del modelo** | `TechMindInference.desde_artefactos()` | Una vez, al arrancar el proceso de FastAPI |
| **Predicción** | `TechMindInference.predecir()` | En cada petición HTTP |
| **Contrato de salida** | `RespuestaContenido` | Espejo del modelo Pydantic del backend |

`TechMindInference` recibe **todas** sus dependencias en el constructor. Eso la vuelve instanciable
desde cualquier proceso, testeable con dobles de prueba, y —no menor— permite tener dos versiones del
modelo cargadas a la vez durante un despliegue gradual.

### Una trampa fácil de pasar por alto

Es tentador declarar "la clase no lee globales" mirando solo su cuerpo. Pero si un método llama a una
función auxiliar que **sí** las lee, la dependencia sigue ahí, solo que escondida un nivel más abajo.
La clase parece portable, se importa desde el backend, y falla con `NameError` en la primera
petición.

En este notebook el riesgo era real: `rankear_keywords()` leía `nlp`, `kw_model` y `extractor_yake`
del ámbito global, y `explicar_prediccion()` leía `GANADOR`, `modelo_a`/`modelo_b`, `CENTROIDES` y
`CATEGORIAS`. Ninguna de esas variables existe en `app/services/nlp_service.py`.

La corrección fue dar a ambas funciones **parámetros de sólo-palabra clave** para inyectar esas
dependencias, con los globales como valor por defecto. El notebook las sigue llamando igual; la clase
las llama pasando lo que recibió en su constructor. El test `test_sin_globales` de §7.5 verifica que
la separación se sostiene: instancia la clase con dependencias explícitas y predice con los nombres
globales del entrenamiento borrados del ámbito.
''')

md(r'''
## Diagrama 7 — Flujo de inferencia (`POST /contenido`)

```mermaid
sequenceDiagram
    autonumber
    participant C as Cliente
    participant F as FastAPI<br/>routes/contenido.py
    participant P as Pydantic<br/>ContenidoRequest
    participant S as TechMindInference<br/>nlp_service.py
    participant SP as spaCy
    participant E as SBERT + caché
    participant M as Clasificador
    participant CH as ChromaDB
    participant PG as PostgreSQL

    C->>F: POST /contenido<br/>{titulo, texto}
    F->>P: Validación de tipos
    alt tipos inválidos
        P-->>C: 422 Unprocessable Entity
    end
    P->>S: predecir(titulo, texto)

    S->>S: validar_entrada() · 7 controles
    alt documento inválido
        S-->>F: ErrorValidacion(codigo, mensaje)
        F-->>C: 422 con código de error tipificado
    end

    S->>S: detectar_idioma()
    alt idioma no soportado
        S-->>C: 422 idioma_no_soportado
    end

    S->>S: limpiar_texto() · NFKC · ruido
    S->>SP: nlp(texto_limpio)
    SP-->>S: Doc · lemas · POS · entidades TECH

    S->>E: codificar(texto_limpio)
    alt acierto de caché
        E-->>S: vector recuperado
    else fallo de caché
        E->>E: encode() · 12 capas Transformer
        E-->>S: vector 384-d
    end

    par Cálculos sobre el mismo vector
        S->>M: predict_proba(vector)
        M-->>S: categoria + probabilidad
    and
        S->>S: rankear_keywords(doc reutilizado)
    and
        S->>CH: query(vector, n=3)
        CH-->>S: documentos relacionados
    end

    opt explicabilidad solicitada
        S->>M: ablación por término
        M-->>S: contribuciones
    end

    S-->>F: RespuestaContenido
    F->>PG: INSERT metadatos
    F->>CH: add(vector) si es contenido nuevo
    F-->>C: 200 OK<br/>{categoria, probabilidad,<br/>informacion_adicional, ...}
```

**Lo que el diagrama vuelve explícito:** el bloque `par` marca las tres operaciones que consumen el
**mismo vector** ya calculado. Es la materialización de la decisión de §3.2 — un `encode()` por
petición, no tres. Y el `Doc` de spaCy se calcula una vez y se pasa a `rankear_keywords()`, evitando
la segunda pasada que tenía el prototipo anterior.
''')

code(r'''
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
''')

code(r'''
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
            "estado": "operativo",
            "version_pipeline": self.metadatos.get("version", self.cfg.version),
            "clasificador": self.tipo_clasificador,
            "categorias": self.categorias,
            "idiomas_soportados": list(self.cfg.idioma.idiomas_soportados),
            "modelo_embeddings": self.cfg.embeddings.modelo,
            "indice_vectorial": (self.coleccion.count()
                                 if self.coleccion is not None else None),
            "explicabilidad": self.centroides is not None,
        }


servicio = TechMindInference.desde_objetos(
    modelo_clasificacion=MODELO_SERVIDO,
    label_encoder=le,
    modelo_embeddings=modelo_embeddings,
    pipeline_nlp=nlp,
    cfg=CFG,
    tipo_clasificador=TIPO_CLASIFICADOR,
    modelo_keybert=kw_model,
    extractor_yake=extractor_yake,
    mapa_tecnologias=_MAPA_TECNOLOGIAS,
    metadatos=METADATOS,
    topic_model=topic_model,
    coleccion_vectorial=coleccion,
    etiquetas_topico=ETIQUETAS_TOPICO,
    centroides=CENTROIDES,
    cache=CACHE,
    conexion_sqlite=con_sqlite,
)

# El diccionario de tecnologías se guarda aparte para que `desde_artefactos()`
# pueda reconstruir el mapa de capitalización sin depender del notebook.
(CFG.rutas.models / "tecnologias.json").write_text(
    json.dumps(TECNOLOGIAS, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(servicio.salud(), ensure_ascii=False, indent=2))
''')

md(r'''
## 7.3 Ejemplos de uso

El brief exige un **mínimo de tres ejemplos**. El primero es literalmente el del enunciado, para que
la correspondencia con el requisito sea verificable de un vistazo; los otros dos cubren categorías
distintas (Ciencia de Datos y DevOps) para evidenciar que el sistema discrimina y no colapsa siempre
a la misma etiqueta.
''')

code(r'''
# @title 7.3 — Tres ejemplos de uso (requisito del brief)
EJEMPLOS = [
    {   # 1 — el del enunciado del brief, textual
        "titulo": "Introducción a Spring Boot",
        "texto": ("En este contenido se presentan los conceptos básicos para la creación de "
                  "APIs REST utilizando Java y Spring Boot."),
    },
    {   # 2 — Ciencia de Datos
        "titulo": "Clasificación de texto con Scikit-Learn",
        "texto": ("Este tutorial explica cómo entrenar un modelo de regresión logística sobre "
                  "una matriz TF-IDF para clasificar documentos por tema, evaluando el "
                  "desempeño con validación cruzada y métricas de precisión y recall."),
    },
    {   # 3 — DevOps
        "titulo": "Despliegue continuo con Docker y Kubernetes",
        "texto": ("Guía práctica para containerizar una aplicación, publicar la imagen en un "
                  "registro y orquestar el despliegue en un clúster de Kubernetes mediante un "
                  "pipeline de integración continua."),
    },
]

for i, ejemplo in enumerate(EJEMPLOS, 1):
    salida = servicio.predecir(**ejemplo)
    print("=" * 94)
    print(f"EJEMPLO {i} — POST /contenido")
    print("-" * 94)
    print("REQUEST:")
    print(json.dumps(ejemplo, ensure_ascii=False, indent=2))
    print("\nRESPONSE:")
    print(salida.a_json())
    print()
''')

md(r'''
## 7.4 Procesamiento en lote

Recurso opcional del brief. La propiedad de diseño relevante es que **un documento inválido no aborta
el lote**: `predecir_lote` captura la excepción por fila, registra el código de error y continúa. Es
el comportamiento correcto para una carga masiva — el usuario que sube un CSV de 500 filas con tres
malformadas quiere las 497 procesadas y un informe de las tres, no un error genérico y cero
resultados.

El lote de demostración incluye deliberadamente tres casos límite: texto demasiado corto, título
ausente y un documento en inglés (idioma no soportado en esta versión).
''')

code(r'''
# @title 7.4 — Procesamiento en lote y manejo de errores
lote_demo = EJEMPLOS + [
    {"titulo": "Entrada inválida", "texto": "corto"},
    {"titulo": None, "texto": "Un texto válido pero sin título asociado, lo que debe rechazarse."},
    {"titulo": "Texto en inglés", "texto": "This article explains how to build REST APIs "
                                           "using the Spring Boot framework in Java."},
]

resultado_lote = servicio.predecir_lote(lote_demo, incluir_relacionados=False,
                                        incluir_explicacion=False)
resultado_lote.to_csv(CFG.rutas.processed / "resultado_lote.csv", index=False)

print("Procesamiento en lote — los errores se reportan por fila, sin abortar el lote:\n")
display(resultado_lote)

n_ok = int(resultado_lote["error"].isna().sum())
print(f"\nProcesados: {n_ok}/{len(lote_demo)} | Rechazados: {len(lote_demo) - n_ok}")
''')

md(r'''
## 7.5 Pruebas de sanidad

Recurso opcional del brief, y la última línea de defensa antes de entregar. Siete pruebas que
verifican las propiedades que no deben romperse nunca:

| Prueba | Qué protege |
|---|---|
| Contrato JSON mínimo | Que los tres campos del brief existan, con el tipo correcto y sean serializables |
| Validación rechaza | Que los seis casos inválidos representativos efectivamente se rechacen |
| Determinismo | Que dos llamadas idénticas devuelvan exactamente lo mismo |
| Búsqueda ordenada | Que los resultados vengan por similitud descendente |
| Detección de idioma | Que español e inglés se distingan correctamente |
| Caché de embeddings | Que la segunda llamada sea un acierto y devuelva el mismo vector |
| Reproducibilidad | Que refijar las semillas no altere la predicción |
| Sin globales | Que la clase prediga con los nombres del entrenamiento borrados del ámbito |
| Auditoría e historial | Que cada inferencia se registre y que el historial acumule |
| **Carga en frío** | Que `desde_artefactos()` —la ruta del backend— produzca entidades y keywords |
| **Paridad frío/caliente** | Que la carga desde disco dé el mismo resultado que la del notebook |
| **Composición de entrada** | Que entrenamiento e inferencia compongan el texto igual |

### Por qué las tres últimas existen

Las pruebas anteriores compartían un punto ciego: **todas usaban `desde_objetos()`**, la vía del
notebook, donde el pipeline de spaCy ya trae el `EntityRuler` que le añadió §4.1. Ninguna ejercitaba
`desde_artefactos()`, que es precisamente la que el backend usa en producción y la que §9.4 documenta
como punto de entrada oficial.

Esa vía reconstruye spaCy con `spacy.load()`, que devuelve el **modelo base**: sin `EntityRuler`, sin
etiqueta `TECH`. El efecto era silencioso y de dos niveles: `entidades_tecnicas` habría salido vacío,
y —más grave— `rankear_keywords()` habría perdido la señal de reglas, que es la de mayor peso
(`peso_entidades = 1.5`, la única con precisión perfecta por construcción). El `informacion_adicional`
servido por la API habría sido peor que el que el notebook enseña en la demo, sin que nada fallara.

Un punto ciego que solo se cierra probando la ruta real, no una equivalente.
''')

code(r'''
# @title 7.5 — Pruebas de sanidad del contrato
def test_contrato_minimo() -> str:
    """Verifica que la respuesta cumple el contrato exigido por el brief."""
    r = servicio.predecir(**EJEMPLOS[0]).a_dict()
    assert {"categoria", "probabilidad", "informacion_adicional"}.issubset(r), \
        "faltan campos obligatorios del brief"
    assert isinstance(r["categoria"], str) and r["categoria"] in servicio.categorias
    assert isinstance(r["probabilidad"], float) and 0.0 <= r["probabilidad"] <= 1.0
    assert isinstance(r["informacion_adicional"], list)
    assert all(isinstance(k, str) for k in r["informacion_adicional"])
    json.dumps(r, ensure_ascii=False)   # debe ser serializable
    return "OK"


def test_validacion_rechaza() -> str:
    """Verifica que cada control de validación rechaza lo que debe rechazar."""
    casos = [
        (None, "Un texto suficientemente largo para superar el mínimo de caracteres."),
        ("Título", None),
        ("Título", ""),
        ("Título", "corto"),
        ("Título", 12345),
        ("Título", "a" * (CFG.validacion.texto_max_chars + 1)),
    ]
    for titulo, texto in casos:
        try:
            servicio.predecir(titulo, texto)
            return f"FALLO: no rechazó {(str(titulo)[:20], str(texto)[:20])}"
        except ErrorValidacion:
            continue
    return "OK"


def test_determinismo() -> str:
    """Verifica que dos predicciones sobre el mismo texto coinciden exactamente."""
    a = servicio.predecir(**EJEMPLOS[1], incluir_relacionados=False, incluir_explicacion=False)
    b = servicio.predecir(**EJEMPLOS[1], incluir_relacionados=False, incluir_explicacion=False)
    return "OK" if (a.categoria == b.categoria
                    and a.probabilidad == b.probabilidad
                    and a.informacion_adicional == b.informacion_adicional) else "FALLO"


def test_busqueda_ordenada() -> str:
    """Verifica que la búsqueda semántica devuelve resultados ordenados por similitud."""
    r = buscar_semantica("bases de datos relacionales", n=3)
    return "OK" if len(r) == 3 and r["similitud"].is_monotonic_decreasing else "FALLO"


def test_idioma() -> str:
    """Verifica que la detección de idioma reconoce español e inglés correctamente."""
    es = detectar_idioma("Este documento explica cómo construir servicios web con Java.")
    en = detectar_idioma("This document explains how to build web services using Java.")
    return "OK" if es.codigo == "es" and en.codigo == "en" else f"FALLO: {es.codigo}/{en.codigo}"


def test_cache_embeddings() -> str:
    """Verifica que la caché devuelve exactamente el mismo vector en la segunda llamada."""
    texto = ["Texto de prueba para verificar el comportamiento de la caché de embeddings."]
    v1 = CACHE.codificar(texto, modelo_embeddings)
    aciertos_antes = CACHE.aciertos
    v2 = CACHE.codificar(texto, modelo_embeddings)
    return ("OK" if np.allclose(v1, v2) and CACHE.aciertos > aciertos_antes
            else "FALLO: la caché no se está usando")


def test_reproducibilidad() -> str:
    """Verifica que el modelo produce la misma probabilidad tras refijar las semillas."""
    p1 = servicio.predecir(**EJEMPLOS[2], incluir_relacionados=False,
                           incluir_explicacion=False).probabilidad
    fijar_semillas(verboso=False)
    p2 = servicio.predecir(**EJEMPLOS[2], incluir_relacionados=False,
                           incluir_explicacion=False).probabilidad
    return "OK" if p1 == p2 else f"FALLO: {p1} != {p2}"


def test_sin_globales() -> str:
    """Verifica que la capa de inferencia no depende de globales de entrenamiento.

    Instancia un servicio nuevo con dependencias explícitas y predice mientras los
    nombres globales que usaba el entrenamiento están ocultos. Si algún método
    seguía leyéndolos, el `NameError` aparece aquí y no en producción.
    """
    import builtins
    servicio_aislado = TechMindInference.desde_objetos(
        modelo_clasificacion=MODELO_SERVIDO,
        label_encoder=le, modelo_embeddings=modelo_embeddings, pipeline_nlp=nlp,
        cfg=CFG, tipo_clasificador=TIPO_CLASIFICADOR,
        modelo_keybert=kw_model, extractor_yake=extractor_yake,
        mapa_tecnologias=_MAPA_TECNOLOGIAS, centroides=CENTROIDES, cache=CACHE,
    )
    # Oculta temporalmente los nombres globales que la clase NO debería necesitar.
    ocultos = ["GANADOR", "CENTROIDES", "CATEGORIAS", "kw_model",
               "extractor_yake", "_MAPA_TECNOLOGIAS", "modelo_a", "modelo_b"]
    respaldo = {n: globals().pop(n) for n in ocultos if n in globals()}
    try:
        r = servicio_aislado.predecir(**EJEMPLOS[0], incluir_relacionados=False)
        assert r.categoria in servicio_aislado.categorias
        assert r.informacion_adicional, "el ranking de keywords devolvió vacío"
        assert r.explicacion.get("terminos_a_favor") is not None
        return "OK"
    except NameError as exc:
        return f"FALLO: aún lee una global -> {exc}"
    finally:
        globals().update(respaldo)


def test_auditoria_sqlite() -> str:
    """Verifica que cada predicción queda registrada en `predicciones_api`."""
    antes = con_sqlite.execute("SELECT COUNT(*) FROM predicciones_api").fetchone()[0]
    servicio.predecir(**EJEMPLOS[0], incluir_relacionados=False, incluir_explicacion=False)
    despues = con_sqlite.execute("SELECT COUNT(*) FROM predicciones_api").fetchone()[0]
    return "OK" if despues == antes + 1 else f"FALLO: {antes} -> {despues}"


def test_historial_versiones() -> str:
    """Verifica que el historial es de solo-anexado y no sobrescribe corridas previas."""
    ruta = CFG.rutas.models / CFG.persistencia.archivo_historial
    if not ruta.exists():
        return "FALLO: no existe el archivo de historial"
    n_lineas = len([l for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()])
    n_filas = con_sqlite.execute("SELECT COUNT(*) FROM versiones_modelo").fetchone()[0]
    return "OK" if n_lineas >= 1 and n_filas >= 1 else f"FALLO: {n_lineas} líneas, {n_filas} filas"


def test_carga_en_frio() -> str:
    """Ejercita la ruta que usa el backend: `desde_artefactos()` desde disco.

    Es la prueba que faltaba. Todas las demás usan `desde_objetos()` con los
    objetos del notebook, donde el pipeline de spaCy ya tiene el EntityRuler
    porque lo añadió §4.1. La ruta de producción reconstruye spaCy desde cero
    con `spacy.load()`, que devuelve el modelo BASE: si nadie vuelve a añadir el
    ruler, `entidades_tecnicas` sale vacío y el ranking de keywords pierde su
    señal de mayor peso. La demo pasaría y el backend no.
    """
    servicio_frio = TechMindInference.desde_artefactos(CFG.rutas.models, cfg=CFG)

    if "entity_ruler" not in servicio_frio.nlp.pipe_names:
        return "FALLO: el pipeline cargado en frío no tiene EntityRuler"

    r = servicio_frio.predecir(**EJEMPLOS[0], incluir_explicacion=False)

    if not r.entidades_tecnicas:
        return "FALLO: entidades_tecnicas vacío en carga en frío"
    if not r.informacion_adicional:
        return "FALLO: informacion_adicional vacío en carga en frío"
    if r.categoria not in servicio_frio.categorias:
        return f"FALLO: categoría inesperada '{r.categoria}'"
    return "OK"


def test_paridad_frio_caliente() -> str:
    """Verifica que la carga en frío y la del notebook dan el mismo resultado.

    Si divergen, algún artefacto no se está reconstruyendo igual — que es
    exactamente el modo de fallo del EntityRuler perdido.
    """
    frio = TechMindInference.desde_artefactos(CFG.rutas.models, cfg=CFG)
    a = servicio.predecir(**EJEMPLOS[0], incluir_relacionados=False, incluir_explicacion=False)
    b = frio.predecir(**EJEMPLOS[0], incluir_relacionados=False, incluir_explicacion=False)
    if a.categoria != b.categoria:
        return f"FALLO: categoría {a.categoria} vs {b.categoria}"
    if set(a.entidades_tecnicas) != set(b.entidades_tecnicas):
        return (f"FALLO: entidades {a.entidades_tecnicas} vs {b.entidades_tecnicas}")
    if a.informacion_adicional != b.informacion_adicional:
        return (f"FALLO: keywords {a.informacion_adicional} vs {b.informacion_adicional}")
    return "OK"


def test_config_bien_direccionada() -> str:
    """Verifica que cada acceso `CFG.bloque.campo` del notebook existe de verdad.

    Cubre un punto ciego real. El análisis de símbolos comprueba que `CFG` está
    definida, pero no que `CFG.nlp.filtrar_keywords_por_pos` exista: eso solo se
    descubre al ejecutar, con un `AttributeError`. Pasó — el campo vivía en
    `ConfigKeywords` y se leía desde `ConfigNLP`—, y como la configuración está
    repartida en doce dataclases es un error fácil de repetir al añadir opciones.

    Returns:
        "OK" o el detalle de los accesos que no resuelven.
    """
    ruta = Path("techmind_eda_modelado.ipynb")
    if not ruta.exists():
        return "OK"   # sin el .ipynb a mano no hay nada que escanear

    # Se analiza el ÁRBOL SINTÁCTICO, no el texto. Una búsqueda por expresión
    # regular encontraría también las menciones dentro de docstrings y
    # comentarios —incluida la de esta misma función— y daría falsos positivos.
    accesos = set()
    for celda in json.loads(ruta.read_text(encoding="utf-8"))["cells"]:
        if celda["cell_type"] != "code":
            continue
        fuente = re.sub(r"get_ipython\(\)\.system\(", "print(", "".join(celda["source"]))
        try:
            arbol = ast.parse(fuente)
        except SyntaxError:
            continue
        for nodo in ast.walk(arbol):
            # Patrón buscado: Attribute(Attribute(Name('CFG'), bloque), campo)
            if (isinstance(nodo, ast.Attribute)
                    and isinstance(nodo.value, ast.Attribute)
                    and isinstance(nodo.value.value, ast.Name)
                    and nodo.value.value.id in ("cfg", "CFG")):
                accesos.add((nodo.value.attr, nodo.attr))

    malos = []
    for bloque, campo in sorted(accesos):
        sub = getattr(CFG, bloque, None)
        if sub is None or not hasattr(sub, campo):
            correcto = [n for n in vars(CFG) if hasattr(getattr(CFG, n), campo)]
            malos.append(f"CFG.{bloque}.{campo}"
                         + (f" (está en {', '.join(correcto)})" if correcto else ""))
    if malos:
        return f"FALLO: {len(malos)} mal dirigidos — " + "; ".join(malos[:3])
    log.info(f"Configuración verificada: {len(accesos)} accesos anidados resuelven.")
    return "OK"


def test_ejemplos_del_brief() -> str:
    """Verifica que los ejemplos de la demo caen en la categoría correcta.

    Es la prueba que faltaba, y su ausencia costó cara: en una corrida con
    F1-macro de 0.9047 —excelente sobre el papel— **dos de los tres ejemplos
    del brief se clasificaban mal**. Una métrica agregada promedia sobre el
    conjunto de prueba y no dice nada sobre los tres documentos concretos que
    el jurado va a escribir en la demo.

    No falla el pipeline: informa. Un fallo aquí casi siempre significa que el
    corpus no cubre el vocabulario del ejemplo, no que el modelo esté roto.
    """
    esperado = {
        "Introducción a Spring Boot": "Backend",
        "Clasificación de texto con Scikit-Learn": "Data Science",
        "Despliegue continuo con Docker y Kubernetes": "DevOps",
    }
    fallos = []
    for ejemplo in EJEMPLOS:
        r = servicio.predecir(**ejemplo, incluir_relacionados=False,
                              incluir_explicacion=False)
        esp = esperado.get(ejemplo["titulo"])
        if esp and r.categoria != esp:
            fallos.append(f"'{ejemplo['titulo'][:28]}' -> {r.categoria} "
                          f"(esperado {esp}, p={r.probabilidad})")
    if fallos:
        for f in fallos:
            log.error(f"Ejemplo del brief mal clasificado: {f}")
        return f"FALLO: {len(fallos)}/{len(esperado)} mal — " + " | ".join(fallos)
    return "OK"


def test_titulo_aporta_entidades() -> str:
    """Verifica que una tecnología presente solo en el título se detecta igual.

    Caso real que motivó la prueba: «Clasificación de texto con Scikit-Learn»
    menciona la librería únicamente en el título. Al excluir el título del texto
    del clasificador —para evitar el desajuste con el entrenamiento— se rompió
    sin querer la detección de entidades, y el ejemplo 2 devolvía
    `entidades_tecnicas` vacío. El título vuelve a alimentar al EntityRuler y a
    KeyBERT, que no se entrenan y por tanto no sufren desajuste.
    """
    if not CFG.nlp.incluir_titulo_en_entidades:
        return "OK"   # política desactivada a propósito
    r = servicio.predecir(
        "Clasificación de texto con Scikit-Learn",
        "Este tutorial explica cómo entrenar un modelo de regresión logística sobre "
        "una matriz TF-IDF para clasificar documentos por tema.",
        incluir_relacionados=False, incluir_explicacion=False)
    if not r.entidades_tecnicas:
        return "FALLO: entidades vacías con la tecnología solo en el título"
    return "OK"


def test_composicion_entrada_unica() -> str:
    """Verifica que entrenamiento e inferencia componen la entrada igual.

    Ambos pasan por `componer_entrada`, gobernada por un único flag. Esta prueba
    fija ese contrato: si alguien vuelve a concatenar el título a mano en un
    solo lado, falla.
    """
    t, x = "Título de prueba", "Cuerpo del documento de prueba."
    esperado = f"{t}. {x}" if CFG.nlp.incluir_titulo_en_texto else x
    return "OK" if componer_entrada(t, x, CFG) == esperado else "FALLO"


def test_calibracion_llega_a_produccion() -> str:
    """Verifica que el modelo calibrado es el que sirve, no solo el que se mide.

    La calibración se evalúa en 5.4.1b y se adopta si mejora el Brier score. Pero
    adoptarla significa poco si el servicio de inferencia sigue instanciado con el
    modelo sin calibrar: la tabla de métricas mejoraría y las probabilidades que ve
    el usuario del endpoint seguirían siendo las de antes. Esta prueba compara la
    identidad del objeto, no su comportamiento, porque el fallo es de cableado.
    """
    if not CALIBRACION_APLICADA:
        return "OK (calibración no adoptada; se sirve el modelo original)"

    if servicio.clf is not MODELO_SERVIDO:
        return "FALLO: el servicio no usa MODELO_SERVIDO"
    if not isinstance(servicio.clf, CalibratedClassifierCV):
        return (f"FALLO: se adoptó la calibración pero el servicio tiene un "
                f"{type(servicio.clf).__name__}")

    # El artefacto en disco debe coincidir: es lo que cargará el backend FastAPI.
    ruta = CFG.rutas.models / "modelo_clasificacion.joblib"
    if ruta.exists():
        if not isinstance(joblib.load(ruta), CalibratedClassifierCV):
            return "FALLO: el .joblib serializado no está calibrado"

    # Y la confianza debe reflejarlo: sin calibrar la media rondaba 0.50.
    r = servicio.predecir("Despliegue continuo con Docker y Kubernetes",
                           "Guía práctica para containerizar una aplicación y "
                           "publicarla mediante integración continua.")
    return f"OK (calibrado, p={r.probabilidad:.3f})"


PRUEBAS = [
    ("contrato JSON mínimo", test_contrato_minimo),
    ("validación rechaza entradas inválidas", test_validacion_rechaza),
    ("determinismo de la predicción", test_determinismo),
    ("búsqueda semántica ordenada", test_busqueda_ordenada),
    ("detección de idioma", test_idioma),
    ("caché de embeddings", test_cache_embeddings),
    ("reproducibilidad tras refijar semillas", test_reproducibilidad),
    ("inferencia sin globales de entrenamiento", test_sin_globales),
    ("auditoría de predicciones en SQLite", test_auditoria_sqlite),
    ("historial de versiones append-only", test_historial_versiones),
    ("carga en frío desde artefactos (ruta del backend)", test_carga_en_frio),
    ("paridad entre carga en frío y notebook", test_paridad_frio_caliente),
    ("configuración bien direccionada", test_config_bien_direccionada),
    ("ejemplos del brief bien clasificados", test_ejemplos_del_brief),
    ("el título aporta entidades técnicas", test_titulo_aporta_entidades),
    ("composición única de la entrada", test_composicion_entrada_unica),
    ("la calibración llega a producción", test_calibracion_llega_a_produccion),
]

print("PRUEBAS DE SANIDAD\n")
fallos = 0
for nombre, prueba in PRUEBAS:
    try:
        estado = prueba()
    except Exception as exc:
        estado = f"ERROR: {type(exc).__name__}: {exc}"
    if estado != "OK":
        fallos += 1
        log.error(f"Prueba fallida — {nombre}: {estado}")
    print(f"  [{estado[:6]:<6}] {nombre}")

print(f"\n{len(PRUEBAS) - fallos}/{len(PRUEBAS)} pruebas superadas")
if fallos:
    log.warning(f"{fallos} prueba(s) fallida(s): revisa el log antes de entregar.")
''')

md(r'''
## 7.6 Exportación del módulo para el backend

Última pieza del handoff: volcar a un archivo `.py` las clases y funciones que el backend necesita,
para que `app/services/nlp_service.py` haga `from techmind_core import TechMindInference` en vez de
copiar código a mano desde el notebook.

### Por qué no se usa `inspect.getsource`

Era el camino obvio y **no funciona en un notebook**, por un motivo que merece explicarse porque es
fácil tropezar con él otra vez.

`inspect.getsource` localiza el código de forma distinta según el objeto:

- Para una **función**, consulta `func.__code__.co_filename`. IPython registra el texto de cada celda
  en `linecache` bajo un nombre ficticio (`<ipython-input-7-...>`), así que lo encuentra.
- Para una **clase**, busca el archivo del módulo donde se definió: `__main__.__file__`. En un
  notebook **ese atributo no existe**, y la llamada falla con `OSError`.

El resultado es un fallo asimétrico y silencioso: las cinco funciones se exportaban y las seis
clases no, dejando un `techmind_core.py` que parecía correcto —tenía 11 KB— pero al que le faltaba
justamente `TechMindInference`, que es la razón de ser del archivo.

**La solución: leer el historial de celdas.** IPython conserva en la variable `In` el texto fuente de
cada celda ejecutada. Recorriéndolo y analizándolo con `ast`, se localiza cada definición por su
nombre y se extrae íntegra —incluidos sus decoradores—, sin depender de `linecache` ni de que el
módulo tenga archivo. Funciona igual para clases, funciones y constantes de módulo.

**Y se verifica lo generado.** La celda analiza el archivo resultante con `ast.parse` y comprueba que
cada símbolo esperado esté realmente dentro. Un export parcial deja de ser un aviso fácil de pasar
por alto para convertirse en un fallo explícito con la lista de lo que falta.
''')

code(r'''
# @title 7.6 — Exportación de techmind_core.py para el backend
COMILLAS = chr(34) * 3   # evita anidar comillas triples dentro de esta celda

CABECERA_CORE = "\n".join([
    COMILLAS,
    "techmind_core.py — Capa de inferencia de TechMind.",
    "",
    "Generado automáticamente por techmind_eda_modelado.ipynb §7.6",
    f"Versión del pipeline : {CFG.version}",
    f"Huella configuración : {CFG.huella()[:16]}",
    f"Fecha de generación  : {pd.Timestamp.now().isoformat()[:19]}",
    "",
    "Uso en el backend (app/services/nlp_service.py)::",
    "",
    "    from techmind_core import TechMindInference",
    "",
    "    servicio = TechMindInference.desde_artefactos('models/')",
    "    respuesta = servicio.predecir(titulo, texto)",
    "",
    "Este archivo contiene únicamente lo necesario para INFERENCIA. El",
    "entrenamiento vive en el notebook y no se replica aquí: el backend",
    "consume artefactos, no los produce.",
    COMILLAS,
    "",
])

# Orden deliberado: cada símbolo debe aparecer después de aquello de lo que depende.
FUENTES_A_EXPORTAR = [
    # Constantes de módulo (los patrones compilados que usan las funciones)
    "RE_MOJIBAKE", "RE_CONTROL", "CARACTER_REEMPLAZO",
    "RE_HTML", "RE_URL", "RE_REFS", "RE_PARENTESIS", "RE_ESPACIOS", "RE_RUIDO",
    "_STOPWORDS_REFERENCIA",
    # Validación
    "CodigoError", "ResultadoValidacion", "ErrorValidacion",
    "_normalizar_espacios", "_es_utf8_valido", "_detectar_corrupcion",
    "validar_entrada", "exigir_valido",
    # Idioma
    "Idioma", "_detectar_por_stopwords", "detectar_idioma",
    # Preprocesamiento
    "componer_entrada", "limpiar_texto", "preprocesar",
    # Contrato e inferencia
    "RespuestaContenido", "TechMindInference",
]


def _fuentes_de_celdas() -> list:
    """Devuelve el texto fuente de todas las celdas ejecutadas.

    IPython acumula en la variable `In` el código de cada celda de la sesión.
    Es la única vía fiable para recuperar el fuente de una CLASE definida en un
    notebook: `inspect.getsource` falla con ellas porque busca
    `__main__.__file__`, que no existe fuera de un script.

    Returns:
        Lista de cadenas, una por celda ejecutada.
    """
    try:
        historial = get_ipython().user_ns.get("In", [])
        return [c for c in historial if isinstance(c, str) and c.strip()]
    except Exception:
        return []


def _extraer_definicion(nombre: str, fuentes: list) -> str:
    """Extrae la definición completa de un símbolo analizando las celdas con `ast`.

    Reconoce clases, funciones y asignaciones de módulo, e incluye los
    decoradores (`@dataclass`) que preceden a la definición. Recorre las celdas
    en orden inverso para quedarse con la última versión, que es la vigente si
    una celda se reejecutó.

    Args:
        nombre: Nombre del símbolo a extraer.
        fuentes: Lista de códigos fuente de celdas.

    Returns:
        El bloque de código como cadena, o "" si no se encontró.

    Example:
        >>> _extraer_definicion("limpiar_texto", _fuentes_de_celdas())[:3]
        'def'
    """
    for src in reversed(fuentes):
        try:
            arbol = ast.parse(src)
        except SyntaxError:
            continue
        for nodo in arbol.body:
            coincide = False
            if isinstance(nodo, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                coincide = nodo.name == nombre
            elif isinstance(nodo, ast.Assign):
                coincide = any(isinstance(d, ast.Name) and d.id == nombre
                               for d in nodo.targets)
            if not coincide:
                continue
            lineas = src.splitlines()
            decoradores = getattr(nodo, "decorator_list", [])
            inicio = min([d.lineno for d in decoradores] + [nodo.lineno]) - 1
            return "\n".join(lineas[inicio:nodo.end_lineno])
    return ""


PREAMBULO = (
    "import ast, hashlib, json, logging, re, time, unicodedata\n"
    "from dataclasses import dataclass, field, asdict\n"
    "from pathlib import Path\n"
    "from typing import Any, Callable, Sequence\n\n"
    "import joblib\n"
    "import numpy as np\n"
    "import pandas as pd\n\n"
    "log = logging.getLogger('techmind')\n\n"
    "# DEPENDENCIAS QUE EL BACKEND DEBE PROPORCIONAR:\n"
    "#   - Config           : reconstruible desde models/config.json\n"
    "#   - rankear_keywords : §5.2.3 del notebook\n"
    "#   - explicar_prediccion, CacheEmbeddings : §5.5.2 y §5.1.2\n"
    "# Ver la tabla de handoff en §9.4.\n"
)

_fuentes = _fuentes_de_celdas()
partes, exportados, omitidos = [CABECERA_CORE, PREAMBULO], [], []

for nombre in FUENTES_A_EXPORTAR:
    bloque = _extraer_definicion(nombre, _fuentes)
    if bloque:
        partes.append("\n\n" + bloque + "\n")
        exportados.append(nombre)
    else:
        omitidos.append(nombre)
        log.warning(f"No se localizó '{nombre}' en el historial de celdas; se omite.")

ruta_core = CFG.rutas.models / "techmind_core.py"
ruta_core.write_text("".join(partes), encoding="utf-8")

# --- Verificación: el archivo debe ser Python válido y contener lo prometido ---
contenido = ruta_core.read_text(encoding="utf-8")
try:
    arbol_final = ast.parse(contenido)
    sintaxis_ok = True
    definidos_en_archivo = {
        n.name for n in arbol_final.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    } | {
        d.id for n in arbol_final.body if isinstance(n, ast.Assign)
        for d in n.targets if isinstance(d, ast.Name)
    }
except SyntaxError as exc:
    sintaxis_ok = False
    definidos_en_archivo = set()
    log.error(f"El módulo exportado no es Python válido: línea {exc.lineno}: {exc.msg}")

faltantes = [n for n in FUENTES_A_EXPORTAR if n not in definidos_en_archivo]

print(f"Módulo exportado : {ruta_core}")
print(f"Tamaño           : {ruta_core.stat().st_size / 1024:.1f} KB")
print(f"Sintaxis válida  : {'sí' if sintaxis_ok else 'NO'}")
print(f"Símbolos pedidos : {len(FUENTES_A_EXPORTAR)}")
print(f"Símbolos escritos: {len(exportados)}")

if faltantes:
    print(f"\n  FALTAN {len(faltantes)} símbolo(s): {faltantes}")
    print(f"  Causa habitual: la celda que los define no se ha ejecutado en esta sesión.")
    print(f"  Ejecuta el notebook de principio a fin y vuelve a correr esta celda.")
    log.error(f"Exportación incompleta: faltan {faltantes}")
else:
    print(f"\n  Los {len(exportados)} símbolos están presentes y el archivo compila.")
    log.info(f"Módulo de inferencia exportado y verificado: {ruta_core}")
''')
