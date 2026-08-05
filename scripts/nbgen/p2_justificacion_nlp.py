"""Sección 3 — Justificación técnica de modelos. Sección 4 — Preprocesamiento NLP."""

from .core import md, code

# ============================== 3. JUSTIFICACIÓN ==============================
md(r'''
---
# 3. Justificación técnica de las decisiones de modelado

Esta sección responde, para cada tecnología del pipeline, cinco preguntas: **qué problema resuelve**,
**por qué fue elegida**, **qué ventajas aporta**, **qué alternativas existían** y **por qué no se
eligieron**. Es un resumen operativo de `Technology_Architecture.md` §5 a §10, escrito desde la
perspectiva de quien va a ejecutar el pipeline.

El criterio que atraviesa todas las decisiones es el mismo: *¿qué stack permite entregar, en el
tiempo disponible, un sistema funcional, demostrable y con una base de código que un evaluador
técnico reconozca como profesional?* No es el mismo criterio que en una arquitectura empresarial a
diez años, y conviene decirlo de frente.
''')

md(r'''
## Diagrama 3 — Arquitectura NLP: qué modelo interviene en qué punto

```mermaid
flowchart LR
    TXT["Texto limpio"] --> SPACY

    subgraph SPACY["spaCy · es_core_news_sm"]
        direction TB
        S1["Tokenizador"]
        S2["POS tagger"]
        S3["Lematizador"]
        S4["NER + EntityRuler"]
        S1 --> S2 --> S3
        S1 --> S4
    end

    SPACY -->|"lemas filtrados por POS"| TFIDF["TF-IDF<br/>scikit-learn<br/>vector disperso ~8k dims"]
    SPACY -->|"entidades TECH"| RANK
    TXT -->|"prosa completa"| SBERT

    subgraph SBERT["Sentence-Transformers<br/>paraphrase-multilingual-MiniLM-L12-v2"]
        E1["Encoder BERT multilingüe"]
        E2["Mean pooling"]
        E3["Normalización L2"]
        E1 --> E2 --> E3
    end

    SBERT -->|"vector denso 384 dims"| KEYBERT["KeyBERT<br/>similitud coseno<br/>doc vs. n-gramas"]
    SBERT -->|"mismo vector"| CLF["Regresión Logística<br/>categoria + probabilidad"]
    SBERT -->|"mismo vector"| BT["BERTopic<br/>UMAP + HDBSCAN + c-TF-IDF"]
    SBERT -->|"mismo vector"| CHROMA[("ChromaDB<br/>índice HNSW")]

    TFIDF --> CLF
    TXT --> YAKE["YAKE<br/>heurísticas estadísticas"]

    KEYBERT --> RANK["Ranking híbrido<br/>Reciprocal Rank Fusion"]
    YAKE --> RANK

    RANK --> OUT["informacion_adicional"]
    CLF --> OUT2["categoria · probabilidad"]
    BT --> OUT3["tema"]
    CHROMA --> OUT4["relacionados"]

    style SBERT fill:#e8f4ea,stroke:#2d6a4f
    style SPACY fill:#e7eef7,stroke:#2a6f97
```

**Lo que el diagrama hace evidente:** un único vector de 384 dimensiones alimenta cuatro capacidades
distintas —keywords, clasificación, clustering y búsqueda—. Esa reutilización es la decisión
arquitectónica central del proyecto: reduce el costo de cómputo por documento a un solo `encode()`, y
elimina la necesidad de mantener tres pipelines de *feature engineering* separados y coherentes entre sí.
''')

md(r'''
## 3.1 spaCy — Procesamiento lingüístico

**Qué problema resuelve.** Antes de vectorizar hay que convertir prosa en unidades léxicas
normalizadas: tokenizar, descartar palabras funcionales sin señal temática, reducir cada palabra a su
forma canónica y quedarse con las categorías gramaticales que cargan contenido. Sin esto, la matriz
TF-IDF se llena de artículos y preposiciones, y `programación`/`programaciones`/`programar` ocupan
tres dimensiones distintas cuando conceptualmente son una.

**Por qué fue elegida.** Cuatro razones concretas: (1) resuelve tokenización, POS, lematización y NER
en **una sola pasada** `nlp(texto)`, frente a NLTK que exige ensamblar módulos independientes;
(2) su núcleo está en Cython, lo que importa porque el preprocesamiento está en el camino crítico de
cada petición a `POST /contenido`; (3) tiene modelos oficiales en español (`es_core_news_sm`), y el
corpus del brief es español; (4) es extensible vía `nlp.add_pipe`, que es exactamente lo que necesita
el `EntityRuler` de §4.1.

**Ventajas.** API unificada y orientada a objetos (`Doc`, `Token`, `Span`). Procesamiento por lotes
con `nlp.pipe`, que amortiza el overhead. Documentación orientada a producción. Componentes
personalizados sin tocar el modelo base.

**Alternativas.** NLTK.

**Por qué no NLTK.** Tres motivos: su API está fragmentada en módulos (`nltk.tokenize`, `nltk.stem`,
`nltk.tag`) que obligan a construir a mano el pipeline que spaCy entrega integrado; su rendimiento es
notablemente menor por estar implementada casi enteramente en Python puro, lo que impacta la latencia
del endpoint; y sus recursos para español son menos completos y menos mantenidos. NLTK sigue siendo
excelente para exploración didáctica, pero no como motor de producción.
''')

md(r'''
## 3.2 Sentence-Transformers — Embeddings semánticos

**Qué problema resuelve.** Comparar textos por **significado** y no por coincidencia literal de
palabras. *"API REST con Java y Spring Boot"* y *"servicio backend en Java construido con el
framework Spring"* comparten poquísimas palabras exactas: para TF-IDF son casi ortogonales, aunque
hablan de lo mismo. Un embedding sitúa ambos textos cerca en el espacio vectorial porque el modelo
fue entrenado para capturar relaciones de significado.

**Por qué fue elegida.** Porque **un solo vector alimenta cuatro capacidades**: clasificación (§5.2),
extracción de keywords vía KeyBERT (§5.1), clustering vía BERTopic (§5.6) y búsqueda semántica vía
ChromaDB (§6). Mantener una representación en vez de tres reduce drásticamente la complejidad del
sistema. Además: los modelos MiniLM corren en **CPU** sin GPU, el modelo elegido es **multilingüe**
—lo que deja abierta la puerta al inglés sin cambiar de espacio vectorial— y la API es una sola
llamada `model.encode(texto)` sin lógica de agregación.

**Ventajas.** Calidad semántica muy superior a TF-IDF en similitud y recomendación. Modelos pequeños
(~120 MB) con latencia aceptable en tiempo real. Catálogo amplio en Hugging Face Hub. Compatibilidad
directa con ChromaDB y con KeyBERT.

**Alternativas.** TF-IDF puro, Word2Vec/GloVe, embeddings vía API comercial (OpenAI, Cohere).

**Por qué no las alternativas.**
- *TF-IDF puro*: no captura sinónimos ni paráfrasis, lo que limitaría gravemente la recomendación y
  la búsqueda semántica. **No se descartó del todo**: se mantiene como baseline de comparación en
  §5.2 y como motor interno de c-TF-IDF en BERTopic.
- *Word2Vec/GloVe*: producen vectores de **palabra**, no de oración; requerirían promediar los
  vectores del documento, lo que degrada la calidad frente a un modelo entrenado específicamente para
  oraciones completas. Además, obtener vectores de calidad en español añade descargas grandes sin
  ganancia clara.
- *API comercial*: altísima calidad, pero introduce dependencia de red externa durante una demo en
  vivo, costo por token sin presupuesto asignado y una API key adicional que gestionar además de las
  credenciales de OCI. El riesgo operativo no se justifica.

**La desventaja que sí duele.** Un modelo multilingüe genérico no captura jerga técnica muy
específica tan bien como uno afinado sobre corpus técnico. Lo mitigamos combinando embeddings con el
`EntityRuler` de reglas (§4.1), y lo dejamos anotado como trabajo futuro (§9).
''')

md(r'''
## 3.3 KeyBERT — Extracción de palabras clave

**Qué problema resuelve.** El brief pide `informacion_adicional: ["Java", "Spring Boot", "API REST"]`
— una lista corta de términos técnicos verdaderamente representativos, no las palabras más
frecuentes. En textos de una o dos oraciones, "más frecuente" y "más representativo" casi nunca
coinciden.

**Por qué fue elegida.** Porque **reutiliza el modelo de embeddings ya cargado**: no añade una
dependencia nueva ni un segundo modelo en memoria. La misma instancia de Sentence-Transformers que
calcula el vector del documento sirve para rankear las frases candidatas por similitud coseno contra
ese vector. En documentos cortos —el caso del brief— esto supera claramente a TF-IDF, cuyo IDF
necesita un corpus grande para ser informativo.

**Ventajas.** No requiere corpus de referencia: funciona documento por documento, incluso sobre
contenido nunca visto. Soporta n-gramas configurables, capturando términos compuestos como
"Spring Boot" o "API REST". Ofrece MMR (*Maximal Marginal Relevance*) para diversificar y evitar
devolver "API" y "API REST" como dos keywords casi idénticas.

**Alternativas.** TF-IDF, YAKE.

**Por qué no las alternativas.** TF-IDF depende de un corpus de referencia representativo para
calcular el IDF, y el corpus de un hackathon es pequeño y heterogéneo. YAKE funciona bien en textos
cortos con heurísticas (posición, capitalización, dispersión) pero **no comparte infraestructura** con
el resto del pipeline: obligaría a mantener una lógica de keywords aislada, sin beneficiarse de la
comprensión semántica ya construida.

**Lo que hicimos en la práctica.** No elegimos uno y descartamos los demás: implementamos **los tres**
y los fusionamos con *Reciprocal Rank Fusion* (§5.1), añadiendo como cuarta señal las entidades del
`EntityRuler`. Cada método aporta algo que los otros no: KeyBERT capta relevancia semántica, YAKE
capta lo distintivo sin necesitar corpus, y el EntityRuler garantiza —por construcción, con precisión
de reglas— que "Spring Boot" nunca se pierda.
''')

md(r'''
## 3.4 BERTopic — Descubrimiento de tópicos

**Qué problema resuelve.** Agrupar automáticamente los documentos por temática **sin definir de
antemano cuántos temas hay ni etiquetarlos a mano**. Es distinto de la clasificación: la
clasificación asigna a una taxonomía que nosotros impusimos al recolectar; el clustering descubre la
estructura que los datos realmente tienen, que puede no coincidir — y ese desacuerdo es información
valiosa, no un error.

**Por qué fue elegida.** Tres razones: (1) **no exige fijar `k`**; usa HDBSCAN, que descubre el
número de grupos por densidad, evitando una decisión arbitraria cuando no se sabe cuántos temas hay;
(2) **genera etiquetas legibles automáticamente** vía c-TF-IDF ("Tema 3: Java, Spring Boot, API
REST"), directamente presentables sin inspección manual; (3) **reutiliza los mismos embeddings** ya
calculados, sin cómputo adicional.

**Ventajas.** Maneja el ruido explícitamente: un documento que no encaja en ningún tema se marca como
*outlier* (tópico `-1`) en vez de forzarlo dentro de un grupo al que no pertenece. Maneja clusters de
forma y tamaño irregular, cosa que KMeans no puede porque asume grupos esféricos y equilibrados.
Trae visualizaciones incorporadas.

**Alternativas.** KMeans sobre embeddings, LDA.

**Por qué no las alternativas.** KMeans exige `k` de antemano y asume geometría esférica, supuesto
que rara vez se cumple con contenido técnico heterogéneo (siempre hay más artículos de un tema que de
otro). LDA opera sobre bolsas de palabras, no sobre embeddings: perdería toda la señal semántica que
ya pagamos por calcular. **KMeans no se descarta del todo**: lo mantenemos en §5.6 como comparación
metodológica, con selección de `k` por codo y silueta, para demostrar que la elección de BERTopic fue
razonada y no por defecto.

**La desventaja honesta.** Con corpus pequeños HDBSCAN puede marcar una fracción alta de documentos
como ruido. Lo mitigamos bajando `min_topic_size` en proporción al tamaño del corpus, y por eso
mantenemos KMeans como respaldo documentado.
''')

md(r'''
## 3.5 ChromaDB — Base de datos vectorial

**Qué problema resuelve.** Dado un vector de consulta, encontrar eficientemente los `N` vectores más
cercanos de la colección. Una base relacional no puede hacerlo: `ORDER BY` sobre una columna numérica
no resuelve una búsqueda por similitud coseno en 384 dimensiones. La fuerza bruta funciona con cien
documentos y deja de funcionar con cien mil. Las bases vectoriales indexan con estructuras
especializadas —grafos HNSW— que resuelven vecinos aproximados en tiempo sub-lineal.

**Por qué fue elegida.** Porque es **autoalojable**: corre embebida o como contenedor dentro de la
misma infraestructura del equipo, sin depender de un proveedor SaaS externo. Esto importa
específicamente porque el brief exige integración con **OCI**: meter un servicio vectorial de un
tercero no aporta nada a ese requisito y sí añade un punto de falla durante la demo en vivo. Además
es gratuita, tiene una API de Python que se aprende en minutos y se despliega como un contenedor más
del `docker-compose.yml`.

**Ventajas.** Persiste **metadatos junto al vector**, lo que permite filtrar la búsqueda semántica por
categoría en una sola consulta (`documentos similares a X, pero solo de Backend`) sin ir a PostgreSQL
en el camino crítico. Sin dependencia de red externa. Integración directa con embeddings de
Sentence-Transformers.

**Alternativas.** Pinecone, FAISS, `pgvector` sobre PostgreSQL.

**Por qué no las alternativas.**
- *Pinecone*: requiere cuenta y API key externas a OCI, tiene costo más allá de una capa gratuita
  limitada, y añade dependencia de red durante la demo. Excelente producto, contexto equivocado.
- *FAISS*: **es una librería de indexación, no una base de datos**. No persiste metadatos, no expone
  servidor ni API lista para usar, no filtra por metadatos. Adoptarla obligaría a reimplementar a
  mano lo que ChromaDB ya resuelve. Nota: ChromaDB puede usar HNSW/FAISS como motor interno, así que
  el proyecto se beneficia indirectamente de esa tecnología sin pagar el costo de integrarla.
- *`pgvector`*: unificaría metadatos y vectores en un solo motor, lo cual es atractivo a largo plazo.
  Se descarta **para el MVP** por simplicidad operativa, y queda anotado en §9 como el camino natural
  de consolidación si el proyecto continúa.

**Cómo convive con PostgreSQL.** No compiten, se complementan. PostgreSQL responde *"¿qué documentos
son de categoría Backend?"*; ChromaDB responde *"¿qué documentos se parecen semánticamente a este?"*.
Un mismo documento vive en ambos, unido por un identificador común — el patrón estándar
*vector store + metadata store*.
''')

md(r'''
## 3.6 Tabla de decisión consolidada

| Componente | Elegido | Alternativas descartadas | Criterio decisivo |
|---|---|---|---|
| NLP | **spaCy** | NLTK | Pipeline integrado + rendimiento Cython + modelos ES oficiales |
| Embeddings | **Sentence-Transformers (SBERT)** | TF-IDF puro, Word2Vec/GloVe, API comercial | Un vector reutilizado por 4 capacidades, CPU, sin dependencia externa |
| Keywords | **KeyBERT** (+ YAKE + EntityRuler, fusionados) | TF-IDF, YAKE en solitario | Reutiliza el modelo de embeddings; no necesita corpus de referencia |
| Clustering | **BERTopic** | KMeans, LDA | No exige `k`, etiquetas legibles automáticas, maneja outliers |
| Clasificador | **Regresión Logística** | SVM, Random Forest, fine-tuning de BERT | Probabilidades calibradas (`predict_proba`), interpretable, entrena en segundos |
| Base vectorial | **ChromaDB** | Pinecone, FAISS, pgvector | Autoalojable en OCI, gratuita, metadatos junto al vector |
| Serialización | **joblib** | pickle | Eficiente con arrays de numpy; es lo que pide el brief |

**Sobre la Regresión Logística**, que el brief sugiere explícitamente: la elegimos porque produce
**probabilidades**, y el contrato de salida exige un campo `probabilidad`. Un SVM da distancias al
hiperplano, no probabilidades, y calibrarlas requiere `CalibratedClassifierCV` y un split adicional.
Random Forest da probabilidades pero mal calibradas en problemas de alta dimensión y pocos datos. Un
fine-tuning de BERT superaría a todos en precisión, pero requiere GPU y horas de entrenamiento que un
hackathon no tiene — queda anotado en §9.
''')

# ============================== 4. NLP ==============================
md(r'''
---
# 4. Preprocesamiento NLP
### Etapa 2 del diagrama

> **Tokenización & Stopwords → Lematización → Filtrado por POS Tagging → Reconocimiento de entidades**

Convertimos prosa validada en las dos representaciones que el modelado necesita: una secuencia de
lemas filtrados (insumo de TF-IDF) y un conjunto de entidades técnicas (insumo del ranking de
keywords). Una sola pasada de spaCy produce ambas.

**Lematización, no stemming.** El diagrama del brief ofrece ambas. Elegimos lematización porque el
stemming trunca por reglas ortográficas y produce raíces que no son palabras
(*"programación" → "program"*), lo cual degrada la legibilidad de las keywords que el brief espera
devolver al usuario en `informacion_adicional`. La lematización devuelve la forma canónica del
diccionario, presentable tal cual.

**Aquí es donde se baja a minúsculas** — en el lema, no antes. La razón la explicamos en §2.4: el POS
tagger y el NER de spaCy usan la capitalización como señal, y `Java` en minúscula deja de ser
reconocible como nombre propio. Bajar a minúsculas antes de spaCy destruye información que spaCy
necesita; hacerlo después, sobre el lema, es gratis.
''')

md(r'''
## 4.1 Pipeline de spaCy y reconocimiento de entidades técnicas

**El problema concreto que resuelve el `EntityRuler`.** El modelo `es_core_news_sm` fue entrenado
sobre noticias y texto general en español: reconoce personas, lugares y organizaciones, pero no sabe
que "Spring Boot" o "FastAPI" son tecnologías. Sobre el ejemplo del brief, el NER estadístico no
etiqueta ninguna de las tres tecnologías que el enunciado espera ver en `informacion_adicional`. Es
una limitación conocida y documentada en `Technology_Architecture.md` §5.

**La solución.** Un `EntityRuler` insertado **antes** del NER estadístico, con un diccionario de
~140 tecnologías agrupadas por dominio. Es un componente basado en reglas, con dos propiedades que
lo hacen valioso aquí:

- **Precisión perfecta por construcción.** Si "Docker" está en el diccionario y aparece en el texto,
  se detecta. No hay incertidumbre que ponderar — y por eso el ranking de keywords (§5.2) le asigna
  el peso más alto de las cuatro señales.
- **Cobertura finita y explícita.** Lo que no está listado, no se detecta. Es una limitación real
  (§9.2, punto 4), pero es *visible y auditable*: se sabe exactamente qué reconoce el sistema, cosa
  que no ocurre con un modelo estadístico.

El orden importa: `before="ner"` hace que las reglas tengan prioridad sobre el modelo estadístico,
evitando que "Java" se etiquete como `LOC` (la isla) en vez de `TECH`.
''')

code(r'''
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
''')

md(r'''
## 4.2 Las cuatro cajas de la etapa 2, en una sola pasada

`preprocesar()` recorre el `Doc` **una vez** y produce simultáneamente los cuatro artefactos que el
diagrama presenta como pasos secuenciales. No es una micro-optimización caprichosa: recorrer cuatro
veces la misma estructura para aplicar cuatro filtros que ya tienen toda la información disponible en
el primer recorrido es trabajo redundante en el camino crítico de cada petición HTTP.

Las decisiones de filtrado y su porqué:

- **Se descartan** espacios, puntuación, números y dígitos: no aportan señal temática y ensucian el
  vocabulario de TF-IDF.
- **Se descartan las stopwords de spaCy** más una lista propia (`stopwords_extra`) de verbos y
  sustantivos genéricos —"ser", "hacer", "forma", "caso"— que sobreviven al filtro estándar pero
  aparecen en todas las categorías por igual, es decir, no discriminan.
- **Se descartan tokens de menos de 3 caracteres**, que en español técnico son casi siempre
  preposiciones o siglas ambiguas.
- **`lemas_pos` conserva solo `NOUN`, `PROPN` y `ADJ`**: el contenido temático de un texto técnico
  vive en sus sustantivos y adjetivos. Los verbos, en documentación técnica, son mayoritariamente de
  proceso genérico ("permite", "utiliza", "configura") y aportan poco a la discriminación entre
  categorías.
''')

code(r'''
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
''')

md(r'''
## 4.3 Trazabilidad del pipeline

Antes de confiar en resultados agregados sobre cientos de documentos, conviene ver el pipeline
completo actuando sobre **un** documento conocido — el ejemplo literal del enunciado del brief. Si
alguna etapa está mal configurada, aquí se ve; en una métrica promediada, no.

La tabla token a token del final es deliberada: muestra qué conserva y qué descarta cada filtro, y es
el material que permite discutir con el jurado *por qué* el sistema eligió esas keywords.
''')

code(r'''
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
''')

md(r'''
## 4.4 EDA sobre el texto ya procesado

Segunda ronda de exploración, ahora sobre el texto transformado. Responde dos preguntas que solo
tienen sentido después del preprocesamiento:

1. **¿Cuánto ruido eliminamos?** La reducción de tokens por etapa cuantifica el efecto del filtrado.
   Una reducción demasiado baja indicaría que el filtro no está actuando; una demasiado alta, que
   estamos descartando señal útil.
2. **¿Son las categorías léxicamente separables?** Los términos más frecuentes por categoría son una
   señal cualitativa previa al entrenamiento: si dos categorías comparten sus ocho términos
   principales, el clasificador léxico las va a confundir, y lo veremos confirmado en la matriz de
   confusión de §5.4.2.
''')

code(r'''
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
display(SOLAPAMIENTO.head(6).reset_index(drop=True))

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
''')
