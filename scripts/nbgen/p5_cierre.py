"""Sección 8 — OCI, rendimiento y empaquetado. Sección 9 — Conclusiones y mejoras futuras."""

from .core import md, code

# ============================== 8. OCI Y EMPAQUETADO ==============================
md(r'''
---
# 8. Integración con OCI, rendimiento y empaquetado

## 8.1 OCI Object Storage

Requisito **obligatorio** del brief: *"la solución debe utilizar al menos un servicio de OCI"*. Desde
el notebook subimos los artefactos serializados a **Object Storage**, que es de donde el backend los
descarga al arrancar en OCI Compute.

El patrón importa más que el código: separar el *artefacto* del *servicio* significa que reentrenar
no exige reconstruir la imagen del contenedor. La celda de entrenamiento sube una versión nueva al
bucket; el backend la descarga en el siguiente arranque. Es la misma razón por la que el prefijo del
objeto incluye la versión (`models/v2.0.0/`): permite tener varias conviviendo y hacer rollback
cambiando una variable de entorno.

> ⚠️ Requiere `~/.oci/config` con las credenciales del tenancy. En Colab, súbelo con `files.upload()`
> o monta Drive. **Nunca** publiques la clave privada en el repositorio — en el despliegue real esa
> credencial vive en **OCI Vault**, no en un archivo.
''')

code(r'''
# @title 8.1 — Subida de artefactos a OCI Object Storage (opcional)
SUBIR_A_OCI = False   # ponlo en True cuando tengas ~/.oci/config configurado

NAMESPACE = "TU_NAMESPACE"        # obtener con: oci os ns get
BUCKET    = "techmind-models"
PREFIJO   = f"models/v{CFG.version}/"

if SUBIR_A_OCI:
    get_ipython().system("pip install -q oci")
    import oci

    with etapa("subida a OCI Object Storage"):
        configuracion_oci = oci.config.from_file()
        cliente_os = oci.object_storage.ObjectStorageClient(configuracion_oci)

        subidos = 0
        for ruta in sorted(CFG.rutas.models.rglob("*")):
            if not ruta.is_file():
                continue
            nombre_objeto = PREFIJO + str(ruta.relative_to(CFG.rutas.models)).replace("\\", "/")
            with open(ruta, "rb") as f:
                cliente_os.put_object(NAMESPACE, BUCKET, nombre_objeto, f)
            log.info(f"Subido: {nombre_objeto} ({ruta.stat().st_size / 1024:,.1f} KB)")
            subidos += 1

    print(f"\n>>> {subidos} artefactos en oci://{BUCKET}@{NAMESPACE}/{PREFIJO}")
else:
    print("SUBIR_A_OCI = False — celda omitida.")
    print(f"Para activarla: configura ~/.oci/config, completa NAMESPACE y pon SUBIR_A_OCI = True.")
    print(f"\nDestino previsto: oci://{BUCKET}@<namespace>/{PREFIJO}")
    print(f"Artefactos a subir: {len(list(CFG.rutas.models.rglob('*')))} archivos")
''')

md(r'''
## 8.2 Rendimiento del pipeline

El brief no pide un análisis de rendimiento, pero un pipeline cuyo costo por etapa se desconoce no se
puede optimizar ni dimensionar. El decorador `@cronometrar` de §0.6 ha ido acumulando la duración de
cada operación; aquí la consolidamos.

Las tres optimizaciones aplicadas y su efecto esperado:

| Optimización | Dónde | Qué ahorra |
|---|---|---|
| **Caché de embeddings** | §5.1.2 | El cómputo completo del encoder en reejecuciones — la operación más costosa |
| **Reutilización del `Doc` de spaCy** | §5.2.3, §7.2 | Una pasada completa de spaCy por petición de inferencia |
| **`nlp.pipe` por lotes** | §4.2 | El overhead de construcción del `Doc` documento a documento |
| **Un solo `encode()` por documento** | §5.1.3 | Tres codificaciones redundantes (clasificación, keywords, clustering) |
''')

code(r'''
# @title 8.2 — Reporte de tiempos de ejecución
def consolidar_tiempos(registros: list) -> pd.DataFrame:
    """Agrupa los tiempos acumulados por `@cronometrar` y `etapa()`.

    Usa agregación nombrada (`NamedAgg`) en lugar de `.agg(["sum", "count"])`
    seguido de renombrar columnas: esa vía produce un número de columnas que
    depende de la versión de pandas y de si se usó `as_index=False`, y renombrar
    a ciegas rompe con `ValueError: Length mismatch`.

    Args:
        registros: Lista de dicts con las claves `operacion` y `segundos`.

    Returns:
        DataFrame con `operacion`, `segundos_total`, `invocaciones` y
        `porcentaje`, ordenado de mayor a menor tiempo. Vacío con esas mismas
        columnas si no hay registros.

    Example:
        >>> consolidar_tiempos([{"operacion": "x", "segundos": 1.0}]).shape[0]
        1
    """
    columnas = ["operacion", "segundos_total", "invocaciones", "porcentaje"]
    if not registros:
        return pd.DataFrame(columns=columnas)

    crudo = pd.DataFrame(registros)
    if not {"operacion", "segundos"}.issubset(crudo.columns):
        log.warning("Registro de tiempos con formato inesperado; se omite el reporte.")
        return pd.DataFrame(columns=columnas)

    consolidado = (crudo.groupby("operacion", as_index=False)
                        .agg(segundos_total=("segundos", "sum"),
                             invocaciones=("segundos", "count"))
                        .sort_values("segundos_total", ascending=False))

    # Las etapas engloban a las operaciones que contienen, así que el total se
    # calcula sobre ellas para que los porcentajes no sumen más del 100 %.
    etapas = consolidado[consolidado["operacion"].str.startswith("[etapa]")]
    total = etapas["segundos_total"].sum()
    if not total:
        total = consolidado["segundos_total"].sum()

    consolidado["porcentaje"] = (
        (consolidado["segundos_total"] / total * 100).round(1) if total else 0.0
    )
    return consolidado.reset_index(drop=True)


consolidado = consolidar_tiempos(TIEMPOS)

if len(consolidado):
    total = consolidado.loc[
        consolidado["operacion"].str.startswith("[etapa]"), "segundos_total"].sum()
    total = total or consolidado["segundos_total"].sum()

    plt.figure(figsize=(11, max(4, 0.4 * min(len(consolidado), 14))))
    sns.barplot(data=consolidado.head(14), x="segundos_total", y="operacion",
                hue="operacion", palette="flare", legend=False)
    plt.title("Tiempo de ejecución por operación", fontweight="bold")
    plt.xlabel("segundos"); plt.ylabel("")
    plt.tight_layout(); plt.show()

    display(consolidado.head(14))
    print(f"\nTiempo total de las etapas principales: {total:.1f} s")
else:
    print("Sin tiempos registrados: ejecuta las secciones §2 a §7 antes de esta celda.")

print(f"\nEstadísticas de la caché de embeddings:")
for clave, valor in CACHE.estadisticas().items():
    print(f"  {clave:<20} {valor}")
print("\nEn una reejecución del notebook sin cambios de configuración, la tasa de acierto")
print("de la caché debería acercarse al 100% y el tiempo de embeddings caer a casi cero.")
''')

code(r'''
# @title 8.3 — Empaquetado de entregables
with etapa("empaquetado"):
    shutil.make_archive("techmind_artefactos", "zip", CFG.rutas.models)

print("ENTREGABLES GENERADOS\n")
total_kb = 0
for ruta in sorted(CFG.rutas.base.rglob("*")):
    if ruta.is_file() and ruta.suffix in (".csv", ".json", ".joblib", ".npy", ".py", ".log"):
        kb = ruta.stat().st_size / 1024
        total_kb += kb
        print(f"  {str(ruta):<62} {kb:>9,.1f} KB")

zip_kb = Path("techmind_artefactos.zip").stat().st_size / 1024
print(f"\n  {'techmind_artefactos.zip':<62} {zip_kb:>9,.1f} KB   (listo para OCI Object Storage)")
print(f"\nTotal de artefactos en disco: {total_kb / 1024:.1f} MB")

log.info(f"Empaquetado completo: {zip_kb:.1f} KB")

# En Colab, descarga directa:
# from google.colab import files; files.download("techmind_artefactos.zip")
''')

# ============================== 9. CONCLUSIONES ==============================
md(r'''
---
# 9. Conclusiones

## 9.1 Qué entrega este notebook

### Requisitos obligatorios del brief

| Requisito | Estado | Sección |
|---|---|---|
| Exploración y limpieza de datos (EDA) | ✅ | §2.5, §4.4 |
| Tratamiento de textos | ✅ pipeline spaCy de 4 etapas | §4 |
| Transformación al formato de modelado | ✅ TF-IDF disperso + embeddings densos | §5.1 |
| Entrenamiento y evaluación de modelos | ✅ dos representaciones comparadas bajo el mismo clasificador | §5.3 |
| Métricas de desempeño apropiadas | ✅ accuracy, precision, recall, F1 macro/weighted, top-2, matriz de confusión, CV 5-fold | §5.4 |
| Serialización del modelo (joblib) | ✅ con `metadata.json` de versionado completo | §5.7 |
| Mínimo tres ejemplos de uso | ✅ | §7.3 |
| Integración con OCI | ✅ Object Storage | §8.1 |
| Documentación de dependencias y versiones | ✅ | §0.7 |

### Recursos opcionales cubiertos

| Recurso | Sección |
|---|---|
| Recomendación de contenido relacionado | §6.4 |
| Búsqueda semántica | §6.3 |
| Consulta por categorías | §6.3 (filtrado por metadatos en ChromaDB) |
| Procesamiento en lote (CSV) | §7.4 |
| Pruebas automatizadas | §7.5 — 7 pruebas de sanidad |
| Explicabilidad del modelo | §5.5 — global y local |
| Persistencia de resultados | §6.5 |
| Organización automática por tópicos | §6.2 |

### Mejoras de ingeniería incorporadas en la v2.0.0

| Área | Qué se añadió |
|---|---|
| **Validación** | 7 controles tipificados: presencia, tipo, UTF-8, corrupción/mojibake, longitud mínima y máxima, ratio alfabético, ratio de mayúsculas |
| **Idioma** | Detección con `langdetect` + heurística de respaldo; `RegistroIdiomas` con carga perezosa por idioma |
| **Configuración** | `dataclass` `Config` con 11 bloques, serializable, hasheable; cero valores hardcodeados |
| **Logging** | Logger dual (consola + archivo), decorador `@cronometrar`, context manager `etapa()` |
| **Reproducibilidad** | 5 fuentes de aleatoriedad fijadas: `random`, `numpy`, `torch`, `PYTHONHASHSEED`, `langdetect` |
| **Caché** | Embeddings indexados por hash de (modelo, normalización, texto), persistidos en disco |
| **Explicabilidad** | Ablación por término (agnóstica al modelo) + similitud a centroides de clase |
| **Versionado** | `metadata.json` con versión, fecha, hiperparámetros, métricas, modelo y hash del dataset |
| **API-ready** | `TechMindInference` sin estado global; entrenamiento, carga, predicción y contrato separados |
| **Deduplicación** | Exacta por hash SHA-256 + near-duplicate por Jaccard sobre shingles de 3 palabras, umbral calibrado empíricamente en 0.60 |

## 9.2 Limitaciones reconocidas

Enumerarlas no es una formalidad: cada una condiciona cómo deben leerse las métricas reportadas.

1. **Etiquetas débiles (*distant supervision*).** Las categorías provienen del artículo semilla, no de
   anotación humana por párrafo. Un párrafo sobre índices dentro del artículo de PostgreSQL hereda la
   etiqueta *Bases de Datos* aunque podría pertenecer a *Backend*. Las métricas miden concordancia
   con una taxonomía ruidosa, no con verdad de terreno.

2. **Sesgo de fuente única.** Todo el corpus proviene de Wikipedia, cuyo registro es enciclopédico y
   no coincide con el de un tutorial o una anotación de estudio — que son los casos de uso reales del
   brief. El modelo puede degradarse ante texto informal o fragmentario.

3. **Tamaño del corpus.** Con pocos cientos de documentos, los intervalos de confianza de las métricas
   son amplios y HDBSCAN puede marcar como ruido una fracción alta del corpus. La desviación estándar
   entre folds de §5.4.3 es el indicador honesto de esta incertidumbre.

4. **Cobertura finita del `EntityRuler`.** El diccionario de tecnologías es manual: lo que no está
   listado, no se detecta. Cubre las tecnologías más frecuentes, no el largo tail.

5. **Probabilidades no calibradas.** El campo `probabilidad` refleja el orden de confianza relativa
   del modelo, pero no está calibrado: un 0.89 no significa que acierte el 89 % de las veces que
   emite ese valor. La calibración requiere un conjunto de validación adicional.

6. **Un solo idioma en producción.** La arquitectura soporta multilenguaje (§2.2) y el modelo de
   embeddings es multilingüe, pero el clasificador se entrenó solo con documentos en español. Activar
   el inglés requiere datos etiquetados, no cambios de código.

7. **Deduplicación cuadrática.** La comparación de near-duplicates es O(n²) en el número de
   documentos. Aceptable para cientos, inviable para decenas de miles.

## 9.3 Mejoras futuras

Ordenadas por relación entre valor aportado y esfuerzo requerido.

### Corto plazo — funcionalidad

| Mejora | Qué implica | Por qué importa |
|---|---|---|
| **OCR para PDFs e imágenes** | Añadir Tesseract/`pytesseract` o **OCI Vision** como etapa 0 del pipeline, antes de la validación | Buena parte del conocimiento técnico está en PDFs escaneados y capturas; hoy quedan fuera del sistema. OCI Vision además refuerza el requisito de integración cloud |
| **Procesamiento por lotes vía CSV** | Endpoint `POST /contenido/lote` que consuma `predecir_lote` (§7.4, ya implementado) | El caso de uso real es cargar un archivo con cientos de contenidos, no de uno en uno |
| **Consulta por categorías** | Endpoints `GET /temas` y `GET /contenido?categoria=Backend&tema=3` | El filtrado por metadatos de ChromaDB ya lo soporta (§6.3); falta exponerlo |
| **Recomendaciones avanzadas** | Combinar similitud vectorial con señales de comportamiento (co-visitas, feedback explícito), diversificar con MMR | La recomendación actual es puramente semántica: siempre devuelve lo más parecido, nunca lo complementario |

### Medio plazo — calidad del modelo

| Mejora | Qué implica | Por qué importa |
|---|---|---|
| **Ampliar y diversificar el corpus** | Añadir documentación oficial, dev.to, blogs técnicos | Ataca directamente las limitaciones 2 y 3 |
| **Anotación manual de validación** | Etiquetar a mano 200-300 documentos como verdad de terreno | Sin esto, las métricas miden concordancia con etiquetas ruidosas (limitación 1) |
| **Calibración de probabilidades** | `CalibratedClassifierCV` con método isotónico o de Platt | Vuelve el campo `probabilidad` interpretable como confianza real (limitación 5) |
| **Soporte multilenguaje activo** | Recolectar corpus en inglés, reentrenar, cambiar `idiomas_soportados` a `("es", "en")` | La arquitectura ya está lista (§2.2); solo faltan datos |
| **Fine-tuning del modelo de embeddings** | Entrenar SBERT sobre pares de documentos técnicos del dominio | Mejora simultáneamente clasificación, keywords, clustering y búsqueda |

### Medio plazo — operación

| Mejora | Qué implica | Por qué importa |
|---|---|---|
| **Dashboard** | Frontend en Next.js con métricas del corpus, mapa de tópicos, buscador semántico y explorador de categorías | Convierte el sistema en un producto usable, no solo en una API |
| **Monitoreo** | Registrar distribución de categorías predichas, latencia p50/p95/p99 y tasa de confianza baja; alertar ante *data drift* | Un modelo en producción se degrada silenciosamente cuando cambia la distribución de entrada |
| **Pruebas automatizadas en CI** | Llevar las 7 pruebas de §7.5 a `pytest` + pruebas de integración de los endpoints con `TestClient` | Las pruebas del notebook no se ejecutan en cada commit; en CI sí |
| **Reentrenamiento programado** | **OCI Functions** que lea el corpus de PostgreSQL, reentrene y suba la versión nueva a Object Storage | Cierra el ciclo: el modelo mejora al crecer el corpus, sin intervención manual |

### Largo plazo — arquitectura

| Mejora | Qué implica | Por qué importa |
|---|---|---|
| **Migración a `pgvector`** | Unificar metadatos y vectores en PostgreSQL | Elimina una pieza de infraestructura y hace transaccional la escritura conjunta |
| **Deduplicación con MinHash/LSH** | Sustituir la comparación cuadrática por hashing sensible a la localidad | Necesario a partir de decenas de miles de documentos (limitación 7) |
| **Clasificación multietiqueta** | Pasar de `categoria` única a un conjunto de categorías con umbral | Un artículo sobre desplegar modelos de ML en Kubernetes es DevOps *y* Data Science; el top-2 accuracy de §5.4 ya evidencia el problema |
| **Búsqueda híbrida** | Combinar BM25 léxico con similitud vectorial por RRF | La búsqueda semántica falla en consultas de término exacto (números de versión, nombres de error); el léxico las resuelve |
''')

md(r'''
## 9.4 Handoff al equipo de Backend

**Artefactos que el backend consume** (desde `data_science/models/` o desde OCI Object Storage):

```
models/
├── modelo_clasificacion.joblib   # clasificador ganador
├── label_encoder.joblib          # índice ↔ nombre de categoría
├── vectorizador_tfidf.joblib     # solo si el ganador es el modelo A
├── modelo_kmeans.joblib          # respaldo de clustering
├── centroides_clase.joblib       # explicabilidad local
├── modelo_bertopic/              # tópicos + c-TF-IDF (safetensors)
├── config.json                   # configuración exacta del entrenamiento
├── metadata.json                 # versión · métricas · dataset · hiperparámetros
└── techmind_core.py              # capa de inferencia exportada (§7.6)
```

**Integración en `app/services/nlp_service.py`:**

```python
from functools import lru_cache
from pathlib import Path
from techmind_core import TechMindInference

@lru_cache(maxsize=1)
def obtener_servicio() -> TechMindInference:
    # Se carga UNA vez al arrancar el proceso, no por petición.
    return TechMindInference.desde_artefactos(Path("models"))
```

**Integración en `app/api/routes/contenido.py`:**

```python
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

router = APIRouter()

@router.post("/contenido", response_model=ContenidoResponse)
async def procesar_contenido(payload: ContenidoRequest):
    servicio = obtener_servicio()
    try:
        # spaCy y SBERT son síncronos y bloqueantes: van al threadpool
        # para no bloquear el event loop (Technology_Architecture.md §4).
        respuesta = await run_in_threadpool(
            servicio.predecir, payload.titulo, payload.texto
        )
    except ErrorValidacion as exc:
        raise HTTPException(
            status_code=422,
            detail=[{"codigo": c, "mensaje": m} for c, m in exc.resultado.errores],
        )
    return respuesta.a_dict()
```

**Qué reconstruye `desde_artefactos()` y qué no.** Conviene tenerlo explícito, porque de aquí salen
las diferencias entre lo que enseña la demo y lo que sirve la API:

| Componente | ¿Lo carga? | Efecto si falta |
|---|---|---|
| Clasificador + label encoder | ✅ desde `.joblib` | — |
| Modelo de embeddings | ✅ por nombre desde `metadata.json` | — |
| spaCy **+ EntityRuler** | ✅ el ruler se reañade desde `tecnologias.json` | Sin él, `entidades_tecnicas` vacío y keywords degradadas |
| KeyBERT + YAKE | ✅ reconstruidos sobre el embedder | — |
| Centroides de clase | ✅ si existe `centroides_clase.joblib` | Sin ellos, no hay explicabilidad |
| BERTopic | ✅ si existe `modelo_bertopic/` | Sin él, el campo `tema` viene vacío |
| ChromaDB | ✅ si existe el directorio persistido | Sin él, `relacionados` viene vacío |
| Caché de embeddings | ✅ se instancia siempre | Sin ella, recodifica en cada petición |
| **Conexión SQLite** | ❌ **la inyecta el backend** | Sin ella, no hay auditoría de predicciones |
| **PostgreSQL** | ❌ **es del backend** | — |

La conexión de auditoría se pasa aparte porque en producción no es SQLite sino PostgreSQL:

```python
servicio = TechMindInference.desde_artefactos(Path("models"))
servicio.con = conexion_de_auditoria   # opcional; si es None, no audita
```

**Tres advertencias para el equipo de backend:**

1. **No cargar los modelos por petición.** `SentenceTransformer(...)` tarda segundos y ocupa cientos
   de megabytes. Se carga una vez al arrancar, vía `lru_cache` o el evento `startup` de FastAPI.

2. **No ejecutar `predecir()` directamente dentro de un `async def`.** spaCy y SBERT son síncronos y
   bloquean el event loop, dejando al servidor sin atender otras peticiones. Usar
   `run_in_threadpool`, o declarar la ruta como `def` síncrona y dejar que Starlette lo gestione.

3. **Verificar la compatibilidad de versión al arrancar.** Comparar `metadata.json["version"]` contra
   la versión que el backend espera, y fallar ruidosamente si difieren. Un artefacto de una versión
   incompatible produce predicciones silenciosamente erróneas, que es el peor modo de fallo posible.
''')

code(r'''
# @title 9.5 — Resumen final de la ejecución
print("=" * 78)
print(f"  TECHMIND — PIPELINE DE CIENCIA DE DATOS v{CFG.version}")
print("=" * 78)

print(f"\n  CORPUS")
print(f"    documentos procesados    : {len(df)}")
print(f"    categorías               : {len(CATEGORIAS)}")
print(f"    rechazados en validación : {len(df_rechazados)}")
print(f"    duplicados eliminados    : {REPORTE_DEDUP['duplicados_exactos'] + REPORTE_DEDUP['near_duplicates']}")
print(f"    idiomas                  : {dict(df['idioma'].value_counts())}")
print(f"    hash del dataset         : {METADATOS['dataset']['hash_sha256'][:32]}...")

print(f"\n  MODELO")
print(f"    clasificador             : {METADATOS['modelo']['tipo_clasificador']}")
print(f"    embeddings               : {CFG.embeddings.modelo}")
print(f"    accuracy   (test)        : {METADATOS['metricas']['conjunto_prueba']['accuracy']:.4f}")
print(f"    precision  (macro)       : {METADATOS['metricas']['conjunto_prueba']['precision_macro']:.4f}")
print(f"    recall     (macro)       : {METADATOS['metricas']['conjunto_prueba']['recall_macro']:.4f}")
print(f"    f1_macro   (test)        : {METADATOS['metricas']['conjunto_prueba']['f1_macro']:.4f}")
_cv = METADATOS["metricas"]["validacion_cruzada"]
if _cv["media"] is not None:
    print(f"    f1_macro   (CV {_cv['folds_efectivos']}-fold)   : {_cv['media']:.4f} "
          f"± {_cv['desviacion']:.4f}")
else:
    print(f"    f1_macro   (CV)          : no disponible (corpus insuficiente)")
_top2 = METADATOS["metricas"]["conjunto_prueba"].get("top2_accuracy")
if _top2 is not None and not pd.isna(_top2):
    print(f"    top-2 accuracy           : {_top2:.4f}")
else:
    print(f"    top-2 accuracy           : no aplica ({len(CATEGORIAS)} categorías)")

# La cifra que se presenta debe ser la fiable, no la más alta.
if _cv["media"] is not None:
    _brecha = METADATOS["metricas"]["conjunto_prueba"]["f1_macro"] - _cv["media"]
    _f1t = METADATOS["metricas"]["conjunto_prueba"]["f1_macro"]
    _ee = (max(_f1t * (1 - _f1t), 1e-6) / max(len(idx_test), 1)) ** 0.5
    if abs(_brecha) > 2 * max(_cv["desviacion"], _ee):
        print(f"\n    >> CIFRA A PRESENTAR: {_cv['media']:.4f} ± {_cv['desviacion']:.4f} "
              f"(validación cruzada)")
        print(f"       El test se aparta {_brecha:+.4f} y no es representativo con "
              f"{len(idx_test)} documentos.")

print(f"\n  ORGANIZACIÓN")
print(f"    tópicos BERTopic         : {METADATOS['modelo']['n_topicos_bertopic']}")
print(f"    clusters KMeans          : {K_OPTIMO}")
print(f"    documentos indexados     : {coleccion.count()}")

print(f"\n  INGENIERÍA")
print(f"    huella de configuración  : {CFG.huella()[:32]}...")
print(f"    caché de embeddings      : {CACHE.estadisticas()['vectores_en_cache']} vectores "
      f"(acierto {CACHE.estadisticas()['tasa_acierto']:.0%})")
print(f"    operaciones cronometradas: {len(TIEMPOS)}")
print(f"    pruebas superadas        : {len(PRUEBAS) - fallos}/{len(PRUEBAS)}")
print(f"    log de ejecución         : {CFG.rutas.logs / 'pipeline.log'}")

print("\n" + "=" * 78)
print("  Pipeline completado. Artefactos listos para el backend en FastAPI.")
print("=" * 78)

log.info(f"PIPELINE COMPLETADO — v{CFG.version} — "
         f"f1_macro={METADATOS['metricas']['conjunto_prueba']['f1_macro']:.4f}")
''')
