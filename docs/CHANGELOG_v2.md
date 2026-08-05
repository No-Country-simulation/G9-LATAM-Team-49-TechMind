# TechMind — Notebook de Ciencia de Datos · v2.0.0

Registro de cambios frente a la v1.0.0 (57 celdas), punto por punto de la lista de mejoras.

**Notebook:** 115 celdas (57 de código, 58 markdown) · **Generador:** `build_nb.py` + paquete `nbgen/`

---

## Addendum — cambios tras comparar con `TECHMI_1_IPY_REVISADO.ipynb`

La auditoría independiente (versión A) resolvió el mismo checklist con un enfoque quirúrgico. La
comparación reveló **tres huecos reales en esta versión**, todos adoptados, y **un defecto propio**
que la comparación destapó. Detalle completo en `Comparativa_TechMind.pdf`.

| # | Cambio | Origen | Sección |
|---|---|---|---|
| 1 | **Persistencia relacional SQLite** — 6 tablas con índices, poblado idempotente, 4 consultas de demostración | Adoptado de la versión A | §6.6 (nueva) |
| 2 | **Historial de versiones append-only** — tabla `versiones_modelo` + `historial_versiones.jsonl`. `metadata.json` intacto para el backend | Adoptado de la versión A | §6.7 (nueva) |
| 3 | **Auditoría de predicciones** — cada inferencia registra fecha, categoría, probabilidad, latencia y versión; no bloqueante | Adoptado de la versión A | §7.2 |
| 4 | **Separación a favor / en contra** en la explicabilidad local | Adoptado de la versión A | §5.5.2 |
| 5 | **Eliminación de dependencias transitivas de globales** | Hallazgo propio | §5.2.3, §5.5.2, §7.2 |

### Sobre el punto 5

La afirmación original de que `TechMindInference` «no lee ninguna variable global» era cierta
mirando el cuerpo de la clase y **falsa en la práctica**: sus métodos llamaban a
`rankear_keywords()` y `explicar_prediccion()`, que sí leían `nlp`, `kw_model`, `extractor_yake`,
`GANADOR`, `modelo_a`/`modelo_b`, `CENTROIDES` y `CATEGORIAS`. Importada desde
`app/services/nlp_service.py`, la clase habría fallado con `NameError` en la primera petición.

Ambas funciones reciben ahora sus dependencias como parámetros de sólo-palabra clave, con los
globales como valor por defecto: el notebook las sigue llamando igual, la clase pasa lo que recibió
en su constructor. La prueba `test_sin_globales` (§7.5) instancia el servicio y predice con esos
nombres borrados del ámbito — si alguno se colara de nuevo, falla.

El defecto se encontró porque la auditoría documentaba con honestidad la misma limitación en su
propio wrapper de inferencia. Aplicar ese criterio a esta versión fue lo que lo destapó.

### Verificación adicional

Ejecución real de las celdas de SQLite con datos sintéticos: 7 comprobaciones superadas — creación
de las 6 tablas, poblado correcto, **idempotencia** (reejecutar no duplica), historial acumulativo
(3 corridas → 3 filas + 3 líneas JSONL), auditoría con marca de confianza baja, y consultas
relacionales operativas.

---

## 1. Validación de entrada · **implementado**

Antes existía un único control (`len(texto) >= 20`) embebido en la función de inferencia. Ahora hay
un validador dedicado en §2.1 con **siete controles** y códigos de error tipificados.

| Requisito | Implementación | Código de error |
|---|---|---|
| Título obligatorio | Control 1, con verificación de tipo | `campo_faltante`, `tipo_invalido` |
| Texto obligatorio | Control 1, con verificación de tipo | `campo_faltante`, `tipo_invalido` |
| Longitud mínima y máxima de ambos | Controles 4 y 5, umbrales en `CFG.validacion` | `longitud_insuficiente`, `longitud_excesiva` |
| Codificación UTF-8 | Control 2, round-trip `encode/decode` — detecta surrogates sin emparejar | `codificacion_invalida` |
| Documentos vacíos | Control 1, tras normalizar espacios | `campo_vacio` |
| Caracteres inválidos o corruptos | Control 3: mojibake, U+FFFD, caracteres de control | `texto_corrupto` |
| Mensajes de error claros | Cada error lleva código + mensaje explicativo con el valor concreto | — |

**Dos decisiones de diseño relevantes:**

- `validar_entrada()` **no lanza excepciones**: devuelve un `ResultadoValidacion` con el diagnóstico
  completo. Separar *detección* de *reacción* permite que el notebook filtre en silencio y que el
  backend traduzca a HTTP 422. `exigir_valido()` es la variante que sí lanza.
- **Dos modos de severidad**: `inferencia` (mínimo 20 caracteres) y `corpus` (mínimo 250). Un
  documento pobre de entrenamiento daña todo el modelo; uno pobre de inferencia solo su predicción.

Se añadieron dos controles no pedidos: **ratio de caracteres no alfabéticos** (detecta tablas,
volcados de log y binarios) y **ratio de mayúsculas** (advertencia no bloqueante: degrada el POS
tagger). Celda §2.1b ejecuta una batería de 14 casos como evidencia.

## 2. Pipeline de preprocesamiento · **completado**

Estaba al ~85 %. Lo que faltaba:

| Etapa | Estado previo | Ahora |
|---|---|---|
| Limpieza de texto | ✅ | Sin cambios |
| Normalización Unicode | ✅ NFKC | Documentado el porqué del orden respecto al filtrado de ruido |
| Conversión a minúsculas | ⚠️ implícita | **Explícita y justificada**: se aplica en el lema (§4.2), no antes de spaCy, para no privar al POS tagger y al NER de la señal de capitalización |
| Eliminación de caracteres especiales | ✅ | Documentado qué se preserva (`C#`, `C++`, `Node.js`, `CI/CD`) y por qué |
| Espacios redundantes | ✅ | Sin cambios |
| Tokenización / Lematización / Stopwords | ✅ | Sin cambios funcionales |
| **Eliminación de duplicados** | ⚠️ solo exactos | **Exactos por SHA-256 + near-duplicates por Jaccard sobre shingles** |
| **Manejo de valores nulos** | ⚠️ parcial | `limpiar_texto` acepta `None`, `NaN` y no-cadenas sin fallar |

**Calibración empírica del near-duplicate.** La configuración inicial que escribí (ventana de 5
palabras, umbral 0.92) resultó no atrapar nada que el hash exacto no capturara ya. Lo medí sobre un
documento real de 35 palabras:

| ventana `n` | 1 palabra cambiada | 3 palabras | 8 palabras | solapamiento del 50 % |
|---|---|---|---|---|
| **3** | 0.83 | **0.61** | 0.27 | **0.36** |
| 5 | 0.72 | 0.51 | 0.15 | 0.33 |

El umbral debe quedar por debajo de "pocas palabras cambiadas" y por encima de "comparten un
pasaje". Con `n=3` esas regiones se separan limpiamente (0.61 vs 0.36) → **umbral 0.60**. La tabla y
el razonamiento están en el notebook, §2.4.

## 3. Detección de idioma · **implementado desde cero**

- **Nivel 1:** `langdetect` con umbral de confianza configurable. `DetectorFactory.seed = 0` para
  volverlo determinista, requisito de reproducibilidad.
- **Nivel 2:** heurística de ratio de stopwords, sin dependencias, si `langdetect` falta o falla.
- **Arquitectura multilenguaje:** clase `RegistroIdiomas` con carga perezosa del pipeline de spaCy
  por idioma. Activar inglés requiere cambiar `idiomas_soportados` a `("es", "en")` y descargar
  `en_core_web_sm` — **no** requiere cambios de código.
- El modelo de embeddings ya es multilingüe, así que español e inglés comparten espacio vectorial y
  son directamente comparables sin traducción.

Lo único que falta para el inglés son datos etiquetados, no arquitectura. Queda explícito en §9.2.

## 4. Organización del notebook · **completado**

- **Introducción** en prosa: el problema, qué hace el sistema, qué entrega el notebook.
- **Objetivos**: 10 objetivos con criterio de cumplimiento y sección donde se verifica.
- **Alcance y no-alcance** explícitos.
- **Explicación antes de cada sección**: se añadieron 12 bloques markdown que faltaban (§4.1–§4.4,
  §6.1–§6.4, §7.3–§7.6). Cada uno explica *qué problema resuelve* antes del código.
- **Conclusiones** en §9: entregables, limitaciones reconocidas y mejoras futuras.
- Numeración de celdas alineada con los encabezados markdown.

## 5. Diagramas Mermaid · **7 diagramas** (se pedían 6)

| # | Diagrama | Sección |
|---|---|---|
| 1 | Flujo completo del pipeline (4 etapas) | Portada |
| 2 | Flujo de ingesta y validación | §2 |
| 3 | Arquitectura NLP — qué modelo interviene dónde | §3 |
| 4 | Flujo de embeddings con caché | §5 |
| 5 | Flujo de entrenamiento | §5.3 |
| 6 | Flujo de búsqueda semántica y recomendación | §6 |
| 7 | Flujo de inferencia (`sequenceDiagram` de `POST /contenido`) | §7 |

Colab no renderiza Mermaid nativamente; §0.8 incluye una utilidad opcional vía mermaid.ink, con
degradación a código fuente si no hay red.

## 6. Justificación de modelos · **implementado**

Sección §3 nueva, con las cinco preguntas para spaCy, Sentence-Transformers, KeyBERT, BERTopic y
ChromaDB: qué problema resuelve, por qué se eligió, ventajas, alternativas y por qué se descartaron.
Incluye tabla de decisión consolidada y justificación de la Regresión Logística (produce
probabilidades calibrables; SVM da distancias, no probabilidades).

## 7. Evaluación de modelos · **completado**

Estaban accuracy, F1, matriz de confusión, CV y classification report. Se añadieron **precision y
recall explícitos** (macro y weighted) en la tabla comparativa, más:

- Explicación de cuándo engaña cada métrica.
- Justificación de F1-macro como métrica de decisión.
- Lectura dirigida de los 5 pares de categorías más confundidos.
- Alerta automática si la desviación entre folds supera 0.10 (resultado no estable).

## 8. Versionado del modelo · **completado**

`metadata.json` pasa de 12 campos a seis bloques estructurados:

| Bloque | Contenido |
|---|---|
| Versión | Versión semántica + **huella SHA-256 de la configuración** |
| Fecha | Timestamp ISO 8601 |
| Parámetros | Hiperparámetros del clasificador + configuración completa serializada |
| Métricas | Test, CV por fold, por categoría y de clustering |
| Modelo | Representación ganadora, embeddings, spaCy, idiomas, categorías |
| Dataset | **Hash SHA-256 del corpus**, tamaño, distribución, idiomas, reporte de dedup |

El hash del dataset es lo que permite responder con certeza *"¿este modelo se entrenó con estos
datos?"*.

## 9. Configuración centralizada · **implementado**

~25 valores hardcodeados → `dataclass Config` con 11 bloques (`ConfigValidacion`, `ConfigIdioma`,
`ConfigNLP`, `ConfigEmbeddings`, `ConfigKeywords`, `ConfigTFIDF`, `ConfigClasificacion`,
`ConfigClustering`, `ConfigVectorial`, `ConfigCorpus`, `ConfigRutas`).

Tres propiedades: **serializable** (`config.json` acompaña a los modelos), **tipada** (falla en la
celda de configuración, no a mitad del entrenamiento) y **hasheable** (`CFG.huella()` invalida la
caché de embeddings al cambiar un parámetro relevante).

## 10. Logging · **implementado**

Logger `techmind` con doble salida: consola (compacta) y `logs/pipeline.log` (con módulo y línea).
Dos primitivas de instrumentación: decorador `@cronometrar` y context manager `etapa()`.

Cubre los ocho eventos pedidos: inicio de entrenamiento, inicio de procesamiento, tiempo de
ejecución, errores, advertencias, generación de embeddings, clasificación y clustering.

## 11. Reproducibilidad · **completado**

De 1 fuente a **5**: `random`, `numpy`, `torch` (+ cuDNN determinista), `PYTHONHASHSEED` y
`langdetect`. Documentado que `PYTHONHASHSEED` solo surte efecto en subprocesos nuevos, y que
scikit-learn no tiene semilla global (se propaga `CFG.random_state` a cada estimador).

## 12. Entity Recognition · **ya existía, ampliado**

El `EntityRuler` estaba correctamente implementado. Se amplió de ~90 a ~140 tecnologías agrupadas por
dominio, se parametrizó desde `CFG`, y se añadió la explicación de por qué se inserta `before="ner"`
(evita que "Java" se etiquete como isla en vez de tecnología).

## 13. Caché de embeddings · **implementado desde cero**

Clase `CacheEmbeddings` con persistencia en disco. Clave = SHA-256 de (modelo ‖ normalización ‖
texto), de modo que cambiar de modelo invalida las entradas afectadas automáticamente. Expone
`estadisticas()` con tasa de acierto. Convierte el ciclo de desarrollo de minutos a segundos.

## 14. Persistencia · **completado**

| Producto | Antes | Ahora |
|---|---|---|
| Documentos procesados | ✅ | Sin cambios |
| Embeddings | ✅ | Sin cambios |
| **Keywords** | ❌ | Columna en `corpus_final.csv` + **índice invertido** `indice_keywords.json` |
| **Categorías y resultados de clasificación** | ❌ | `resultados_clasificacion.csv` con predicción, probabilidad, partición, acierto y flag de confianza baja |
| Caché de embeddings | ❌ | `cache/embeddings_cache.joblib` |
| Tiempos de ejecución | ❌ | `logs/tiempos_ejecucion.csv` |

## 15. Optimización · **implementado**

| Optimización | Qué ahorra |
|---|---|
| Caché de embeddings | El cómputo del encoder completo en reejecuciones |
| **Reutilización del `Doc` de spaCy** | Una pasada completa de spaCy por petición — el prototipo llamaba `nlp()` dos veces |
| `nlp.pipe` por lotes | Overhead de construcción del `Doc` |
| Un solo `encode()` por documento | Tres codificaciones redundantes |

§8.2 genera un reporte de tiempos por operación para identificar cuellos de botella.

## 16. Explicabilidad · **completado**

Antes: solo coeficientes globales del modelo TF-IDF — que **puede no ser el modelo ganador**, en cuyo
caso no explicaba nada.

Ahora, dos niveles:

- **Global**: coeficientes por categoría (diagnóstico del modelo, detecta aprendizaje espurio).
- **Local**: **ablación por término**, agnóstica al modelo. Elimina cada keyword, recalcula la
  probabilidad de la categoría predicha y reporta la caída. Cuesta `n+1` predicciones en lugar de
  las `2ⁿ` de SHAP.
- **Señal complementaria**: similitud contra los centroides de cada categoría. Cuando ambas señales
  discrepan, la discrepancia es en sí misma un diagnóstico.

La explicación se expone en el campo `explicacion` de la respuesta JSON.

## 17. API Ready · **implementado**

`procesar_contenido()` dependía de ~10 variables globales del notebook — funcionaba en el kernel y
fallaba al importarla desde `nlp_service.py`.

Ahora `TechMindInference` **sin estado global**, con las cuatro responsabilidades separadas:

| Responsabilidad | Dónde |
|---|---|
| Entrenamiento | §5 del notebook, offline |
| Carga del modelo | `TechMindInference.desde_artefactos()`, una vez al arrancar |
| Predicción | `.predecir()`, por petición |
| Contrato de salida | `RespuestaContenido` |

Incluye `.predecir_lote()` (no aborta ante filas inválidas) y `.salud()` para un `GET /health`.
§9.4 trae el código de integración con FastAPI y tres advertencias operativas (no cargar modelos por
petición; usar `run_in_threadpool`; verificar compatibilidad de versión al arrancar).

## 18. Documentación · **completado**

Todas las funciones con docstring en formato Google: objetivo, `Args`, `Returns`, `Raises` cuando
aplica, y `Example` ejecutable.

## 19. Buenas prácticas · **completado**

- **Modularización**: `build_nb.py` monolítico de 1550 líneas → paquete `nbgen/` con `core.py` + 6
  módulos de sección. Editar una sección ya no implica tocar el resto.
- **Sin duplicación**: la lógica de RRF, validación y limpieza existe en un solo lugar.
- **Nombres consistentes**: español en todo el dominio, `snake_case`, prefijos por sección.
- **Type hints** en todas las firmas.
- **PEP 8**: líneas ≤ 100, dos líneas entre definiciones de nivel superior.

## 20. Mejoras futuras · **implementado**

§9.3, organizadas en cuatro horizontes por relación valor/esfuerzo. Cubre los ocho puntos pedidos:
OCR (con OCI Vision como opción que además refuerza el requisito cloud), procesamiento por lotes,
dashboard, monitoreo con detección de *data drift*, pruebas automatizadas en CI, soporte
multilenguaje, recomendaciones avanzadas y consulta por categorías. Se añadieron: calibración de
probabilidades, fine-tuning de embeddings, migración a `pgvector`, MinHash/LSH, clasificación
multietiqueta y búsqueda híbrida BM25 + vectorial.

---

## Verificación ejecutada

| Comprobación | Resultado |
|---|---|
| JSON del notebook válido (nbformat 4) | ✅ 115 celdas |
| Sintaxis de cada celda de código (`ast.parse`) | ✅ 0 errores en 57 celdas |
| Orden de definición de símbolos | ✅ 0 referencias usadas antes de definirse |
| Bloques Mermaid | ✅ 7 |
| Ejecución real de §0 y §2 en sandbox | ✅ 9 celdas sin error |
| Batería funcional (nulos, NFKC, dedup, 13 casos de validación, UTF-8, idioma, config, logging) | ✅ 11/11 + 2 extras |

**Dependencias del sandbox de verificación:** no incluye spaCy, sentence-transformers, KeyBERT,
BERTopic ni ChromaDB, así que §4–§8 se validaron por análisis estático (sintaxis y resolución de
símbolos), no por ejecución. La ejecución completa requiere Colab o un entorno con el stack
instalado.

## Cómo regenerar el notebook

```bash
python build_nb.py          # regenera techmind_eda_modelado.ipynb desde nbgen/
```

Para editar una sección, se toca su módulo en `nbgen/` y se vuelve a ejecutar. El orden de los
imports en `build_nb.py` **es** el orden de las celdas.
