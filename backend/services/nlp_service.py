import logging
import os
from functools import lru_cache
from pathlib import Path

from ml.core import TechMindInference
from services.oci_storage import asegurar_artefactos, diagnostico

log = logging.getLogger("techmind.servicio")

# Raíz de la aplicación: la carpeta que CONTIENE models/ y chroma_db/.
# `core.ConfigRutas.base` es Path("."), así que ChromaDB se resuelve contra el
# directorio de trabajo. En el contenedor, WORKDIR=/app y ambos son volúmenes.
RAIZ = Path(os.getenv("APP_ROOT", "."))
DIRECTORIO_MODELOS = Path(os.getenv("MODELS_DIR", str(RAIZ / "models")))


@lru_cache(maxsize=1)
def obtener_servicio() -> TechMindInference:
    """Carga el servicio de inferencia una sola vez.

    Antes de construirlo se asegura de que el paquete del modelo esté en
    disco: si no lo está, lo descarga de OCI Object Storage. El `lru_cache`
    garantiza que los modelos se cargan una única vez por proceso.

    Raises:
        FileNotFoundError: Si los artefactos no están ni se pueden obtener.
            El mensaje indica las tres vías para resolverlo.
    """
    asegurar_artefactos(RAIZ)
    log.info(f"Cargando artefactos desde {DIRECTORIO_MODELOS.resolve()}")
    return TechMindInference.desde_artefactos(DIRECTORIO_MODELOS)


def estado_modelo() -> dict:
    """Estado de carga del modelo, para el endpoint de salud.

    No lanza excepciones: devuelve siempre un diccionario describiendo la
    situación real. Distingue "cargado" de "cargado y completo": la API
    responde igual sin BERTopic o sin ChromaDB, pero con `tema` y
    `relacionados` vacíos, y eso conviene que se vea desde fuera.
    """
    paquete = diagnostico(RAIZ)

    try:
        servicio = obtener_servicio()
    except Exception as exc:
        return {"cargado": False,
                "detalle": f"{type(exc).__name__}: {exc}",
                "paquete": paquete}

    return {
        "cargado": True,
        "categorias": list(getattr(servicio, "categorias", [])),
        "version_artefactos": getattr(servicio, "metadatos", {}).get("version", "desconocida"),
        "topicos": getattr(servicio, "topic_model", None) is not None,
        "recomendaciones": getattr(servicio, "coleccion", None) is not None,
        "paquete": paquete,
    }
