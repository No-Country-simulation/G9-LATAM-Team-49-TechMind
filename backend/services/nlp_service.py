import logging
import os
from functools import lru_cache
from pathlib import Path

from ml.core import TechMindInference
from services.oci_storage import asegurar_artefactos

log = logging.getLogger("techmind.servicio")

# Ruta de los artefactos. En el contenedor es /app/models (WORKDIR + volumen).
# Se puede sobreescribir con MODELS_DIR para tests o despliegues alternativos.
DIRECTORIO_MODELOS = Path(os.getenv("MODELS_DIR", "models"))


@lru_cache(maxsize=1)
def obtener_servicio() -> TechMindInference:
    """Carga el servicio de inferencia una sola vez.

    Antes de construirlo se asegura de que los artefactos esten en disco:
    si no lo estan, los descarga de OCI Object Storage. El `lru_cache`
    garantiza que los modelos se cargan una unica vez por proceso.

    Raises:
        FileNotFoundError: Si los artefactos no estan ni se pueden obtener.
            El mensaje indica exactamente como resolverlo.
    """
    asegurar_artefactos(DIRECTORIO_MODELOS)
    log.info(f"Cargando artefactos desde {DIRECTORIO_MODELOS.resolve()}")
    return TechMindInference.desde_artefactos(DIRECTORIO_MODELOS)


def estado_modelo() -> dict:
    """Estado de carga del modelo, para el endpoint de salud.

    No lanza excepciones: devuelve siempre un diccionario describiendo la
    situacion real, cargado o no.
    """
    try:
        servicio = obtener_servicio()
    except Exception as exc:
        return {"cargado": False, "detalle": f"{type(exc).__name__}: {exc}"}

    return {
        "cargado": True,
        "categorias": list(getattr(servicio, "categorias", [])),
        "version_artefactos": getattr(servicio, "metadatos", {}).get("version", "desconocida"),
    }
