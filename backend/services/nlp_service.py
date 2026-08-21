import os
from functools import lru_cache
from pathlib import Path

from ml.core import TechMindInference
from services.oci_storage import asegurar_artefactos


def _directorio_modelos() -> Path:
    return Path(os.getenv("MODELS_DIR", "models"))


@lru_cache(maxsize=1)
def obtener_servicio() -> TechMindInference:
    """
    Garantiza que los artefactos estén disponibles y carga
    el servicio de inferencia una sola vez.
    """
    directorio = _directorio_modelos()
    asegurar_artefactos(directorio)
    return TechMindInference.desde_artefactos(directorio)


def estado_modelo() -> dict:
    """
    Devuelve información básica sobre la disponibilidad del modelo.
    """
    directorio = _directorio_modelos()

    requeridos = [
        "metadata.json",
        "modelo_clasificacion.joblib",
        "label_encoder.joblib",
        "config.json",
        "centroides_clase.joblib",
        "tecnologias.json",
    ]

    faltantes = [
        nombre
        for nombre in requeridos
        if not (directorio / nombre).is_file()
    ]

    return {
        "cargado": len(faltantes) == 0,
        "directorio": str(directorio),
        "faltantes": faltantes,
    }
