from functools import lru_cache
from pathlib import Path
from ml.core import TechMindInference

@lru_cache(maxsize=1)
def obtener_servicio() -> TechMindInference:
    """
    Carga el servicio de inferencia una sola vez usando los artefactos en la carpeta models/.
    """
    return TechMindInference.desde_artefactos(Path("models"))
