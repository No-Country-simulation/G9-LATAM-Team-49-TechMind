import logging
import os
from pathlib import Path
from urllib.parse import quote

import requests


log = logging.getLogger(__name__)


# Artefactos necesarios para una inferencia completa.
ARTEFACTOS_OBLIGATORIOS = [
    "metadata.json",
    "modelo_clasificacion.joblib",
    "label_encoder.joblib",
    "config.json",
    "centroides_clase.joblib",
    "tecnologias.json",
]

# Artefactos adicionales presentes en el modelo actual.
ARTEFACTOS_OPCIONALES = [
    "modelo_kmeans.joblib",
    "vectorizador_tfidf.joblib",
    "historial_versiones.jsonl",
    "modelo_bertopic/config.json",
    "modelo_bertopic/ctfidf_config.json",
    "modelo_bertopic/ctfidf.safetensors",
    "modelo_bertopic/topic_embeddings.safetensors",
    "modelo_bertopic/topics.json",
]


def _faltantes(directorio: Path, artefactos: list[str]) -> list[str]:
    return [
        nombre
        for nombre in artefactos
        if not (directorio / nombre).is_file()
    ]


def _url_objeto(base_url: str, nombre: str) -> str:
    return f"{base_url.rstrip('/')}/{quote(nombre, safe='/')}"


def _descargar(base_url: str, nombre: str, destino: Path) -> None:
    url = _url_objeto(base_url, nombre)
    destino.parent.mkdir(parents=True, exist_ok=True)

    temporal = destino.with_name(destino.name + ".part")

    log.info("Descargando artefacto %s desde OCI Object Storage", nombre)

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(10, 120),
        ) as respuesta:
            respuesta.raise_for_status()

            with temporal.open("wb") as archivo:
                for bloque in respuesta.iter_content(chunk_size=1024 * 1024):
                    if bloque:
                        archivo.write(bloque)

        temporal.replace(destino)

    except Exception:
        temporal.unlink(missing_ok=True)
        raise


def asegurar_artefactos(directorio: Path) -> None:
    """
    Garantiza que los artefactos necesarios estén disponibles.

    Si ya existen localmente, no realiza ninguna descarga.

    Si faltan, intenta descargarlos desde una Pre-Authenticated
    Request de OCI Object Storage definida mediante OCI_PAR_URL.
    """

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    faltantes = _faltantes(directorio, ARTEFACTOS_OBLIGATORIOS)

    if not faltantes:
        log.info("Los artefactos del modelo ya existen en %s", directorio)
        return

    par_url = os.getenv("OCI_PAR_URL", "").strip()

    if not par_url:
        raise FileNotFoundError(
            "Faltan artefactos del modelo: "
            + ", ".join(faltantes)
            + ". OCI_PAR_URL no está configurada."
        )

    for nombre in ARTEFACTOS_OBLIGATORIOS:
        destino = directorio / nombre
        if destino.exists():
            continue

        try:
            _descargar(par_url, nombre, destino)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"No se pudo descargar {nombre} desde OCI Object Storage"
            ) from exc

    # Los opcionales mejoran tópicos, explicabilidad y otras funciones,
    # pero su ausencia no debe impedir que la API arranque.
    for nombre in ARTEFACTOS_OPCIONALES:
        destino = directorio / nombre

        if destino.exists():
            continue

        try:
            _descargar(par_url, nombre, destino)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                log.warning("Artefacto opcional no encontrado en OCI: %s", nombre)
                continue
            raise
        except requests.RequestException as exc:
            log.warning(
                "No se pudo descargar el artefacto opcional %s: %s",
                nombre,
                exc,
            )

    faltantes = _faltantes(directorio, ARTEFACTOS_OBLIGATORIOS)

    if faltantes:
        raise FileNotFoundError(
            "Después de consultar OCI siguen faltando artefactos: "
            + ", ".join(faltantes)
        )

    log.info("Artefactos del modelo recuperados correctamente desde OCI.")
