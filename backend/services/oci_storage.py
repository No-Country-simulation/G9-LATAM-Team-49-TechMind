"""Descarga de los artefactos del modelo desde OCI Object Storage.

Resuelve el problema de arranque del despliegue: los artefactos entrenados
(joblib de varios MB) no se versionan en Git, asi que el contenedor necesita
obtenerlos de algun sitio al arrancar. Ese "algun sitio" es un bucket de
Object Storage, que ademas cubre el requisito obligatorio de integracion
con OCI del brief.

Tres modos, en este orden de preferencia:

1. **PAR (Pre-Authenticated Request)** — `OCI_PAR_URL`. Una URL firmada por
   OCI con caducidad. No necesita SDK, ni credenciales, ni permisos IAM:
   solo `requests`. Es el modo recomendado para un hackathon.

2. **SDK con Instance Principals** — `OCI_NAMESPACE` + `OCI_BUCKET`. La
   instancia de Compute se autentica con su propia identidad; no hay claves
   en disco ni en el repositorio. Requiere un Dynamic Group y una Policy
   (ver docs/OCI_DEPLOYMENT.md). Es el modo correcto en produccion.

3. **SDK con fichero de configuracion** — `~/.oci/config`. Util en local.

Si los artefactos ya estan en disco no se descarga nada, de modo que el
despliegue clasico (entrenar en la VM y montar `./models`) sigue funcionando
sin tocar ninguna variable de entorno.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("techmind.oci")

# Los seis artefactos que `TechMindInference.desde_artefactos()` necesita.
# Debe coincidir con ARTEFACTOS_ESPERADOS de scripts/entrenar.py.
ARTEFACTOS = (
    "metadata.json",
    "modelo_clasificacion.joblib",
    "label_encoder.joblib",
    "config.json",
    "centroides_clase.joblib",
    "tecnologias.json",
)

TIMEOUT = int(os.getenv("OCI_TIMEOUT", "120"))


def _faltantes(directorio: Path) -> list[str]:
    """Devuelve los artefactos que no estan en disco."""
    return [a for a in ARTEFACTOS if not (directorio / a).exists()]


def _descargar_par(base: str, nombre: str, destino: Path) -> None:
    """Descarga un objeto a traves de una Pre-Authenticated Request."""
    import requests

    url = f"{base.rstrip('/')}/{nombre}"
    resp = requests.get(url, timeout=TIMEOUT, stream=True)
    resp.raise_for_status()
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    with tmp.open("wb") as fh:
        for bloque in resp.iter_content(chunk_size=1 << 20):
            fh.write(bloque)
    tmp.replace(destino)  # escritura atomica: nunca un artefacto a medias


def _cliente_sdk():
    """Cliente de Object Storage: Instance Principals y, si no, ~/.oci/config."""
    import oci

    try:
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        log.info("OCI: autenticando con Instance Principals")
        return oci.object_storage.ObjectStorageClient(config={}, signer=signer)
    except Exception as exc:
        log.info(f"OCI: Instance Principals no disponible ({type(exc).__name__}); "
                 f"probando ~/.oci/config")
        config = oci.config.from_file(
            file_location=os.getenv("OCI_CONFIG_FILE", "~/.oci/config"),
            profile_name=os.getenv("OCI_CONFIG_PROFILE", "DEFAULT"),
        )
        return oci.object_storage.ObjectStorageClient(config)


def _descargar_sdk(cliente, namespace: str, bucket: str, prefijo: str,
                   nombre: str, destino: Path) -> None:
    """Descarga un objeto usando el SDK de OCI."""
    objeto = f"{prefijo}{nombre}" if prefijo else nombre
    resp = cliente.get_object(namespace, bucket, objeto)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    with tmp.open("wb") as fh:
        for bloque in resp.data.raw.stream(1 << 20, decode_content=False):
            fh.write(bloque)
    tmp.replace(destino)


def asegurar_artefactos(directorio: Path) -> Path:
    """Garantiza que los artefactos del modelo estan en `directorio`.

    No hace nada si ya estan. Si faltan, los descarga de Object Storage segun
    la configuracion disponible.

    Args:
        directorio: Carpeta local de artefactos (normalmente `models/`).

    Returns:
        El mismo `directorio`, ya poblado.

    Raises:
        FileNotFoundError: Si faltan artefactos y no hay forma de obtenerlos.
            El mensaje explica exactamente que hacer.
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    faltan = _faltantes(directorio)
    if not faltan:
        log.info(f"Artefactos presentes en {directorio.resolve()}")
        return directorio

    par = os.getenv("OCI_PAR_URL", "").strip()
    namespace = os.getenv("OCI_NAMESPACE", "").strip()
    bucket = os.getenv("OCI_BUCKET", "").strip()
    prefijo = os.getenv("OCI_PREFIX", "").strip()

    if par:
        log.info(f"Descargando {len(faltan)} artefacto(s) via PAR")
        for nombre in faltan:
            log.info(f"  -> {nombre}")
            _descargar_par(par, nombre, directorio / nombre)

    elif namespace and bucket:
        log.info(f"Descargando {len(faltan)} artefacto(s) de "
                 f"oci://{bucket}@{namespace}/{prefijo}")
        cliente = _cliente_sdk()
        for nombre in faltan:
            log.info(f"  -> {nombre}")
            _descargar_sdk(cliente, namespace, bucket, prefijo,
                           nombre, directorio / nombre)

    else:
        raise FileNotFoundError(
            f"Faltan los artefactos del modelo en {directorio.resolve()}: "
            f"{', '.join(faltan)}. Opciones: (a) entrenar en local con "
            f"'python scripts/entrenar.py --offline' y montar la carpeta "
            f"models/; (b) definir OCI_PAR_URL con una Pre-Authenticated "
            f"Request del bucket; (c) definir OCI_NAMESPACE y OCI_BUCKET y "
            f"dar permisos de lectura a la instancia via Dynamic Group."
        )

    restantes = _faltantes(directorio)
    if restantes:
        raise FileNotFoundError(
            f"La descarga termino pero siguen faltando: {', '.join(restantes)}. "
            f"Revisa que OCI_PREFIX ('{prefijo}') apunte a la carpeta correcta "
            f"del bucket y que los seis objetos esten subidos."
        )

    log.info("Artefactos descargados correctamente")
    return directorio
