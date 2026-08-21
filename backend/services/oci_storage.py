"""Descarga del paquete del modelo desde OCI Object Storage.

Los artefactos entrenados no se versionan en Git (`.gitignore` excluye
`models/*` y `chroma_db/`), así que un `git clone` en la VM deja ambas
carpetas vacías. Este módulo las repuebla al arrancar el contenedor, y de
paso cubre el requisito obligatorio de integración con OCI.

El paquete NO son seis ficheros sueltos
---------------------------------------
`TechMindInference.desde_artefactos()` carga, además del clasificador:

  · `models/modelo_bertopic/`  → un DIRECTORIO. Sin él, `tema` sale siempre
    como "(sin tema definido)".
  · `chroma_db/`               → la base vectorial. Sin ella, `relacionados`
    sale siempre vacío.

Por eso la descarga se guía por un **manifiesto** (`manifest.json`) que
enumera cada fichero con su ruta relativa, su tamaño y su hash. Lo genera
`scripts/preparar_paquete_modelo.py` en el momento de subir. Así el paquete
puede crecer o cambiar de forma sin tocar este módulo, y los subdirectorios
se reconstruyen tal cual.

Modos de acceso, en orden de preferencia:

1. **PAR** (`OCI_PAR_URL`) — URL firmada por OCI. Sin SDK ni permisos IAM.
2. **SDK con Instance Principals** (`OCI_NAMESPACE` + `OCI_BUCKET`).
3. **SDK con `~/.oci/config`** — para local.

Si el paquete ya está completo en disco no se descarga nada, de modo que
entrenar en la VM y montar las carpetas sigue funcionando sin configurar
ninguna variable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("techmind.oci")

MANIFIESTO = "manifest.json"

# Mínimo imprescindible para que `desde_artefactos()` no lance FileNotFoundError.
# Se usa para decidir si hace falta descargar y como plan B si no hay manifiesto.
ARTEFACTOS_MINIMOS = (
    "models/metadata.json",
    "models/modelo_clasificacion.joblib",
    "models/label_encoder.joblib",
    "models/config.json",
    "models/centroides_clase.joblib",
    "models/tecnologias.json",
)

# Presentes = funcionalidad completa. Ausentes = la API responde igual, pero
# con `tema` y `relacionados` vacíos. Se avisa en el log, no se aborta.
ARTEFACTOS_OPCIONALES = (
    "models/modelo_bertopic",
    "chroma_db",
)

TIMEOUT = int(os.getenv("OCI_TIMEOUT", "120"))


# --------------------------------------------------------------------- #
# Estado local                                                           #
# --------------------------------------------------------------------- #

def _faltan_minimos(raiz: Path) -> list[str]:
    return [r for r in ARTEFACTOS_MINIMOS if not (raiz / r).exists()]


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _hay_que_descargar(destino: Path, entrada: dict) -> bool:
    """True si el fichero local no existe o no coincide con el manifiesto."""
    if not destino.exists():
        return True
    if entrada.get("bytes") is not None and destino.stat().st_size != entrada["bytes"]:
        return True
    return False


def diagnostico(raiz: Path) -> dict:
    """Qué partes del paquete están presentes. Para logs y para /health."""
    raiz = Path(raiz)
    return {
        "minimos_completos": not _faltan_minimos(raiz),
        "faltan": _faltan_minimos(raiz),
        "bertopic": (raiz / "models" / "modelo_bertopic").exists(),
        "chroma": (raiz / "chroma_db").exists(),
    }


# --------------------------------------------------------------------- #
# Transporte                                                             #
# --------------------------------------------------------------------- #

def _descargar_par(base: str, ruta: str, destino: Path) -> None:
    """Descarga un objeto a través de una Pre-Authenticated Request."""
    import requests

    url = f"{base.rstrip('/')}/{ruta}"
    resp = requests.get(url, timeout=TIMEOUT, stream=True)
    resp.raise_for_status()
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    with tmp.open("wb") as fh:
        for bloque in resp.iter_content(chunk_size=1 << 20):
            fh.write(bloque)
    tmp.replace(destino)  # atómico: nunca queda un artefacto a medias


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
                   ruta: str, destino: Path) -> None:
    objeto = f"{prefijo}{ruta}" if prefijo else ruta
    resp = cliente.get_object(namespace, bucket, objeto)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    with tmp.open("wb") as fh:
        for bloque in resp.data.raw.stream(1 << 20, decode_content=False):
            fh.write(bloque)
    tmp.replace(destino)


def _construir_descargador():
    """Devuelve `(fn(ruta, destino), descripcion)` o `(None, motivo)`."""
    par = os.getenv("OCI_PAR_URL", "").strip()
    namespace = os.getenv("OCI_NAMESPACE", "").strip()
    bucket = os.getenv("OCI_BUCKET", "").strip()
    prefijo = os.getenv("OCI_PREFIX", "").strip()

    if par:
        return (lambda ruta, destino: _descargar_par(par, ruta, destino),
                "PAR (Pre-Authenticated Request)")

    if namespace and bucket:
        cliente = _cliente_sdk()
        return (lambda ruta, destino: _descargar_sdk(
                    cliente, namespace, bucket, prefijo, ruta, destino),
                f"SDK sobre oci://{bucket}@{namespace}/{prefijo}")

    return None, ("no hay ni OCI_PAR_URL ni OCI_NAMESPACE/OCI_BUCKET "
                  "definidos en el entorno")


# --------------------------------------------------------------------- #
# Punto de entrada                                                       #
# --------------------------------------------------------------------- #

def asegurar_artefactos(raiz: Path) -> Path:
    """Garantiza que el paquete del modelo está bajo `raiz`.

    `raiz` es la carpeta que CONTIENE `models/` y `chroma_db/` — en el
    contenedor, `/app`. No confundir con la carpeta `models/` en sí.

    Raises:
        FileNotFoundError: Si faltan artefactos mínimos y no se pueden
            obtener. El mensaje explica las tres vías para resolverlo.
    """
    raiz = Path(raiz)
    raiz.mkdir(parents=True, exist_ok=True)

    faltan = _faltan_minimos(raiz)
    if not faltan:
        est = diagnostico(raiz)
        if not est["bertopic"]:
            log.warning("Falta models/modelo_bertopic: el campo 'tema' vendrá vacío.")
        if not est["chroma"]:
            log.warning("Falta chroma_db: el campo 'relacionados' vendrá vacío.")
        log.info(f"Paquete del modelo presente en {raiz.resolve()}")
        return raiz

    descargar, descripcion = _construir_descargador()
    if descargar is None:
        raise FileNotFoundError(
            f"Faltan artefactos del modelo en {raiz.resolve()}: "
            f"{', '.join(faltan)} — y {descripcion}. Opciones: "
            f"(a) entrenar en local con 'python scripts/entrenar.py --offline' "
            f"y montar models/ y chroma_db/; "
            f"(b) definir OCI_PAR_URL con una Pre-Authenticated Request; "
            f"(c) definir OCI_NAMESPACE y OCI_BUCKET y dar permiso de lectura "
            f"a la instancia mediante un Dynamic Group."
        )

    log.info(f"Descargando el paquete del modelo mediante {descripcion}")

    # --- Manifiesto ---
    ruta_manifiesto = raiz / MANIFIESTO
    try:
        descargar(MANIFIESTO, ruta_manifiesto)
        manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
        entradas = manifiesto["archivos"]
        log.info(f"Manifiesto v{manifiesto.get('version', '?')}: "
                 f"{len(entradas)} ficheros")
    except Exception as exc:
        # Compatibilidad con subidas hechas sin manifiesto.
        log.warning(f"No se pudo leer {MANIFIESTO} ({type(exc).__name__}): "
                    f"se descargarán solo los artefactos mínimos. El campo "
                    f"'tema' y las recomendaciones quedarán vacíos. Regenera "
                    f"el paquete con scripts/preparar_paquete_modelo.py.")
        entradas = [{"ruta": r} for r in ARTEFACTOS_MINIMOS]

    # --- Ficheros ---
    descargados = omitidos = 0
    for entrada in entradas:
        ruta = entrada["ruta"]
        destino = raiz / ruta
        if not _hay_que_descargar(destino, entrada):
            omitidos += 1
            continue
        log.info(f"  ↓ {ruta}")
        descargar(ruta, destino)
        descargados += 1

        esperado = entrada.get("sha256")
        if esperado and _sha256(destino) != esperado:
            raise FileNotFoundError(
                f"El hash de {ruta} no coincide con el del manifiesto. "
                f"La descarga está corrupta o el bucket tiene una versión "
                f"distinta de la que se empaquetó."
            )

    log.info(f"Paquete listo: {descargados} descargados, {omitidos} ya presentes")

    restantes = _faltan_minimos(raiz)
    if restantes:
        raise FileNotFoundError(
            f"La descarga terminó pero siguen faltando: {', '.join(restantes)}. "
            f"Revisa que OCI_PREFIX ('{os.getenv('OCI_PREFIX', '')}') apunte a "
            f"la carpeta correcta del bucket y que el paquete se subiera "
            f"completo con scripts/preparar_paquete_modelo.py."
        )

    est = diagnostico(raiz)
    if not est["bertopic"]:
        log.warning("Sin models/modelo_bertopic: el campo 'tema' vendrá vacío.")
    if not est["chroma"]:
        log.warning("Sin chroma_db: el campo 'relacionados' vendrá vacío.")

    return raiz
