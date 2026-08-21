"""Empaqueta el modelo entrenado para subirlo a OCI Object Storage.

Por qué existe
--------------
`scripts/entrenar.py` deja los artefactos repartidos en dos carpetas:

    models/        metadata.json, *.joblib, config.json, tecnologias.json
                   y modelo_bertopic/  <- un DIRECTORIO, no un fichero
    chroma_db/     la base vectorial con el corpus indexado

Subir "los seis artefactos" a mano deja fuera `modelo_bertopic/` y
`chroma_db/`, y la API arranca sin errores pero devuelve `tema` y
`relacionados` siempre vacíos. Este script recoge el paquete completo y
genera un `manifest.json` que el contenedor usa para reconstruirlo tal cual.

Uso
---
    python scripts/preparar_paquete_modelo.py
    python scripts/preparar_paquete_modelo.py --sin-chroma
    python scripts/preparar_paquete_modelo.py --salida dist_modelo

Luego, con la OCI CLI configurada:

    oci os object bulk-upload \
      --bucket-name techmind-models \
      --src-dir ./dist_modelo \
      --object-prefix paquete/v2.0.0/ \
      --content-type auto --overwrite
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Lo que `TechMindInference.desde_artefactos()` exige para arrancar.
MINIMOS = (
    "models/metadata.json",
    "models/modelo_clasificacion.joblib",
    "models/label_encoder.joblib",
    "models/config.json",
    "models/centroides_clase.joblib",
    "models/tecnologias.json",
)

# Sin esto la API funciona, pero devuelve campos vacíos.
OPCIONALES = {
    "models/modelo_bertopic": "el campo 'tema' vendrá vacío",
    "chroma_db": "el campo 'relacionados' vendrá vacío",
}

# No tiene sentido subirlos: son subproductos del entrenamiento.
EXCLUIR = {".gitkeep", "historial_versiones.jsonl"}
EXCLUIR_SUFIJOS = {".parcial", ".log", ".pyc"}


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def recolectar(origen: Path, prefijo: str) -> list[Path]:
    """Ficheros bajo `origen`, en rutas relativas a la raíz del proyecto."""
    if not origen.exists():
        return []
    return sorted(
        p for p in origen.rglob("*")
        if p.is_file()
        and p.name not in EXCLUIR
        and p.suffix not in EXCLUIR_SUFIJOS
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default="dist_modelo",
                    help="Carpeta destino del paquete (por defecto: dist_modelo)")
    ap.add_argument("--sin-chroma", action="store_true",
                    help="No incluir chroma_db (las recomendaciones quedarán vacías)")
    ap.add_argument("--version", default=None,
                    help="Versión a registrar. Por defecto, la de metadata.json")
    args = ap.parse_args()

    salida = RAIZ / args.salida
    print("=" * 62)
    print("  TechMind — empaquetado del modelo para Object Storage")
    print("=" * 62)

    # --- Comprobar los mínimos ---
    faltan = [r for r in MINIMOS if not (RAIZ / r).exists()]
    if faltan:
        print("\n  FALTAN artefactos imprescindibles:")
        for r in faltan:
            print(f"    - {r}")
        print("\n  Entrena primero:  python scripts/entrenar.py --offline")
        return 1
    print(f"\n  OK: los {len(MINIMOS)} artefactos mínimos están presentes")

    # --- Avisar de los opcionales ---
    for ruta, consecuencia in OPCIONALES.items():
        if ruta == "chroma_db" and args.sin_chroma:
            print(f"  OMITIDO por --sin-chroma: {ruta} → {consecuencia}")
            continue
        if not (RAIZ / ruta).exists():
            print(f"  AVISO: falta {ruta} → {consecuencia}")

    # --- Recolectar ---
    carpetas = ["models"] + ([] if args.sin_chroma else ["chroma_db"])
    ficheros: list[Path] = []
    for c in carpetas:
        ficheros.extend(recolectar(RAIZ / c, c))

    if not ficheros:
        print("\n  No hay nada que empaquetar.")
        return 1

    # --- Copiar y construir el manifiesto ---
    if salida.exists():
        shutil.rmtree(salida)
    salida.mkdir(parents=True)

    version = args.version
    if version is None:
        meta = json.loads((RAIZ / "models" / "metadata.json").read_text(encoding="utf-8"))
        version = meta.get("version", "0.0.0")

    entradas = []
    total = 0
    for origen in ficheros:
        rel = origen.relative_to(RAIZ).as_posix()
        destino = salida / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen, destino)
        tam = origen.stat().st_size
        total += tam
        entradas.append({"ruta": rel, "bytes": tam, "sha256": sha256(origen)})

    manifiesto = {
        "version": version,
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_archivos": len(entradas),
        "bytes_totales": total,
        "archivos": entradas,
    }
    (salida / "manifest.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Resumen ---
    print(f"\n  PAQUETE LISTO en {salida}")
    print("  " + "-" * 58)
    por_carpeta: dict[str, list[int]] = {}
    for e in entradas:
        c = e["ruta"].split("/")[0]
        por_carpeta.setdefault(c, []).append(e["bytes"])
    for c, tams in sorted(por_carpeta.items()):
        print(f"    {c:<16} {len(tams):>4} ficheros   {sum(tams)/1024:>9,.1f} KB")
    print("  " + "-" * 58)
    print(f"    {'TOTAL':<16} {len(entradas):>4} ficheros   {total/1024:>9,.1f} KB")
    print(f"\n  versión: {version}")

    print("\n  Siguiente paso — subir a Object Storage:\n")
    print(f"    oci os object bulk-upload \\")
    print(f"      --bucket-name techmind-models \\")
    print(f"      --src-dir ./{args.salida} \\")
    print(f"      --object-prefix paquete/v{version}/ \\")
    print(f"      --content-type auto --overwrite\n")
    print("  Y en el .env de la VM, la PAR debe terminar en:")
    print(f"    .../o/paquete/v{version}/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
