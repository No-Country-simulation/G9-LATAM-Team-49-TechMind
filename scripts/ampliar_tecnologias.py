"""Añade patrones de tecnologías al artefacto ya desplegado.

Por que existe
--------------
`TECNOLOGIAS` vive en `backend/ml/core.py`, pero el EntityRuler no lo lee de
ahi en produccion: lo lee de `models/tecnologias.json`, un artefacto que genera
`scripts/entrenar.py`. Cambiar `core.py` solo surte efecto en el siguiente
entrenamiento.

Este script parchea el artefacto directamente, de modo que un simple reinicio
de la API basta para reconocer las nuevas tecnologias. No hace falta reentrenar
ni reconstruir la imagen de Docker.

Uso
---
    python scripts/ampliar_tecnologias.py                # anade la lista por defecto
    python scripts/ampliar_tecnologias.py --listar       # solo muestra lo que hay
    python scripts/ampliar_tecnologias.py --anadir Rust Zig

Ejecutar desde la raiz del proyecto. Despues:

    docker compose restart api

Deja una copia en `models/tecnologias.json.bak` antes de tocar nada.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARTEFACTO = RAIZ / "models" / "tecnologias.json"

# Sistemas operativos e infraestructura de sistema: la lista original cubria
# muy bien lenguajes, frameworks y cloud, pero no tenia ni un solo sistema
# operativo. Un texto sobre servidores o administracion de sistemas salia con
# el panel de entidades tecnicas vacio.
POR_DEFECTO = [
    "Linux", "Ubuntu", "Debian", "CentOS", "Red Hat", "Fedora", "Alpine",
    "Windows", "Windows Server", "macOS", "Unix", "WSL",
    "Bash", "Shell", "PowerShell", "systemd", "cron", "SSH", "Firewall",
    "VirtualBox", "VMware", "Podman", "máquina virtual",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listar", action="store_true",
                    help="Muestra los patrones actuales y sale")
    ap.add_argument("--anadir", nargs="*", default=None,
                    help="Patrones concretos a anadir (por defecto: sistemas operativos)")
    args = ap.parse_args()

    if not ARTEFACTO.exists():
        print(f"  No existe {ARTEFACTO}.")
        print("  Entrena primero:  python scripts/entrenar.py --offline")
        return 1

    actuales = json.loads(ARTEFACTO.read_text(encoding="utf-8"))

    if args.listar:
        print(f"{len(actuales)} patrones en {ARTEFACTO.name}:\n")
        for i, t in enumerate(sorted(actuales), 1):
            print(f"  {i:>3}. {t}")
        return 0

    nuevos = args.anadir if args.anadir else POR_DEFECTO
    faltan = [n for n in nuevos if n not in actuales]

    print("=" * 58)
    print("  Ampliacion del diccionario de tecnologias")
    print("=" * 58)
    print(f"\n  Artefacto : {ARTEFACTO}")
    print(f"  Antes     : {len(actuales)} patrones")

    if not faltan:
        print("\n  Nada que anadir: todos los patrones ya estaban presentes.")
        return 0

    shutil.copy2(ARTEFACTO, ARTEFACTO.with_suffix(".json.bak"))
    print(f"  Copia     : {ARTEFACTO.name}.bak")

    actuales.extend(faltan)
    ARTEFACTO.write_text(
        json.dumps(actuales, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Despues   : {len(actuales)} patrones")
    print(f"\n  Anadidos ({len(faltan)}):")
    for t in faltan:
        print(f"    + {t}")

    print("\n  Para que surta efecto, reinicia la API:")
    print("    docker compose restart api")
    print("\n  La API tarda 2-3 minutos en volver a estar disponible.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
