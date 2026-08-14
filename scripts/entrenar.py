"""Entrena el modelo y serializa los artefactos en models/, sin depender de Jupyter.

Reemplaza a scripts/extracted_notebook.py, que solo funciona dentro de un kernel
de IPython. Envuelve a run_eda_and_training() de pipeline_notebook.py.

Uso:
    python scripts/entrenar.py --offline     # sin red, usa corpus_fallback.csv
    python scripts/entrenar.py               # con scraping de Wikipedia
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

ARTEFACTOS_ESPERADOS = [
    "metadata.json",
    "modelo_clasificacion.joblib",
    "label_encoder.joblib",
    "config.json",
    "centroides_clase.joblib",
    "tecnologias.json",
]


def comprobar_requisitos(args) -> list:
    """Comprueba las precondiciones antes de gastar minutos de entrenamiento."""
    faltan = []

    semillas = RAIZ / args.semillas
    if not semillas.exists():
        faltan.append(f"No existe el archivo de semillas: {args.semillas}")

    if args.offline:
        fallback = RAIZ / "corpus_fallback.csv"
        if not fallback.exists():
            faltan.append("Falta corpus_fallback.csv. Generalo con: "
                          "python scripts/build_fallback.py")

    try:
        import spacy
        spacy.load("es_core_news_sm")
    except OSError:
        faltan.append("Falta el modelo de spaCy. Instalalo con: "
                      "python -m spacy download es_core_news_sm")
    except ImportError:
        faltan.append("spaCy no esta instalado. Ejecuta: pip install -r requirements.txt")

    return faltan


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offline", action="store_true",
                   help="usa corpus_fallback.csv en lugar de scrapear Wikipedia")
    p.add_argument("--semillas", default="configs/semillas_wikipedia.csv",
                   help="ruta al CSV de semillas (por defecto: configs/semillas_wikipedia.csv)")
    p.add_argument("--solo-comprobar", action="store_true",
                   help="verifica requisitos y sale, sin entrenar")
    args = p.parse_args()

    print("=" * 62)
    print("  TechMind — entrenamiento y serializacion de artefactos")
    print("=" * 62)
    print(f"  modo     : {'OFFLINE (corpus de respaldo)' if args.offline else 'ONLINE (scraping)'}")
    print(f"  semillas : {args.semillas}")
    print(f"  salida   : {RAIZ / 'models'}")

    print("\n--- Comprobando requisitos ---")
    faltan = comprobar_requisitos(args)
    if faltan:
        for f in faltan:
            print(f"  FALTA: {f}")
        return 1
    print("  OK: todos los requisitos presentes")

    if args.solo_comprobar:
        print("\n--solo-comprobar activo: no se entrena.")
        return 0

    import pipeline_notebook as P

    # Config es un dataclass frozen: hay que reemplazarlo, no mutarlo.
    corpus = dataclasses.replace(
        P.CFG.corpus,
        usar_fallback=args.offline,
        archivo_semillas=args.semillas,
    )
    P.CFG = dataclasses.replace(P.CFG, corpus=corpus)
    # `display()` es un builtin de IPython que no existe fuera de Jupyter.
    # El pipeline conserva una llamada activa (§5.3.1, distribución por
    # partición). Se inyecta un equivalente en el espacio de nombres del
    # módulo para no tener que tocar el pipeline.

    P.display = lambda x: print(x.to_string() if hasattr(x, "to_string") else x)

    print("\n--- Entrenando (esto tarda varios minutos) ---")
    t0 = time.perf_counter()
    try:
        P.run_eda_and_training()
    except Exception as exc:
        print(f"\nFALLO EL ENTRENAMIENTO: {type(exc).__name__}: {exc}")
        return 1
    dt = time.perf_counter() - t0

# El pipeline no serializa tecnologias.json y metadata.json tampoco lo trae.
    # Sin el, desde_artefactos() no puede reconstruir el EntityRuler y las
    # entidades tecnicas vendrian siempre vacias. La lista vive en core.py.
    import json
    sys.path.insert(0, str(RAIZ / "src"))
    from app.ml.core import TECNOLOGIAS

    destino = RAIZ / "models"
    destino.mkdir(exist_ok=True)
    (destino / "tecnologias.json").write_text(
        json.dumps(TECNOLOGIAS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  tecnologias.json escrito ({len(TECNOLOGIAS)} patrones)")

    print(f"\n--- Verificando artefactos ({dt/60:.1f} min) ---")
    destino = RAIZ / "models"
    ausentes = [a for a in ARTEFACTOS_ESPERADOS if not (destino / a).exists()]
    for a in ARTEFACTOS_ESPERADOS:
        existe = (destino / a).exists()
        print(f"  [{'OK   ' if existe else 'FALTA'}] {a}")

    if ausentes:
        print(f"\nEntrenamiento terminado pero faltan {len(ausentes)} artefacto(s).")
        return 1

    print("\n" + "=" * 62)
    print("  LISTO. Arranca la API con:")
    print('    $env:PYTHONPATH="src"; uvicorn app.main:app --reload')
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())