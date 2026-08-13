"""Comprueba el estado de las correcciones de TechMind.

Uso:  python scripts/verificar.py
Ejecutar desde la raiz del proyecto, con el entorno virtual activo.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DIRS = ["datasets", "models", "logs", "cache", "chroma_db"]
SIMBOLOS = ["limpiar_texto", "componer_entrada", "preprocesar", "rankear_keywords",
            "explicar_prediccion", "RespuestaContenido", "CacheEmbeddings",
            "TechMindInference", "validar_entrada", "detectar_idioma", "CFG"]

ok_total = True


def titulo(t):
    print(f"\n{'=' * 58}\n  {t}\n{'=' * 58}")


titulo("1. El import no debe crear directorios ni imprimir nada")
for d in DIRS:
    shutil.rmtree(d, ignore_errors=True)
try:
    import app.ml.core as C
except Exception as exc:
    print(f"  FALLO al importar: {type(exc).__name__}: {exc}")
    sys.exit(1)

creados = [d for d in DIRS if Path(d).exists()]
if creados:
    print(f"  FALLO: se crearon directorios -> {creados}")
    ok_total = False
else:
    print("  OK: ningun directorio creado")

titulo("2. Todos los simbolos deben resolver")
for n in SIMBOLOS:
    existe = hasattr(C, n)
    print(f"  [{'OK   ' if existe else 'FALTA'}] {n}")
    if not existe:
        ok_total = False

titulo("3. RespuestaContenido debe tener doc_id y tiempo_ms")
campos = getattr(C.RespuestaContenido, "__dataclass_fields__", {})
for n in ("doc_id", "tiempo_ms"):
    existe = n in campos
    print(f"  [{'OK   ' if existe else 'FALTA'}] {n}")
    if not existe:
        ok_total = False

titulo("4. predecir() debe aceptar n_keywords e id_externo")
import inspect
params = inspect.signature(C.TechMindInference.predecir).parameters
for n in ("n_keywords", "id_externo"):
    existe = n in params
    print(f"  [{'OK   ' if existe else 'FALTA'}] {n}")
    if not existe:
        ok_total = False

titulo("5. 'procesar' NO debe existir (el metodo real es 'predecir')")
if hasattr(C.TechMindInference, "procesar"):
    print("  FALLO: existe un metodo 'procesar' inesperado")
    ok_total = False
else:
    print("  OK: solo existe 'predecir'")

print(f"\n{'=' * 58}")
print("  RESULTADO:", "TODO OK" if ok_total else "HAY FALLOS — revisa arriba")
print(f"{'=' * 58}\n")
sys.exit(0 if ok_total else 1)
