"""Genera techmind_eda_modelado.ipynb a partir de los módulos de nbgen/.

El notebook ya no se escribe a mano: se compone importando, en orden, los
módulos de sección del paquete `nbgen`. Cada import ejecuta las llamadas a
`md()` y `code()` de esa sección, que van acumulando celdas en `nbgen.core.CELDAS`.

Uso::

    python build_nb.py

Para editar una sección concreta, se toca su módulo y se vuelve a ejecutar este
script. El orden de los imports ES el orden de las celdas del notebook.
"""

from nbgen.core import build

# El orden de importación define el orden de las celdas. No reordenar sin
# revisar las dependencias entre símbolos (una celda no puede usar algo que
# se define más abajo).
import nbgen.p0_intro_config      # §0  — portada, objetivos, config, semillas, logging
import nbgen.p1_ingesta           # §1  — contrato de datos
                                  # §2  — validación, idioma, ingesta, limpieza, EDA
import nbgen.p2_justificacion_nlp # §3  — justificación técnica de modelos
                                  # §4  — preprocesamiento NLP
import nbgen.p3_modelado          # §5  — representaciones, keywords, clasificación,
                                  #       evaluación, explicabilidad, clustering, versionado
import nbgen.p4_kb_inferencia     # §6  — base vectorial, búsqueda, persistencia
                                  # §7  — capa de inferencia API-ready
import nbgen.p5_cierre            # §8  — OCI, rendimiento, empaquetado
                                  # §9  — conclusiones y mejoras futuras


if __name__ == "__main__":
    resumen = build("techmind_eda_modelado.ipynb")
    print(f"OK: {resumen['total']} celdas "
          f"({resumen['code']} de código, {resumen['markdown']} markdown)")
