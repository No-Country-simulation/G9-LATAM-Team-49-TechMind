"""Núcleo del generador de notebooks de TechMind.

Mantiene la lista de celdas y expone las tres primitivas que usan los módulos
de sección: `md()`, `code()` y `build()`.

Cada módulo `p*_*.py` importa `md`/`code` desde aquí y va acumulando celdas en
el orden en que `build_nb.py` los importa. La separación en módulos existe para
que una sección se pueda editar sin tocar el resto del generador.
"""

from __future__ import annotations

import json
from pathlib import Path

CELDAS: list = []


def md(texto: str) -> None:
    """Agrega una celda markdown al notebook en construcción."""
    CELDAS.append(("markdown", texto.strip("\n")))


def code(texto: str) -> None:
    """Agrega una celda de código al notebook en construcción."""
    CELDAS.append(("code", texto.strip("\n")))


def build(ruta="techmind_eda_modelado.ipynb") -> dict:
    """Serializa las celdas acumuladas a un archivo .ipynb (nbformat 4).

    Args:
        ruta: Ruta de salida del notebook.

    Returns:
        Diccionario con el conteo de celdas por tipo.

    Example:
        >>> build("salida.ipynb")
        {'total': 90, 'code': 55, 'markdown': 35}
    """
    nb = {
        "cells": [
            {
                "cell_type": tipo,
                "metadata": {},
                "source": txt.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if tipo == "code" else {}),
            }
            for tipo, txt in CELDAS
        ],
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True,
                "name": "techmind_eda_modelado.ipynb",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    Path(ruta).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    return {
        "total": len(CELDAS),
        "code": sum(1 for t, _ in CELDAS if t == "code"),
        "markdown": sum(1 for t, _ in CELDAS if t == "markdown"),
    }
