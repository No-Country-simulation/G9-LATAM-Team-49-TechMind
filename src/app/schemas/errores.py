from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

class CodigoError:
    """Códigos de error de validación. El backend los mapea a respuestas HTTP 422."""
    CAMPO_FALTANTE       = "campo_faltante"
    TIPO_INVALIDO        = "tipo_invalido"
    CAMPO_VACIO          = "campo_vacio"
    CODIFICACION         = "codificacion_invalida"
    TEXTO_CORRUPTO       = "texto_corrupto"
    MUY_CORTO            = "longitud_insuficiente"
    MUY_LARGO            = "longitud_excesiva"
    CONTENIDO_DEGENERADO = "contenido_degenerado"
    IDIOMA_NO_SOPORTADO  = "idioma_no_soportado"


@dataclass
class ResultadoValidacion:
    """Resultado de validar un documento de entrada.

    Attributes:
        valido: True si no se detectó ningún error bloqueante.
        errores: Lista de (codigo, mensaje) que impiden procesar el documento.
        advertencias: Lista de (codigo, mensaje) que no impiden el procesamiento.
        titulo: Título normalizado (espacios colapsados, sin caracteres de control).
        texto: Texto normalizado.
        metricas: Estadísticos calculados durante la validación.

    Example:
        >>> r = validar_entrada("Spring Boot", "Framework de Java para APIs REST." * 3)
        >>> r.valido
        True
    """
    valido: bool = True
    errores: list = field(default_factory=list)
    advertencias: list = field(default_factory=list)
    titulo: str = ""
    texto: str = ""
    metricas: dict = field(default_factory=dict)

    def agregar_error(self, codigo: str, mensaje: str) -> None:
        """Registra un error bloqueante y marca el resultado como inválido."""
        self.errores.append((codigo, mensaje))
        self.valido = False

    def agregar_advertencia(self, codigo: str, mensaje: str) -> None:
        """Registra una advertencia no bloqueante."""
        self.advertencias.append((codigo, mensaje))

    def mensaje(self) -> str:
        """Devuelve todos los errores concatenados en una sola línea legible."""
        return " | ".join(f"[{c}] {m}" for c, m in self.errores) or "sin errores"



@dataclass
class Idioma:
    """Resultado de la detección de idioma de un documento.

    Attributes:
        codigo: Código ISO 639-1 detectado ("es", "en", "und" si no se pudo determinar).
        confianza: Probabilidad asignada por el detector, en [0, 1].
        metodo: "langdetect", "heuristica_stopwords" o "no_disponible".
        soportado: True si el idioma está en `CFG.idioma.idiomas_soportados`.
    """
    codigo: str = "und"
    confianza: float = 0.0
    metodo: str = "no_disponible"
    soportado: bool = False

    def a_dict(self) -> dict:
        """Serializa el resultado para incluirlo en la respuesta JSON de la API."""
        return {"codigo": self.codigo, "confianza": round(self.confianza, 4),
                "metodo": self.metodo, "soportado": self.soportado}


# --- Nivel 2: stopwords por idioma para la heurística de respaldo ---------
_STOPWORDS_REFERENCIA = {
    "es": {"de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
           "un", "para", "con", "no", "una", "su", "al", "es", "lo", "como", "más",
           "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre"},
    "en": {"the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "as", "was",
           "with", "be", "by", "on", "not", "he", "this", "are", "or", "his", "from",
           "at", "which", "but", "have", "an", "they", "you", "were", "their", "one"},
}


def _detectar_por_stopwords(texto: str) -> Idioma:
    """Detecta idioma por proporción de stopwords conocidas (respaldo sin dependencias).

    Args:
        texto: Texto a analizar.

    Returns:
        Un `Idioma` con método "heuristica_stopwords".
    """
    tokens = re.findall(r"[a-záéíóúüñ]+", texto.lower())
    if len(tokens) < 5:
        return Idioma(metodo="heuristica_stopwords")

    ratios = {
        idioma: sum(1 for t in tokens if t in palabras) / len(tokens)
        for idioma, palabras in _STOPWORDS_REFERENCIA.items()
    }
    mejor = max(ratios, key=ratios.get)
    total = sum(ratios.values())
    confianza = ratios[mejor] / total if total > 0 else 0.0
    return Idioma(codigo=mejor if ratios[mejor] > 0.05 else "und",
                  confianza=round(confianza, 4),
                  metodo="heuristica_stopwords",
                  soportado=mejor in CFG.idioma.idiomas_soportados)


@dataclass
class IdiomaDetectado:
    codigo: str = "und"
    confianza: float = 0.0
    metodo: str = "no_disponible"
    soportado: bool = False

    def a_dict(self) -> dict:
        return {"codigo": self.codigo, "confianza": round(self.confianza, 4),
                "metodo": self.metodo, "soportado": self.soportado}
