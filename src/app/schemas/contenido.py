from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContenidoRequest(BaseModel):
    titulo: str = Field(..., min_length=3, max_length=300,
                        json_schema_extra={"example": "Spring Boot vs FastAPI"})
    texto: str = Field(..., min_length=20, max_length=50_000,
                       json_schema_extra={"example": "Spring Boot es un framework de Java."})
    n_keywords: Optional[int] = Field(None, ge=1, le=50)
    id_externo: Optional[str] = Field(None, max_length=64)


class ContenidoResponse(BaseModel):
    """Refleja exactamente RespuestaContenido.a_dict() de app.ml.core."""
    categoria: str
    probabilidad: float
    informacion_adicional: List[Any]

    doc_id: str = ""
    tiempo_ms: float = 0.0
    titulo: str = ""
    idioma: Dict[str, Any] = {}
    tema: Dict[str, Any] = {}
    entidades_tecnicas: List[str] = []
    distribucion_categorias: Dict[str, float] = {}
    metricas_texto: Dict[str, Any] = {}
    explicacion: Dict[str, Any] = {}
    relacionados: List[Dict[str, Any]] = []
    advertencias: List[Dict[str, str]] = []