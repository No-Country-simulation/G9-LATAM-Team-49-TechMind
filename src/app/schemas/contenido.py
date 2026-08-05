from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ContenidoRequest(BaseModel):
    titulo: str = Field(..., description="Título del documento", example="Spring Boot vs FastAPI")
    texto: str = Field(..., description="Cuerpo completo del texto", example="Spring Boot es un framework de Java. FastAPI es para Python.")
    n_keywords: Optional[int] = Field(15, description="Número de keywords a extraer")
    id_externo: Optional[str] = Field(None, description="ID externo opcional para tracking")

class ContenidoResponse(BaseModel):
    doc_id: str
    categoria: str
    probabilidad: float
    tiempo_ms: float
    keywords: List[Dict[str, Any]]
    entidades_tecnicas: List[str]
    tema: Optional[str] = None
    relacionados: List[Dict[str, Any]] = []
