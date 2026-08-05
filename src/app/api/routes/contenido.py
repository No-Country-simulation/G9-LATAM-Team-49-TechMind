from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.schemas.contenido import ContenidoRequest, ContenidoResponse
from app.services.nlp_service import obtener_servicio

router = APIRouter()

@router.post("/contenido", response_model=ContenidoResponse)
async def procesar_contenido(payload: ContenidoRequest):
    servicio = obtener_servicio()
    try:
        # spaCy y SBERT son síncronos y bloqueantes: van al threadpool
        # para no bloquear el event loop.
        respuesta = await run_in_threadpool(
            servicio.predecir, payload.titulo, payload.texto
        )
    except Exception as exc:
        # Aquí capturamos la excepción de validación u otras.
        if type(exc).__name__ == 'ErrorValidacion':
            raise HTTPException(
                status_code=422,
                detail=[{"codigo": c, "mensaje": m} for c, m in exc.resultado.errores],
            )
        raise HTTPException(status_code=500, detail=str(exc))
    
    return respuesta.a_dict() if hasattr(respuesta, 'a_dict') else respuesta
