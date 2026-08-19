from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ml.core import ErrorValidacion
from schemas.contenido import ContenidoRequest, ContenidoResponse
from services.nlp_service import obtener_servicio

router = APIRouter()


@router.post("/contenido", response_model=ContenidoResponse)
async def procesar_contenido(payload: ContenidoRequest):
    servicio = obtener_servicio()
    try:
        respuesta = await run_in_threadpool(
            servicio.predecir, payload.titulo, payload.texto,
            n_keywords=payload.n_keywords, id_externo=payload.id_externo,
        )
    except ErrorValidacion as exc:
        raise HTTPException(
            status_code=422,
            detail=[{"codigo": c, "mensaje": m} for c, m in exc.resultado.errores],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return respuesta.a_dict()
