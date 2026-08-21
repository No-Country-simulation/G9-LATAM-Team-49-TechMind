from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ml.core import ErrorValidacion
from schemas.contenido import ContenidoRequest, ContenidoResponse
from services.nlp_service import obtener_servicio

router = APIRouter()


@router.post("/contenido", response_model=ContenidoResponse)
async def procesar_contenido(payload: ContenidoRequest):
    # La carga del modelo va DENTRO del try: si los artefactos no estan
    # disponibles esto lanza FileNotFoundError y, sin este bloque, el cliente
    # recibiria un 500 con traza en vez de un error controlado.
    try:
        servicio = obtener_servicio()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=[{"codigo": "modelo_no_disponible", "mensaje": str(exc)}],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=[{"codigo": "modelo_no_cargado",
                     "mensaje": f"{type(exc).__name__}: {exc}"}],
        )

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
