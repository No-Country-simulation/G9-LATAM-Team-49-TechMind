import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.contenido import router as contenido_router
from services.nlp_service import estado_modelo, obtener_servicio

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("techmind.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precarga los modelos al arrancar, no en la primera peticion.
    if os.getenv("PRECARGAR_MODELOS", "1").strip().lower() in ("1", "true", "yes", "si"):
        try:
            obtener_servicio()
            log.info("Modelos precargados correctamente")
        except Exception as exc:
            # No abortamos el arranque a proposito: la API sigue en pie para
            # poder diagnosticar por /health, pero /health devolvera 503 para
            # que ningun monitor la de por sana.
            log.error(f"No se pudieron precargar los modelos: "
                      f"{type(exc).__name__}: {exc}")
    yield


app = FastAPI(
    title="TechMind API",
    version="2.0.0",
    description=(
        "Organizacion inteligente de contenido tecnico: clasificacion "
        "tematica, extraccion de palabras clave, deteccion de entidades "
        "tecnicas y modelado de topicos."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if o.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(contenido_router, prefix="/api/v1")


@app.get("/health/live", tags=["salud"])
def liveness():
    """Liveness: el proceso responde. No dice nada sobre el modelo."""
    return {"status": "ok"}


@app.get("/health", tags=["salud"])
def health_check():
    """Readiness: 200 solo si el modelo esta realmente cargado.

    Antes devolvia siempre 200, incluso con los artefactos ausentes, lo que
    hacia que el servicio pareciera sano mientras el endpoint principal
    fallaba. Ahora responde 503 con el motivo exacto.
    """
    estado = estado_modelo()
    if estado["cargado"]:
        return {"status": "ok", "modelo": estado}
    return JSONResponse(
        status_code=503,
        content={"status": "degradado", "modelo": estado},
    )
