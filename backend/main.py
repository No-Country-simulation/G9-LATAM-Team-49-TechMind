import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.contenido import router as contenido_router
from services.nlp_service import estado_modelo, obtener_servicio


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger("techmind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precarga los modelos al arrancar, no en la primera petición.
    if os.getenv("PRECARGAR_MODELOS", "1").strip().lower() in (
        "1", "true", "yes", "si"
    ):
        try:
            obtener_servicio()
            log.info("Modelos precargados correctamente")
        except Exception as exc:
            log.warning("No se pudieron precargar los modelos: %s", exc)

    yield


app = FastAPI(
    title="TechMind API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(contenido_router, prefix="/api/v1")


@app.get("/health/live")
def health_live():
    """Indica únicamente que el proceso de FastAPI está vivo."""
    return {"status": "ok"}


@app.get("/health")
def health_check():
    """Comprueba que los artefactos necesarios del modelo estén disponibles."""
    modelo = estado_modelo()

    if not modelo["cargado"]:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "modelo": modelo,
            },
        )

    return {
        "status": "ok",
        "modelo": modelo,
    }
