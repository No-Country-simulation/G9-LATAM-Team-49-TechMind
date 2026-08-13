import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.contenido import router as contenido_router
from app.services.nlp_service import obtener_servicio


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precarga los modelos al arrancar, no en la primera peticion.
    if os.getenv("PRECARGAR_MODELOS", "1") == "1":
        try:
            obtener_servicio()
        except Exception as exc:
            print(f"AVISO: no se pudieron precargar los modelos ({exc})")
    yield


app = FastAPI(title="TechMind API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(contenido_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "ok"}