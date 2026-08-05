from fastapi import FastAPI
from app.api.routes.contenido import router as contenido_router

app = FastAPI(
    title="TechMind API",
    description="API de NLP para clasificación de textos técnicos.",
    version="2.0.0"
)

app.include_router(contenido_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok"}
