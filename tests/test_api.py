from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

RESPUESTA = {
    "categoria": "DevOps", "probabilidad": 0.81,
    "informacion_adicional": [{"keyword": "docker", "score": 0.9}],
    "doc_id": "test-123", "tiempo_ms": 10.5, "titulo": "Prueba",
    "idioma": {"codigo": "es"}, "tema": {"id": 1, "etiqueta": "contenedores"},
    "entidades_tecnicas": ["Docker"], "distribucion_categorias": {"DevOps": 0.81},
    "metricas_texto": {"n_tokens": 20}, "explicacion": {},
    "relacionados": [], "advertencias": [],
}


@patch("main.estado_modelo")
def test_health_check(mock_estado):
    mock_estado.return_value = {
        "cargado": True,
        "directorio": "models",
        "faltantes": [],
    }

    r = client.get("/health")
    body = r.json()

    assert r.status_code == 200
    assert body["status"] == "ok"
    assert body["modelo"]["cargado"] is True
    assert body["modelo"]["faltantes"] == []


@patch("api.routes.contenido.obtener_servicio")
def test_procesar_contenido_success(mock_servicio):
    doble = MagicMock()
    doble.predecir.return_value.a_dict.return_value = RESPUESTA
    mock_servicio.return_value = doble

    payload = {"titulo": "Prueba de NLP",
               "texto": "Texto de prueba suficientemente largo para pasar las validaciones."}
    r = client.post("/api/v1/contenido", json=payload)

    assert r.status_code == 200
    assert r.json()["categoria"] == "DevOps"
    assert r.json()["doc_id"] == "test-123"


def test_texto_corto_rechazado_por_pydantic():
    r = client.post("/api/v1/contenido", json={"titulo": "Hola", "texto": "corto"})
    assert r.status_code == 422


def test_cors_expuesto():
    r = client.options("/api/v1/contenido", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
def test_health_live():
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@patch("main.estado_modelo")
def test_health_devuelve_503_si_modelo_no_disponible(mock_estado):
    mock_estado.return_value = {
        "cargado": False,
        "directorio": "models",
        "faltantes": ["metadata.json"],
    }

    r = client.get("/health")
    body = r.json()

    assert r.status_code == 503
    assert body["status"] == "degraded"
    assert body["modelo"]["cargado"] is False
    assert "metadata.json" in body["modelo"]["faltantes"]


@patch("api.routes.contenido.obtener_servicio")
def test_contenido_devuelve_503_si_falta_modelo(mock_servicio):
    mock_servicio.side_effect = FileNotFoundError("metadata.json")

    payload = {
        "titulo": "Prueba de modelo",
        "texto": "Texto suficientemente largo para comprobar el manejo de modelos faltantes.",
    }

    r = client.post("/api/v1/contenido", json=payload)

    assert r.status_code == 503
    assert "Modelo no disponible" in r.json()["detail"]
