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


def test_liveness_siempre_responde():
    """El proceso esta vivo, independientemente del estado del modelo."""
    r = client.get("/health/live")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@patch("services.nlp_service.obtener_servicio")
def test_health_ok_con_modelo_cargado(mock_servicio):
    doble = MagicMock()
    doble.categorias = ["Backend", "DevOps"]
    doble.metadatos = {"version": "2.0.0"}
    mock_servicio.return_value = doble

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["modelo"]["cargado"] is True


@patch("services.nlp_service.obtener_servicio")
def test_health_503_sin_artefactos(mock_servicio):
    """Regresion: /health devolvia 200 aunque el modelo no estuviera cargado."""
    mock_servicio.side_effect = FileNotFoundError("No se encontro metadata.json")

    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degradado"
    assert r.json()["modelo"]["cargado"] is False


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


@patch("api.routes.contenido.obtener_servicio")
def test_contenido_503_si_faltan_artefactos(mock_servicio):
    """Regresion: sin artefactos se devolvia un 500 con traza, no un error controlado."""
    mock_servicio.side_effect = FileNotFoundError("Faltan los artefactos del modelo")

    payload = {"titulo": "Prueba de NLP",
               "texto": "Texto de prueba suficientemente largo para pasar las validaciones."}
    r = client.post("/api/v1/contenido", json=payload)

    assert r.status_code == 503
    assert r.json()["detail"][0]["codigo"] == "modelo_no_disponible"


def test_texto_corto_rechazado_por_pydantic():
    r = client.post("/api/v1/contenido", json={"titulo": "Hola", "texto": "corto"})
    assert r.status_code == 422


def test_cors_expuesto():
    r = client.options("/api/v1/contenido", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
