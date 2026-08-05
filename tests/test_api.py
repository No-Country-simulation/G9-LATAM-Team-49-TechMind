import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.schemas.contenido import ContenidoRequest, ContenidoResponse

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.api.routes.contenido.obtener_servicio")
def test_procesar_contenido_success(mock_obtener_servicio):
    # Mocking the ML service to prevent loading large models
    mock_service = MagicMock()
    
    # Mock the predecir method to return a valid response structure
    mock_response = MagicMock()
    mock_response.a_dict.return_value = {
        "doc_id": "test-123",
        "categoria": "General",
        "probabilidad": 0.99,
        "tiempo_ms": 10.5,
        "keywords": [{"keyword": "test", "score": 0.9}, {"keyword": "nlp", "score": 0.8}],
        "entidades_tecnicas": ["FastAPI", "Python"],
        "tema": "Tech",
        "relacionados": []
    }
    mock_service.predecir.return_value = mock_response
    mock_obtener_servicio.return_value = mock_service

    payload = {
        "titulo": "Prueba de NLP",
        "texto": "Este es un texto de prueba suficientemente largo para pasar las validaciones minimas de longitud del sistema."
    }

    response = client.post("/api/v1/contenido", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "test-123"
    assert data["categoria"] == "General"
    assert len(data["keywords"]) == 2
    
    # Verify the mock was called with correct arguments
    mock_service.predecir.assert_called_once_with(payload["titulo"], payload["texto"])
