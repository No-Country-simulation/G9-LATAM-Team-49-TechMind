"""Test de integracion: ejercita el pipeline REAL, sin MagicMock.

Es la prueba que habria detectado los bloqueantes #1 y #2 y el hallazgo N1.
Usa dobles simples (no MagicMock) para no cargar modelos de GB, pero ejecuta
todo el codigo de ml.core: validacion, idioma, limpieza, spaCy, keywords,
clasificacion y serializacion.
"""
import numpy as np
import pytest
import spacy

from ml.core import CFG, ErrorValidacion, TechMindInference
from schemas.contenido import ContenidoResponse

TEXTO = ("Guia practica para containerizar una aplicacion Python con Docker y "
         "publicarla en Kubernetes mediante un pipeline de integracion continua.")


class _LE:
    classes_ = np.array(["Backend", "DevOps", "Data"])
    def inverse_transform(self, i): return self.classes_[i]

class _CLF:
    def predict_proba(self, X): return np.array([[0.12, 0.81, 0.07]])

class _Emb:
    def encode(self, t, **k): return np.ones((len(t), 8), dtype="float32")

class _KB:
    def extract_keywords(self, *a, **k):
        return [("docker", 0.9), ("kubernetes", 0.8), ("contenedores", 0.6)]

class _Yake:
    def extract_keywords(self, texto): return [("despliegue continuo", 0.1)]


@pytest.fixture(scope="module")
def servicio():
    nlp = spacy.blank("es")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "TECH", "pattern": t}
                        for t in ["Docker", "Kubernetes", "Python"]])
    return TechMindInference(
        modelo_clasificacion=_CLF(), label_encoder=_LE(), modelo_embeddings=_Emb(),
        pipeline_nlp=nlp, cfg=CFG, tipo_clasificador="sbert+logreg",
        modelo_keybert=_KB(), extractor_yake=_Yake(),
        mapa_tecnologias={"docker": "Docker"}, metadatos={"version": "2.0.0"})


def test_predecir_resuelve_todos_los_nombres(servicio):
    """Bloqueante #1: predecir() no puede lanzar NameError."""
    r = servicio.predecir("Despliegue continuo con Docker", TEXTO,
                          incluir_explicacion=False, incluir_relacionados=False)
    assert r.categoria == "DevOps"
    assert 0.0 <= r.probabilidad <= 1.0
    assert r.doc_id and r.tiempo_ms > 0


def test_respuesta_valida_contra_el_schema(servicio):
    """Bloqueante #2: a_dict() debe satisfacer ContenidoResponse."""
    r = servicio.predecir("Despliegue continuo con Docker", TEXTO,
                          incluir_explicacion=False, incluir_relacionados=False)
    ContenidoResponse(**r.a_dict())          # no debe lanzar ValidationError


def test_n_keywords_se_respeta(servicio):
    r = servicio.predecir("Docker", TEXTO, n_keywords=2,
                          incluir_explicacion=False, incluir_relacionados=False)
    assert len(r.informacion_adicional) <= 2


def test_id_externo_se_propaga_a_doc_id(servicio):
    r = servicio.predecir("Docker", TEXTO, id_externo="abc-123",
                          incluir_explicacion=False, incluir_relacionados=False)
    assert r.doc_id == "abc-123"


def test_texto_corto_es_rechazado(servicio):
    with pytest.raises(ErrorValidacion):
        servicio.predecir("Hola", "corto")


def test_metodos_publicos_existen(servicio):
    """Hallazgo N1: 'procesar' no existe; el metodo real es 'predecir'."""
    assert hasattr(servicio, "predecir")
    assert not hasattr(servicio, "procesar")