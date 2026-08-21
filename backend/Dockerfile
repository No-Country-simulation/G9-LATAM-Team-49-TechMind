# Python 3.12: DEBE coincidir con la version en la que se entreno el modelo.
#
# Los .joblib son pickles: guardan referencias a las clases de scikit-learn
# tal y como existian al serializar. Entrenar en Windows con Python 3.12 y
# scikit-learn 1.9 y luego cargar en Python 3.10 con la version que pip
# resuelva ese dia produce, en el mejor caso, un InconsistentVersionWarning
# y predicciones distintas; en el peor, un AttributeError al deserializar.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.lock.txt (generado con pip freeze en el entorno donde se
# entreno) tiene prioridad sobre requirements.txt, que lleva rangos abiertos.
# Los corchetes hacen que el COPY no falle si el fichero no existe todavia.
COPY requirements.txt requirements.lock.tx[t] ./

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN if [ -f requirements.lock.txt ]; then \
        echo ">>> Instalando versiones ancladas (requirements.lock.txt)"; \
        pip install --no-cache-dir -r requirements.lock.txt; \
    else \
        echo ">>> AVISO: sin requirements.lock.txt, se usan rangos abiertos"; \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Modelo linguistico de spaCy, disponible sin red en tiempo de ejecucion.
RUN python -m spacy download es_core_news_sm

# El modelo de embeddings se descarga de HuggingFace en el PRIMER predecir().
# Dejarlo para el arranque significa depender de que HF este accesible y no
# aplique rate limit justo durante la demo (el log de entrenamiento ya avisa:
# "You are sending unauthenticated requests to the HF Hub"). Se hornea aqui,
# en tiempo de build, y en produccion la red deja de ser un factor.
ENV HF_HOME=/opt/hf
ENV SENTENCE_TRANSFORMERS_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('modelo de embeddings horneado en la imagen')"

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# El pipeline avisa de que sin esto el orden de los topicos de BERTopic puede
# variar entre reinicios. Fijarlo hace el arranque reproducible.
ENV PYTHONHASHSEED=42

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
