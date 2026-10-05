# Advocate Remuneration Assistant: the Streamlit app as a CPU-only image.
# The LLM runs elsewhere: Groq by default, or a vLLM server when LLM_BASE_URL is set
# (see docker-compose.yml). Build: docker build -t advocate-assistant .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel bundles several GB of CUDA libraries
# that this app never uses (it only runs a small embedding model on CPU).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image, so containers start without a download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

# Run as a non-root user. Chroma writes to its SQLite file on open, so the user owns /app.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
