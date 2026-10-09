FROM python:3.11-slim

WORKDIR /app

COPY requirements.api.txt .

# CPU Torch first so later deps (sentence-transformers) reuse it instead of the CUDA wheel.
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.api.txt \
 && python -c "import torch; assert '+cpu' in torch.__version__, torch.__version__"

COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
