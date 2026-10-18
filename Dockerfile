FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheel to minimize container size (<400MB vs >4GB CUDA)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch>=2.2.0

# Install remaining dependencies from requirements.txt
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt > requirements-notorch.txt && \
    pip install --no-cache-dir -r requirements-notorch.txt && \
    rm requirements-notorch.txt

# Copy project source and configuration
COPY pyproject.toml .
COPY src/ src/
COPY run_live_bot.py evaluate.py train.py ./

# Install package in standard mode
RUN pip install --no-cache-dir .

# Configuration defaults
ENV PYTHONUNBUFFERED=1 \
    ULTIMO_SERVER_HOST="127.0.0.1" \
    ULTIMO_SERVER_PORT="8080" \
    BOT_NAME="DeepUltimo_v1" \
    BOT_STRATEGY="heuristic"

# Default entrypoint runs the live TCP bot against target Ultimo server
ENTRYPOINT ["python", "run_live_bot.py"]
CMD ["--host=127.0.0.1", "--port=8080", "--strategy=heuristic"]
