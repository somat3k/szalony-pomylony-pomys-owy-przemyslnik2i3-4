# HoloLang — Python runtime Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install runtime and optional extras
COPY pyproject.toml README.md ./
COPY hololang/ ./hololang/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e ".[all]"

CMD ["hololang", "info"]
