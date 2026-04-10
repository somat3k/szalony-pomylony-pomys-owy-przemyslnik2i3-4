# =============================================================================
# HoloLang / Pillar – Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder)  – compile C Pillar core → libbolo.so
#   Stage 2 (runtime)  – slim Python image with HoloLang installed + .so
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – C Pillar core builder
# ─────────────────────────────────────────────────────────────────────────────
FROM gcc:13-bookworm AS builder

WORKDIR /build

# Copy C core source (may not exist yet; COPY with --chown skips missing dirs)
COPY core/ ./core/

# Build libbolo.a + libbolo.so if source files are present; no-op otherwise.
# This keeps the image buildable even before the C core is implemented.
RUN set -eux; \
    SRCS="$(find core -name '*.c' 2>/dev/null | tr '\n' ' ')"; \
    if [ -n "$SRCS" ]; then \
        gcc -O2 -Wall -std=c11 -fPIC -Icore \
            $SRCS \
            -shared -o core/libbolo.so; \
        ar rcs core/libbolo.a $(echo $SRCS | sed 's/\.c/.o/g' || true); \
        echo "C Pillar core built."; \
    else \
        echo "No C sources found – skipping C core build."; \
        mkdir -p core && touch core/.empty; \
    fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – HoloLang runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="HoloLang Pillar"
LABEL org.opencontainers.image.description="C-based Pillar VM with HoloLang interpreter, tensor runtime, and gRPC network"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT"

# System dependencies: SQLite3 (offline storage), libcurl (IPFS client), tini
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-dev \
        libcurl4 \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy built C library from builder stage
COPY --from=builder /build/core/ ./core/

# Install Python package
COPY pyproject.toml README.md ./
COPY hololang/ ./hololang/
RUN pip install --no-cache-dir -e ".[all]"

# Copy examples and tests (non-root user will run these)
COPY examples/ ./examples/
COPY tests/    ./tests/

# Create a non-root user for safe execution
RUN useradd -m -u 1000 holonaut && chown -R holonaut:holonaut /workspace
USER holonaut

# Expose standard service ports
# 50051 – gRPC (PillarService)
# 8080  – WebSocket
# 8000  – REST API / Webhook sink
EXPOSE 50051 8080 8000

# Use tini as init to reap zombie processes (important for shell_vm children)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default: start interactive REPL (override with e.g. hololang run examples/full_system.hl)
CMD ["hololang", "repl"]
