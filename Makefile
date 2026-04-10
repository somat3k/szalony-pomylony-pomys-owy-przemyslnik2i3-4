# =============================================================================
# HoloLang / Pillar – Makefile
#
# Targets follow a .hl-aware + .NET-style convention:
#   make build        – install Python package in editable mode (≈ dotnet build)
#   make run          – run the default full_system.hl entry point
#   make run FILE=    – run a specific .hl file (e.g.  make run FILE=examples/tensor_graph.hl)
#   make test         – run pytest suite              (≈ dotnet test)
#   make check FILE=  – syntax-check a .hl file      (≈ dotnet build --dry-run)
#   make repl         – launch the interactive shell_vm REPL
#   make info         – print HoloLang runtime info
#   make skills       – list registered skills
#   make canvas       – open MDI canvas (if display available)
#   make clean        – remove build artefacts        (≈ dotnet clean)
#   make docker-build – build Docker image
#   make docker-up    – bring up full Compose stack
#   make docker-down  – tear down Compose stack
#   make docker-run FILE= – run a .hl file inside the container
#   make core         – build the C Pillar core (libbolo.a / libbolo.so)
#   make core-clean   – clean C build artefacts
#   make publish      – build wheel + sdist          (≈ dotnet pack)
# =============================================================================

# --------------------------------------------------------------------------- #
# Configurable variables
# --------------------------------------------------------------------------- #
PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip
PYTEST      ?= $(PYTHON) -m pytest
HOLOLANG    ?= hololang

ENTRY       ?= examples/full_system.hl
FILE        ?= $(ENTRY)

IMAGE_NAME  ?= hololang/pillar
IMAGE_TAG   ?= latest
COMPOSE     ?= docker compose

CORE_DIR    ?= core
CC          ?= gcc
CFLAGS      ?= -O2 -Wall -Wextra -std=c11 -fPIC

# --------------------------------------------------------------------------- #
# Default target
# --------------------------------------------------------------------------- #
.DEFAULT_GOAL := help

# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #
.PHONY: help
help:
	@echo ""
	@echo "  HoloLang / Pillar – build system"
	@echo ""
	@echo "  Python / .hl targets"
	@echo "  ─────────────────────────────────────────────────────────────────"
	@echo "  build           Install package in editable mode (pip install -e)"
	@echo "  build-all       Install with all optional extras"
	@echo "  run             Run default .hl entry point  (FILE=$(ENTRY))"
	@echo "  run FILE=<path> Run a specific .hl file"
	@echo "  check FILE=     Syntax-check a .hl file"
	@echo "  test            Run pytest test suite"
	@echo "  test-v          Run pytest (verbose)"
	@echo "  repl            Launch interactive HoloLang REPL (shell_vm)"
	@echo "  info            Print HoloLang runtime info"
	@echo "  skills          List registered skills"
	@echo "  canvas          Open MDI canvas"
	@echo "  clean           Remove Python build artefacts"
	@echo ""
	@echo "  C Pillar core targets"
	@echo "  ─────────────────────────────────────────────────────────────────"
	@echo "  core            Build libbolo.a + libbolo.so from core/"
	@echo "  core-clean      Remove C build artefacts"
	@echo ""
	@echo "  Docker / Compose targets"
	@echo "  ─────────────────────────────────────────────────────────────────"
	@echo "  docker-build    Build Docker image ($(IMAGE_NAME):$(IMAGE_TAG))"
	@echo "  docker-up       Start full Compose stack (detached)"
	@echo "  docker-down     Stop and remove Compose stack"
	@echo "  docker-run      Run FILE=<path> inside container"
	@echo "  docker-repl     Launch REPL inside container"
	@echo "  docker-logs     Tail Compose logs"
	@echo ""
	@echo "  Packaging"
	@echo "  ─────────────────────────────────────────────────────────────────"
	@echo "  publish         Build wheel + sdist"
	@echo ""

# --------------------------------------------------------------------------- #
# Python / HoloLang targets
# --------------------------------------------------------------------------- #
.PHONY: build
build:
	$(PIP) install -e ".[dev]"

.PHONY: build-all
build-all:
	$(PIP) install -e ".[all,dev]"

.PHONY: run
run:
	$(HOLOLANG) run $(FILE)

.PHONY: check
check:
	$(HOLOLANG) check $(FILE)

.PHONY: test
test:
	$(PYTEST) tests/

.PHONY: test-v
test-v:
	$(PYTEST) tests/ -v

.PHONY: repl
repl:
	$(HOLOLANG) repl

.PHONY: info
info:
	$(HOLOLANG) info

.PHONY: skills
skills:
	$(HOLOLANG) skills

.PHONY: canvas
canvas:
	$(HOLOLANG) canvas

.PHONY: clean
clean:
	rm -rf build dist *.egg-info .eggs __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"    -delete 2>/dev/null || true

# --------------------------------------------------------------------------- #
# C Pillar core
# --------------------------------------------------------------------------- #
CORE_SRCS := $(wildcard $(CORE_DIR)/**/*.c) $(wildcard $(CORE_DIR)/*.c)
CORE_OBJS := $(CORE_SRCS:.c=.o)

.PHONY: core
core: $(CORE_DIR)/libbolo.a $(CORE_DIR)/libbolo.so

$(CORE_DIR)/libbolo.a: $(CORE_OBJS)
	ar rcs $@ $^

$(CORE_DIR)/libbolo.so: $(CORE_OBJS)
	$(CC) -shared -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -I$(CORE_DIR) -c $< -o $@

.PHONY: core-clean
core-clean:
	rm -f $(CORE_OBJS) $(CORE_DIR)/libbolo.a $(CORE_DIR)/libbolo.so

# --------------------------------------------------------------------------- #
# Docker / Compose targets
# --------------------------------------------------------------------------- #
.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: docker-up
docker-up:
	$(COMPOSE) up -d

.PHONY: docker-down
docker-down:
	$(COMPOSE) down

.PHONY: docker-run
docker-run:
	docker run --rm \
		-v "$(CURDIR)/examples:/workspace/examples:ro" \
		-v "$(CURDIR)/core:/workspace/core:ro" \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		hololang run /workspace/$(FILE)

.PHONY: docker-repl
docker-repl:
	docker run --rm -it \
		-v "$(CURDIR)/examples:/workspace/examples:ro" \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		hololang repl

.PHONY: docker-logs
docker-logs:
	$(COMPOSE) logs -f

# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #
.PHONY: publish
publish:
	$(PIP) install --upgrade build
	$(PYTHON) -m build

# --------------------------------------------------------------------------- #
# Composite convenience targets
# --------------------------------------------------------------------------- #
.PHONY: all
all: build core

.PHONY: ci
ci: build test
