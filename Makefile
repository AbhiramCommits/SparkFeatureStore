SHELL := /bin/bash
MONTHS ?= 1

HAS_UV := $(shell command -v uv 2>/dev/null)

ifeq ($(strip $(HAS_UV)),)
RUN_PY := .venv/bin/python
else
RUN_PY := uv run
endif

.PHONY: help up down fetch bronze bronze-local test lint format venv

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## Create local venv + install pinned deps (pip fallback when uv is absent)
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

up: ## Build images and start spark-master, 2 workers, minio, postgres
	docker compose up -d --build

down: ## Stop the stack (volumes are kept)
	docker compose down

fetch: ## Download NYC TLC Yellow Taxi parquet (MONTHS=1 by default, MONTHS=12 for full 2023)
	$(RUN_PY) scripts/fetch_data.py --months $(MONTHS)

bronze: ## Run bronze ingest on the docker spark cluster (make up && make fetch first)
	mkdir -p data/bronze
	chmod -R a+rwX data
	docker compose exec -T spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		--deploy-mode client \
		/opt/sparkfeaturestore/ingest/bronze.py --env docker --profile cluster

bronze-local: ## Run bronze ingest in local[*] mode on the host (needs Java 8/11/17)
	$(RUN_PY) ingest/bronze.py --env local --profile local

test: ## Run unit tests
	$(RUN_PY) -m pytest tests -q

lint: ## Lint with ruff + check formatting with black
	$(RUN_PY) -m ruff check .
	$(RUN_PY) -m black --check .

format: ## Auto-fix lint issues + apply black formatting
	$(RUN_PY) -m ruff check --fix .
	$(RUN_PY) -m black .

ifeq ($(strip $(HAS_UV)),)
fetch: venv
bronze-local: venv
test: venv
lint: venv
endif
