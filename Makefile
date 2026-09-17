SHELL := /bin/bash
MONTHS ?= 1
FRAMEWORK ?= sklearn
KIND_CLUSTER ?= sparkfeaturestore
KUBECTL ?= kubectl --context kind-$(KIND_CLUSTER)
K8S_DIR ?= k8s
EXECUTOR_INSTANCES ?= 2
EXECUTOR_MEMORY ?= 1g
EXECUTOR_CORES ?= 2

HAS_UV := $(shell command -v uv 2>/dev/null)

ifeq ($(strip $(HAS_UV)),)
RUN_PY := .venv/bin/python
else
RUN_PY := uv run
endif

.PHONY: help up down fetch bronze bronze-local silver silver-local bench-skew bench-partitions train-sklearn train-torch k8s-up k8s-seed k8s-bronze k8s-features k8s-train k8s-down test lint format venv

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

silver: ## Build silver feature groups on the docker spark cluster (make up && make bronze first)
	mkdir -p data/silver
	chmod -R a+rwX data
	docker compose exec -T spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		--deploy-mode client \
		/opt/sparkfeaturestore/features/build_silver.py --env docker --profile cluster

silver-local: ## Build silver features in local[*] mode on the host
	$(RUN_PY) features/build_silver.py --env local --profile local

bench-skew: ## Skew benchmark: salted vs unsalted aggregation (Spark REST metrics)
	docker compose exec -T spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		--deploy-mode client \
		/opt/sparkfeaturestore/bench/skew_bench.py --env docker

bench-partitions: ## Sweep spark.sql.shuffle.partitions x AQE (8 runs, ~10-15 min)
	@for p in 50 200 800 2000; do \
	  for a in true false; do \
	    docker compose exec -T spark-master /opt/spark/bin/spark-submit \
	      --master spark://spark-master:7077 --deploy-mode client \
	      /opt/sparkfeaturestore/bench/partition_sweep.py --env docker \
	      --partitions $$p --aqe $$a; \
	  done; \
	done

train-sklearn: ## Train the sklearn gradient-boosting model on silver features
	$(RUN_PY) training/sklearn_job.py --env local

train-torch: ## Train the PyTorch MLP on silver features
	$(RUN_PY) training/torch_job.py --env local

k8s-up: ## Create kind cluster, build+load images, apply manifests, seed raw data
	@kind get clusters 2>/dev/null | grep -qx $(KIND_CLUSTER) || kind create cluster --config $(K8S_DIR)/kind-config.yaml
	docker build -f docker/Dockerfile.spark -t sparkfeaturestore/spark:3.5.9 .
	docker build -f docker/Dockerfile.train -t sparkfeaturestore/train:0.1.0 .
	kind load docker-image sparkfeaturestore/spark:3.5.9 --name $(KIND_CLUSTER)
	kind load docker-image sparkfeaturestore/train:0.1.0 --name $(KIND_CLUSTER)
	$(KUBECTL) apply -f $(K8S_DIR)/namespace.yaml -f $(K8S_DIR)/rbac.yaml -f $(K8S_DIR)/configmap.yaml
	sed -e 's/CHANGEME_MINIO_USER/minioadmin/' -e 's/CHANGEME_MINIO_PASSWORD/minioadmin/' \
	    -e 's/CHANGEME_POSTGRES_PASSWORD/sparkfs/' $(K8S_DIR)/secrets.example.yaml | $(KUBECTL) apply -f -
	$(KUBECTL) apply -f $(K8S_DIR)/postgres.yaml -f $(K8S_DIR)/minio.yaml
	$(KUBECTL) wait --for=condition=ready pod -l app=postgres -n spark-jobs --timeout=300s
	$(KUBECTL) wait --for=condition=ready pod -l app=minio -n spark-jobs --timeout=300s
	$(MAKE) k8s-seed

k8s-seed: ## Create the datalake bucket and upload the raw parquet to MinIO
	docker run --rm -v $(CURDIR)/data/raw:/data --entrypoint /bin/sh \
	  quay.io/minio/mc:RELEASE.2024-11-17T19-35-25Z -c "\
	    mc alias set k8s-minio http://host.docker.internal:30900 minioadmin minioadmin && \
	    mc mb --ignore-existing k8s-minio/datalake && \
	    mc cp /data/yellow_tripdata_2023-01.parquet k8s-minio/datalake/raw/ && \
	    mc ls k8s-minio/datalake/raw"

k8s-bronze: ## Run the bronze ingest on k8s (Spark native scheduler, cluster mode)
	$(KUBECTL) delete job spark-submit-bronze -n spark-jobs --ignore-not-found
	$(KUBECTL) delete pod -n spark-jobs -l job=bronze-trips --ignore-not-found
	sed -e 's/EXECUTOR_INSTANCES/$(EXECUTOR_INSTANCES)/' -e 's/spark.executor.memory=1g/spark.executor.memory=$(EXECUTOR_MEMORY)/' \
	    $(K8S_DIR)/spark-submit-bronze.yaml | $(KUBECTL) apply -f -
	@for i in $$(seq 1 60); do \
	  $(KUBECTL) get pod -n spark-jobs -l job=bronze-trips --no-headers 2>/dev/null | grep -q . && break; \
	  sleep 5; \
	done
	$(KUBECTL) wait --for=jsonpath='{.status.phase}'=Succeeded pod -n spark-jobs -l job=bronze-trips --timeout=1500s
	$(KUBECTL) logs -n spark-jobs -l job=bronze-trips --tail=30

k8s-features: k8s-bronze ## Run bronze + silver feature pipeline on k8s
	$(KUBECTL) delete job spark-submit-features -n spark-jobs --ignore-not-found
	$(KUBECTL) delete pod -n spark-jobs -l job=silver-features --ignore-not-found
	sed -e 's/EXECUTOR_INSTANCES/$(EXECUTOR_INSTANCES)/' -e 's/spark.executor.memory=1g/spark.executor.memory=$(EXECUTOR_MEMORY)/' \
	    $(K8S_DIR)/spark-submit-features.yaml | $(KUBECTL) apply -f -
	@for i in $$(seq 1 60); do \
	  $(KUBECTL) get pod -n spark-jobs -l job=silver-features --no-headers 2>/dev/null | grep -q . && break; \
	  sleep 5; \
	done
	$(KUBECTL) wait --for=jsonpath='{.status.phase}'=Succeeded pod -n spark-jobs -l job=silver-features --timeout=2400s
	$(KUBECTL) logs -n spark-jobs -l job=silver-features --tail=40

k8s-train: ## Run a training Job on k8s (FRAMEWORK=sklearn|torch, default sklearn)
	$(KUBECTL) delete job train-job -n spark-jobs --ignore-not-found
	sed 's/FRAMEWORK_VALUE/$(FRAMEWORK)/' $(K8S_DIR)/train-job.yaml | $(KUBECTL) apply -f -
	@for i in $$(seq 1 240); do \
	  state=$$($(KUBECTL) get job train-job -n spark-jobs -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null); \
	  [ -n "$$state" ] && break; sleep 10; \
	done
	$(KUBECTL) get job train-job -n spark-jobs -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q True \
	  || { $(KUBECTL) logs job/train-job -n spark-jobs --tail=30; exit 1; }
	$(KUBECTL) logs job/train-job -n spark-jobs --tail=30

k8s-down: ## Delete the kind cluster
	kind delete cluster --name $(KIND_CLUSTER)

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
