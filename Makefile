-include .env
export

PYTHON ?= .venv/bin/python
export PROJECT_ROOT := $(shell pwd)
API_BASE ?= http://127.0.0.1:8080
JSON_FMT = if command -v jq >/dev/null 2>&1; then jq; else cat; fi
INTERACTIONS_CSV ?= ./data/db/banner_interactions.csv
USERS_CSV ?= ./data/db/users.csv
BANNERS_CSV ?= ./data/db/banners.csv
RETRIEVAL_ARTIFACTS_DIR ?= artifacts/pytorch_retrieval
RANKER_ARTIFACTS_DIR ?= artifacts/pytorch_ranker


deepfmrun: ranker-train


retrieval-train:
	$(PYTHON) -m src.retrieval.twotower_minimal \
		--data-path $(INTERACTIONS_CSV) \
		--output-dir $(RETRIEVAL_ARTIFACTS_DIR)


retrieval-refresh:
	curl -sS -X POST $(API_BASE)/api/v1/retrieval/refresh | /bin/sh -c '$(JSON_FMT)'


retrieval-reload:
	curl -sS -X POST $(API_BASE)/api/v1/retrieval/reload | /bin/sh -c '$(JSON_FMT)'


retrieval-train-reload:
	$(MAKE) retrieval-train
	$(MAKE) retrieval-reload


ranker-train:
	$(PYTHON) -m src.ranker.deepfm.train_deepfm \
		--interactions-csv $(INTERACTIONS_CSV) \
		--users-csv $(USERS_CSV) \
		--banners-csv $(BANNERS_CSV) \
		--output-dir $(RANKER_ARTIFACTS_DIR)


infra-up:
	@docker compose up -d redis postgres

postgres-up:
	docker compose up -d postgres

api-up:
	@python -m uvicorn backend.cmd.api.main:app --host 127.0.0.1 --port 8080

api-health:
	curl -sS $(API_BASE)/health | /bin/sh -c '$(JSON_FMT)'

smoke-retrieval:
	curl -sS -X POST $(API_BASE)/api/v1/retrieval \
		-H "Content-Type: application/json" \
		-d '{"user_id":"u_00007","top_k":5}' | /bin/sh -c '$(JSON_FMT)'

smoke-retrieval-refresh:
	$(MAKE) retrieval-refresh

smoke-retrieval-reload:
	$(MAKE) retrieval-reload

smoke-retrieval-all:
	$(MAKE) api-health
	$(MAKE) smoke-retrieval
	$(MAKE) smoke-retrieval-refresh
	$(MAKE) smoke-retrieval-reload

smoke-recommendations:
	curl -sS -X POST $(API_BASE)/api/v1/recommendations \
		-H "Content-Type: application/json" \
		-d '{"user_id":"u_00007","top_k":5,"score_mode":"value","retrieval_artifacts_dir":"artifacts/pytorch_retrieval","retrieval_top_n":100}' | /bin/sh -c '$(JSON_FMT)'

smoke-serving-all:
	$(MAKE) api-health
	$(MAKE) smoke-retrieval
	$(MAKE) smoke-retrieval-refresh
	$(MAKE) smoke-retrieval-reload
	$(MAKE) smoke-recommendations

test-backend:
	.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

test-retrieval:
	.venv/bin/python -m unittest \
		tests.backend.test_retrieval_service \
		tests.backend.test_retrieval_api \
		tests.backend.test_recommendations_with_retrieval \
		-v

stack-up:
	docker compose up -d api redis postgres


infra-down:
	docker compose down


infra-ps:
	docker compose ps


infra-logs:
	docker compose logs -f api redis postgres


api-logs:
	docker compose logs -f api


redis-logs:
	docker compose logs -f redis


postgres-logs:
	docker compose logs -f postgres


amc-create:
	alembic init backend/migrations

amc-up:
	@alembic upgrade head

amc-action:
	@if [ -z "$(action)" ]; then \
		echo "Отсутствует необходимый параметр action. Пример: make migrate-action action=up"; \
		exit 1; \
	fi; \
	alembic revision -m "$(action)"
