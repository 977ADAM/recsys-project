-include .env
export

export PROJECT_ROOT := $(shell pwd)


up:
	./.venv/bin/python main.py train-ranker \
		--output-dir ctr_artifacts \
		--iterations 500 \
		--valid-days 14

deepfmrun:
	python src/pipeline/deepfm/train_deepfm.py \
	--interactions-csv ./data/db/banner_interactions.csv \
	--users-csv ./data/db/users.csv \
	--banners-csv ./data/db/banners.csv \
	--output-dir deepfm_artifacts


infra-up:
	docker compose up -d redis postgres

postgres-up:
	docker compose up -d postgres

api-up:
	docker compose up --build api

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
