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


api-up:
	docker compose up -d api


inference-up:
	docker compose up -d inference


stack-up:
	docker compose up -d api inference redis postgres


infra-down:
	docker compose down


infra-ps:
	docker compose ps


infra-logs:
	docker compose logs -f api inference redis postgres


api-logs:
	docker compose logs -f api


inference-logs:
	docker compose logs -f inference


redis-logs:
	docker compose logs -f redis


postgres-logs:
	docker compose logs -f postgres
