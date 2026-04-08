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