up:
	python src/pipeline/train.py \
		--interactions-csv ./data/db/banner_interactions.csv \
		--users-csv ./data/db/users.csv \
		--banners-csv ./data/db/banners.csv \
		--output-dir ctr_artifacts \
		--iterations 500 \
		--valid-days 14
