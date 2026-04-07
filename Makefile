up:
	./.venv/bin/python main.py train-ranker \
		--output-dir ctr_artifacts \
		--iterations 500 \
		--valid-days 14
