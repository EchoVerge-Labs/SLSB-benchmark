.PHONY: install lint test benchmark

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest -q tests/

benchmark:
	slsb run --upstream facebook/wav2vec2-xls-r-300m --tasks asr,sid --seeds 0 --out results/
