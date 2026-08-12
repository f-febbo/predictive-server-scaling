.PHONY: help setup data eda phase1 test clean

help:
	@echo "make setup   - install dependencies into .venv via uv"
	@echo "make phase1  - download the trace, build the series, and run the EDA"
	@echo "make data    - download the trace and build the processed series"
	@echo "make eda     - figure + summary statistics into results/"
	@echo "make test    - run the test suite"

setup:
	uv sync --extra dev

data:
	uv run python experiments/build_dataset.py

eda:
	uv run python experiments/eda.py

# Phase 1 acceptance: one command yields the processed series and the figure.
phase1: data eda

test:
	uv run pytest tests/ -W error::RuntimeWarning

clean:
	rm -rf data/processed/* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
