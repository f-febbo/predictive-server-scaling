.PHONY: help setup data eda phase1 baselines backtest frontier results test clean

help:
	@echo "make setup      - install dependencies into .venv via uv"
	@echo "make data       - download the trace and build the processed series"
	@echo "make eda        - figure + summary statistics into results/"
	@echo "make phase1     - data + eda"
	@echo "make baselines  - tune and sweep the reactive baselines (training split)"
	@echo "make backtest   - backtest the forecasters on the held-out split"
	@echo "make frontier   - the headline predictive-vs-reactive comparison"
	@echo "make results    - everything above, in order"
	@echo "make test       - run the test suite"

setup:
	uv sync --extra dev

data:
	uv run python experiments/build_dataset.py

eda:
	uv run python experiments/eda.py

phase1: data eda

baselines:
	uv run python -m experiments.phase3_baselines

# Writes the forecasts the frontier sweep and the AWS deployment both read.
backtest:
	uv run python -m experiments.phase4_backtest

frontier:
	uv run python -m experiments.phase4_frontier

# Everything in results/, from a clean checkout. Order matters: the frontier
# sweep consumes the forecasts the backtest writes.
results: phase1 baselines backtest frontier

test:
	uv run pytest tests/ -W error::RuntimeWarning

clean:
	rm -rf data/processed/* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
