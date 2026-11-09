.PHONY: install test bench serve web clean

install:
	uv sync

test:
	uv run pytest tests/ -v

bench:
	uv run python -m bench.run_bench

serve:
	uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

web:
	cd web && python -m http.server 3000

clean:
	rm -rf __pycache__ .pytest_cache docs/results/*.png

