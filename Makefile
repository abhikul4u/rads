.PHONY: help smoke smoke-quick smoke-quickest install clean lint

help:
	@echo "RADS Layer 3 — dev commands"
	@echo ""
	@echo "  make smoke         Full smoke test (~90s on CPU, faster on GPU)"
	@echo "  make smoke-quick   Skip ONNX export (~70s)"
	@echo "  make smoke-quickest  Skip ONNX + training step (~50s)"
	@echo "  make install       Install requirements"
	@echo "  make clean         Remove build artifacts (NOT artifacts/runs)"
	@echo ""

smoke:
	python tests/smoke_test.py

smoke-quick:
	python tests/smoke_test.py --quick

smoke-quickest:
	python tests/smoke_test.py --quick --no-train-step

install:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf runs/  # Ultralytics' default scratch dir if tests leak there
