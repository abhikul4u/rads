# =============================================================================
# Author: Rutuja Kulkarni
# RADS Layer 3 — Developer Makefile
# =============================================================================
# Convenience targets for the RADS model-development pipeline. The smoke* family
# runs tests/smoke_test.py, a fast end-to-end regression check that exercises
# the build/train/export path so a broken commit is caught before a long GPU run
# is launched on RunPod. install/clean handle dependency setup and scratch
# cleanup. All recipe lines below are unchanged — only comments were added.

# Declare phony targets so make never confuses them with same-named files.
.PHONY: help smoke smoke-quick smoke-quickest install clean lint

# help: default-friendly target that prints the available commands and their
# approximate runtimes (CPU timings shown; GPU is faster).
help:
	@echo "RADS Layer 3 — dev commands"
	@echo ""
	@echo "  make smoke         Full smoke test (~90s on CPU, faster on GPU)"
	@echo "  make smoke-quick   Skip ONNX export (~70s)"
	@echo "  make smoke-quickest  Skip ONNX + training step (~50s)"
	@echo "  make install       Install requirements"
	@echo "  make clean         Remove build artifacts (NOT artifacts/runs)"
	@echo ""

# smoke: full regression suite — builds each model config, runs a short training
# step, and exercises the ONNX export path. The most thorough of the three.
smoke:
	python tests/smoke_test.py

# smoke-quick: same as smoke but skips the (slow) ONNX export stage.
smoke-quick:
	python tests/smoke_test.py --quick

# smoke-quickest: skips both ONNX export and the training step — the fastest
# sanity check that configs parse and models instantiate.
smoke-quickest:
	python tests/smoke_test.py --quick --no-train-step

# install: install all pinned Python dependencies from requirements.txt.
install:
	pip install -r requirements.txt

# clean: remove transient build/cache artifacts and Ultralytics' default scratch
# `runs/` dir. Does NOT delete the curated artifacts/ outputs (trained weights).
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf runs/  # Ultralytics' default scratch dir if tests leak there
