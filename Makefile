PYTHON ?= python3

.PHONY: install test smoke build
install:
	$(PYTHON) -m pip install -e '.[dev]'
test:
	$(PYTHON) -m pytest -q
smoke:
	$(PYTHON) -m protomaml smoke
build:
	$(PYTHON) -m build
