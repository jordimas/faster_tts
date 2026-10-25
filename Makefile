PYTHON ?= python3
export TAG

.PHONY: docs build check-package check-release-version

docs:
	$(PYTHON) scripts/generate_docs.py

build:
	rm -rf dist
	$(PYTHON) -m build

check-package: build
	$(PYTHON) -m twine check dist/*

check-release-version:
	$(PYTHON) scripts/check_release_version.py
