.PHONY: docs

# Install the docs extra first: python -m pip install -e '.[docs]'
# Exclude implementation submodules; __all__ exposes the supported public API.
docs:
	python -m pdoc --docformat google --no-show-source -t docs/templates faster_tts '!faster_tts.' -o docs
	touch docs/.nojekyll
