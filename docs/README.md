# API reference

Read the [Python API reference](api.md) directly on GitHub.

Install the pre-commit hook once per checkout:

```bash
python -m pip install -e '.[docs]'
pre-commit install
```

The hook regenerates Markdown when API sources change. If it updates the file,
stage `docs/api.md` and retry the commit. CI verifies that the reference is current.

To rebuild manually with the package installed:

```bash
python scripts/generate_docs.py
```

`make docs` runs the same script. It reads public signatures and Google-style
docstrings without downloading weights or initializing the native runtime.
Edit source docstrings; generated Markdown needs no manual maintenance.
