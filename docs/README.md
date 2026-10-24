# API reference

Open [index.html](index.html) in a browser after checking out this repository.
GitHub's repository view displays HTML source; to read the rendered pages online,
configure GitHub Pages to publish the `/docs` folder from your default branch.

The HTML is generated from public API signatures and docstrings with pdoc.
After changing the public API or its docstrings, regenerate it from the repo root:

```bash
python -m pip install -e '.[docs]'
make docs
```

Use the Python environment where you installed the docs extra. With uv, run
`uv run --extra docs make docs` instead. Include regenerated HTML in the same
commit as the API change. Model weights, GPU access, and Drama 3 native bindings
are not needed to build the reference.

Only the four classes exported by `faster_tts` are included. Internal modules
are deliberately excluded. `.nojekyll` allows Pages to serve generated assets directly.
