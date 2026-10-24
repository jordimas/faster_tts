# Working on faster_tts

## Simplicity

- Use as little code as possible. Prefer existing functionality and direct
  solutions; add abstractions or dependencies only when they are needed.

## Public API

- The supported top-level API is listed in `faster_tts/__init__.py::__all__`:
  `Model`, `SynthesisResult`, `Drama3Prompt`, and `Drama3PromptBuilder`.
- Preserve public names, constructor/method signatures, parameter defaults,
  result fields, array shapes/dtypes, and documented behavior unless the task
  explicitly calls for a breaking change. Explicitly flag any changes to public
  names, signatures, defaults, or result fields in the final response, and
  identify any compatibility breaks.
- Prefer additive, backward-compatible changes. Existing keyword arguments and
  positional calls must continue to work.
- Review `tests/test_drama3_python_api.py` and related focused tests when changing
  behavior. Add or update meaningful compatibility coverage for API changes.
- There is currently no automated API snapshot check. Do not assume existing
  tests detect every signature or default change.

## Documentation

- Be succinct in documentation. Include details users need and avoid repetition.
- Keep usage examples and setup guidance in `README.md`.
- Document the public API in Google-style docstrings. Explain parameters,
  defaults, return values, exceptions, array shapes/dtypes, and units where relevant.
- The API reference under `docs/` is generated HTML and is versioned with the
  source. Edit docstrings rather than generated pages.
- Install documentation tools with `python -m pip install -e '.[docs]'`, then
  run `make docs` in that environment, or use `uv run --extra docs make docs`.
- Regenerate the HTML after public API or docstring changes and include it with
  the source changes. Instructions for readers/builds live in `docs/README.md`.
- Keep documentation builds independent of model downloads, GPU access, and
  initialization of the experimental native Drama 3 runtime.

## Validation and scope

- Run focused tests appropriate to the change. Some runtime checks require a
  matching experimental CT2 build or local checkpoints; report unavailable checks.
- CI runs `ruff check faster_tts/ tests/ benchmark/` and
  `ruff format --check faster_tts/ tests/ benchmark/`.
- For documentation-only changes, build the reference and check its local links
  and public API coverage; model synthesis is not required.
- Preserve unrelated work, especially local samples, benchmark outputs, and
  experimental files. Do not regenerate model assets as part of routine changes.
