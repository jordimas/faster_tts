"""Resolve local converted models, model names, or Hugging Face snapshots.

Repository IDs accept an optional ``hf://`` prefix. Explicit local paths never
trigger downloads. This module does not load an inference engine.
"""

import json
import os
import re
from importlib import import_module
from pathlib import Path

DEFAULT_MODEL = "jordimas/drama-3-ct2"
MODEL_ALIASES = {"drama-3": DEFAULT_MODEL}


ALLOW_PATTERNS = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "conversion.json",
    "LICENSE",
    "LICENSE.*",
    "LICENCE",
    "LICENCE.*",
    "NOTICE",
    "NOTICE.*",
    "README.md",
)


def _validate(directory):
    directory = Path(directory).expanduser().resolve()
    for name in ("model.bin", "config.json", "tokenizer.json"):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty converted Drama 3 asset: {path}. "
                "Provide a complete CTranslate2-converted checkpoint; official "
                "safetensors and GGUF checkpoints cannot be loaded directly."
            )
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            f"Invalid converted model config: {directory / 'config.json'}"
        ) from exc
    source = config.get("source_config") if isinstance(config, dict) else None
    if (
        not isinstance(config, dict)
        or config.get("model_type") != "drama3"
        or type(config.get("ct2_layout_version")) is not int
        or config["ct2_layout_version"] != 1
        or not isinstance(source, dict)
        or not isinstance(source.get("text_config"), dict)
        or not source["text_config"]
        or not isinstance(source.get("audio_decoder_config"), dict)
        or not source["audio_decoder_config"]
    ):
        raise ValueError(
            f"Unsupported converted Drama 3 config in {directory}: expected "
            "model_type='drama3', ct2_layout_version=1 and source_config with "
            "text_config/audio_decoder_config. Convert the checkpoint with the matching converter."
        )
    try:
        tokenizer = json.loads(
            (directory / "tokenizer.json").read_text(encoding="utf-8")
        )
        if not isinstance(tokenizer, dict) or not isinstance(
            tokenizer.get("model"), dict
        ):
            raise ValueError("tokenizer model object missing")  # noqa: TRY004 - invalid JSON content
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            f"Invalid tokenizer JSON: {directory / 'tokenizer.json'}"
        ) from exc
    return directory


def resolve_model_path(
    model=DEFAULT_MODEL, *, revision=None, cache_dir=None, local_files_only=False
):
    """Return a validated local converted checkpoint directory as a ``Path``.

    Accept a local directory, ``drama-3``, or a Hub repository ID (with optional
    ``hf://`` prefix). Existing directories take precedence. Path objects and
    explicit paths (absolute, ./, ../, ~/) stay local. revision, cache_dir and
    local_files_only pass to snapshot_download; no fallback model is loaded.

    Validation checks required nonempty assets and the supported config envelope,
    not the full binary payload, tokenizer vocabulary, or numerical correctness;
    the matching native loader performs those checks when loading the model.
    """
    if not isinstance(model, (str, os.PathLike)):
        raise TypeError(
            "model must be a local path, model name, or organization/repository string"
        )
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision must be a nonempty string when provided")
    local = Path(model).expanduser()
    if local.exists():
        if not local.is_dir():
            raise NotADirectoryError(f"Model path is not a directory: {local}")
        return _validate(local)
    if not isinstance(model, str) or model.startswith(("/", "./", "../", "~/")):
        raise FileNotFoundError(f"Local model directory does not exist: {local}")
    explicit_hub = model.startswith("hf://")
    repository = (
        model[len("hf://") :] if explicit_hub else MODEL_ALIASES.get(model, model)
    )
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository
    ):
        raise ValueError(
            "Use a known model name (drama-3), organization/repository "
            "with revision supplied separately, or an explicit local path"
        )
    try:
        hub = import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError(
            "Hub model resolution requires the huggingface-hub package. "
            "Install it with: pip install huggingface-hub ; or provide an existing local converted model."
        ) from exc
    try:
        snapshot = hub.snapshot_download(
            repo_id=repository,
            revision=revision,
            cache_dir=os.fspath(Path(cache_dir).expanduser())
            if cache_dir is not None
            else None,
            local_files_only=local_files_only,
            allow_patterns=list(ALLOW_PATTERNS),
        )
    except Exception as exc:
        mode = "local cache" if local_files_only else "Hugging Face Hub"
        raise RuntimeError(
            f"Cannot resolve converted model {model} from {mode} "
            f"at revision {revision or 'default'}. Check the repository, access, "
            "revision and cache availability; no fallback model was loaded."
        ) from exc
    return _validate(snapshot)
