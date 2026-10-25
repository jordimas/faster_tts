"""Check that TAG matches the package version (Python 3.11+)."""

import os

import tomllib

with open("pyproject.toml", "rb") as metadata:
    version = tomllib.load(metadata)["project"]["version"]
tag = os.environ.get("TAG", "")
if not tag:
    raise SystemExit("Set TAG, for example: make check-release-version TAG=v0.0.1")
if tag != f"v{version}":
    raise SystemExit(f"Tag {tag!r} does not match package version {version!r}")
