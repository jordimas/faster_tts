"""Small round-trip fixture for the standalone native INT8 exporter."""

import json
import struct
import subprocess
import tempfile
from pathlib import Path


def string(value):
    raw = value.encode() + b"\0"
    return struct.pack("<H", len(raw)) + raw


def fixture(directory):
    tensors = [
        (
            "backbone/layers_0/mlp/experts_0/gate_proj/weight",
            (2, 2),
            1,
            struct.pack("4b", 0x38, 0x40, -72, -64),
        ),
        (
            "backbone/layers_0/mlp/experts_0/gate_proj/weight_scale_inv",
            (1, 1),
            0,
            struct.pack("<f", 2),
        ),
        (
            "backbone/layers_0/mlp/router/weight",
            (2, 2),
            0,
            struct.pack("<4f", 0.125, -0.375, 0.5, 0.75),
        ),
        ("backbone/norm/weight", (2,), 0, struct.pack("<2f", 0.25, -0.5)),
        (
            "codec/quantizer/pwconv1/weight",
            (2, 2),
            0,
            struct.pack("<4f", 0.125, -0.375, 0.5, 0.75),
        ),
    ]
    raw = (
        struct.pack("<I", 6)
        + string("Drama3Spec")
        + struct.pack("<II", 1, len(tensors))
    )
    for name, shape, dtype, payload in tensors:
        raw += string(name) + struct.pack("<B", len(shape))
        raw += b"".join(struct.pack("<I", dim) for dim in shape)
        raw += struct.pack("<BI", dtype, len(payload)) + payload
    raw += (
        struct.pack("<I", 1) + string("backbone/alias/weight") + string(tensors[0][0])
    )
    directory.mkdir()
    (directory / "model.bin").write_bytes(raw)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "drama3",
                "ct2_layout_version": 1,
                "source_config": {},
                "quantization_policy": "experts_fp8_sensitive_float32",
                "fp8_format": "e4m3fn_block128",
            }
        )
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ct2-export-check-") as tmp:
        root = Path(tmp)
        fixture(root / "source")
        subprocess.run(
            ["/tmp/faster-tts-export-int8", str(root / "source"), str(root / "int8")],
            check=True,
        )
        subprocess.run(
            [
                "/tmp/faster-tts-export-int8",
                str(root / "int8"),
                str(root / "roundtrip"),
            ],
            check=True,
        )
        assert (root / "int8/model.bin").read_bytes() == (
            root / "roundtrip/model.bin"
        ).read_bytes()
        config = json.loads((root / "int8/config.json").read_text())
        assert (
            config["quantization_policy"] == "standard" and "fp8_format" not in config
        )
        print(
            "Native INT8 export/load/export round-trip passed, including alias and generated scales."
        )
