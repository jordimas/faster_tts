"""Benchmark the local official Drama 3 PyTorch implementation alongside benchmark.py."""

import importlib.util
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

from benchmark_common import benchmark_parser, measure, run_logged, sweep
from benchmark_results import Results
from check_quality import run_checker


def main():
    tts = Path(__file__).resolve().parents[2]
    parser = benchmark_parser(__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path(os.environ.get("HF_MODEL", tts / "models/drama-3-official")),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            os.environ.get("HF_SOURCE", tts / "../../audio/vendor/fish-speech")
        ),
    )
    parser.add_argument(
        "--compute-type",
        choices=("bfloat16", "float16", "float32"),
        help="default: benchmark all three floating-point types",
    )
    args = parser.parse_args()
    args.check_quality = args.check_quality or args.check == "quality"
    if args.runs < 1 or args.threads < 1 or args.device_index < 0:
        parser.error(
            "runs and threads must be positive; device-index must be nonnegative"
        )
    if not args.text.strip():
        parser.error("text must contain non-whitespace characters")
    if args.check_quality:
        if importlib.util.find_spec("faster_whisper") is None:
            parser.error(
                "quality checking requires faster-whisper in this Python environment"
            )
        from check_quality import normalized_words

        if not normalized_words(args.text):
            parser.error("quality checking requires at least one reference word")
    if not args.show_progress:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TQDM_DISABLE"] = "1"

    run_logged(args, "hf")

    import torch

    results = Results(args, "hf", torch.__version__)

    if args.device is None:
        sweep(
            results,
            [["--device", "cpu"]]
            + [
                ["--device", "cuda", "--device-index", str(i)]
                for i in range(torch.cuda.device_count())
            ],
        )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and args.device_index >= torch.cuda.device_count():
        parser.error("selected GPU is unavailable")
    if not args.compute_worker:
        print(
            f"\nBackend: HF (PyTorch)\nDevice: {args.device}:{args.device_index}"
            + ("; Memory profile: default; uncapped" if args.device == "cuda" else ""),
            flush=True,
        )
    if args.compute_type is None:
        sweep(
            results,
            [
                ["--compute-type", dtype, "--compute-worker"]
                for dtype in ("bfloat16", "float16", "float32")
            ],
        )
    if args.device == "cuda":
        properties = torch.cuda.get_device_properties(args.device_index)
        results.gpu_hardware = {
            "name": properties.name,
            "runtime": "rocm" if torch.version.hip else "cuda",
            "runtime_version": torch.version.hip or torch.version.cuda,
            "total_memory_bytes": properties.total_memory,
            "total_memory_gib": properties.total_memory / 2**30,
        }
    results.start()
    if not (args.source / "fish_speech").is_dir():
        parser.error(f"official Fish Speech source not found: {args.source}")
    if not (args.model / "codec.pth").is_file():
        parser.error(f"official checkpoint not found: {args.model}")
    sys.path.insert(0, str(args.source.resolve()))
    # Use exactly the sentence boundaries used by the CT2 API.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from fish_speech.models.text2semantic.fish_omni.backbone import (
        CHECKPOINT_BACKBONE_PREFIX,
        CHECKPOINT_LM_HEAD_KEY,
        FishOmniBackbone,
        load_backbone_weights,
    )
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_gemm_supported
    from fish_speech.models.text2semantic.fish_omni.generate import (
        build_drama3_prompt,
        generate_drama3,
    )
    from fish_speech.models.text2semantic.fish_omni.model import (
        AUDIO_DECODER_PREFIXES,
        FishOmniModel,
        iter_checkpoint_tensors,
    )
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder,
        load_audio_decoder_weights,
    )
    from fish_speech.models.text2semantic.inference import (
        decode_to_audio,
        load_codec_model,
    )
    from tokenizers import Tokenizer

    from faster_tts.chunking import split_text

    torch.set_num_threads(args.threads)
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device_index}"
    dtype = getattr(torch, args.compute_type)
    config = FishOmniConfig.from_dict(
        json.loads((args.model / "config.json").read_text())
    )
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    print(f"\nCompute type: {args.compute_type}", flush=True)

    with torch.inference_mode():
        # Retain the sample runner's CPU selected-expert dequantization path.
        # GPU FP8 kernels require hardware supported by the official source.
        fp8_experts = config.fp8_experts and (
            args.device == "cpu" or fp8_gemm_supported([device])
        )
        print(
            f"Expert storage: {'FP8' if fp8_experts else args.compute_type}", flush=True
        )
        with torch.device("meta"):
            backbone = FishOmniBackbone(
                config.text, fp8_experts=fp8_experts, dtype=dtype
            )
            tower = FishOmniAudioDecoder(config, dtype=dtype)
        backbone.materialize([device])
        load_backbone_weights(
            backbone,
            iter_checkpoint_tensors(args.model, skip_prefixes=AUDIO_DECODER_PREFIXES),
        )
        tower.to_empty(device=device)
        load_audio_decoder_weights(
            tower,
            iter_checkpoint_tensors(
                args.model,
                skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY),
            ),
        )
        model = FishOmniModel(config, backbone, tower, tokenizer).eval()
        # CT2 waveform decoding also stays on CPU in float32, including GPU runs.
        codec = load_codec_model(args.model / "codec.pth", "cpu", torch.float32)

        def synchronize():
            if args.device == "cuda":
                torch.cuda.synchronize(args.device_index)

        def synthesize():
            completed, outputs = [], []
            remaining = 512
            chunks = split_text(args.text)
            finish = "length"
            for chunk in chunks:
                if remaining == 0:
                    break
                history = completed[:1] + (completed[-1:] if len(completed) > 1 else [])
                prompt = build_drama3_prompt(
                    unicodedata.normalize("NFC", chunk),
                    tokenizer,
                    config,
                    history=history,
                )
                torch.manual_seed(0)
                frames, finish = generate_drama3(
                    model,
                    prompt,
                    max_new_tokens=remaining,
                    temperature=0.7,
                    top_k=30,
                    top_p=0.7,
                    prefill_chunk_size=16,
                )
                if not len(frames):
                    raise RuntimeError("Incomplete synthesis: model returned no audio")
                codes = frames[:, 1:].T.contiguous().long().cpu()
                outputs.append(decode_to_audio(codes, codec).cpu().float().numpy())
                remaining -= len(frames)
                if finish != "eos":
                    break
                completed.append((unicodedata.normalize("NFC", chunk), codes))
            finish = "eos" if len(completed) == len(chunks) else "length"
            return np.concatenate(outputs), codec.sample_rate, finish

        synthesize()  # Exclude model/codec loading and one full warmup.
        synchronize()
        with tempfile.TemporaryDirectory() as audio_dir:
            measure(results, synthesize, audio_dir, synchronize)
            model = backbone = tower = codec = None
            if args.check_quality:
                run_checker(
                    args,
                    audio_dir,
                    f"HF {args.device}:{args.device_index} {args.compute_type} default",
                )


if __name__ == "__main__":
    main()
