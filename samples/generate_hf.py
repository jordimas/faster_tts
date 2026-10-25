"""Generate the sample prompts with the official Hugging Face / PyTorch model."""

import argparse
import json
import os
import sys
import unicodedata
import wave
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    prompts = json.loads((directory / "prompts.json").read_text(encoding="utf-8"))
    audio = directory.parents[1] / "audio"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path(os.environ.get("HF_MODEL", audio / "models/drama-3")),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("HF_SOURCE", audio / "vendor/fish-speech")),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--compute-type", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=768)
    parser.add_argument("--max-retries", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-dir", type=Path, default=directory / "hf")
    parser.add_argument(
        "--sample",
        choices=[prompt["name"] for prompt in prompts],
        help="generate only the named sample",
    )
    args = parser.parse_args()
    if args.sample:
        prompts = [prompt for prompt in prompts if prompt["name"] == args.sample]
    if args.threads < 1 or args.max_frames < 1 or args.device_index < 0:
        parser.error(
            "threads and max-frames must be positive; device-index must be nonnegative"
        )
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be a nonnegative uint64 integer")
    if not (args.source / "fish_speech").is_dir():
        parser.error(
            f"official Fish Speech source not found: {args.source}; set --source or HF_SOURCE"
        )
    if not (args.model / "codec.pth").is_file():
        parser.error(
            f"official checkpoint not found: {args.model}; set the model argument or HF_MODEL"
        )

    import numpy as np
    import torch
    from tokenizers import Tokenizer

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and args.device_index >= torch.cuda.device_count():
        parser.error("selected GPU is unavailable")
    sys.path.insert(0, str(args.source.resolve()))
    sys.path.insert(0, str(directory.parent))
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
    print(f"Loading HF model: {args.model} ({device}, {args.compute_type})", flush=True)
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

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for sample in prompts:
            print(f"Generating {sample['name']}...", flush=True)
            completed, outputs = [], []
            remaining = args.max_frames
            chunks = split_text(sample["text"])
            for index, chunk in enumerate(chunks):
                if remaining == 0:
                    raise RuntimeError(
                        f"{sample['name']}: frame budget exhausted before chunk {index + 1}"
                    )
                text = unicodedata.normalize("NFC", chunk)
                history = completed[:1] + (completed[-1:] if len(completed) > 1 else [])
                prompt = build_drama3_prompt(text, tokenizer, config, history=history)
                for attempt in range(args.max_retries + 1):
                    seed = (args.seed + attempt) % 2**64
                    torch.manual_seed(seed)
                    frames, finish = generate_drama3(
                        model,
                        prompt,
                        max_new_tokens=remaining,
                        temperature=0.7,
                        top_k=30,
                        top_p=0.7,
                        prefill_chunk_size=16,
                    )
                    print(
                        f"  chunk={index + 1}, seed={seed}, frames={len(frames)}, finish={finish}",
                        flush=True,
                    )
                    if not len(frames):
                        raise RuntimeError(f"{sample['name']}: model returned no audio")
                    if finish == "eos":
                        break
                if finish != "eos":
                    raise RuntimeError(
                        f"{sample['name']}: generation reached the {args.max_frames}-frame limit; "
                        "try another --seed or simplify the prompt. No WAV written for this sample."
                    )
                codes = frames[:, 1:].T.contiguous().long().cpu()
                audio = decode_to_audio(codes, codec).cpu().float().numpy().reshape(-1)
                if not audio.size or not np.isfinite(audio).all():
                    raise RuntimeError(f"{sample['name']}: invalid audio")
                outputs.append(audio)
                remaining -= len(frames)
                completed.append((text, codes))
            audio = np.concatenate(outputs)
            pcm = np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2")
            output_path = args.output_dir / f"{sample['name']}.wav"
            with wave.open(str(output_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(codec.sample_rate)
                output.writeframes(pcm.tobytes())
            print(
                f"Saved {output_path} ({audio.size / codec.sample_rate:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
