"""Benchmark the local official Drama 3 PyTorch implementation alongside benchmark.py."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
from statistics import mean, stdev
import subprocess
import sys
import tempfile
from time import perf_counter
import unicodedata
import wave

from benchmark_results import Results, run_logged


def main():
    tts = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', type=Path,
                        default=Path(os.environ.get('HF_MODEL', tts / 'models/drama-3-official')))
    parser.add_argument('--source', type=Path,
                        default=Path(os.environ.get('HF_SOURCE', tts / '../../audio/vendor/fish-speech')))
    parser.add_argument('--text', default='Hello! Welcome to our application.')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'),
                        help='default: benchmark CPU and every available GPU')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--compute-type', choices=('bfloat16', 'float16', 'float32'),
                        help='default: benchmark all three floating-point types')
    parser.add_argument('--compute-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--show-progress', action='store_true')
    parser.add_argument('--check-quality', action='store_true')
    parser.add_argument('--check', choices=('quality',), help='alias for --check-quality')
    parser.add_argument("--output-json", type=Path, default=Path("hf.json"),
                        help="JSON output (default: ./hf.json; overwritten on each invocation)")
    parser.add_argument("--log-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.check_quality = args.check_quality or args.check == 'quality'
    if args.runs < 1 or args.threads < 1 or args.device_index < 0:
        parser.error('runs and threads must be positive; device-index must be nonnegative')
    if not args.text.strip():
        parser.error('text must contain non-whitespace characters')
    if args.check_quality:
        if importlib.util.find_spec('faster_whisper') is None:
            parser.error('quality checking requires faster-whisper in this Python environment')
        from check_quality import normalized_words
        if not normalized_words(args.text):
            parser.error('quality checking requires at least one reference word')
    if not args.show_progress:
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['TQDM_DISABLE'] = '1'

    run_logged(args, "hf")

    import torch

    results = Results(args, "hf", torch.__version__)

    def sweep(configurations):
        failed = False
        for options in configurations:
            status = results.run(options)
            if status:
                print(f"{' '.join(options)} failed (exit {status})", flush=True)
                failed = True
        raise SystemExit(1 if failed else 0)

    if args.device is None:
        sweep([['--device', 'cpu']] + [
            ['--device', 'cuda', '--device-index', str(i)]
            for i in range(torch.cuda.device_count())])
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and args.device_index >= torch.cuda.device_count():
        parser.error('selected GPU is unavailable')
    if not args.compute_worker:
        print(f'\nBackend: HF (PyTorch)\nDevice: {args.device}:{args.device_index}'
              + ('; Memory profile: default; uncapped' if args.device == 'cuda' else ''), flush=True)
    if args.compute_type is None:
        sweep([['--compute-type', dtype, '--compute-worker']
               for dtype in ('bfloat16', 'float16', 'float32')])
    if args.device == 'cuda':
        properties = torch.cuda.get_device_properties(args.device_index)
        results.gpu_hardware = {
            'name': properties.name,
            'runtime': 'rocm' if torch.version.hip else 'cuda',
            'runtime_version': torch.version.hip or torch.version.cuda,
            'total_memory_bytes': properties.total_memory,
            'total_memory_gib': properties.total_memory / 2**30,
        }
    results.start()
    if not (args.source / 'fish_speech').is_dir():
        parser.error(f'official Fish Speech source not found: {args.source}')
    if not (args.model / 'codec.pth').is_file():
        parser.error(f'official checkpoint not found: {args.model}')
    sys.path.insert(0, str(args.source.resolve()))
    # Use exactly the sentence boundaries used by the CT2 API.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from faster_tts.chunking import split_text
    import numpy as np
    from tokenizers import Tokenizer
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.backbone import (
        FishOmniBackbone, load_backbone_weights,
        CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY,
    )
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder, load_audio_decoder_weights,
    )
    from fish_speech.models.text2semantic.fish_omni.model import (
        FishOmniModel, iter_checkpoint_tensors, AUDIO_DECODER_PREFIXES,
    )
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt, generate_drama3
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_gemm_supported
    from fish_speech.models.text2semantic.inference import load_codec_model, decode_to_audio

    torch.set_num_threads(args.threads)
    device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device_index}'
    dtype = getattr(torch, args.compute_type)
    config = FishOmniConfig.from_dict(json.loads((args.model / 'config.json').read_text()))
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    print(f'\nCompute type: {args.compute_type}', flush=True)

    with torch.inference_mode():
        # Retain the sample runner's CPU selected-expert dequantization path.
        # GPU FP8 kernels require hardware supported by the official source.
        fp8_experts = config.fp8_experts and (args.device == 'cpu' or fp8_gemm_supported([device]))
        print(f'Expert storage: {"FP8" if fp8_experts else args.compute_type}', flush=True)
        with torch.device('meta'):
            backbone = FishOmniBackbone(config.text, fp8_experts=fp8_experts, dtype=dtype)
            tower = FishOmniAudioDecoder(config, dtype=dtype)
        backbone.materialize([device])
        load_backbone_weights(backbone, iter_checkpoint_tensors(
            args.model, skip_prefixes=AUDIO_DECODER_PREFIXES))
        tower.to_empty(device=device)
        load_audio_decoder_weights(tower, iter_checkpoint_tensors(
            args.model, skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY)))
        model = FishOmniModel(config, backbone, tower, tokenizer).eval()
        # CT2 waveform decoding also stays on CPU in float32, including GPU runs.
        codec = load_codec_model(args.model / 'codec.pth', 'cpu', torch.float32)

        def synchronize():
            if args.device == 'cuda':
                torch.cuda.synchronize(args.device_index)

        def synthesize():
            completed, outputs = [], []
            remaining = 512
            chunks = split_text(args.text)
            finish = 'length'
            for chunk in chunks:
                if remaining == 0:
                    break
                history = completed[:1] + (completed[-1:] if len(completed) > 1 else [])
                prompt = build_drama3_prompt(unicodedata.normalize('NFC', chunk), tokenizer,
                                            config, history=history)
                torch.manual_seed(0)
                frames, finish = generate_drama3(model, prompt, max_new_tokens=remaining,
                                                temperature=0.7, top_k=30, top_p=0.7,
                                                prefill_chunk_size=16)
                if not len(frames):
                    raise RuntimeError('Incomplete synthesis: model returned no audio')
                codes = frames[:, 1:].T.contiguous().long().cpu()
                outputs.append(decode_to_audio(codes, codec).cpu().float().numpy())
                remaining -= len(frames)
                if finish != 'eos':
                    break
                completed.append((unicodedata.normalize('NFC', chunk), codes))
            finish = 'eos' if len(completed) == len(chunks) else 'length'
            return np.concatenate(outputs), finish

        synthesize()  # Exclude model/codec loading and one full warmup.
        synchronize()
        times, durations = [], []
        run_results = []
        with tempfile.TemporaryDirectory() as audio_dir:
            for i in range(args.runs):
                synchronize()
                start = perf_counter()
                audio, finish = synthesize()
                synchronize()
                elapsed = perf_counter() - start
                duration = audio.size / codec.sample_rate
                if args.runs > 1:
                    print(f'Run {i + 1}: {elapsed:.3f}s, audio={duration:.3f}s, '
                          f'RTF={elapsed / duration:.3f}, finish_reason={finish}')
                if finish != 'eos':
                    raise SystemExit('Frame limit reached; shorten --text.')
                run_results.append({'time_seconds': elapsed, 'audio_seconds': duration,
                                    'rtf': elapsed / duration, 'finish_reason': finish})
                times.append(elapsed)
                durations.append(duration)
                if args.check_quality:
                    with wave.open(str(Path(audio_dir) / f'run-{i + 1}.wav'), 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(codec.sample_rate)
                        wav.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
                del audio
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_gib = peak_rss / (1024**3 if sys.platform == 'darwin' else 1024**2)
            print(f'Mean: {mean(times):.3f}s; std: {stdev(times) if len(times) > 1 else 0:.3f}s; '
                  f'RTF: {sum(times) / sum(durations):.3f}; peak RSS: {peak_gib:.3f} GiB', flush=True)
            results.success(run_results, peak_gib)
            del model, backbone, tower, codec
            if args.check_quality:
                subprocess.run([sys.executable, str(Path(__file__).with_name('check_quality.py')),
                                '--text', args.text, '--threads', str(args.threads),
                                '--label', f'HF {args.device}:{args.device_index} {args.compute_type} default',
                                *(['--show-progress'] if args.show_progress else []),
                                *[str(Path(audio_dir) / f'run-{i + 1}.wav') for i in range(args.runs)]],
                               check=True)


if __name__ == '__main__':
    main()
