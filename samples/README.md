# Samples

Requires `faster_tts` and a Drama 3-capable CTranslate2 4.8.4+ build.
Run commands from the repository root.

## Audio chat

```bash
python samples/audio_chat.py
```

Type text and press Enter to hear it. Use `/help` for speaker and delivery tags,
or `/quit` to exit. Playback requires `aplay`, `paplay`, `afplay`, or `ffplay` on PATH.
The default model downloads on first use.

Add `--device cuda --compute-type float16` for GPU, `--model /path/to/model`
for a local checkpoint, or `--max-duration-s 60` to raise the default 24-second
limit per input. The CLI reports truncated audio.

## Article narration

`article_voice.py` turns a UTF-8 text file into a narrated WAV, splitting sentences
and carrying speech history between them.

```bash
python samples/article_voice.py article.txt --output article.wav
```

Selects the device automatically and defaults to the input filename with a `.wav`
extension. The audio budget is 120 seconds; use `--max-duration-s` to change it.
Reaching the limit saves partial narration.

## Audio samples

Generate emotion, two-speaker, and multilingual examples from [prompts.json](prompts.json).

Run `python samples/generate.py`. WAV files are generated locally and excluded from Git.
For Fish Speech/PyTorch, use
`python samples/generate_hf.py /path/to/drama-3 --source /path/to/fish-speech`
in an environment with its dependencies installed.
