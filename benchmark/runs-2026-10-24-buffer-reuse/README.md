# CT2 CPU INT8 — buffer reuse comparison

| Measurement | Previous wheel | Rebuilt wheel | Change |
| --- | ---: | ---: | ---: |
| Timed synthesis (s) | 73.164 | 73.960 | +0.795 s (+1.1%) |
| Peak process RSS (GiB) | 164.483 | 68.306 | -96.176 GiB (58.5% lower) |

Same CPU host, checkpoint cache, text, 8 threads, 1 warmup and 1 timed run. No memory caps or allocator thresholds. Peak RSS includes loading and warmup. Both timed runs produced 2.786 seconds of audio and finished with EOS. Audio content/quality was not compared. One timed run per build does not establish whether the small timing difference is repeatable.

Fix: CTranslate2 commit `613c119c`. The latest remote `drama3-tts` was checked before recompilation; it had no newer fix. [Build provenance](build.json), [new result](ct2-cpu-int8.json), [previous result](../runs-2026-10-24-cpu-1run/ct2-cpu-int8.json), [console log](ct2-cpu-int8.log), [memory investigation](../memory-investigation/README.md).

CPU int8_float32 rerun: 74.741 s, 69.927 GiB peak RSS. Previous wheel: 73.536 s, 164.474 GiB. One warmup and one timed run, 8 threads, no RAM caps or allocator thresholds. HF has no matching int8_float32 configuration in this baseline.
