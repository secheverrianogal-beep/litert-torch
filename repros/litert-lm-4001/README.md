# LiteRT-LM GPU FP16 digit-corruption reproduction harness

This harness probes the numeric-copy corruption reported in [google-ai-edge/LiteRT-LM #4001](https://github.com/google-ai-edge/LiteRT-LM/issues/4001) and extended to Gemma 4 E2B / Adreno 740 in [this comment](https://github.com/google-ai-edge/LiteRT-LM/issues/4001#issuecomment-6337675113).

## Run on Android/Termux

Use the exact local `.litertlm` bundle and note its SHA-256. The reporter says GPU engine creation works on `litert-lm` / `litert-lm-api` nightly `0.19.0.dev20261015`; `0.19.0.dev20261017` and later fail at engine creation for E2B in [issue #4125](https://github.com/google-ai-edge/LiteRT-LM/issues/4125), which is separate from the inference corruption.

```bash
python litert_lm_fp16_repro.py /path/to/gemma-4-E2B-it.litertlm
```

Default matrix: CPU, GPU default activations, GPU/FP16 and GPU/FLOAT32. Targets are 1,972, 1,987 and 2,002 approximate tokens. To broaden the sweep:

```bash
python litert_lm_fp16_repro.py /path/to/gemma-4-E2B-it.litertlm \
  --targets 1826,1950,1972,1987,2002,2050,2362 \
  --output results_ampliados.jsonl
```

Use `python litert_lm_fp16_repro.py --self-test` to test harness logic without loading LiteRT-LM or a model.

## Interpretation

The target count is tokenized raw system text plus raw user text; it excludes model chat-template/control-token overhead. The JSONL records the requested and observed counts separately, the full bundle SHA-256, runtime/package versions, raw outputs, exact ordered decimal comparison, and the runtime's KV-cache count after generation. Treat engine creation failures separately from generation failures.

This is a minimal one-turn probe, not the reporter's private multi-turn transcript. The script was syntax-checked and its local padding/comparison self-test passed; it has not been executed against the `.litertlm` model on Adreno hardware in this environment. No root cause or fix is claimed.
