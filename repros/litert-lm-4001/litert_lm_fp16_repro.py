#!/usr/bin/env python3
"""Deterministic probe for LiteRT-LM GPU FP16 numeric corruption near 2k tokens.

This is a minimal one-turn reproduction harness, not the original reporter's
private multi-turn transcript. It compares CPU, GPU default activations, explicit
GPU FLOAT16 and GPU FLOAT32 using identical decimal-copy tasks. It records both
the requested raw-text token estimate and the runtime's post-generation KV-cache
count; do not conflate those counts.

Example (Android/Termux, on a build where GPU engine creation works):
  python litert_lm_fp16_repro.py /data/adb/models/litert-lm/gemma-4-E2B-it.litertlm

For the E2B nightly engine-creation regression on 20261017+, see LiteRT-LM #4125.
Use the last known working GPU nightly from the reporter's matrix (20261015) to
probe the separate inference corruption in #4001.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

SYSTEM_MESSAGE = (
    "You copy data exactly. Preserve every digit and decimal point. "
    "Do not calculate, round, normalize, or infer missing values."
)
TASK = (
    'Copy these five decimal values exactly into a JSON object with key "values". '
    'Return only JSON. Input values: 341.8, 288.8, 255.2, 123.4, 456.7. '
    'Preserve every digit and decimal point; do not calculate or normalize.'
)
PAD_ATOMS = [
    " Reference note: the source record remains unchanged.",
    " This sentence is neutral context for a controlled length sweep.",
    " Copying means preserving all characters in the source data.",
    " No operation is requested by this background note.",
    " The same neutral record can be read without transformation.",
    " This is context-only text and contains no task instructions.",
    " The record remains stable; its values must not be modified.",
    " Plain text follows only to increase the input length reproducibly.",
]
SHORT_ATOMS = [" note", " text", " data", " item", " record", " context", " unchanged", " copy", " a", " the", " x", " ."]
EXPECTED = ["341.8", "288.8", "255.2", "123.4", "456.7"]
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.])")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def count_tokens(engine: Any, text: str) -> int:
    return len(engine.tokenize(text))


def build_prompt_for_target(engine: Any, target: int) -> tuple[str, int, bool]:
    """Pad prompt to target tokenized SYSTEM+user-text count, if feasible.

    Chat-template/control tokens are not included in this tokenizer estimate.
    The measured KV-cache count is captured separately after generation.
    """
    filler = ""

    def prompt_tokens(pad: str) -> int:
        return count_tokens(engine, SYSTEM_MESSAGE + "\n" + pad + TASK)

    current = prompt_tokens(filler)
    if current > target:
        return TASK, current, False

    # Fill most of the gap by binary-searching repetitions of one stable phrase.
    coarse_atom = PAD_ATOMS[0]
    one_atom_count = prompt_tokens(coarse_atom)
    coarse_cost = max(1, one_atom_count - current)
    estimate = max(0, (target - current) // coarse_cost)
    low, high = 0, estimate + 2
    while low < high:
        mid = (low + high + 1) // 2
        candidate_count = prompt_tokens(coarse_atom * mid)
        if candidate_count <= target:
            low = mid
        else:
            high = mid - 1
    filler = coarse_atom * low
    current = prompt_tokens(filler)

    # Finish the remainder with short neutral atoms, recounting for token merges.
    for _ in range(64):
        if current == target:
            return filler + TASK, current, True
        best: tuple[int, str] | None = None
        for atom in SHORT_ATOMS + PAD_ATOMS[1:]:
            candidate_count = prompt_tokens(filler + atom)
            if current < candidate_count <= target:
                if best is None or candidate_count > best[0]:
                    best = (candidate_count, atom)
        if best is None:
            break
        current = best[0]
        filler += best[1]
    actual = prompt_tokens(filler)
    return filler + TASK, actual, actual == target


def extract_output_text(response: Any) -> str:
    contents = getattr(response, "contents", None)
    if contents is not None:
        return str(contents)
    if isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    return str(response)


def evaluate(output: str) -> dict[str, Any]:
    actual = NUMBER_RE.findall(output)
    return {
        "expected_decimal_strings": EXPECTED,
        "observed_decimal_strings": actual,
        "exact_decimal_sequence": actual == EXPECTED,
    }


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    try:
        import litert_lm
    except ImportError as exc:
        print(
            "ERROR: litert_lm is not installed in this Python environment. "
            "Run this script in Android/Termux where the LiteRT-LM nightly wheel is installed.",
            file=sys.stderr,
        )
        print(f"Import details: {exc}", file=sys.stderr)
        return 2

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"ERROR: model bundle not found: {model_path}", file=sys.stderr)
        return 2

    model_sha = sha256_file(model_path)
    package_versions = {
        "litert-lm": package_version("litert-lm"),
        "litert-lm-api": package_version("litert-lm-api"),
    }
    modes = args.modes.split(",")
    valid_modes = {"cpu", "gpu-default", "gpu-fp16", "gpu-fp32"}
    unknown = set(modes) - valid_modes
    if unknown:
        raise ValueError(f"Unknown mode(s): {', '.join(sorted(unknown))}")
    targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    if not targets or any(x <= 0 for x in targets):
        raise ValueError("--targets must be comma-separated positive integers")

    print(json.dumps({
        "event": "run_start",
        "model_basename": model_path.name,
        "model_sha256": model_sha,
        "platform": platform.platform(),
        "python": sys.version,
        "package_versions": package_versions,
        "modes": modes,
        "targets": targets,
        "target_definition": "token count of raw SYSTEM_MESSAGE + newline + user text; excludes chat-template/control-token overhead",
        "output_jsonl": str(Path(args.output).expanduser().resolve()),
    }, ensure_ascii=False), flush=True)

    for mode in modes:
        backend = litert_lm.Backend.CPU() if mode == "cpu" else litert_lm.Backend.GPU()
        activation = None
        if mode == "gpu-fp16":
            activation = litert_lm.ActivationDataType.FLOAT16
        elif mode == "gpu-fp32":
            activation = litert_lm.ActivationDataType.FLOAT32
        init_started = time.monotonic()
        try:
            engine_kwargs: dict[str, Any] = {
                "model_path": str(model_path),
                "backend": backend,
                "max_num_tokens": args.max_num_tokens,
            }
            if activation is not None:
                engine_kwargs["activation_data_type"] = activation
            engine = litert_lm.Engine(**engine_kwargs)
            engine_init_seconds = time.monotonic() - init_started
        except Exception as exc:
            record = {
                "event": "engine_init_failed",
                "mode": mode,
                "model_basename": model_path.name,
                "model_sha256": model_sha,
                "package_versions": package_versions,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_jsonl(Path(args.output).expanduser(), record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            continue

        for target in targets:
            prompt, prompt_count_estimate, exact_target = build_prompt_for_target(engine, target)
            combined = SYSTEM_MESSAGE + "\n" + prompt
            raw_prompt_count = count_tokens(engine, prompt)
            case_started = time.monotonic()
            record: dict[str, Any] = {
                "event": "case_result",
                "mode": mode,
                "model_basename": model_path.name,
                "model_sha256": model_sha,
                "package_versions": package_versions,
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "target_combined_raw_tokens": target,
                "observed_combined_raw_tokens": count_tokens(engine, combined),
                "target_hit_exactly": exact_target,
                "raw_user_text_tokens": raw_prompt_count,
                "chat_template_tokens_included_in_target": False,
                "engine_init_seconds": round(engine_init_seconds, 4),
                "prompt": prompt,
            }
            try:
                with engine.create_conversation(
                    system_message=SYSTEM_MESSAGE,
                    sampler_config=litert_lm.SamplerConfig(temperature=0.0, seed=0),
                    enable_speculative_decoding=False,
                ) as conversation:
                    response = conversation.send_message(prompt, max_output_tokens=128)
                    output = extract_output_text(response)
                    record["raw_output"] = output
                    record.update(evaluate(output))
                    try:
                        record["runtime_token_count_after_generation"] = int(conversation.token_count)
                    except Exception as exc:
                        record["runtime_token_count_error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        benchmark = conversation.get_benchmark_info()
                        record["benchmark"] = {
                            k: getattr(benchmark, k)
                            for k in getattr(benchmark, "__dataclass_fields__", {})
                        }
                    except Exception as exc:
                        record["benchmark_unavailable"] = f"{type(exc).__name__}: {exc}"
                record["elapsed_seconds"] = round(time.monotonic() - case_started, 4)
                record["status"] = "ok"
            except Exception as exc:
                record.update({
                    "status": "generation_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": round(time.monotonic() - case_started, 4),
                })
            write_jsonl(Path(args.output).expanduser(), record)
            print(json.dumps({k: v for k, v in record.items() if k != "prompt"}, ensure_ascii=False), flush=True)
        del engine
    print(f"Finished. Append-only JSONL results: {Path(args.output).expanduser().resolve()}")
    return 0


def self_test() -> int:
    class FakeEngine:
        @staticmethod
        def tokenize(text: str) -> list[str]:
            return re.findall(r"\S+", text)
    fake = FakeEngine()
    for target in (80, 100, 120):
        prompt, count, exact = build_prompt_for_target(fake, target)
        measured = count_tokens(fake, SYSTEM_MESSAGE + "\n" + prompt)
        if measured != count or measured > target:
            raise AssertionError((target, count, measured))
        print(f"target={target}, measured={measured}, exact={exact}")
    assert evaluate(json.dumps({"values": EXPECTED}))["exact_decimal_sequence"] is True
    assert evaluate(json.dumps({"values": ["341.8", "2888.8", "255.2", "123.4", "456.7"]}))["exact_decimal_sequence"] is False
    print("Self-test passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="path to the local .litertlm bundle")
    parser.add_argument("--targets", default="1972,1987,2002", help="comma-separated approximate raw token targets")
    parser.add_argument("--modes", default="cpu,gpu-default,gpu-fp16,gpu-fp32", help="cpu,gpu-default,gpu-fp16,gpu-fp32")
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--output", default="litert_lm_fp16_repro_results.jsonl")
    parser.add_argument("--self-test", action="store_true", help="self-test without LiteRT-LM/model")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.model:
        parser.error("model bundle path is required unless --self-test is used")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
