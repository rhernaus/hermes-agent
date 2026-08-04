#!/usr/bin/env python3
"""Bounded Tier B context-compression benchmark harness.

``prepare`` and every public scoring helper are deterministic and provider-free.
``collect`` is a separate fail-closed live boundary and is never entered by
importing this module.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import uuid
from argparse import ArgumentParser
from contextlib import contextmanager
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator


FIXTURE_SCHEMA = "context-compression-tier-b-fixture/1"
SCHEDULE_SCHEMA = "context-compression-tier-b-schedule/1"
ANSWER_SCHEMA = "context-compression-tier-b-answer/1"
RAW_SCHEMA = "context-compression-tier-b-raw/1"
SCORES_SCHEMA = "context-compression-tier-b-scores/1"
COMPARISON_SCHEMA = "context-compression-tier-b-comparison/1"
REPORT_COMPLETE = "TIER_B_REPORT_COMPLETE"
REPORT_INCOMPLETE = "TIER_B_REPORT_INCOMPLETE"

PROVIDER = "openai-codex"
API_MODE = "codex_responses"
SUMMARY_MODEL = "gpt-5.6-luna"
DOWNSTREAM_MODEL = "gpt-5.6-sol"
BASELINE_SHA = "d5e135a51353c2dbc489d5c2583158b22d8efd7b"
CANDIDATE_SHA = "f07664bb9a19788ec426db2eb8b8ec8d9572b21d"
CHECKPOINTS = (0, 1, 2, 4, 8)
REPEATS = (1, 2, 3)
EXPECTED_SUMMARY_ATTEMPTS = 162
EXPECTED_DOWNSTREAM_ATTEMPTS = 90
EXPECTED_TOTAL_ATTEMPTS = 252
TIMEOUT_SECONDS = 300
LIVE_ACK = "CONTEXT_COMPRESSION_TIER_B_AUTHORIZED"
STARTING_HEAD = "953491781ba8ec39cf4b5e15c9654eb7066b33f5"
EVALUATION_BRANCH = "eval/compaction-tier-a"
SOURCE_MANIFEST_SCHEMA = "context-compression-tier-b-source-manifest/1"
RUNTIME_ATTESTATION_SCHEMA = "context-compression-tier-b-runtime-attestation/1"
SUBJECT_PATH = "agent/context_compressor.py"
SPLIT_LIMIT_CHARS = 12_000
CONTEXT_LENGTH = 128_000
CURRENT_TOKENS = 100_000
TAIL_TOKEN_BUDGET = 600
FOCUS_TOPIC = "tier-b synthetic state continuity"
MEMORY_CONTEXT = "tier-b synthetic benchmark memory; no external facts"
BUILD_BASE_IMAGE_ID = (
    "sha256:8539546b37868ca348618a8aa147ecfb68eb0caa8e597f98e649b42ed4e5c805"
)
BUILD_BASE_PROVENANCE = (
    "ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@"
    "sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b"
)

_ALLOWED_CANDIDATE_PATHS = (
    "evaluation/context_compression_tier_b.py",
    "evaluation/fixtures/context-compression-tier-b.json",
    "tests/test_context_compression_tier_b.py",
    "evaluation/Containerfile.context-compression-tier-b",
    "evaluation/context_compression_tier_b_runtime.sh",
)
_PRODUCT_DIFF_PATHS = (
    "agent/context_compressor.py",
    "tests/agent/test_compression_multipass.py",
    "tests/agent/test_context_compressor.py",
)
_CREDENTIAL_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "NOUS_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "CEREBRAS_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "CODEX_HOME",
)

_SOURCE_ROLES = (
    "user",
    "assistant",
    "user",
    "assistant",
    "tool",
    "assistant",
    "user",
    "assistant",
    "user",
    "assistant",
    "user",
    "assistant",
)
_PLACEMENTS = frozenset({"summary_head", "summary_middle", "summary_tail", "raw_tail"})
_DEPENDENCIES = frozenset({"summary", "raw_tail"})
_SECRET_PATTERN = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._-]{8,}|"
    r"(?:api[_-]?key|password|secret|token)\s*[:=]\s*[^\s\]}\",]{4,})"
)


def canonical_bytes(value: Any) -> bytes:
    """Return the single canonical JSON representation used for identities."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_fixture(path: str | Path) -> dict[str, Any]:
    """Load and fully validate the frozen synthetic fixture."""
    try:
        fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("FIXTURE_MALFORMED") from exc
    validate_fixture(fixture)
    return fixture


def _scenario(fixture: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in fixture.get("scenarios", [])
        if item.get("scenario_id") == scenario_id
    ]
    if len(matches) != 1:
        raise ValueError("FIXTURE_SCENARIO_ID_INVALID")
    return matches[0]


def _padded_content(prefix: str, filler: str, target: int) -> str:
    if len(prefix) > target:
        raise ValueError("FIXTURE_FACT_SENTENCE_TRUNCATED")
    if not filler:
        raise ValueError("FIXTURE_FILLER_INVALID")
    repeats = (target - len(prefix) + len(filler) - 1) // len(filler)
    content = (prefix + filler * repeats)[:target]
    if not content.startswith(prefix) or len(content) != target:
        raise ValueError("FIXTURE_CONTENT_LENGTH_MISMATCH")
    return content


def expand_scenario(
    fixture: dict[str, Any],
    scenario_id: str,
    repeat: int,
) -> dict[str, Any]:
    """Expand one immutable source transcript without model or provider access."""
    scenario = _scenario(fixture, scenario_id)
    nonce = fixture["repeat_nonces"].get(str(repeat))
    if not nonce:
        raise ValueError("FIXTURE_REPEAT_INVALID")
    corpus = fixture["corpus"]
    system_prefix = corpus["system_text"]
    system = {
        "row_id": "SYSTEM",
        "role": "system",
        "content": _padded_content(
            system_prefix,
            " Synthetic system context.",
            int(corpus["system_target_chars"]),
        ),
    }
    facts_by_row: dict[str, list[dict[str, Any]]] = {}
    for fact in scenario["facts"]:
        facts_by_row.setdefault(fact["row_id"], []).append(fact)

    rows: list[dict[str, Any]] = [system]
    for spec in scenario["source_rows"]:
        row_id = spec["row_id"]
        role = spec["role"]
        base = spec["text"].replace("{repeat_nonce}", nonce)
        fact_text = " ".join(item["sentence"] for item in facts_by_row.get(row_id, []))
        if role == "tool":
            prefix = f"{base} cycle=0; row={row_id};"
            content = _padded_content(
                prefix,
                ".",
                int(corpus["tool_target_chars"]),
            )
        else:
            prefix = " ".join(
                item
                for item in (
                    base,
                    fact_text,
                    scenario["narrative"],
                    f"cycle=0; row={row_id};",
                )
                if item
            )
            filler = f" {scenario['narrative']} cycle=0; row={row_id};"
            content = _padded_content(
                prefix,
                filler,
                int(corpus["source_target_chars"]),
            )
        row: dict[str, Any] = {
            "row_id": row_id,
            "role": role,
            "content": content,
        }
        if spec.get("tool_call"):
            row["tool_calls"] = [
                {
                    "id": scenario["tool_call_id"],
                    "type": "function",
                    "function": {
                        "name": scenario["tool_name"],
                        "arguments": scenario["tool_arguments"],
                    },
                }
            ]
        if spec.get("tool_result"):
            row["tool_call_id"] = scenario["tool_call_id"]
        rows.append(row)
    return {
        "scenario_id": scenario_id,
        "repeat": repeat,
        "rows": rows,
        "facts": scenario["facts"],
    }


def expand_growth_rows(
    fixture: dict[str, Any],
    scenario_id: str,
    cycle: int,
) -> list[dict[str, Any]]:
    """Expand the exact six alternating, answer-neutral growth rows."""
    if cycle not in range(1, 9):
        raise ValueError("FIXTURE_CYCLE_INVALID")
    scenario = _scenario(fixture, scenario_id)
    corpus = fixture["corpus"]
    rows: list[dict[str, Any]] = []
    for index in range(int(corpus["growth_rows_per_cycle"])):
        row_id = f"G{index:02d}"
        role = "user" if index % 2 == 0 else "assistant"
        prefix = (
            f"{corpus['growth_text']} {scenario['narrative']} "
            f"cycle={cycle}; row={row_id};"
        )
        content = _padded_content(
            prefix,
            f" {scenario['narrative']} cycle={cycle}; row={row_id};",
            int(corpus["growth_target_chars"]),
        )
        rows.append({"row_id": row_id, "role": role, "content": content})
    return rows


def validate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Validate shape, safety, exact row contracts, and declared dependencies."""
    if not isinstance(fixture, dict) or fixture.get("schema_version") != FIXTURE_SCHEMA:
        raise ValueError("FIXTURE_SCHEMA_MISMATCH")
    identity = fixture.get("identity")
    expected_identity = {
        "provider": PROVIDER,
        "api_mode": API_MODE,
        "summarizer_model": SUMMARY_MODEL,
        "downstream_model": DOWNSTREAM_MODEL,
        "baseline_revision": BASELINE_SHA,
        "candidate_revision": CANDIDATE_SHA,
        "expected_summary_attempts": EXPECTED_SUMMARY_ATTEMPTS,
        "expected_downstream_attempts": EXPECTED_DOWNSTREAM_ATTEMPTS,
        "expected_total_attempts": EXPECTED_TOTAL_ATTEMPTS,
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("FIXTURE_IDENTITY_MISMATCH")
    if identity.get("checkpoints") != list(CHECKPOINTS) or identity.get("repeats") != 3:
        raise ValueError("FIXTURE_MATRIX_MISMATCH")
    order = fixture.get("scenario_order")
    scenarios = fixture.get("scenarios")
    if (
        not isinstance(order, list)
        or not isinstance(scenarios, list)
        or len(scenarios) != 3
    ):
        raise ValueError("FIXTURE_SCENARIO_COUNT_MISMATCH")
    if order != [item.get("scenario_id") for item in scenarios] or len(set(order)) != 3:
        raise ValueError("FIXTURE_SCENARIO_ORDER_MISMATCH")
    if tuple(fixture.get("corpus", {}).get("source_roles", [])) != _SOURCE_ROLES:
        raise ValueError("FIXTURE_ROLE_SHAPE_MISMATCH")
    if fixture.get("repeat_nonces") != {
        "1": "TBNONCE-R1-7K4M2Q",
        "2": "TBNONCE-R2-9P6V3D",
        "3": "TBNONCE-R3-5X8C1H",
    }:
        raise ValueError("FIXTURE_NONCE_MISMATCH")

    dependency_counts = {"summary": 0, "raw_tail": 0}
    for scenario in scenarios:
        rows = scenario.get("source_rows")
        if not isinstance(rows, list) or len(rows) != 12:
            raise ValueError("FIXTURE_SOURCE_ROW_COUNT_MISMATCH")
        if [item.get("row_id") for item in rows] != [f"T{i:02d}" for i in range(12)]:
            raise ValueError("FIXTURE_ROW_ID_MISMATCH")
        if tuple(item.get("role") for item in rows) != _SOURCE_ROLES:
            raise ValueError("FIXTURE_ROLE_SHAPE_MISMATCH")
        if not rows[3].get("tool_call") or not rows[4].get("tool_result"):
            raise ValueError("FIXTURE_TOOL_PAIR_MISMATCH")
        facts = scenario.get("facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError("FIXTURE_FACTS_MISSING")
        fact_ids = [item.get("field_id") for item in facts]
        expected_ids = (
            list(scenario.get("required_state_order", []))
            + list(scenario.get("identifiers_order", []))
            + ["recommended_next_action"]
        )
        if sorted(fact_ids) != sorted(expected_ids) or len(set(fact_ids)) != len(
            fact_ids
        ):
            raise ValueError("FIXTURE_FACT_FIELD_MISMATCH")
        for fact in facts:
            if fact.get("placement") not in _PLACEMENTS:
                raise ValueError("FIXTURE_FACT_PLACEMENT_INVALID")
            dependency = fact.get("initial_dependency")
            if dependency not in _DEPENDENCIES:
                raise ValueError("FIXTURE_FACT_DEPENDENCY_INVALID")
            if (fact["placement"] == "raw_tail") != (dependency == "raw_tail"):
                raise ValueError("FIXTURE_FACT_DEPENDENCY_MISMATCH")
            dependency_counts[dependency] += 1
        fixture_text = canonical_bytes(scenario).decode("utf-8")
        if _SECRET_PATTERN.search(fixture_text):
            raise ValueError("FIXTURE_SECRET_PATTERN")
        for repeat in REPEATS:
            expanded = expand_scenario(fixture, scenario["scenario_id"], repeat)
            transcript = canonical_bytes(expanded["rows"]).decode("utf-8")
            nonce = fixture["repeat_nonces"][str(repeat)]
            if transcript.count(nonce) != 1:
                raise ValueError("FIXTURE_NONCE_COUNT_MISMATCH")
            for fact in facts:
                row = next(
                    item
                    for item in expanded["rows"]
                    if item.get("row_id") == fact["row_id"]
                )
                if fact["sentence"] not in row["content"]:
                    raise ValueError("FIXTURE_FACT_SENTENCE_TRUNCATED")
        for cycle in range(1, 9):
            expand_growth_rows(fixture, scenario["scenario_id"], cycle)
    return {
        "scenario_count": 3,
        "source_row_count_per_scenario": 12,
        "growth_rows_per_cycle": 6,
        "initial_dependency_counts": dependency_counts,
    }


def _logical_call_id(record: dict[str, Any]) -> str:
    identity = {
        key: record[key]
        for key in (
            "trajectory_id",
            "call_role",
            "cycle",
            "summary_pass_ordinal",
            "question_id",
        )
    }
    return "call-" + sha256_bytes(canonical_bytes(identity))


def build_execution_schedule(
    fixture: dict[str, Any],
    fixture_sha256: str,
) -> dict[str, Any]:
    """Build the frozen 18-trajectory, 252-call serialized schedule."""
    validate_fixture(fixture)
    if not re.fullmatch(r"[0-9a-f]{64}", fixture_sha256):
        raise ValueError("FIXTURE_HASH_INVALID")
    identities = fixture["identity"]
    trajectories: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    position = 0
    for repeat in REPEATS:
        for scenario_index, scenario_id in enumerate(fixture["scenario_order"]):
            cell = (repeat - 1) * 3 + scenario_index
            revision_order = (
                ("baseline", "candidate")
                if cell % 2 == 0
                else ("candidate", "baseline")
            )
            scenario = _scenario(fixture, scenario_id)
            for revision_label in revision_order:
                revision_sha = (
                    identities["baseline_revision"]
                    if revision_label == "baseline"
                    else identities["candidate_revision"]
                )
                trajectory_id = f"{scenario_id}-r{repeat}-{revision_label}"
                trajectory = {
                    "trajectory_position": len(trajectories) + 1,
                    "trajectory_id": trajectory_id,
                    "scenario_id": scenario_id,
                    "repeat": repeat,
                    "revision_label": revision_label,
                    "revision_sha": revision_sha,
                    "expected_summary_calls": 8 if revision_label == "baseline" else 10,
                    "expected_downstream_calls": 5,
                }
                trajectories.append(trajectory)

                def append_call(
                    role: str,
                    cycle: int,
                    summary_pass_ordinal: int | None = None,
                ) -> None:
                    nonlocal position
                    position += 1
                    snapshot_id = f"snapshot:{trajectory_id}:cycle:{cycle}"
                    record: dict[str, Any] = {
                        "schedule_position": position,
                        "trajectory_id": trajectory_id,
                        "scenario_id": scenario_id,
                        "repeat": repeat,
                        "revision_label": revision_label,
                        "revision_sha": revision_sha,
                        "call_role": role,
                        "cycle": cycle,
                        "summary_pass_ordinal": summary_pass_ordinal,
                        "question_id": scenario["question_id"]
                        if role == "downstream"
                        else None,
                        "snapshot_id": snapshot_id,
                        "provider": PROVIDER,
                        "api_mode": API_MODE,
                        "model": SUMMARY_MODEL
                        if role == "summary"
                        else DOWNSTREAM_MODEL,
                        "timeout_seconds": TIMEOUT_SECONDS,
                    }
                    if role == "downstream":
                        sampling_key = {
                            "fixture_sha256": fixture_sha256,
                            "scenario_id": scenario_id,
                            "repeat": repeat,
                            "revision_sha": revision_sha,
                            "cycle": cycle,
                            "question_id": scenario["question_id"],
                        }
                        record["sampling_key"] = sampling_key
                        record["sampling_key_sha256"] = sha256_bytes(
                            canonical_bytes(sampling_key)
                        )
                    else:
                        dependency_key = {
                            "fixture_sha256": fixture_sha256,
                            "scenario_id": scenario_id,
                            "repeat": repeat,
                            "revision_sha": revision_sha,
                            "cycle": cycle,
                            "summary_pass_ordinal": summary_pass_ordinal,
                        }
                        record["dependency_key"] = dependency_key
                    record["logical_call_id"] = _logical_call_id(record)
                    calls.append(record)

                append_call("downstream", 0)
                for cycle in range(1, 9):
                    pass_count = (
                        3 if revision_label == "candidate" and cycle == 1 else 1
                    )
                    for summary_pass in range(1, pass_count + 1):
                        append_call("summary", cycle, summary_pass)
                    if cycle in CHECKPOINTS:
                        append_call("downstream", cycle)
    summary_count = sum(item["call_role"] == "summary" for item in calls)
    downstream_count = sum(item["call_role"] == "downstream" for item in calls)
    if (
        len(trajectories) != 18
        or summary_count != EXPECTED_SUMMARY_ATTEMPTS
        or downstream_count != EXPECTED_DOWNSTREAM_ATTEMPTS
        or len(calls) != EXPECTED_TOTAL_ATTEMPTS
    ):
        raise ValueError("SCHEDULE_ARITHMETIC_MISMATCH")
    schedule = {
        "schema_version": SCHEDULE_SCHEMA,
        "fixture_sha256": fixture_sha256,
        "expected_counts": {
            "trajectories": 18,
            "summary": summary_count,
            "downstream": downstream_count,
            "total": len(calls),
        },
        "trajectories": trajectories,
        "calls": calls,
    }
    schedule["schedule_sha256"] = sha256_bytes(canonical_bytes(schedule))
    return schedule


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,159}$")
_ANSWER_TOP_KEYS = (
    "schema_version",
    "scenario_id",
    "question_id",
    "required_state",
    "identifiers",
    "recommended_next_action",
)
_ANSWER_FIELD_KEYS = ("applicable", "status", "value")
_ANSWER_STATUSES = frozenset({"known", "unknown", "omitted", "not_applicable"})


def _string_leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _string_leaves(key)
            yield from _string_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_leaves(item)


def _nonce_leakage_count(value: Any, nonce: str) -> int:
    return sum(nonce in leaf for leaf in _string_leaves(value))


def _invalid_answer_result(
    response_bytes: bytes,
    error_codes: list[str],
    nonce: str,
) -> dict[str, Any]:
    decoded = response_bytes.decode("utf-8", errors="ignore")
    return {
        "valid": False,
        "response_bytes": len(response_bytes),
        "response_sha256": sha256_bytes(response_bytes),
        "error_codes": sorted(set(error_codes)),
        "metrics": {
            "task_format_valid": 0,
            "trial_nonce_leakage_count": decoded.count(nonce),
        },
    }


def _strict_field_valid(field: Any) -> bool:
    if not isinstance(field, dict) or tuple(field) != _ANSWER_FIELD_KEYS:
        return False
    applicable = field.get("applicable")
    status = field.get("status")
    value = field.get("value")
    if not isinstance(applicable, bool) or status not in _ANSWER_STATUSES:
        return False
    if applicable:
        if status == "known":
            return isinstance(value, str) and bool(value)
        if status in {"unknown", "omitted"}:
            return value is None
        return False
    return status == "not_applicable" and value is None


def _safe_observed_token(value: str, declared: set[str]) -> bool:
    return (
        value == "[REDACTED]"
        or value in declared
        or bool(_SAFE_TOKEN_RE.fullmatch(value))
    )


def _known_values(answer: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for section in ("required_state", "identifiers"):
        for field_id, field in answer[section].items():
            if field["applicable"] and field["status"] == "known":
                values[field_id] = field["value"]
    action = answer["recommended_next_action"]
    if action["applicable"] and action["status"] == "known":
        values["recommended_next_action"] = action["value"]
    return values


def _condition_true(condition: dict[str, Any], values: dict[str, str]) -> bool:
    observed = values.get(str(condition.get("field")))
    if "equals" in condition:
        return observed == condition["equals"]
    if "in" in condition:
        return observed in condition["in"]
    return False


def _contradiction_count(
    scenario: dict[str, Any],
    values: dict[str, str],
) -> int:
    count = 0
    for rule in scenario["contradiction_rules"]:
        if "all" in rule:
            matched = all(_condition_true(item, values) for item in rule["all"])
        else:
            matched = any(_condition_true(item, values) for item in rule["any"])
        count += int(matched)
    return count


def normalize_and_score_answer(
    fixture: dict[str, Any],
    *,
    scenario_id: str,
    repeat: int,
    response_bytes: bytes,
) -> dict[str, Any]:
    """Strictly normalize and score one answer without persisting unsafe text."""
    validate_fixture(fixture)
    scenario = _scenario(fixture, scenario_id)
    nonce = fixture["repeat_nonces"].get(str(repeat), "")
    if not nonce:
        return _invalid_answer_result(response_bytes, ["REPEAT_INVALID"], "")
    try:
        answer = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return _invalid_answer_result(
            response_bytes,
            ["MALFORMED_MODEL_RESPONSE"],
            nonce,
        )
    if not isinstance(answer, dict):
        return _invalid_answer_result(
            response_bytes,
            ["STRICT_ANSWER_SCHEMA_INVALID"],
            nonce,
        )
    expected_required = scenario["required_state_order"]
    expected_identifiers = scenario["identifiers_order"]
    format_valid = (
        tuple(answer) == _ANSWER_TOP_KEYS
        and answer.get("schema_version") == ANSWER_SCHEMA
        and answer.get("scenario_id") == scenario_id
        and answer.get("question_id") == scenario["question_id"]
        and isinstance(answer.get("required_state"), dict)
        and list(answer.get("required_state", {})) == expected_required
        and isinstance(answer.get("identifiers"), dict)
        and list(answer.get("identifiers", {})) == expected_identifiers
        and all(
            _strict_field_valid(item)
            for item in answer.get("required_state", {}).values()
        )
        and all(
            _strict_field_valid(item) for item in answer.get("identifiers", {}).values()
        )
        and _strict_field_valid(answer.get("recommended_next_action"))
    )
    if not format_valid:
        return _invalid_answer_result(
            response_bytes,
            ["STRICT_ANSWER_SCHEMA_INVALID"],
            nonce,
        )
    facts = {item["field_id"]: item for item in scenario["facts"]}
    declared: set[str] = {item["expected"] for item in scenario["facts"]}
    for alternatives in scenario["stale_alternatives"].values():
        declared.update(alternatives)
    declared.update(scenario["prohibited_actions"])
    for field in (
        list(answer["required_state"].values())
        + list(answer["identifiers"].values())
        + [answer["recommended_next_action"]]
    ):
        if field["status"] == "known" and not _safe_observed_token(
            field["value"], declared
        ):
            return _invalid_answer_result(
                response_bytes,
                ["UNSAFE_OBSERVED_VALUE"],
                nonce,
            )

    flattened: list[tuple[str, dict[str, Any], str]] = []
    for section in ("required_state", "identifiers"):
        flattened.extend(
            (field_id, field, section) for field_id, field in answer[section].items()
        )
    flattened.append((
        "recommended_next_action",
        answer["recommended_next_action"],
        "action",
    ))
    applicability_correct = sum(
        field["applicable"] is True
        and field["status"] == "known"
        and isinstance(field["value"], str)
        for _, field, _ in flattened
    )
    required_correct = sum(
        field["status"] == "known" and field["value"] == facts[field_id]["expected"]
        for field_id, field, section in flattened
        if section == "required_state"
    )
    identifier_correct = sum(
        field["status"] == "known" and field["value"] == facts[field_id]["expected"]
        for field_id, field, section in flattened
        if section == "identifiers"
    )
    stale_count = sum(
        field["status"] == "known"
        and field["value"] in scenario["stale_alternatives"].get(field_id, [])
        for field_id, field, _ in flattened
    )
    uncertainty_count = sum(
        field["applicable"] and field["status"] == "unknown"
        for _, field, _ in flattened
    )
    omission_count = sum(
        field["applicable"] and field["status"] == "omitted"
        for _, field, _ in flattened
    )
    prohibited = set(scenario["prohibited_actions"])
    wrong_known = 0
    for field_id, field, _section in flattened:
        if field["status"] != "known" or field["value"] == facts[field_id]["expected"]:
            continue
        value = field["value"]
        if value in scenario["stale_alternatives"].get(field_id, []):
            continue
        if field_id == "recommended_next_action" and value in prohibited:
            continue
        if nonce in value:
            continue
        wrong_known += 1
    values = _known_values(answer)
    metrics = {
        "applicability_correct_numerator": applicability_correct,
        "applicability_total_denominator": len(flattened),
        "required_state_correct_numerator": required_correct,
        "required_state_applicable_denominator": len(expected_required),
        "identifier_provenance_correct_numerator": identifier_correct,
        "identifier_provenance_applicable_denominator": len(expected_identifiers),
        "stale_resurrection_count": stale_count,
        "contradiction_count": _contradiction_count(scenario, values),
        "unsafe_prohibited_action_count": int(
            values.get("recommended_next_action") in prohibited
        ),
        "explicit_uncertainty_count": uncertainty_count,
        "explicit_omission_count": omission_count,
        "wrong_known_value_count": wrong_known,
        "task_format_valid": 1,
        "trial_nonce_leakage_count": _nonce_leakage_count(answer, nonce),
    }
    return {
        "valid": True,
        "normalized_answer": answer,
        "response_sha256": sha256_bytes(response_bytes),
        "metrics": metrics,
        "error_codes": [],
    }


def nearest_rank(values: list[int], percentile: int) -> int:
    """Return the frozen one-based nearest-rank percentile."""
    if not values or percentile < 1 or percentile > 100:
        raise ValueError("PERCENTILE_INPUT_INVALID")
    ordered = sorted(values)
    rank = math.ceil((percentile / 100) * len(ordered))
    return ordered[rank - 1]


def integer_distribution(
    values: list[int],
    *,
    repeat_values: list[int],
) -> dict[str, Any]:
    """Summarize integer metrics without floating-point rounding."""
    if not values:
        return {"status": "not_applicable", "repeat_values": list(repeat_values)}
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in values
    ):
        raise ValueError("DISTRIBUTION_VALUE_INVALID")
    return {
        "status": "available",
        "minimum": min(values),
        "p50": nearest_rank(values, 50),
        "p95": nearest_rank(values, 95),
        "maximum": max(values),
        "repeat_values": list(repeat_values),
    }


_QUALITY_TOTAL_FIELDS = (
    "applicability_correct_numerator",
    "applicability_total_denominator",
    "required_state_correct_numerator",
    "required_state_applicable_denominator",
    "identifier_provenance_correct_numerator",
    "identifier_provenance_applicable_denominator",
)
_QUALITY_DIMENSIONS = {
    "applicability": (
        "applicability_correct_numerator",
        "applicability_total_denominator",
    ),
    "required_state": (
        "required_state_correct_numerator",
        "required_state_applicable_denominator",
    ),
    "identifier_provenance": (
        "identifier_provenance_correct_numerator",
        "identifier_provenance_applicable_denominator",
    ),
}
_COUNT_METRIC_FIELDS = (
    "stale_resurrection_count",
    "contradiction_count",
    "unsafe_prohibited_action_count",
    "explicit_uncertainty_count",
    "explicit_omission_count",
    "wrong_known_value_count",
    "task_format_valid",
    "trial_nonce_leakage_count",
    "summary_provider_call_count",
    "downstream_provider_call_count",
    "blocked_retry_count",
    "blocked_fallback_count",
)
_TELEMETRY_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_miss_tokens",
    "quota",
    "monetary_cost",
)
_REVISION_ORDER = {"baseline": 0, "candidate": 1}


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _revision_label(record: dict[str, Any]) -> str | None:
    if not isinstance(record, dict):
        return None
    label = record.get("revision_label")
    if label in _REVISION_ORDER:
        return str(label)
    revision = record.get("revision_sha")
    if revision == BASELINE_SHA:
        return "baseline"
    if revision == CANDIDATE_SHA:
        return "candidate"
    return None


def _sample_sort_key(
    sample: dict[str, Any], scenario_order: list[str]
) -> tuple[int, int, int, int, str]:
    label = _revision_label(sample)
    scenario = sample.get("scenario_id")
    try:
        scenario_index = scenario_order.index(scenario)
    except ValueError:
        scenario_index = len(scenario_order)
    return (
        _REVISION_ORDER.get(str(label), len(_REVISION_ORDER)),
        scenario_index,
        int(sample.get("cycle", -1)),
        int(sample.get("repeat", -1)),
        str(sample.get("question_id", "")),
    )


def _fraction_document(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _repeat_distribution(
    rows: list[dict[str, Any]],
    value_field: str,
) -> dict[str, Any]:
    values: list[int] = []
    by_repeat: dict[int, list[int]] = {}
    for row in rows:
        value = row.get(value_field)
        if not _nonnegative_integer(value):
            continue
        values.append(value)
        repeat = row.get("repeat")
        if isinstance(repeat, int) and not isinstance(repeat, bool):
            by_repeat.setdefault(repeat, []).append(value)
    repeat_values = [
        nearest_rank(by_repeat[repeat], 50)
        for repeat in REPEATS
        if by_repeat.get(repeat)
    ]
    return integer_distribution(values, repeat_values=repeat_values)


def _unique_transaction_rows(
    rows: list[dict[str, Any]],
    value_field: str,
    *,
    include_cycle: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key: tuple[Any, ...] = (
            row.get("trajectory_id"),
            row.get("scenario_id"),
            row.get("repeat"),
            _revision_label(row),
        )
        if include_cycle:
            key += (row.get("cycle"),)
        value = row.get(value_field)
        previous = grouped.get(key)
        if previous is not None and previous.get(value_field) != value:
            raise ValueError("TRANSACTION_LATENCY_INCONSISTENT")
        grouped[key] = row
    return list(grouped.values())


def _cell_latency(call_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = [row for row in call_rows if row.get("call_role") == "summary"]
    downstream = [row for row in call_rows if row.get("call_role") == "downstream"]
    trajectory = _unique_transaction_rows(
        call_rows,
        "trajectory_duration_ns",
        include_cycle=False,
    )
    return {
        "luna_call": _repeat_distribution(summary, "duration_ns"),
        "sol_call": _repeat_distribution(downstream, "duration_ns"),
        "compaction_transaction": _repeat_distribution(
            _unique_transaction_rows(summary, "transaction_duration_ns"),
            "transaction_duration_ns",
        ),
        "downstream_transaction": _repeat_distribution(
            _unique_transaction_rows(downstream, "transaction_duration_ns"),
            "transaction_duration_ns",
        ),
        "full_trajectory": _repeat_distribution(
            trajectory,
            "trajectory_duration_ns",
        ),
    }


def _telemetry_rollup(call_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in _TELEMETRY_FIELDS:
        status_field = f"{field}_status"
        status_counts: dict[str, int] = {}
        available_rows: list[dict[str, Any]] = []
        for row in call_rows:
            status = row.get(status_field)
            if not isinstance(status, str) or not status:
                status = "missing"
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "available":
                if not _nonnegative_integer(row.get(field)):
                    raise ValueError("TELEMETRY_VALUE_INVALID")
                available_rows.append(row)
        result[field] = {
            "distribution": _repeat_distribution(available_rows, field),
            "status_counts": dict(sorted(status_counts.items())),
        }
    return result


def _derive_sample_call_counts(
    sample: dict[str, Any],
    calls: list[dict[str, Any]],
    attempt_totals: dict[str, Any],
) -> None:
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        return
    label = _revision_label(sample)
    matching = [
        call
        for call in calls
        if _revision_label(call) == label
        and call.get("scenario_id") == sample.get("scenario_id")
        and call.get("repeat") == sample.get("repeat")
        and call.get("cycle") == sample.get("cycle")
    ]
    metrics.setdefault(
        "summary_provider_call_count",
        sum(call.get("call_role") == "summary" for call in matching),
    )
    metrics.setdefault(
        "downstream_provider_call_count",
        sum(call.get("call_role") == "downstream" for call in matching),
    )
    if attempt_totals.get("blocked_retry") == 0:
        metrics.setdefault("blocked_retry_count", 0)
    if attempt_totals.get("blocked_fallback") == 0:
        metrics.setdefault("blocked_fallback_count", 0)


def _build_cells(
    samples: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    scenario_order: list[str],
    errors: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for sample in samples:
        label = _revision_label(sample)
        scenario_id = sample.get("scenario_id")
        cycle = sample.get("cycle")
        question_id = sample.get("question_id")
        if (
            label not in _REVISION_ORDER
            or scenario_id not in scenario_order
            or cycle not in CHECKPOINTS
            or not isinstance(question_id, str)
        ):
            errors.append("SCORE_SAMPLE_INVALID")
            continue
        grouped.setdefault(
            (label, str(scenario_id), int(cycle), question_id), []
        ).append(sample)
    cells: list[dict[str, Any]] = []
    for label in ("baseline", "candidate"):
        for scenario_id in scenario_order:
            for cycle in CHECKPOINTS:
                expected_question = (
                    str(
                        next(
                            sample.get("question_id")
                            for sample in samples
                            if sample.get("scenario_id") == scenario_id
                        )
                    )
                    if any(
                        sample.get("scenario_id") == scenario_id for sample in samples
                    )
                    else ""
                )
                rows = grouped.get((label, scenario_id, cycle, expected_question), [])
                repeat_map = {row.get("repeat"): row for row in rows}
                complete = (
                    len(rows) == 3
                    and set(repeat_map) == set(REPEATS)
                    and all(row.get("valid") is True for row in rows)
                    and all(
                        _nonnegative_integer(row.get("metrics", {}).get(field))
                        for row in rows
                        for field in (*_QUALITY_TOTAL_FIELDS, *_COUNT_METRIC_FIELDS)
                    )
                )
                if rows and not complete:
                    errors.append("INCOMPLETE_SCORE_CELL")
                quality: dict[str, dict[str, int]] | None = None
                count_metrics: dict[str, dict[str, Any]] = {}
                if complete:
                    ordered = [repeat_map[repeat] for repeat in REPEATS]
                    quality = {
                        name: {
                            "numerator": sum(
                                row["metrics"][numerator] for row in ordered
                            ),
                            "denominator": sum(
                                row["metrics"][denominator] for row in ordered
                            ),
                        }
                        for name, (
                            numerator,
                            denominator,
                        ) in _QUALITY_DIMENSIONS.items()
                    }
                    if any(value["denominator"] <= 0 for value in quality.values()):
                        errors.append("SAMPLE_METRIC_INVALID")
                        complete = False
                        quality = None
                    count_metrics = {
                        field: {
                            "sum": sum(row["metrics"][field] for row in ordered),
                            "repeat_values": [row["metrics"][field] for row in ordered],
                        }
                        for field in _COUNT_METRIC_FIELDS
                    }
                cell_calls = [
                    call
                    for call in calls
                    if _revision_label(call) == label
                    and call.get("scenario_id") == scenario_id
                    and call.get("cycle") == cycle
                ]
                trajectory_calls = [
                    call
                    for call in calls
                    if _revision_label(call) == label
                    and call.get("scenario_id") == scenario_id
                ]
                latency = _cell_latency(cell_calls)
                latency["full_trajectory"] = _repeat_distribution(
                    _unique_transaction_rows(
                        trajectory_calls,
                        "trajectory_duration_ns",
                        include_cycle=False,
                    ),
                    "trajectory_duration_ns",
                )
                cells.append({
                    "revision_label": label,
                    "scenario_id": scenario_id,
                    "cycle": cycle,
                    "question_id": expected_question,
                    "status": "complete" if complete else "incomplete",
                    "valid_repeats": sorted(
                        int(repeat)
                        for repeat, row in repeat_map.items()
                        if isinstance(repeat, int) and row.get("valid") is True
                    ),
                    "quality": quality,
                    "count_metrics": count_metrics,
                    "latency": latency,
                    "telemetry": _telemetry_rollup(cell_calls),
                })
    return cells


def _build_revision_aggregates(
    cells: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    revisions: list[dict[str, Any]] = []
    for label in ("baseline", "candidate"):
        revision_cells = [cell for cell in cells if cell["revision_label"] == label]
        complete_cells = [
            cell for cell in revision_cells if cell.get("status") == "complete"
        ]
        macro: dict[str, Any] = {}
        pooled: dict[str, Any] = {}
        if len(complete_cells) == 15:
            for name in _QUALITY_DIMENSIONS:
                ratios = [
                    Fraction(
                        cell["quality"][name]["numerator"],
                        cell["quality"][name]["denominator"],
                    )
                    for cell in complete_cells
                ]
                value = sum(ratios, Fraction()) / len(ratios)
                macro[name] = {
                    **_fraction_document(value),
                    "complete_cell_count": len(ratios),
                }
                pooled[name] = {
                    "numerator": sum(
                        cell["quality"][name]["numerator"] for cell in complete_cells
                    ),
                    "denominator": sum(
                        cell["quality"][name]["denominator"] for cell in complete_cells
                    ),
                }
        revision_calls = [call for call in calls if _revision_label(call) == label]
        revisions.append({
            "revision_label": label,
            "status": "complete" if len(complete_cells) == 15 else "incomplete",
            "complete_cell_count": len(complete_cells),
            "macro_quality": macro,
            "pooled_quality": pooled,
            "latency": _cell_latency(revision_calls),
            "telemetry": _telemetry_rollup(revision_calls),
        })
    return revisions


def _incomplete_scores(code: str) -> dict[str, Any]:
    return {
        "schema_version": SCORES_SCHEMA,
        "status": REPORT_INCOMPLETE,
        "error_codes": [code],
        "samples": [],
        "cells": [],
        "revisions": [],
        "quality_totals": {
            "valid_samples": 0,
            **{field: 0 for field in _QUALITY_TOTAL_FIELDS},
        },
    }


def score_raw_evidence(
    raw_bytes: bytes,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    """Total pure scorer over finalized allowlisted raw evidence."""
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return _incomplete_scores("RAW_EVIDENCE_MALFORMED")
    if not isinstance(raw, dict) or raw.get("schema_version") != RAW_SCHEMA:
        return _incomplete_scores("RAW_EVIDENCE_SCHEMA_INVALID")
    samples = raw.get("samples")
    calls = raw.get("calls", [])
    attempt_totals = raw.get("attempt_totals", {})
    if (
        not isinstance(samples, list)
        or not isinstance(calls, list)
        or not isinstance(attempt_totals, dict)
    ):
        return _incomplete_scores("RAW_EVIDENCE_SCHEMA_INVALID")
    error_codes: list[str] = []
    valid_calls = [row for row in calls if isinstance(row, dict)]
    if len(valid_calls) != len(calls):
        error_codes.append("RAW_CALL_INVALID")
    seen: set[str] = set()
    scored_samples: list[dict[str, Any]] = []
    totals = {
        "valid_samples": 0,
        **{field: 0 for field in _QUALITY_TOTAL_FIELDS},
    }
    for sample in samples:
        if not isinstance(sample, dict):
            error_codes.append("RAW_SAMPLE_INVALID")
            continue
        key_hash = sample.get("sampling_key_sha256")
        if not isinstance(key_hash, str):
            error_codes.append("MISSING_SAMPLE_KEY")
            continue
        if key_hash in seen:
            error_codes.append("DUPLICATE_SAMPLE_KEY")
            continue
        seen.add(key_hash)
        scored_sample = json.loads(json.dumps(sample))
        scored_sample.setdefault("revision_label", _revision_label(scored_sample))
        _derive_sample_call_counts(scored_sample, valid_calls, attempt_totals)
        scored_samples.append(scored_sample)
        if scored_sample.get("valid") is True and isinstance(
            scored_sample.get("metrics"), dict
        ):
            totals["valid_samples"] += 1
            for field in _QUALITY_TOTAL_FIELDS:
                value = scored_sample["metrics"].get(field)
                if _nonnegative_integer(value):
                    totals[field] += value
                else:
                    error_codes.append("SAMPLE_METRIC_INVALID")
    if len(seen) != EXPECTED_DOWNSTREAM_ATTEMPTS:
        error_codes.append("MISSING_SAMPLE_KEY")
    if any(sample.get("valid") is not True for sample in scored_samples):
        error_codes.append("INVALID_SAMPLE")
    scenario_order = list(fixture.get("scenario_order", []))
    try:
        scored_samples.sort(key=lambda item: _sample_sort_key(item, scenario_order))
        cells = _build_cells(
            scored_samples,
            valid_calls,
            scenario_order,
            error_codes,
        )
        revisions = _build_revision_aggregates(
            cells,
            valid_calls,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        error_codes.append("AGGREGATION_INPUT_INVALID")
        cells = []
        revisions = []
    exact_attempts = (
        len(valid_calls) == EXPECTED_TOTAL_ATTEMPTS
        and sum(
            isinstance(row, dict) and row.get("call_role") == "summary"
            for row in valid_calls
        )
        == EXPECTED_SUMMARY_ATTEMPTS
        and sum(
            isinstance(row, dict) and row.get("call_role") == "downstream"
            for row in valid_calls
        )
        == EXPECTED_DOWNSTREAM_ATTEMPTS
        and attempt_totals.get("actual") == EXPECTED_TOTAL_ATTEMPTS
        and attempt_totals.get("expected") == EXPECTED_TOTAL_ATTEMPTS
        and attempt_totals.get("blocked_retry") == 0
        and attempt_totals.get("blocked_fallback") == 0
    )
    if raw.get("status") == "complete" and not exact_attempts:
        error_codes.append("ATTEMPT_ACCOUNTING_INVALID")
    complete = (
        raw.get("status") == "complete"
        and len(seen) == EXPECTED_DOWNSTREAM_ATTEMPTS
        and totals["valid_samples"] == EXPECTED_DOWNSTREAM_ATTEMPTS
        and exact_attempts
        and len(cells) == 30
        and all(cell["status"] == "complete" for cell in cells)
        and all(item["status"] == "complete" for item in revisions)
        and not error_codes
    )
    return {
        "schema_version": SCORES_SCHEMA,
        "status": REPORT_COMPLETE if complete else REPORT_INCOMPLETE,
        "error_codes": sorted(set(error_codes)),
        "fixture_schema": fixture.get("schema_version"),
        "samples": scored_samples,
        "quality_totals": totals,
        "cells": cells,
        "revisions": revisions,
        "telemetry_availability": {
            "cache_breakdown": "not_exposed_by_adapter",
            "quota": (raw.get("quota") or {}).get("status", "missing")
            if isinstance(raw.get("quota"), dict)
            else "missing",
            "monetary_cost": (raw.get("cost") or {}).get("status", "missing")
            if isinstance(raw.get("cost"), dict)
            else "missing",
        },
    }


def score_sample_for_comparison(
    sample: dict[str, Any],
    *,
    revision_label: str,
    revision_hashes: dict[str, str],
) -> dict[str, Any]:
    """Bind one scored sample to its revision-derived identity fields."""
    result = json.loads(json.dumps(sample))
    revision_sha = BASELINE_SHA if revision_label == "baseline" else CANDIDATE_SHA
    result["revision_label"] = revision_label
    result["sampling_key"]["revision_sha"] = revision_sha
    result["sampling_key_sha256"] = sha256_bytes(
        canonical_bytes(result["sampling_key"])
    )
    identity = {
        "comparison_identity_sha256": result.get("comparison_identity_sha256"),
        "revision_sha": revision_sha,
        "revision_archive_sha256": revision_hashes["archive"],
        "product_tree_sha256": revision_hashes["tree"],
        "compressor_blob_sha256": revision_hashes["blob"],
    }
    result["score_identity_document"] = identity
    result["score_identity_sha256"] = sha256_bytes(canonical_bytes(identity))
    return result


def _pairing_key(sample: dict[str, Any]) -> bytes:
    key = dict(sample.get("sampling_key") or {})
    key.pop("revision_sha", None)
    return canonical_bytes(key)


def _numeric_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, int]:
    baseline_metrics = baseline.get("metrics") or {}
    candidate_metrics = candidate.get("metrics") or {}
    fields = sorted(set(baseline_metrics) & set(candidate_metrics))
    return {
        field: candidate_metrics[field] - baseline_metrics[field]
        for field in fields
        if isinstance(baseline_metrics[field], int)
        and not isinstance(baseline_metrics[field], bool)
        and isinstance(candidate_metrics[field], int)
        and not isinstance(candidate_metrics[field], bool)
    }


def _score_identity_pair_valid(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    keys = {
        "comparison_identity_sha256",
        "revision_sha",
        "revision_archive_sha256",
        "product_tree_sha256",
        "compressor_blob_sha256",
    }
    baseline_identity = baseline.get("score_identity_document")
    candidate_identity = candidate.get("score_identity_document")
    if (
        not isinstance(baseline_identity, dict)
        or not isinstance(candidate_identity, dict)
        or set(baseline_identity) != keys
        or set(candidate_identity) != keys
        or baseline_identity.get("revision_sha") != BASELINE_SHA
        or candidate_identity.get("revision_sha") != CANDIDATE_SHA
        or baseline_identity.get("comparison_identity_sha256")
        != baseline.get("comparison_identity_sha256")
        or candidate_identity.get("comparison_identity_sha256")
        != candidate.get("comparison_identity_sha256")
        or baseline.get("score_identity_sha256")
        != sha256_bytes(canonical_bytes(baseline_identity))
        or candidate.get("score_identity_sha256")
        != sha256_bytes(canonical_bytes(candidate_identity))
    ):
        return False
    baseline_common = dict(baseline_identity)
    candidate_common = dict(candidate_identity)
    for key in (
        "revision_sha",
        "revision_archive_sha256",
        "product_tree_sha256",
        "compressor_blob_sha256",
    ):
        baseline_common.pop(key)
        candidate_common.pop(key)
    return baseline_common == candidate_common


def _distribution_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, int] | None:
    if baseline.get("status") != "available" or candidate.get("status") != "available":
        return None
    fields = ("minimum", "p50", "p95", "maximum")
    if not all(
        _nonnegative_integer(baseline.get(field))
        and _nonnegative_integer(candidate.get(field))
        for field in fields
    ):
        return None
    return {field: int(candidate[field]) - int(baseline[field]) for field in fields}


def _cell_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    quality: dict[str, Any] = {}
    for name in _QUALITY_DIMENSIONS:
        baseline_value = baseline["quality"][name]
        candidate_value = candidate["quality"][name]
        ratio = Fraction(
            candidate_value["numerator"], candidate_value["denominator"]
        ) - Fraction(baseline_value["numerator"], baseline_value["denominator"])
        quality[name] = {
            "numerator_delta": candidate_value["numerator"]
            - baseline_value["numerator"],
            "denominator_delta": candidate_value["denominator"]
            - baseline_value["denominator"],
            "ratio_delta": _fraction_document(ratio),
            "positive_direction": "more correct",
        }
    counts = {
        field: candidate["count_metrics"][field]["sum"]
        - baseline["count_metrics"][field]["sum"]
        for field in _COUNT_METRIC_FIELDS
    }
    latency = {
        name: delta
        for name in baseline["latency"]
        if (
            delta := _distribution_delta(
                baseline["latency"][name], candidate["latency"][name]
            )
        )
        is not None
    }
    telemetry: dict[str, Any] = {}
    for field in _TELEMETRY_FIELDS:
        delta = _distribution_delta(
            baseline["telemetry"][field]["distribution"],
            candidate["telemetry"][field]["distribution"],
        )
        if delta is not None:
            telemetry[field] = delta
    return {
        "scenario_id": baseline["scenario_id"],
        "cycle": baseline["cycle"],
        "question_id": baseline["question_id"],
        "status": "COMPARABLE",
        "direction": "candidate - baseline",
        "quality": quality,
        "count_metrics": counts,
        "latency": latency,
        "telemetry": telemetry,
    }


def _revision_delta(revisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {item.get("revision_label"): item for item in revisions}
    baseline = by_label.get("baseline")
    candidate = by_label.get("candidate")
    if (
        not isinstance(baseline, dict)
        or not isinstance(candidate, dict)
        or baseline.get("status") != "complete"
        or candidate.get("status") != "complete"
    ):
        return {}
    macro: dict[str, Any] = {}
    pooled: dict[str, Any] = {}
    for name in _QUALITY_DIMENSIONS:
        baseline_macro = baseline["macro_quality"][name]
        candidate_macro = candidate["macro_quality"][name]
        macro[name] = _fraction_document(
            Fraction(candidate_macro["numerator"], candidate_macro["denominator"])
            - Fraction(baseline_macro["numerator"], baseline_macro["denominator"])
        )
        baseline_pooled = baseline["pooled_quality"][name]
        candidate_pooled = candidate["pooled_quality"][name]
        pooled[name] = _fraction_document(
            Fraction(candidate_pooled["numerator"], candidate_pooled["denominator"])
            - Fraction(baseline_pooled["numerator"], baseline_pooled["denominator"])
        )
    latency = {
        name: delta
        for name in baseline["latency"]
        if (
            delta := _distribution_delta(
                baseline["latency"][name], candidate["latency"][name]
            )
        )
        is not None
    }
    telemetry: dict[str, Any] = {}
    for field in _TELEMETRY_FIELDS:
        delta = _distribution_delta(
            baseline["telemetry"][field]["distribution"],
            candidate["telemetry"][field]["distribution"],
        )
        if delta is not None:
            telemetry[field] = delta
    return {
        "direction": "candidate - baseline",
        "macro_quality": macro,
        "pooled_quality": pooled,
        "latency": latency,
        "telemetry": telemetry,
    }


def _compare_score_document(scores: Any) -> dict[str, Any]:
    """Total pure matched comparison over score identities."""
    if not isinstance(scores, dict) or scores.get("schema_version") != SCORES_SCHEMA:
        return {
            "schema_version": COMPARISON_SCHEMA,
            "status": REPORT_INCOMPLETE,
            "error_codes": ["SCORE_DOCUMENT_SCHEMA_INVALID"],
            "pairs": [],
            "cell_deltas": [],
            "revision_deltas": {},
        }
    samples = scores.get("samples")
    if not isinstance(samples, list):
        return {
            "schema_version": COMPARISON_SCHEMA,
            "status": REPORT_INCOMPLETE,
            "error_codes": ["SCORE_DOCUMENT_SCHEMA_INVALID"],
            "pairs": [],
            "cell_deltas": [],
            "revision_deltas": {},
        }
    grouped: dict[bytes, dict[str, dict[str, Any]]] = {}
    errors: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("revision_label") not in {
            "baseline",
            "candidate",
        }:
            errors.append("SCORE_SAMPLE_INVALID")
            continue
        group = grouped.setdefault(_pairing_key(sample), {})
        label = sample["revision_label"]
        if label in group:
            errors.append("DUPLICATE_COMPARISON_SAMPLE")
        else:
            group[label] = sample
    pairs: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        baseline = group.get("baseline")
        candidate = group.get("candidate")
        if baseline is None or candidate is None:
            errors.append("MISSING_COMPARISON_SAMPLE")
            pairs.append({
                "pairing_key_sha256": sha256_bytes(key),
                "status": "NOT_COMPARABLE",
                "delta": None,
            })
            continue
        comparable = (
            baseline.get("valid") is True
            and candidate.get("valid") is True
            and baseline.get("comparison_identity_sha256")
            == candidate.get("comparison_identity_sha256")
            and _score_identity_pair_valid(baseline, candidate)
        )
        if not comparable:
            errors.append("COMPARISON_IDENTITY_MISMATCH")
        pairs.append({
            "pairing_key_sha256": sha256_bytes(key),
            "status": "COMPARABLE" if comparable else "NOT_COMPARABLE",
            "delta": _numeric_delta(baseline, candidate) if comparable else None,
        })
    if len(pairs) != EXPECTED_DOWNSTREAM_ATTEMPTS // 2:
        errors.append("MISSING_COMPARISON_SAMPLE")
    cells = scores.get("cells")
    revisions = scores.get("revisions")
    cell_deltas: list[dict[str, Any]] = []
    if isinstance(cells, list):
        grouped_cells: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
        for cell in cells:
            if not isinstance(cell, dict):
                errors.append("SCORE_CELL_INVALID")
                continue
            key = (
                str(cell.get("scenario_id")),
                int(cell.get("cycle", -1)),
                str(cell.get("question_id")),
            )
            grouped_cells.setdefault(key, {})[str(cell.get("revision_label"))] = cell
        for key in sorted(grouped_cells):
            group = grouped_cells[key]
            baseline_cell = group.get("baseline")
            candidate_cell = group.get("candidate")
            if (
                baseline_cell is None
                or candidate_cell is None
                or baseline_cell.get("status") != "complete"
                or candidate_cell.get("status") != "complete"
            ):
                errors.append("INCOMPLETE_COMPARISON_CELL")
                continue
            try:
                cell_deltas.append(_cell_delta(baseline_cell, candidate_cell))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                errors.append("SCORE_CELL_INVALID")
    else:
        errors.append("SCORE_DOCUMENT_SCHEMA_INVALID")
    revision_deltas = _revision_delta(revisions) if isinstance(revisions, list) else {}
    if scores.get("status") == REPORT_COMPLETE and (
        len(cell_deltas) != 15 or not revision_deltas
    ):
        errors.append("INCOMPLETE_AGGREGATION")
    complete = (
        scores.get("status") == REPORT_COMPLETE
        and len(pairs) == EXPECTED_DOWNSTREAM_ATTEMPTS // 2
        and all(pair["status"] == "COMPARABLE" for pair in pairs)
        and len(cell_deltas) == 15
        and bool(revision_deltas)
        and not errors
    )
    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": REPORT_COMPLETE if complete else REPORT_INCOMPLETE,
        "error_codes": sorted(set(errors)),
        "delta_direction": "candidate - baseline",
        "pairs": pairs,
        "cell_deltas": cell_deltas,
        "revision_deltas": revision_deltas,
        "sample_rows": samples,
        "cells": cells if isinstance(cells, list) else [],
        "revisions": revisions if isinstance(revisions, list) else [],
        "telemetry_availability": scores.get("telemetry_availability", {}),
    }


def compare_score_document(scores: Any) -> dict[str, Any]:
    """Return a schema-valid incomplete comparison for every malformed input."""
    try:
        return _compare_score_document(scores)
    except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError):
        return {
            "schema_version": COMPARISON_SCHEMA,
            "status": REPORT_INCOMPLETE,
            "error_codes": ["SCORE_DOCUMENT_SCHEMA_INVALID"],
            "pairs": [],
            "cell_deltas": [],
            "revision_deltas": {},
        }


def compare_score_bytes(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        value = None
    return compare_score_document(value)


class TerminalBenchmarkStop(BaseException):
    """Uncatchable-by-product terminal stop used at the live wire boundary."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class LiveAuxiliaryGuard:
    """Install the single live pre-wire policy on the frozen product seam."""

    def __init__(
        self,
        expected_calls: list[dict[str, Any]],
        *,
        auxiliary_module: Any,
        compressor_module: Any,
        global_actual_attempts: int,
        attempted_logical_call_ids: set[str],
        append_journal: Callable[[dict[str, Any]], Any],
        trajectory_started_ns: int,
        trajectory_deadline_ns: int,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        attempt_id: str = "attempt-provider-free-test",
    ) -> None:
        self._expected_calls = expected_calls
        self._expected_by_id = {
            str(item.get("logical_call_id")): item for item in expected_calls
        }
        if len(self._expected_by_id) != len(expected_calls):
            raise ValueError("SCHEDULE_INVALID")
        self._auxiliary = auxiliary_module
        self._compressor = compressor_module
        self._append_journal = append_journal
        self._monotonic_ns = monotonic_ns
        self._trajectory_started_ns = trajectory_started_ns
        self._trajectory_deadline_ns = trajectory_deadline_ns
        self._attempt_id = attempt_id
        self._active_call: dict[str, Any] | None = None
        self._active_snapshot_sha256: str | None = None
        self._call_index = 0
        self._attempted = set(attempted_logical_call_ids)
        self.actual_attempts = global_actual_attempts
        self.blocked_retry_count = 0
        self.blocked_fallback_count = 0
        self.unscheduled_logical_call_count = 0
        self.route_mismatch_count = 0
        self.observed_routes: list[tuple[str, str, str]] = []
        self.call_records: list[dict[str, Any]] = []
        self._original_relay = auxiliary_module._relay_sync_completion
        self._original_create = auxiliary_module._CodexCompletionsAdapter.create

    @property
    def attempted_logical_call_ids(self) -> set[str]:
        return set(self._attempted)

    @property
    def call_index(self) -> int:
        return self._call_index

    def next_expected_call(self) -> dict[str, Any]:
        if self._call_index >= len(self._expected_calls):
            raise TerminalBenchmarkStop("UNSCHEDULED_LOGICAL_CALL")
        return self._expected_calls[self._call_index]

    def install(self) -> None:
        guard = self

        def guarded_create(adapter: Any, **kwargs: Any) -> Any:
            return guard.guarded_create(adapter, **kwargs)

        self._auxiliary._relay_sync_completion = self.guarded_relay
        self._auxiliary._CodexCompletionsAdapter.create = guarded_create
        self._compressor.ContextCompressor._fallback_to_main_for_compression = (
            self.block_compressor_fallback
        )
        for name in ("run_codex_stream", "main_responses_create"):
            if callable(getattr(self._auxiliary, name, None)):
                setattr(self._auxiliary, name, self.block_main_agent_transport)

    @contextmanager
    def activate(
        self, call: dict[str, Any], *, snapshot_sha256: str | None
    ) -> Iterator[None]:
        previous_call = self._active_call
        previous_snapshot = self._active_snapshot_sha256
        self._active_call = call
        self._active_snapshot_sha256 = snapshot_sha256
        try:
            yield
        finally:
            self._active_call = previous_call
            self._active_snapshot_sha256 = previous_snapshot

    def _scheduled_call(self) -> dict[str, Any]:
        call = self._active_call
        logical_call_id = str((call or {}).get("logical_call_id"))
        expected = self._expected_by_id.get(logical_call_id)
        if expected is None:
            self.unscheduled_logical_call_count += 1
            raise TerminalBenchmarkStop("UNSCHEDULED_LOGICAL_CALL")
        current = (
            self._expected_calls[self._call_index]
            if self._call_index < len(self._expected_calls)
            else None
        )
        if current is not expected and logical_call_id not in self._attempted:
            self.unscheduled_logical_call_count += 1
            raise TerminalBenchmarkStop("UNSCHEDULED_LOGICAL_CALL")
        return expected

    def guarded_relay(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        provider: str | None = None,
        api_mode: str | None = None,
        create: Callable[..., Any] | None = None,
        fallback_kind: str | None = None,
    ) -> Any:
        expected = self._scheduled_call()
        observed_route = (provider, api_mode, kwargs.get("model"))
        if (
            fallback_kind is not None
            or observed_route != (PROVIDER, API_MODE, expected.get("model"))
            or type(client).__name__ != "CodexAuxiliaryClient"
            or type(client).__module__ != "agent.auxiliary_client"
        ):
            self.blocked_fallback_count += 1
            raise TerminalBenchmarkStop("FALLBACK_ROUTE_BLOCKED")
        return self._original_relay(
            client,
            kwargs,
            provider=provider,
            api_mode=api_mode,
            create=create,
        )

    def _unavailable_usage(self) -> dict[str, str]:
        return {
            "cache_read_tokens": "not_exposed_by_adapter",
            "cache_read_tokens_status": "not_exposed",
            "cache_miss_tokens": "not_exposed_by_adapter",
            "cache_miss_tokens_status": "not_exposed",
            "quota": "not_available",
            "quota_status": "not_available",
            "monetary_cost": "not_available",
            "monetary_cost_status": "not_available",
        }

    def guarded_create(self, adapter: Any, **kwargs: Any) -> Any:
        expected = self._scheduled_call()
        call = self._active_call or {}
        if (
            call.get("provider") != PROVIDER
            or call.get("api_mode") != API_MODE
            or call.get("model") != kwargs.get("model")
            or call.get("model") not in {SUMMARY_MODEL, DOWNSTREAM_MODEL}
            or type(adapter).__name__ != "_CodexCompletionsAdapter"
            or type(adapter).__module__ != "agent.auxiliary_client"
        ):
            self.route_mismatch_count += 1
            raise TerminalBenchmarkStop("ROUTE_MISMATCH")
        if (
            any(
                key in kwargs
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "seed",
                    "max_tokens",
                    "max_output_tokens",
                    "max_completion_tokens",
                )
            )
            or kwargs.get("timeout") != TIMEOUT_SECONDS
        ):
            raise TerminalBenchmarkStop("WIRE_REQUEST_IDENTITY_MISMATCH")
        if getattr(adapter._client, "max_retries", None) != 0:
            raise TerminalBenchmarkStop("WIRE_REQUEST_IDENTITY_MISMATCH")
        base_url = str(getattr(adapter._client, "base_url", "")).rstrip("/")
        if base_url != "https://chatgpt.com/backend-api/codex":
            self.route_mismatch_count += 1
            raise TerminalBenchmarkStop("ROUTE_MISMATCH")
        logical_call_id = str(call["logical_call_id"])
        if logical_call_id in self._attempted:
            self.blocked_retry_count += 1
            raise TerminalBenchmarkStop("HIDDEN_RETRY_BLOCKED")
        if self.actual_attempts >= 300:
            raise TerminalBenchmarkStop("GLOBAL_ATTEMPT_CAP_REACHED")
        now = self._monotonic_ns()
        if now >= self._trajectory_deadline_ns:
            raise TerminalBenchmarkStop("TRAJECTORY_DEADLINE_REACHED")
        self._attempted.add(logical_call_id)
        self.actual_attempts += 1
        self.observed_routes.append((PROVIDER, API_MODE, str(call["model"])))
        self._append_journal({
            "attempt_id": self._attempt_id,
            "logical_call_id": logical_call_id,
            "schedule_position": call["schedule_position"],
            "state": "PLANNED",
            "actual_attempt_ordinal": 1,
        })
        started = self._monotonic_ns()
        prompt_sha256 = sha256_bytes(canonical_bytes(kwargs.get("messages", [])))
        try:
            response = self._original_create(adapter, **kwargs)
        except Exception:
            duration = max(0, self._monotonic_ns() - started)
            self._append_journal({
                "attempt_id": self._attempt_id,
                "logical_call_id": logical_call_id,
                "schedule_position": call["schedule_position"],
                "state": "FAILED",
                "error_code": "PROVIDER_CALL_FAILED",
                "duration_ns": duration,
            })
            raise
        duration = max(0, self._monotonic_ns() - started)
        if duration > 300_000_000_000:
            raise TerminalBenchmarkStop("PROVIDER_ATTEMPT_DEADLINE_REACHED")
        try:
            response_bytes = response.choices[0].message.content.encode("utf-8")
        except (AttributeError, IndexError, TypeError, UnicodeError):
            raise TerminalBenchmarkStop("MALFORMED_MODEL_RESPONSE") from None
        echoed_model = getattr(response, "model", None)
        if echoed_model != call["model"]:
            raise TerminalBenchmarkStop("ADAPTER_REQUEST_ECHO_MISMATCH")
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        record = {
            "attempt_id": self._attempt_id,
            "logical_call_id": logical_call_id,
            "schedule_position": call["schedule_position"],
            "trajectory_id": call.get("trajectory_id"),
            "call_role": call["call_role"],
            "scenario_id": call["scenario_id"],
            "repeat": call["repeat"],
            "revision_label": call.get("revision_label"),
            "revision_sha": call["revision_sha"],
            "cycle": call["cycle"],
            "summary_pass_ordinal": call.get("summary_pass_ordinal"),
            "question_id": call.get("question_id"),
            "provider": PROVIDER,
            "api_mode": API_MODE,
            "requested_model": call["model"],
            "adapter_returned_model": echoed_model,
            "adapter_returned_model_status": "request_echo_not_provider_identity",
            "provider_raw_returned_model": "not_exposed_by_adapter",
            "provider_raw_returned_model_status": "not_exposed",
            "provider_raw_returned_model_reason": "CODEX_ADAPTER_DISCARDS_RAW_MODEL_IDENTITY",
            "effort_source": "model_alias",
            "explicit_reasoning_effort": None,
            "serialized_reasoning_payload": "not_exposed_by_adapter",
            "actual_attempt_ordinal": 1,
            "start_monotonic_ns": started - self._trajectory_started_ns,
            "duration_ns": duration,
            "transaction_duration_ns": duration,
            "trajectory_duration_ns": max(
                0, self._monotonic_ns() - self._trajectory_started_ns
            ),
            "prompt_sha256": prompt_sha256,
            "snapshot_sha256": self._active_snapshot_sha256,
            "response_sha256": sha256_bytes(response_bytes),
            "response_status": "complete",
            "error_code": None,
            "input_tokens": input_tokens,
            "input_tokens_status": "available"
            if _nonnegative_integer(input_tokens)
            else "not_exposed",
            "output_tokens": output_tokens,
            "output_tokens_status": "available"
            if _nonnegative_integer(output_tokens)
            else "not_exposed",
            "total_tokens": total_tokens,
            "total_tokens_status": "available"
            if _nonnegative_integer(total_tokens)
            else "not_exposed",
            **self._unavailable_usage(),
        }
        self.call_records.append(record)
        self._append_journal({
            "attempt_id": self._attempt_id,
            "logical_call_id": logical_call_id,
            "schedule_position": call["schedule_position"],
            "state": "COMPLETED",
            "response_sha256": record["response_sha256"],
            "duration_ns": duration,
        })
        if expected is self._expected_calls[self._call_index]:
            self._call_index += 1
        return response

    def record_compaction_transaction(self, cycle: int, duration_ns: int) -> None:
        if not _nonnegative_integer(duration_ns):
            raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
        for record in self.call_records:
            if record["call_role"] == "summary" and record["cycle"] == cycle:
                record["transaction_duration_ns"] = duration_ns

    def finish_trajectory(self, duration_ns: int) -> None:
        if not _nonnegative_integer(duration_ns):
            raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
        for record in self.call_records:
            record["trajectory_duration_ns"] = duration_ns

    def block_compressor_fallback(self, *_args: Any, **_kwargs: Any) -> None:
        self.blocked_fallback_count += 1
        raise TerminalBenchmarkStop("FALLBACK_TO_MAIN_BLOCKED")

    def block_main_agent_transport(self, *_args: Any, **_kwargs: Any) -> None:
        self.blocked_fallback_count += 1
        raise TerminalBenchmarkStop("MAIN_AGENT_TRANSPORT_BLOCKED")


def plan_resume(
    schedule_calls: list[dict[str, Any]],
    journal: list[dict[str, Any]],
    *,
    available_snapshot_ids: set[str],
) -> dict[str, Any]:
    """Return the one-shot safe resumption plan without changing a journal."""
    states: dict[str, list[str]] = {}
    for record in journal:
        states.setdefault(str(record.get("logical_call_id")), []).append(
            str(record.get("state"))
        )
    execute_once: list[dict[str, Any]] = []
    skipped_completed: list[str] = []
    invalid: list[dict[str, str]] = []
    for call in schedule_calls:
        logical_call_id = call["logical_call_id"]
        call_states = states.get(logical_call_id, [])
        if "COMPLETED" in call_states:
            skipped_completed.append(logical_call_id)
            continue
        if "PLANNED" in call_states:
            invalid.append({
                "logical_call_id": logical_call_id,
                "error_code": "OUTCOME_UNKNOWN",
            })
            continue
        if call["snapshot_id"] not in available_snapshot_ids:
            invalid.append({
                "logical_call_id": logical_call_id,
                "error_code": "SNAPSHOT_STATE_LOST",
            })
            continue
        execute_once.append(call)
    return {
        "execute_once": execute_once,
        "skipped_completed": skipped_completed,
        "invalid": invalid,
    }


def safe_observed_value(value: str) -> str:
    """Persist a safe token verbatim and hash all free-form observations."""
    if _SAFE_TOKEN_RE.fullmatch(value) and not _SECRET_PATTERN.search(value):
        return value
    return "sha256:" + sha256_bytes(value.encode("utf-8", errors="replace"))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


_FINAL_SOURCE_ARTIFACTS = (
    "evaluation/context_compression_tier_b.py",
    "evaluation/fixtures/context-compression-tier-b.json",
    "evaluation/context_compression_tier_b_runtime.sh",
    "evaluation/Containerfile.context-compression-tier-b",
)
_FINAL_INPUT_ARTIFACTS = (
    "source-manifest.json",
    "image-manifest.json",
    "execution-schedule.json",
)
_FINAL_OUTPUT_ARTIFACTS = (
    "context-compression-tier-b-raw.json",
    "context-compression-tier-b-scores.json",
    "context-compression-tier-b-comparison.json",
    "context-compression-tier-b-report.md",
)


def _final_artifact_paths(
    source_root: Path,
    input_root: Path,
    final_root: Path,
) -> dict[str, Path]:
    return {
        **{name: source_root / name for name in _FINAL_SOURCE_ARTIFACTS},
        **{name: input_root / name for name in _FINAL_INPUT_ARTIFACTS},
        **{name: final_root / name for name in _FINAL_OUTPUT_ARTIFACTS},
    }


def _required_hash(value: Any, code: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(code)
    return value


def _derive_final_status(
    artifacts: dict[str, Path],
) -> str:
    source_path = artifacts["source-manifest.json"]
    image_path = artifacts["image-manifest.json"]
    schedule_path = artifacts["execution-schedule.json"]
    source = _load_json(source_path, "FINAL_SOURCE_MANIFEST_INVALID")
    image = _load_json(image_path, "FINAL_IMAGE_MANIFEST_INVALID")
    schedule = _load_json(schedule_path, "FINAL_SCHEDULE_INVALID")
    raw = _load_json(
        artifacts["context-compression-tier-b-raw.json"],
        "FINAL_RAW_INVALID",
    )
    scores = _load_json(
        artifacts["context-compression-tier-b-scores.json"],
        "FINAL_SCORES_INVALID",
    )
    comparison = _load_json(
        artifacts["context-compression-tier-b-comparison.json"],
        "FINAL_COMPARISON_INVALID",
    )
    if (
        source.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or image.get("schema_version") != "context-compression-tier-b-image-manifest/1"
        or schedule.get("schema_version") != SCHEDULE_SCHEMA
        or raw.get("schema_version") != RAW_SCHEMA
        or scores.get("schema_version") != SCORES_SCHEMA
        or comparison.get("schema_version") != COMPARISON_SCHEMA
    ):
        raise ValueError("FINAL_ARTIFACT_SCHEMA_INVALID")
    files = source.get("files")
    if not isinstance(files, dict):
        raise ValueError("FINAL_SOURCE_FILES_INVALID")
    for name in _FINAL_SOURCE_ARTIFACTS:
        path = artifacts[name]
        if not path.is_file() or files.get(name) != _file_identity(path):
            raise ValueError("FINAL_SOURCE_FILE_DRIFT")
    archives = source.get("archives")
    if not isinstance(archives, dict) or set(archives) != {
        "baseline",
        "candidate",
        "evaluation-head",
    }:
        raise ValueError("FINAL_ARCHIVE_IDENTITIES_INVALID")
    for identity in archives.values():
        if not isinstance(identity, dict):
            raise ValueError("FINAL_ARCHIVE_IDENTITIES_INVALID")
        for field in ("archive", "manifest"):
            value = identity.get(field)
            if not isinstance(value, dict):
                raise ValueError("FINAL_ARCHIVE_IDENTITIES_INVALID")
            _required_hash(value.get("sha256"), "FINAL_ARCHIVE_IDENTITIES_INVALID")
    schedule_identity = _required_hash(
        schedule.get("schedule_sha256"), "FINAL_SCHEDULE_INVALID"
    )
    if (
        source.get("schedule_sha256") != schedule_identity
        or image.get("schedule_sha256") != schedule_identity
        or image.get("source_manifest_sha256") != sha256_bytes(source_path.read_bytes())
    ):
        raise ValueError("FINAL_CROSS_IDENTITY_MISMATCH")
    _required_hash(
        image.get("installed_distributions_sha256"),
        "FINAL_DISTRIBUTION_IDENTITY_INVALID",
    )
    try:
        report = artifacts["context-compression-tier-b-report.md"].read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError) as exc:
        raise ValueError("FINAL_REPORT_INVALID") from exc
    complete = (
        raw.get("status") == "complete"
        and scores.get("status") == REPORT_COMPLETE
        and comparison.get("status") == REPORT_COMPLETE
        and f"Status: `{REPORT_COMPLETE}`" in report
    )
    if not complete and f"Status: `{REPORT_INCOMPLETE}`" not in report:
        raise ValueError("FINAL_REPORT_STATUS_MISMATCH")
    return REPORT_COMPLETE if complete else REPORT_INCOMPLETE


def _checksum_payload(artifacts: dict[str, Path], marker: Path) -> bytes:
    paths = {**artifacts, "FINALIZED": marker}
    return "".join(
        f"{sha256_bytes(paths[name].read_bytes())}  {name}\n" for name in sorted(paths)
    ).encode("ascii")


def validate_finalized_evidence(
    source_root: Path,
    input_root: Path,
    final_root: Path,
) -> str:
    """Rehash the complete Stage 8 closure from the bytes named by the CLI."""
    artifacts = _final_artifact_paths(source_root, input_root, final_root)
    status = _derive_final_status(artifacts)
    marker = final_root / "FINALIZED"
    checksum = final_root / "FINALIZED.sha256"
    if marker.read_bytes() != status.encode("ascii") + b"\n":
        raise ValueError("FINAL_MARKER_DRIFT")
    expected = _checksum_payload(artifacts, marker)
    if checksum.read_bytes() != expected:
        raise ValueError("FINAL_CHECKSUM_DRIFT")
    return status


def finalize_evidence(
    source_root: Path,
    input_root: Path,
    final_root: Path,
) -> str:
    """Finalize existing Stage 8 files without reserializing their bytes."""
    artifacts = _final_artifact_paths(source_root, input_root, final_root)
    status = _derive_final_status(artifacts)
    marker = final_root / "FINALIZED"
    checksum = final_root / "FINALIZED.sha256"
    if marker.exists() or checksum.exists():
        raise ValueError("FINALIZATION_ALREADY_EXISTS")
    _atomic_write(marker, status.encode("ascii") + b"\n")
    _atomic_write(checksum, _checksum_payload(artifacts, marker))
    validate_finalized_evidence(source_root, input_root, final_root)
    return status


_TERMINAL_CODES = frozenset({
    "ADAPTER_REQUEST_ECHO_MISMATCH",
    "AUTH_ATTESTATION_INVALID",
    "BENCHMARK_TERMINAL_STOP",
    "CI_LIVE_COLLECTION_FORBIDDEN",
    "COLLECT_CANCELLED",
    "COLLECTION_ALREADY_EXISTS",
    "COLLECTION_STATE_INVALID",
    "COMPRESSION_ABORTED",
    "DUPLICATE_SAMPLE_KEY",
    "ROUTE_MISMATCH",
    "UNSCHEDULED_LOGICAL_CALL",
    "HIDDEN_RETRY_BLOCKED",
    "FALLBACK_ROUTE_BLOCKED",
    "FALLBACK_TO_MAIN_BLOCKED",
    "GLOBAL_ATTEMPT_CAP_REACHED",
    "LIVE_ACK_REQUIRED",
    "MAIN_AGENT_TRANSPORT_BLOCKED",
    "MALFORMED_MODEL_RESPONSE",
    "OUTCOME_UNKNOWN",
    "PROVIDER_ATTEMPT_DEADLINE_REACHED",
    "PROVIDER_OR_RUNTIME_FAILURE",
    "RUNTIME_CONFIG_INVALID",
    "RUNTIME_IMAGE_ID_INVALID",
    "RUNTIME_PREFLIGHT_FAILED",
    "SCHEDULE_INVALID",
    "SCHEDULE_INCOMPLETE",
    "SNAPSHOT_STATE_LOST",
    "SOURCE_RUNTIME_IDENTITY_MISMATCH",
    "TRAJECTORY_DEADLINE_REACHED",
    "WIRE_REQUEST_IDENTITY_MISMATCH",
})


def _write_adverse(output_root: Path, code: str) -> None:
    attempt_id = "attempt-" + uuid.uuid4().hex
    root = output_root / "adverse" / attempt_id
    record = {
        "schema_version": "context-compression-tier-b-adverse/1",
        "status": REPORT_INCOMPLETE,
        "attempt_id": attempt_id,
        "error_code": code,
    }
    record_bytes = json.dumps(record, indent=2, sort_keys=True).encode() + b"\n"
    report_bytes = f"Tier B collection stopped safely: {code}\n".encode("ascii")
    _atomic_write(root / "record.json", record_bytes)
    _atomic_write(root / "report.md", report_bytes)
    checksum = (
        f"{sha256_bytes(record_bytes)}  record.json\n"
        f"{sha256_bytes(report_bytes)}  report.md\n"
    ).encode("ascii")
    _atomic_write(root / "SHA256SUMS", checksum)


_PARTIAL_CALL_FIELDS = frozenset({
    "attempt_id",
    "logical_call_id",
    "schedule_position",
    "trajectory_id",
    "call_role",
    "scenario_id",
    "repeat",
    "revision_label",
    "revision_sha",
    "cycle",
    "summary_pass_ordinal",
    "question_id",
    "provider",
    "api_mode",
    "requested_model",
    "adapter_returned_model",
    "adapter_returned_model_status",
    "provider_raw_returned_model",
    "provider_raw_returned_model_status",
    "provider_raw_returned_model_reason",
    "effort_source",
    "explicit_reasoning_effort",
    "serialized_reasoning_payload",
    "actual_attempt_ordinal",
    "start_monotonic_ns",
    "duration_ns",
    "transaction_duration_ns",
    "trajectory_duration_ns",
    "prompt_sha256",
    "snapshot_sha256",
    "response_sha256",
    "response_status",
    "error_code",
    "input_tokens",
    "input_tokens_status",
    "output_tokens",
    "output_tokens_status",
    "total_tokens",
    "total_tokens_status",
    "cache_read_tokens",
    "cache_read_tokens_status",
    "cache_miss_tokens",
    "cache_miss_tokens_status",
    "quota",
    "quota_status",
    "monetary_cost",
    "monetary_cost_status",
})
_PARTIAL_SAMPLE_FIELDS = frozenset({
    "sampling_key",
    "sampling_key_sha256",
    "scenario_id",
    "repeat",
    "cycle",
    "question_id",
    "revision_label",
    "valid",
    "normalized_answer",
    "response_bytes",
    "response_sha256",
    "error_codes",
    "metrics",
    "comparison_identity_sha256",
    "score_identity_document",
    "score_identity_sha256",
})
_PARTIAL_SNAPSHOT_FIELDS = frozenset({
    "snapshot_id",
    "snapshot_sha256",
    "row_count",
    "compression_count",
    "dependency_key",
    "storage",
})


def _safe_partial_tree(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if _nonnegative_integer(value):
        return value
    if isinstance(value, str):
        if value == "[REDACTED]" or (
            _SAFE_TOKEN_RE.fullmatch(value) and not _SECRET_PATTERN.search(value)
        ):
            return value
        raise ValueError("PARTIAL_UNSAFE_VALUE")
    if isinstance(value, list):
        return [_safe_partial_tree(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not _SAFE_TOKEN_RE.fullmatch(key):
                raise ValueError("PARTIAL_UNSAFE_KEY")
            sanitized[key] = _safe_partial_tree(item)
        return sanitized
    raise ValueError("PARTIAL_UNSAFE_TYPE")


def _allowlisted_partial_records(
    values: Any, fields: frozenset[str]
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    records: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        record: dict[str, Any] = {}
        for key in sorted(set(value) & fields):
            try:
                record[key] = _safe_partial_tree(value[key])
            except ValueError:
                continue
        records.append(record)
    return records


def write_partial_raw_evidence(
    *,
    output_root: Path,
    state_path: Path,
    journal_path: Path,
    error_code: str,
) -> None:
    """Atomically preserve the sanitized durable subset of collection state."""
    code = error_code if error_code in _TERMINAL_CODES else "UNEXPECTED_INTERNAL_ERROR"
    state: dict[str, Any] = {}
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            state = loaded
    except (OSError, UnicodeError, json.JSONDecodeError):
        state = {}
    calls = _allowlisted_partial_records(state.get("calls"), _PARTIAL_CALL_FIELDS)
    samples = _allowlisted_partial_records(state.get("samples"), _PARTIAL_SAMPLE_FIELDS)
    snapshots = _allowlisted_partial_records(
        state.get("snapshots"), _PARTIAL_SNAPSHOT_FIELDS
    )
    attempt_ids = {
        value
        for value in state.get("attempt_ids", [])
        if isinstance(value, str) and re.fullmatch(r"attempt-[0-9a-f]{32}", value)
    }
    journal: list[dict[str, Any]] = []
    try:
        journal = _load_journal(journal_path)
    except TerminalBenchmarkStop:
        journal = []
    attempt_ids.update(
        str(item.get("attempt_id"))
        for item in journal
        if re.fullmatch(r"attempt-[0-9a-f]{32}", str(item.get("attempt_id")))
    )
    planned_ids = {
        str(item.get("logical_call_id"))
        for item in journal
        if item.get("state") == "PLANNED"
    }
    identity = {
        "provider": PROVIDER,
        "api_mode": API_MODE,
        "summarizer_model": SUMMARY_MODEL,
        "downstream_model": DOWNSTREAM_MODEL,
        "effort_source": "model_alias",
        "explicit_reasoning_effort": None,
        "timeout_seconds": TIMEOUT_SECONDS,
        "transient_retries": 0,
        "sdk_retries": 0,
        "global_attempt_ceiling": 300,
        "attempt_ids": sorted(attempt_ids),
    }
    raw = {
        "schema_version": RAW_SCHEMA,
        "status": "incomplete",
        "identity": identity,
        "production_readback": {
            "status": "not_finalized",
            "before_sha256": None,
            "after_sha256": None,
            "equal": None,
        },
        "isolation_proofs": {
            "production_accessed": False,
            "main_agent_transport_calls": 0,
        },
        "schedule_sha256": state.get("schedule_sha256")
        if isinstance(state.get("schedule_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", state["schedule_sha256"])
        else None,
        "expected_counts": {
            "trajectories": 18,
            "summary": EXPECTED_SUMMARY_ATTEMPTS,
            "downstream": EXPECTED_DOWNSTREAM_ATTEMPTS,
            "total": EXPECTED_TOTAL_ATTEMPTS,
        },
        "attempt_totals": {
            "actual": len(planned_ids),
            "expected": EXPECTED_TOTAL_ATTEMPTS,
            "summary": sum(item.get("call_role") == "summary" for item in calls),
            "downstream": sum(item.get("call_role") == "downstream" for item in calls),
            "blocked_retry": int(code == "HIDDEN_RETRY_BLOCKED"),
            "blocked_fallback": int(
                code
                in {
                    "FALLBACK_ROUTE_BLOCKED",
                    "FALLBACK_TO_MAIN_BLOCKED",
                    "MAIN_AGENT_TRANSPORT_BLOCKED",
                }
            ),
            "unscheduled_logical_call": int(code == "UNSCHEDULED_LOGICAL_CALL"),
            "route_mismatch": int(code == "ROUTE_MISMATCH"),
            "adapter_echo_mismatch": int(code == "ADAPTER_REQUEST_ECHO_MISMATCH"),
            "malformed": sum(item.get("valid") is not True for item in samples),
            "timeout": int(
                code
                in {
                    "PROVIDER_ATTEMPT_DEADLINE_REACHED",
                    "TRAJECTORY_DEADLINE_REACHED",
                }
            ),
            "cancelled": int(code == "COLLECT_CANCELLED"),
        },
        "calls": calls,
        "snapshots": snapshots,
        "samples": samples,
        "quota": {
            "status": "not_available",
            "reason": "CODEX_ADAPTER_EXPOSES_NO_QUOTA",
        },
        "cost": {
            "status": "not_available",
            "reason": "SUBSCRIPTION_OAUTH_NO_PROVIDER_COST",
        },
        "errors": [{"code": code}],
    }
    final_path = output_root / "final/context-compression-tier-b-raw.json"
    if final_path.exists():
        try:
            existing = json.loads(final_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "complete":
            return
    _atomic_write(
        final_path,
        json.dumps(raw, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )


def run_collect_entrypoint(
    action: Callable[[], Any],
    output_root: Path,
    emit: Callable[[str], Any],
    *,
    partial_writer: Callable[[str], Any] | None = None,
) -> None:
    """Run collect with sibling sanitized terminal handlers and no traceback."""
    try:
        action()
    except TerminalBenchmarkStop as exc:
        code = exc.code if exc.code in _TERMINAL_CODES else "BENCHMARK_TERMINAL_STOP"
        if partial_writer is not None:
            partial_writer(code)
        _write_adverse(output_root, code)
        emit(code)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        code = "COLLECT_CANCELLED"
        if partial_writer is not None:
            partial_writer(code)
        _write_adverse(output_root, code)
        emit(code)
        raise SystemExit(130) from None
    except Exception:
        code = "UNEXPECTED_INTERNAL_ERROR"
        if partial_writer is not None:
            partial_writer(code)
        _write_adverse(output_root, code)
        emit(code)
        raise SystemExit(2) from None


def run_collect_after_preflight(
    *,
    environment: dict[str, str],
    manifest: dict[str, Any],
    runtime_attestation: dict[str, Any],
    collect_once: Callable[[], Any],
) -> Any:
    """Reject CI, missing acknowledgement, or identity drift before collection."""
    if environment.get("CI") or environment.get("GITHUB_ACTIONS"):
        raise TerminalBenchmarkStop("CI_LIVE_COLLECTION_FORBIDDEN")
    if environment.get("TIER_B_LIVE_ACK") != LIVE_ACK:
        raise TerminalBenchmarkStop("LIVE_ACK_REQUIRED")
    for field in ("source_manifest_sha256", "schedule_sha256"):
        if manifest.get(field) != runtime_attestation.get(field):
            raise TerminalBenchmarkStop("SOURCE_RUNTIME_IDENTITY_MISMATCH")
    if runtime_attestation.get("preflight_passed") is not True:
        raise TerminalBenchmarkStop("RUNTIME_PREFLIGHT_FAILED")
    image_id = runtime_attestation.get("runtime_image_id")
    if not isinstance(image_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_id
    ):
        raise TerminalBenchmarkStop("RUNTIME_IMAGE_ID_INVALID")
    return collect_once()


_SHAPE_DRIVER_SOURCE = r'''
import copy
import json
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

config_path, output_path = sys.argv[1:]

def deny_network(*args, **kwargs):
    raise AssertionError("NETWORK_FORBIDDEN")

socket.create_connection = deny_network
socket.socket.connect = deny_network

def emit(value):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))

try:
    with open(config_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    import agent.context_compressor as cc

    calls = []

    def fake_call_llm(**kwargs):
        calls.append({
            "provider": kwargs.get("provider"),
            "api_mode": kwargs.get("api_mode"),
            "model": kwargs.get("model"),
            "task": kwargs.get("task"),
        })
        body = """## Active Task
Preserve tier-b synthetic state continuity.
## Constraints
- Retain exact synthetic facts and the latest raw tail.
## Completed Actions
1. Deterministic fake-provider shape pass completed.
## Active State
- Synthetic benchmark state remains active.
## Key Decisions
- No provider or network access is permitted during preparation.
## Next Steps
1. Continue the deterministic transcript cycles.
## Resolved Questions
- None.
## Open Questions
- None.
## Critical Context
- The frozen fixture remains the only fact source.
"""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))],
            model=kwargs.get("model"),
            usage=None,
        )

    placement = []
    trajectories = []
    previous_limit = cc.ContextCompressor._SUMMARY_INPUT_MAX_CHARS
    cc.ContextCompressor._SUMMARY_INPUT_MAX_CHARS = cfg["split_limit_chars"]
    try:
        for item in cfg["cases"]:
            compressor = cc.ContextCompressor(
                model=cfg["downstream_model"],
                threshold_percent=0.85,
                protect_first_n=0,
                protect_last_n=1,
                quiet_mode=True,
                summary_model_override=cfg["summary_model"],
                config_context_length=cfg["context_length"],
                provider=cfg["provider"],
                api_mode=cfg["api_mode"],
                abort_on_summary_failure=True,
            )
            compressor.tail_token_budget = cfg["tail_token_budget"]
            compressor.last_prompt_tokens = cfg["current_tokens"]
            messages = copy.deepcopy(item["rows"])
            start = compressor._align_boundary_forward(
                messages, compressor._protect_head_size(messages)
            )
            initial_end = compressor._find_tail_cut_by_tokens(messages, start)
            cycle_one_messages = messages + copy.deepcopy(item["growth"]["1"])
            cycle_one_start = compressor._align_boundary_forward(
                cycle_one_messages,
                compressor._protect_head_size(cycle_one_messages),
            )
            cycle_one_end = compressor._find_tail_cut_by_tokens(
                cycle_one_messages, cycle_one_start
            )
            for fact in item["facts"]:
                row_index = next(
                    index
                    for index, row in enumerate(messages)
                    if row.get("row_id") == fact["row_id"]
                )
                if fact["initial_dependency"] == "raw_tail":
                    passed = row_index >= initial_end
                    observed = "raw_tail" if passed else "summary"
                else:
                    passed = cycle_one_start <= row_index < cycle_one_end
                    observed = "summary" if passed else "raw_tail"
                placement.append({
                    "scenario_id": item["scenario_id"],
                    "repeat": item["repeat"],
                    "field_id": fact["field_id"],
                    "declared_dependency": fact["initial_dependency"],
                    "observed_dependency": observed,
                    "passed": passed,
                })
            per_cycle = []
            before_trajectory = len(calls)
            for cycle in range(1, 9):
                messages.extend(copy.deepcopy(item["growth"][str(cycle)]))
                before_cycle = len(calls)
                with patch.object(cc, "call_llm", fake_call_llm):
                    messages = compressor.compress(
                        messages,
                        current_tokens=cfg["current_tokens"],
                        force=True,
                        focus_topic=cfg["focus_topic"],
                        memory_context=cfg["memory_context"],
                    )
                if compressor._last_compress_aborted:
                    raise AssertionError("SHAPE_COMPRESSION_ABORTED")
                per_cycle.append(len(calls) - before_cycle)
            trajectories.append({
                "trajectory_key": item["trajectory_key"],
                "summary_calls": len(calls) - before_trajectory,
                "calls_per_cycle": per_cycle,
            })
    finally:
        cc.ContextCompressor._SUMMARY_INPUT_MAX_CHARS = previous_limit
    emit({
        "status": "ok",
        "module_sha256": cfg["module_sha256"],
        "placement": placement,
        "trajectories": trajectories,
        "routes": calls,
    })
except BaseException as exc:
    emit({
        "status": "error",
        "error_code": "SHAPE_DRIVER_FAILED",
        "error_class": type(exc).__name__,
    })
    raise SystemExit(3) from None
'''


def _run_git(repo: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("GIT_IDENTITY_INVALID")
    return result.stdout


def _git_text(repo: Path, *arguments: str) -> str:
    return _run_git(repo, *arguments).decode("utf-8", errors="strict").strip()


def _git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _repository_identity(repo: Path) -> dict[str, Any]:
    if (
        not repo.is_dir()
        or _git_text(repo, "branch", "--show-current") != EVALUATION_BRANCH
    ):
        raise ValueError("REPOSITORY_IDENTITY_INVALID")
    head = _git_text(repo, "rev-parse", "HEAD")
    for revision in (STARTING_HEAD, BASELINE_SHA, CANDIDATE_SHA):
        if (
            _git_text(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
            != revision
        ):
            raise ValueError("REVISION_IDENTITY_INVALID")
    if not (
        _git_is_ancestor(repo, STARTING_HEAD, head)
        and _git_is_ancestor(repo, BASELINE_SHA, CANDIDATE_SHA)
        and _git_is_ancestor(repo, CANDIDATE_SHA, head)
    ):
        raise ValueError("REVISION_ANCESTRY_INVALID")
    product_diff = tuple(
        sorted(
            line
            for line in _git_text(
                repo, "diff", "--name-only", f"{BASELINE_SHA}..{CANDIDATE_SHA}"
            ).splitlines()
            if line
        )
    )
    if product_diff != tuple(sorted(_PRODUCT_DIFF_PATHS)):
        raise ValueError("PRODUCT_DIFF_SCOPE_INVALID")
    status_lines = [
        line
        for line in _git_text(
            repo, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line
    ]
    dry_overlay = False
    if status_lines:
        expected = {f"?? {path}" for path in _ALLOWED_CANDIDATE_PATHS}
        if set(status_lines) != expected:
            raise ValueError("WORKTREE_SCOPE_INVALID")
        dry_overlay = True
    return {
        "branch": EVALUATION_BRANCH,
        "evaluation_head": head,
        "dry_candidate_overlay": dry_overlay,
        "live_ready": not dry_overlay,
        "product_diff_paths": list(_PRODUCT_DIFF_PATHS),
    }


def _write_git_archive(repo: Path, revision: str, destination: Path) -> None:
    payload = _run_git(repo, "archive", "--format=tar", revision)
    _atomic_write(destination, payload)


def _append_dry_overlay(repo: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="a") as archive:
        for relative in sorted(_ALLOWED_CANDIDATE_PATHS):
            source = repo / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError("DRY_OVERLAY_PATH_INVALID")
            payload = source.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mode = stat.S_IMODE(source.stat().st_mode)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))


def _safe_member_name(name: str) -> str:
    candidate = name[:-1] if name.endswith("/") else name
    path = Path(candidate)
    if (
        not candidate
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != candidate
    ):
        raise ValueError("ARCHIVE_MEMBER_PATH_INVALID")
    return candidate


def _archive_manifest(archive_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            name = _safe_member_name(member.name)
            if name in seen:
                raise ValueError("ARCHIVE_DUPLICATE_MEMBER")
            seen.add(name)
            if member.isdir():
                kind = "directory"
                payload = b""
            elif member.isfile():
                kind = "file"
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("ARCHIVE_MEMBER_UNREADABLE")
                payload = extracted.read()
            elif member.issym():
                kind = "symlink"
                target = Path(member.linkname)
                resolved = Path(name).parent / target
                if target.is_absolute() or ".." in resolved.parts:
                    raise ValueError("ARCHIVE_LINK_ESCAPE")
                payload = member.linkname.encode("utf-8")
            else:
                raise ValueError("ARCHIVE_MEMBER_TYPE_INVALID")
            records.append({
                "path": name,
                "kind": kind,
                "mode": member.mode,
                "size": len(payload),
                "sha256": sha256_bytes(payload),
            })
    return sorted(records, key=lambda item: item["path"])


def _extract_validated_archive(
    archive_path: Path, destination: Path
) -> list[dict[str, Any]]:
    expected = _archive_manifest(archive_path)
    destination.mkdir(mode=0o700)
    root = destination.resolve()
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            name = _safe_member_name(member.name)
            target = destination / name
            parent = target.parent.resolve()
            if not parent.is_relative_to(root):
                raise ValueError("ARCHIVE_MEMBER_PATH_INVALID")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                os.chmod(target, member.mode)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("ARCHIVE_MEMBER_UNREADABLE")
                with target.open("xb") as handle:
                    shutil.copyfileobj(extracted, handle)
                os.chmod(target, member.mode)
            elif member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
    actual = _tree_manifest(destination)
    if actual != expected:
        raise ValueError("ARCHIVE_TREE_DISAGREEMENT")
    return actual


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
            payload = path.read_bytes()
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            link = os.readlink(path)
            resolved = path.parent / link
            if Path(link).is_absolute() or not resolved.resolve(
                strict=False
            ).is_relative_to(root.resolve()):
                raise ValueError("TREE_LINK_ESCAPE")
            payload = link.encode("utf-8")
        else:
            raise ValueError("TREE_MEMBER_TYPE_INVALID")
        records.append({
            "path": relative,
            "kind": kind,
            "mode": stat.S_IMODE(metadata.st_mode),
            "size": len(payload),
            "sha256": sha256_bytes(payload),
        })
    return records


def _file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"size": len(payload), "sha256": sha256_bytes(payload)}


def _shape_child_environment(home: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in _CREDENTIAL_ENV_VARS:
        environment.pop(name, None)
    environment.update({
        "HOME": str(home),
        "HERMES_HOME": str(home / "hermes"),
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "NO_COLOR": "1",
    })
    environment.pop("PYTHONPATH", None)
    return environment


def _run_shape_preflight(
    fixture: dict[str, Any],
    trees: dict[str, Path],
    temporary_root: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for scenario_id in fixture["scenario_order"]:
        for repeat in REPEATS:
            expanded = expand_scenario(fixture, scenario_id, repeat)
            cases.append({
                "trajectory_key": f"{scenario_id}-r{repeat}",
                "scenario_id": scenario_id,
                "repeat": repeat,
                "rows": expanded["rows"],
                "facts": expanded["facts"],
                "growth": {
                    str(cycle): expand_growth_rows(fixture, scenario_id, cycle)
                    for cycle in range(1, 9)
                },
            })
    temporary_root.mkdir(mode=0o700)
    config_path = temporary_root / "shape-input.json"
    results: dict[str, Any] = {}
    for label in ("baseline", "candidate"):
        module_path = trees[label] / SUBJECT_PATH
        config = {
            "provider": PROVIDER,
            "api_mode": API_MODE,
            "summary_model": SUMMARY_MODEL,
            "downstream_model": DOWNSTREAM_MODEL,
            "split_limit_chars": SPLIT_LIMIT_CHARS,
            "context_length": CONTEXT_LENGTH,
            "current_tokens": CURRENT_TOKENS,
            "tail_token_budget": TAIL_TOKEN_BUDGET,
            "focus_topic": FOCUS_TOPIC,
            "memory_context": MEMORY_CONTEXT,
            "module_sha256": sha256_bytes(module_path.read_bytes()),
            "cases": cases,
        }
        _atomic_write(config_path, canonical_bytes(config))
        output_path = temporary_root / f"{label}.json"
        home = temporary_root / f"{label}-home"
        home.mkdir(mode=0o700)
        result = subprocess.run(
            [sys.executable, "-", str(config_path), str(output_path)],
            input=_SHAPE_DRIVER_SOURCE,
            text=True,
            cwd=trees[label],
            env=_shape_child_environment(home),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not output_path.is_file():
            raise ValueError("SHAPE_PREFLIGHT_FAILED")
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("SHAPE_PREFLIGHT_FAILED") from exc
        if (
            payload.get("status") != "ok"
            or payload.get("module_sha256") != config["module_sha256"]
        ):
            raise ValueError("SHAPE_PREFLIGHT_FAILED")
        expected_calls = 8 if label == "baseline" else 10
        expected_cycle_shape = [1] * 8 if label == "baseline" else [3] + [1] * 7
        trajectories = payload.get("trajectories")
        placements = payload.get("placement")
        routes = payload.get("routes")
        if (
            not isinstance(trajectories, list)
            or len(trajectories) != 9
            or any(
                item.get("summary_calls") != expected_calls
                or item.get("calls_per_cycle") != expected_cycle_shape
                for item in trajectories
            )
            or not isinstance(placements, list)
            or not placements
            or any(item.get("passed") is not True for item in placements)
            or not isinstance(routes, list)
            or len(routes) != expected_calls * 9
        ):
            raise ValueError("SHAPE_PREFLIGHT_DRIFT")
        results[label] = {
            "module_sha256": config["module_sha256"],
            "trajectories": len(trajectories),
            "summary_calls_per_trajectory": expected_calls,
            "summary_calls_total": len(routes),
            "calls_per_cycle": expected_cycle_shape,
            "placement_checks": len(placements),
            "placement_passed": len(placements),
            "provider_calls": 0,
            "network_calls": 0,
        }
    shutil.rmtree(temporary_root)
    return results


def prepare_benchmark(repo_root: Path, runtime_root: Path) -> dict[str, Any]:
    """Create deterministic archives, trees, manifests, shape proof, and schedule."""
    repo = repo_root.resolve(strict=True)
    root = runtime_root.resolve(strict=True)
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("RUNTIME_ROOT_NOT_EMPTY")
    identity = _repository_identity(repo)
    fixture_path = repo / "evaluation/fixtures/context-compression-tier-b.json"
    fixture = load_fixture(fixture_path)
    fixture_sha = sha256_bytes(fixture_path.read_bytes())
    schedule = build_execution_schedule(fixture, fixture_sha)

    input_root = root / "input"
    archive_root = input_root / "archives"
    tree_root = input_root / "trees"
    build_context = root / "build-context" / "evaluation-head"
    output_root = root / "output"
    for child in (
        input_root,
        archive_root,
        tree_root,
        root / "build-context",
        output_root,
    ):
        child.mkdir(mode=0o700)

    revisions = {
        "baseline": BASELINE_SHA,
        "candidate": CANDIDATE_SHA,
        "evaluation-head": identity["evaluation_head"],
    }
    archive_identities: dict[str, Any] = {}
    tree_identities: dict[str, Any] = {}
    trees: dict[str, Path] = {}
    for label, revision in revisions.items():
        archive_path = archive_root / f"{label}.tar"
        _write_git_archive(repo, revision, archive_path)
        if label == "evaluation-head" and identity["dry_candidate_overlay"]:
            _append_dry_overlay(repo, archive_path)
        archive_members = _archive_manifest(archive_path)
        archive_manifest_path = archive_root / f"{label}.manifest.json"
        _atomic_write(
            archive_manifest_path,
            json.dumps(archive_members, indent=2, sort_keys=True).encode("utf-8")
            + b"\n",
        )
        destination = tree_root / label
        tree_members = _extract_validated_archive(archive_path, destination)
        trees[label] = destination
        tree_manifest_path = input_root / f"{label}-tree-manifest.json"
        _atomic_write(
            tree_manifest_path,
            json.dumps(tree_members, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        archive_identities[label] = {
            "revision_sha": revision,
            "archive": _file_identity(archive_path),
            "manifest": _file_identity(archive_manifest_path),
            "member_count": len(archive_members),
        }
        tree_identities[label] = {
            "manifest": _file_identity(tree_manifest_path),
            "tree_sha256": sha256_bytes(canonical_bytes(tree_members)),
            "member_count": len(tree_members),
        }

    _extract_validated_archive(archive_root / "evaluation-head.tar", build_context)
    build_manifest = _tree_manifest(build_context)
    shape = _run_shape_preflight(fixture, trees, root / ".shape-preflight")
    schedule_path = input_root / "execution-schedule.json"
    _atomic_write(
        schedule_path,
        json.dumps(schedule, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    source_files = {
        relative: _file_identity(repo / relative)
        for relative in (
            "evaluation/context_compression_tier_b.py",
            "evaluation/fixtures/context-compression-tier-b.json",
            "evaluation/Containerfile.context-compression-tier-b",
            "evaluation/context_compression_tier_b_runtime.sh",
            "pyproject.toml",
            "uv.lock",
        )
    }
    source_manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "identity": identity,
        "evaluation_head": identity["evaluation_head"],
        "live_ready": identity["live_ready"],
        "revisions": revisions,
        "provider": PROVIDER,
        "api_mode": API_MODE,
        "summarizer_model": SUMMARY_MODEL,
        "downstream_model": DOWNSTREAM_MODEL,
        "build_base_image_id": BUILD_BASE_IMAGE_ID,
        "build_base_provenance": BUILD_BASE_PROVENANCE,
        "expected_counts": schedule["expected_counts"],
        "fixture_sha256": fixture_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file": _file_identity(schedule_path),
        "shape_preflight": shape,
        "archives": archive_identities,
        "trees": tree_identities,
        "build_context": {
            "tree_sha256": sha256_bytes(canonical_bytes(build_manifest)),
            "member_count": len(build_manifest),
        },
        "files": source_files,
    }
    manifest_path = input_root / "source-manifest.json"
    _atomic_write(
        manifest_path,
        json.dumps(source_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return {
        "source_manifest": str(manifest_path.relative_to(root)),
        "source_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "schedule": str(schedule_path.relative_to(root)),
        "schedule_file_sha256": sha256_bytes(schedule_path.read_bytes()),
        "expected_counts": schedule["expected_counts"],
        "shape_preflight": shape,
        "live_ready": identity["live_ready"],
    }


def _load_json(path: Path, error_code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(error_code) from exc
    if not isinstance(value, dict):
        raise ValueError(error_code)
    return value


def write_image_manifest(
    source_manifest_path: Path,
    inspection_path: Path,
    output_path: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Bind one locally built immutable image to the frozen source manifest."""
    source = _load_json(source_manifest_path, "SOURCE_MANIFEST_INVALID")
    inspection = _load_json(inspection_path, "IMAGE_INSPECTION_INVALID")
    image_id = environment.get("TIER_B_IMAGE_ID", "")
    evaluation_head = environment.get("EVALUATION_HEAD", "")
    if (
        source.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or source.get("live_ready") is not True
        or evaluation_head != source.get("evaluation_head")
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
    ):
        raise ValueError("IMAGE_SOURCE_IDENTITY_MISMATCH")
    distributions = inspection.get("distributions")
    if not isinstance(distributions, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or not all(isinstance(part, str) and part for part in item)
        for item in distributions
    ):
        raise ValueError("INSTALLED_DISTRIBUTIONS_INVALID")
    distribution_manifest = sorted(distributions)
    manifest = {
        "schema_version": "context-compression-tier-b-image-manifest/1",
        "tier_b_image_id": image_id,
        "evaluation_head": evaluation_head,
        "source_manifest_sha256": sha256_bytes(source_manifest_path.read_bytes()),
        "schedule_sha256": source["schedule_sha256"],
        "build_base_image_id": BUILD_BASE_IMAGE_ID,
        "build_base_provenance": BUILD_BASE_PROVENANCE,
        "base_layers": json.loads(environment.get("BASE_LAYERS", "[]")),
        "image_layers": json.loads(environment.get("IMAGE_LAYERS", "[]")),
        "architecture": inspection.get("architecture"),
        "python_version": inspection.get("python_version"),
        "uv_version": "0.11.6",
        "sync_command": "uv sync --locked --python 3.13 --extra dev",
        "build_command": (
            "podman build --pull=never --no-cache --network=slirp4netns "
            "--label io.hermes.benchmark=context-compression-tier-b "
            f"--label io.hermes.evaluation-head={evaluation_head} "
            "--file evaluation/Containerfile.context-compression-tier-b "
            f"--tag localhost/hermes-compaction-tier-b:{evaluation_head} ."
        ),
        "source_files": source["files"],
        "evaluation_archive": source["archives"]["evaluation-head"],
        "evaluation_tree": source["trees"]["evaluation-head"],
        "build_context": source["build_context"],
        "installed_distributions": distribution_manifest,
        "installed_distributions_sha256": sha256_bytes(
            canonical_bytes(distribution_manifest)
        ),
        "registry_digest_status": "not_exposed_by_local_image",
        "registry_digests": [],
    }
    _atomic_write(
        output_path,
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return manifest


def verify_runtime_metadata(
    *,
    mounts_payload: str,
    networks_payload: str,
    ports_payload: str,
    environment_payload: str,
    image_id: str,
) -> None:
    """Validate only the fixed task container metadata passed by the helper."""
    try:
        mounts = json.loads(mounts_payload)
        networks = json.loads(networks_payload)
        ports = json.loads(ports_payload)
        environment = json.loads(environment_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("RUNTIME_METADATA_MALFORMED") from exc
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("RUNTIME_IMAGE_ID_INVALID")
    if not isinstance(networks, dict) or set(networks) != {
        "hermes-compaction-tier-b-egress"
    }:
        raise ValueError("RUNTIME_NETWORK_MISMATCH")
    if ports not in ({}, None) or not isinstance(mounts, list):
        raise ValueError("RUNTIME_PORT_OR_MOUNT_MISMATCH")
    expected_mounts = {
        "/benchmark/input": (
            "bind",
            "/home/ron/hermes-compaction-tier-b/runtime/input",
            False,
        ),
        "/benchmark/output": (
            "bind",
            "/home/ron/hermes-compaction-tier-b/runtime/output",
            True,
        ),
        "/benchmark/home": ("tmpfs", "", True),
        "/benchmark/hermes": ("tmpfs", "", True),
        "/benchmark/run": ("tmpfs", "", True),
        "/tmp": ("tmpfs", "", True),
        "/run": ("tmpfs", "", True),
    }
    observed: dict[str, tuple[str, str, bool]] = {}
    for mount in mounts:
        if not isinstance(mount, dict):
            raise ValueError("RUNTIME_MOUNT_MISMATCH")
        destination = str(mount.get("Destination") or "")
        observed[destination] = (
            str(mount.get("Type") or ""),
            str(mount.get("Source") or "") if mount.get("Type") == "bind" else "",
            bool(mount.get("RW")),
        )
    if observed != expected_mounts:
        raise ValueError("RUNTIME_MOUNT_MISMATCH")
    if not isinstance(environment, list):
        raise ValueError("RUNTIME_ENVIRONMENT_MISMATCH")
    env_map = dict(item.split("=", 1) for item in environment if "=" in item)
    required = {
        "HOME": "/benchmark/home",
        "HERMES_HOME": "/benchmark/hermes",
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
        "TIER_B_LIVE_ACK": LIVE_ACK,
    }
    if any(env_map.get(key) != value for key, value in required.items()):
        raise ValueError("RUNTIME_ENVIRONMENT_MISMATCH")
    forbidden_names = set(_CREDENTIAL_ENV_VARS) | {
        "DOCKER_HOST",
        "CONTAINER_HOST",
        "SSH_AUTH_SOCK",
    }
    forbidden_names.discard("CODEX_HOME")
    if "CODEX_HOME" in env_map or any(name in env_map for name in forbidden_names):
        raise ValueError("RUNTIME_FORBIDDEN_ENVIRONMENT")
    serialized = canonical_bytes({"mounts": mounts, "environment": environment})
    if any(
        token in serialized
        for token in (b"/.hermes", b"/.codex", b"podman.sock", b"docker.sock")
    ):
        raise ValueError("RUNTIME_FORBIDDEN_PATH")


def _installed_distribution_manifest() -> list[list[str]]:
    rows: set[tuple[str, str]] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
        ):
            raise ValueError("INSTALLED_DISTRIBUTIONS_INVALID")
        if not _SAFE_TOKEN_RE.fullmatch(name) or not _SAFE_TOKEN_RE.fullmatch(version):
            raise ValueError("INSTALLED_DISTRIBUTIONS_INVALID")
        rows.add((name.lower(), version))
    return [list(item) for item in sorted(rows)]


def _load_manifest_rows(path: Path, code: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(code)
    return value


def _verify_embedded_tree(
    source_root: Path,
    expected_rows: list[dict[str, Any]],
) -> None:
    root = source_root.resolve(strict=True)
    for expected in expected_rows:
        relative = expected.get("path")
        if not isinstance(relative, str) or _safe_member_name(relative) != relative:
            raise ValueError("EMBEDDED_EVALUATION_TREE_DRIFT")
        path = source_root / relative
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ValueError("EMBEDDED_EVALUATION_TREE_DRIFT") from exc
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
            payload = path.read_bytes()
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            link = os.readlink(path)
            resolved = path.parent / link
            if Path(link).is_absolute() or not resolved.resolve(
                strict=False
            ).is_relative_to(root):
                raise ValueError("EMBEDDED_EVALUATION_TREE_DRIFT")
            payload = link.encode("utf-8")
        else:
            raise ValueError("EMBEDDED_EVALUATION_TREE_DRIFT")
        observed = {
            "path": relative,
            "kind": kind,
            "mode": stat.S_IMODE(metadata.st_mode),
            "size": len(payload),
            "sha256": sha256_bytes(payload),
        }
        if observed != expected:
            raise ValueError("EMBEDDED_EVALUATION_TREE_DRIFT")


def _attest_runtime_artifacts(
    manifest_path: Path,
    image_manifest_path: Path,
    schedule_path: Path,
    *,
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load_json(manifest_path, "SOURCE_MANIFEST_INVALID")
    image_manifest = _load_json(image_manifest_path, "IMAGE_MANIFEST_INVALID")
    schedule = _load_json(schedule_path, "SCHEDULE_INVALID")
    expected_counts = {
        "trajectories": 18,
        "summary": EXPECTED_SUMMARY_ATTEMPTS,
        "downstream": EXPECTED_DOWNSTREAM_ATTEMPTS,
        "total": EXPECTED_TOTAL_ATTEMPTS,
    }
    fixture_path = source_root / "evaluation/fixtures/context-compression-tier-b.json"
    fixture = load_fixture(fixture_path)
    fixture_sha256 = sha256_bytes(fixture_path.read_bytes())
    expected_schedule = build_execution_schedule(fixture, fixture_sha256)
    schedule_identity = schedule.get("schedule_sha256")
    evaluation_head = manifest.get("evaluation_head")
    expected_revisions = {
        "baseline": BASELINE_SHA,
        "candidate": CANDIDATE_SHA,
        "evaluation-head": evaluation_head,
    }
    expected_source_identity = {
        "branch": EVALUATION_BRANCH,
        "evaluation_head": evaluation_head,
        "dry_candidate_overlay": False,
        "live_ready": True,
        "product_diff_paths": list(_PRODUCT_DIFF_PATHS),
    }
    if (
        manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("live_ready") is not True
        or not re.fullmatch(r"[0-9a-f]{40}", str(evaluation_head or ""))
        or manifest.get("identity") != expected_source_identity
        or manifest.get("revisions") != expected_revisions
        or manifest.get("provider") != PROVIDER
        or manifest.get("api_mode") != API_MODE
        or manifest.get("summarizer_model") != SUMMARY_MODEL
        or manifest.get("downstream_model") != DOWNSTREAM_MODEL
        or manifest.get("build_base_image_id") != BUILD_BASE_IMAGE_ID
        or manifest.get("build_base_provenance") != BUILD_BASE_PROVENANCE
        or schedule.get("schema_version") != SCHEDULE_SCHEMA
        or schedule.get("expected_counts") != expected_counts
        or schedule != expected_schedule
        or schedule_identity != manifest.get("schedule_sha256")
        or manifest.get("fixture_sha256") != fixture_sha256
        or manifest.get("schedule_file") != _file_identity(schedule_path)
        or manifest.get("expected_counts") != expected_counts
    ):
        raise ValueError("RUNTIME_SOURCE_IDENTITY_MISMATCH")
    source_files = manifest.get("files")
    required_files = {
        "evaluation/context_compression_tier_b.py",
        "evaluation/fixtures/context-compression-tier-b.json",
        "evaluation/context_compression_tier_b_runtime.sh",
        "evaluation/Containerfile.context-compression-tier-b",
        "pyproject.toml",
        "uv.lock",
    }
    if not isinstance(source_files, dict) or set(source_files) != required_files:
        raise ValueError("RUNTIME_SOURCE_FILE_SET_MISMATCH")
    for relative in sorted(required_files):
        path = source_root / relative
        if not path.is_file() or source_files.get(relative) != _file_identity(path):
            raise ValueError("RUNTIME_SOURCE_FILE_DRIFT")
    input_root = manifest_path.parent
    archives = manifest.get("archives")
    trees = manifest.get("trees")
    labels = ("baseline", "candidate", "evaluation-head")
    if (
        not isinstance(archives, dict)
        or set(archives) != set(labels)
        or not isinstance(trees, dict)
        or set(trees) != set(labels)
    ):
        raise ValueError("RUNTIME_ARCHIVE_TREE_IDENTITY_MISMATCH")
    tree_rows_by_label: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        archive_path = input_root / "archives" / f"{label}.tar"
        archive_manifest_path = input_root / "archives" / f"{label}.manifest.json"
        tree_manifest_path = input_root / f"{label}-tree-manifest.json"
        tree_root = input_root / "trees" / label
        archive_identity = archives.get(label)
        tree_identity = trees.get(label)
        if (
            not isinstance(archive_identity, dict)
            or not isinstance(tree_identity, dict)
            or archive_identity.get("revision_sha") != expected_revisions[label]
            or archive_identity.get("archive") != _file_identity(archive_path)
            or archive_identity.get("manifest") != _file_identity(archive_manifest_path)
            or tree_identity.get("manifest") != _file_identity(tree_manifest_path)
        ):
            raise ValueError("RUNTIME_ARCHIVE_TREE_IDENTITY_MISMATCH")
        archive_rows = _load_manifest_rows(
            archive_manifest_path, "RUNTIME_ARCHIVE_MANIFEST_INVALID"
        )
        tree_rows = _load_manifest_rows(
            tree_manifest_path, "RUNTIME_TREE_MANIFEST_INVALID"
        )
        observed_archive_rows = _archive_manifest(archive_path)
        observed_tree_rows = _tree_manifest(tree_root)
        if (
            archive_rows != observed_archive_rows
            or tree_rows != observed_tree_rows
            or archive_rows != tree_rows
            or archive_identity.get("member_count") != len(archive_rows)
            or tree_identity.get("member_count") != len(tree_rows)
            or tree_identity.get("tree_sha256")
            != sha256_bytes(canonical_bytes(observed_tree_rows))
        ):
            raise ValueError("RUNTIME_ARCHIVE_TREE_DRIFT")
        tree_rows_by_label[label] = tree_rows
    evaluation_tree = trees["evaluation-head"]
    if manifest.get("build_context") != {
        "tree_sha256": evaluation_tree.get("tree_sha256"),
        "member_count": evaluation_tree.get("member_count"),
    }:
        raise ValueError("RUNTIME_BUILD_CONTEXT_IDENTITY_MISMATCH")
    _verify_embedded_tree(source_root, tree_rows_by_label["evaluation-head"])
    distributions = image_manifest.get("installed_distributions")
    expected_build_command = (
        "podman build --pull=never --no-cache --network=slirp4netns "
        "--label io.hermes.benchmark=context-compression-tier-b "
        f"--label io.hermes.evaluation-head={evaluation_head} "
        "--file evaluation/Containerfile.context-compression-tier-b "
        f"--tag localhost/hermes-compaction-tier-b:{evaluation_head} ."
    )
    base_layers = image_manifest.get("base_layers")
    image_layers = image_manifest.get("image_layers")
    if (
        image_manifest.get("schema_version")
        != "context-compression-tier-b-image-manifest/1"
        or image_manifest.get("source_manifest_sha256")
        != sha256_bytes(manifest_path.read_bytes())
        or image_manifest.get("schedule_sha256") != schedule_identity
        or image_manifest.get("source_files") != source_files
        or image_manifest.get("evaluation_archive") != archives["evaluation-head"]
        or image_manifest.get("evaluation_tree") != trees["evaluation-head"]
        or image_manifest.get("build_context") != manifest.get("build_context")
        or image_manifest.get("evaluation_head") != manifest.get("evaluation_head")
        or image_manifest.get("build_base_image_id") != BUILD_BASE_IMAGE_ID
        or image_manifest.get("build_base_provenance") != BUILD_BASE_PROVENANCE
        or image_manifest.get("sync_command")
        != "uv sync --locked --python 3.13 --extra dev"
        or image_manifest.get("build_command") != expected_build_command
        or image_manifest.get("architecture") != platform.machine()
        or image_manifest.get("python_version") != platform.python_version()
        or image_manifest.get("uv_version") != "0.11.6"
        or not isinstance(base_layers, list)
        or not base_layers
        or not isinstance(image_layers, list)
        or image_layers[: len(base_layers)] != base_layers
        or image_manifest.get("registry_digest_status") != "not_exposed_by_local_image"
        or image_manifest.get("registry_digests") != []
        or not isinstance(distributions, list)
        or distributions != sorted(distributions)
        or image_manifest.get("installed_distributions_sha256")
        != sha256_bytes(canonical_bytes(distributions))
        or distributions != _installed_distribution_manifest()
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(image_manifest.get("tier_b_image_id", ""))
        )
    ):
        raise ValueError("RUNTIME_IMAGE_ARTIFACT_IDENTITY_MISMATCH")
    return manifest, image_manifest, schedule


def attest_runtime(
    manifest_path: Path,
    image_manifest_path: Path,
    schedule_path: Path,
    run_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Perform in-container identity, configuration, and writability proofs."""
    manifest, image_manifest, schedule = _attest_runtime_artifacts(
        manifest_path,
        image_manifest_path,
        schedule_path,
        source_root=Path(__file__).resolve(strict=True).parents[1],
    )
    try:
        import yaml

        config = yaml.safe_load(
            (Path(os.environ["HERMES_HOME"]) / "config.yaml").read_text(
                encoding="utf-8"
            )
        )
    except (KeyError, OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("RUNTIME_CONFIG_INVALID") from exc
    expected_config = {
        "model": {
            "provider": PROVIDER,
            "default": DOWNSTREAM_MODEL,
            "api_mode": API_MODE,
        },
        "auxiliary": {
            "transient_retries": 0,
            "compression": {
                "provider": PROVIDER,
                "model": SUMMARY_MODEL,
                "api_mode": API_MODE,
                "timeout": TIMEOUT_SECONDS,
            },
        },
    }
    if config != expected_config or os.environ.get("CODEX_HOME"):
        raise ValueError("RUNTIME_CONFIG_INVALID")
    mountinfo = Path("/proc/self/mountinfo").read_text(
        encoding="utf-8", errors="strict"
    )
    if any(
        token in mountinfo
        for token in ("/.hermes", "/.codex", "podman.sock", "docker.sock")
    ):
        raise ValueError("RUNTIME_FORBIDDEN_MOUNT")
    probe_roots = {
        "home": Path(os.environ["HOME"]),
        "hermes": Path(os.environ["HERMES_HOME"]),
        "run": run_root,
        "output": output_root,
    }
    probe_results: dict[str, bool] = {}
    for label, root in probe_roots.items():
        probe = root / ".tier-b-write-probe"
        if probe.exists():
            raise ValueError("RUNTIME_PROBE_COLLISION")
        root.mkdir(parents=True, exist_ok=True)
        with probe.open("xb") as handle:
            handle.write(b"tier-b-benign-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
        probe_results[label] = probe.read_bytes() == b"tier-b-benign-probe\n"
        probe.unlink()
        if probe.exists():
            raise ValueError("RUNTIME_PROBE_REMOVE_FAILED")
    if not all(probe_results.values()):
        raise ValueError("RUNTIME_PROBE_FAILED")
    attestation = {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA,
        "preflight_passed": True,
        "source_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "schedule_sha256": manifest["schedule_sha256"],
        "schedule_file_sha256": sha256_bytes(schedule_path.read_bytes()),
        "image_manifest_sha256": sha256_bytes(image_manifest_path.read_bytes()),
        "runtime_image_id": image_manifest["tier_b_image_id"],
        "source_files_sha256": sha256_bytes(canonical_bytes(manifest["files"])),
        "archive_identities_sha256": sha256_bytes(
            canonical_bytes(manifest["archives"])
        ),
        "tree_identities_sha256": sha256_bytes(canonical_bytes(manifest["trees"])),
        "installed_distributions_sha256": image_manifest[
            "installed_distributions_sha256"
        ],
        "embedded_evaluation_tree_exact": True,
        "configuration_exact": True,
        "write_remove_probes": probe_results,
        "forbidden_mounts_absent": True,
        "codex_home_absent": True,
    }
    _atomic_write(
        run_root / "runtime-attestation.json",
        json.dumps(attestation, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return attestation


def sanitize_auth_status(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Reduce task-tmpfs auth-status text to fixed, non-secret evidence."""
    payload = input_path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    if _SECRET_PATTERN.search(text):
        input_path.unlink(missing_ok=True)
        raise ValueError("AUTH_STATUS_SECRET_PATTERN")
    matching_lines = [
        line.strip() for line in text.splitlines() if PROVIDER in line.lower()
    ]
    if len(matching_lines) != 1:
        input_path.unlink(missing_ok=True)
        raise ValueError("AUTH_POOL_ENTRY_COUNT_INVALID")
    fingerprint = sha256_bytes(
        b"context-compression-tier-b/oauth-account/v1\x00"
        + matching_lines[0].encode("utf-8")
    )
    attestation = {
        "schema_version": "context-compression-tier-b-auth-attestation/1",
        "status": "usable",
        "provider": PROVIDER,
        "pool_entry_count": 1,
        "source_class": "manual:device_code",
        "account_fingerprint_sha256": fingerprint,
    }
    input_path.unlink(missing_ok=True)
    _atomic_write(
        output_path,
        json.dumps(attestation, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return attestation


_LIVE_TRAJECTORY_DRIVER_SOURCE = r"""
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

payload = json.loads(sys.stdin.read())
product_root = Path(payload["product_root"])
sys.path.insert(0, str(product_root))
spec = importlib.util.spec_from_file_location(
    "tier_b_live_harness", payload["evaluator_path"]
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def emit(value):
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
    sys.stdout.flush()


def append_journal(record):
    data = harness.canonical_bytes(record) + b"\n"
    fd = os.open(
        payload["journal_path"],
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def response_content(response):
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise harness.TerminalBenchmarkStop("MALFORMED_MODEL_RESPONSE")
    return content.encode("utf-8")


guard = None
sample_records = []
snapshot_records = []


def emit_error(code):
    emit(
        {
            "status": "error",
            "error_code": code,
            "calls": guard.call_records if guard is not None else [],
            "samples": sample_records,
            "snapshots": snapshot_records,
            "actual_attempts": guard.actual_attempts if guard is not None else int(
                payload.get("global_actual_attempts", 0)
            ),
        }
    )


try:
    import agent.auxiliary_client as auxiliary
    import agent.context_compressor as compressor_module

    fixture = harness.load_fixture(payload["fixture_path"])
    trajectory = payload["trajectory"]
    expected_calls = payload["calls"]
    active_snapshot_sha256 = None
    trajectory_started = time.monotonic_ns()
    trajectory_deadline = trajectory_started + 5_100_000_000_000

    def terminal(code):
        raise harness.TerminalBenchmarkStop(code)

    guard = harness.LiveAuxiliaryGuard(
        expected_calls,
        auxiliary_module=auxiliary,
        compressor_module=compressor_module,
        global_actual_attempts=int(payload["global_actual_attempts"]),
        attempted_logical_call_ids=set(payload["attempted_logical_call_ids"]),
        append_journal=append_journal,
        trajectory_started_ns=trajectory_started,
        trajectory_deadline_ns=trajectory_deadline,
        attempt_id=payload["attempt_id"],
    )
    guard.install()

    scenario_id = trajectory["scenario_id"]
    repeat = trajectory["repeat"]
    expanded = harness.expand_scenario(fixture, scenario_id, repeat)
    messages = copy.deepcopy(expanded["rows"])
    scenario = harness._scenario(fixture, scenario_id)
    summary_ordinal = 0
    current_cycle = 0

    def save_snapshot(cycle):
        global active_snapshot_sha256
        snapshot_id = f"snapshot:{trajectory['trajectory_id']}:cycle:{cycle}"
        snapshot_bytes = harness.canonical_bytes(messages)
        snapshot_sha256 = harness.sha256_bytes(snapshot_bytes)
        path = Path(payload["run_root"]) / "snapshots" / (snapshot_id + ".json")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(snapshot_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o400)
        active_snapshot_sha256 = snapshot_sha256
        snapshot_records.append(
            {
                "snapshot_id": snapshot_id,
                "snapshot_sha256": snapshot_sha256,
                "row_count": len(messages),
                "compression_count": cycle,
                "dependency_key": f"{trajectory['trajectory_id']}:cycle:{cycle}",
                "storage": "container_tmpfs",
            }
        )

    def summary_call(**kwargs):
        global summary_ordinal
        summary_ordinal += 1
        call = guard.next_expected_call()
        if (
            call["call_role"] != "summary"
            or call["cycle"] != current_cycle
            or call["summary_pass_ordinal"] != summary_ordinal
        ):
            terminal("UNSCHEDULED_LOGICAL_CALL")
        with guard.activate(call, snapshot_sha256=active_snapshot_sha256):
            return auxiliary.call_llm(
                task="compression",
                provider=harness.PROVIDER,
                model=harness.SUMMARY_MODEL,
                api_mode=harness.API_MODE,
                timeout=harness.TIMEOUT_SECONDS,
                messages=kwargs.get("messages", []),
            )

    compressor_module.call_llm = summary_call
    previous_limit = compressor_module.ContextCompressor._SUMMARY_INPUT_MAX_CHARS
    compressor_module.ContextCompressor._SUMMARY_INPUT_MAX_CHARS = (
        harness.SPLIT_LIMIT_CHARS
    )
    compressor = compressor_module.ContextCompressor(
        model=harness.DOWNSTREAM_MODEL,
        threshold_percent=0.85,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
        summary_model_override=harness.SUMMARY_MODEL,
        config_context_length=harness.CONTEXT_LENGTH,
        provider=harness.PROVIDER,
        api_mode=harness.API_MODE,
        abort_on_summary_failure=True,
    )
    compressor.tail_token_budget = harness.TAIL_TOKEN_BUDGET
    compressor.last_prompt_tokens = harness.CURRENT_TOKENS

    def downstream_question():
        field = {"applicable": True, "status": "known", "value": "<single safe token>"}
        answer_shape = {
            "schema_version": harness.ANSWER_SCHEMA,
            "scenario_id": scenario_id,
            "question_id": scenario["question_id"],
            "required_state": {key: dict(field) for key in scenario["required_state_order"]},
            "identifiers": {key: dict(field) for key in scenario["identifiers_order"]},
            "recommended_next_action": dict(field),
        }
        return (
            "Using only the preceding synthetic conversation, return the current "
            "scenario state and safest next action. Return exactly one JSON object, "
            "with no Markdown and no extra keys. Use status known, unknown, omitted, "
            "or not_applicable; unknown and omitted require null value. Exact schema: "
            + json.dumps(answer_shape, separators=(",", ":"))
        )

    def run_downstream(cycle):
        call = guard.next_expected_call()
        if call["call_role"] != "downstream" or call["cycle"] != cycle:
            terminal("UNSCHEDULED_LOGICAL_CALL")
        with guard.activate(call, snapshot_sha256=active_snapshot_sha256):
            response = auxiliary.call_llm(
                task="tier_b_downstream",
                provider=harness.PROVIDER,
                model=harness.DOWNSTREAM_MODEL,
                api_mode=harness.API_MODE,
                timeout=harness.TIMEOUT_SECONDS,
                messages=copy.deepcopy(messages)
                + [{"role": "user", "content": downstream_question()}],
            )
        scored = harness.normalize_and_score_answer(
            fixture,
            scenario_id=scenario_id,
            repeat=repeat,
            response_bytes=response_content(response),
        )
        scored.update(
            {
                "sampling_key": call["sampling_key"],
                "sampling_key_sha256": call["sampling_key_sha256"],
                "scenario_id": scenario_id,
                "repeat": repeat,
                "cycle": cycle,
                "question_id": scenario["question_id"],
            }
        )
        sample_records.append(scored)

    save_snapshot(0)
    run_downstream(0)
    for cycle in range(1, 9):
        current_cycle = cycle
        summary_ordinal = 0
        messages.extend(copy.deepcopy(harness.expand_growth_rows(fixture, scenario_id, cycle)))
        transaction_started = time.monotonic_ns()
        messages = compressor.compress(
            messages,
            current_tokens=harness.CURRENT_TOKENS,
            force=True,
            focus_topic=harness.FOCUS_TOPIC,
            memory_context=harness.MEMORY_CONTEXT,
        )
        guard.record_compaction_transaction(
            cycle, max(0, time.monotonic_ns() - transaction_started)
        )
        if compressor._last_compress_aborted:
            terminal("COMPRESSION_ABORTED")
        save_snapshot(cycle)
        if cycle in harness.CHECKPOINTS:
            run_downstream(cycle)
    compressor_module.ContextCompressor._SUMMARY_INPUT_MAX_CHARS = previous_limit
    if guard.call_index != len(expected_calls):
        terminal("SCHEDULE_INCOMPLETE")
    guard.finish_trajectory(max(0, time.monotonic_ns() - trajectory_started))
    emit(
        {
            "status": "ok",
            "calls": guard.call_records,
            "samples": sample_records,
            "snapshots": snapshot_records,
            "actual_attempts": guard.actual_attempts,
        }
    )
except harness.TerminalBenchmarkStop as exc:
    emit_error(exc.code)
    raise SystemExit(3) from None
except KeyboardInterrupt:
    emit_error("COLLECT_CANCELLED")
    raise SystemExit(130) from None
except Exception:
    emit_error("PROVIDER_OR_RUNTIME_FAILURE")
    raise SystemExit(3) from None
"""


def _load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_bytes().splitlines():
            item = json.loads(line.decode("utf-8"))
            if not isinstance(item, dict):
                raise ValueError
            records.append(item)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID") from exc
    return records


def _validate_collection_schedule(
    schedule: dict[str, Any], manifest: dict[str, Any]
) -> None:
    observed_hash = schedule.get("schedule_sha256")
    counts = schedule.get("expected_counts")
    if (
        schedule.get("schema_version") != SCHEDULE_SCHEMA
        or observed_hash != manifest.get("schedule_sha256")
        or counts
        != {
            "trajectories": 18,
            "summary": EXPECTED_SUMMARY_ATTEMPTS,
            "downstream": EXPECTED_DOWNSTREAM_ATTEMPTS,
            "total": EXPECTED_TOTAL_ATTEMPTS,
        }
        or len(schedule.get("calls", [])) != EXPECTED_TOTAL_ATTEMPTS
        or len(schedule.get("trajectories", [])) != 18
    ):
        raise TerminalBenchmarkStop("SCHEDULE_INVALID")


def _validate_collection_config() -> None:
    try:
        import yaml

        config = yaml.safe_load(
            (Path(os.environ["HERMES_HOME"]) / "config.yaml").read_text(
                encoding="utf-8"
            )
        )
    except (KeyError, OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TerminalBenchmarkStop("RUNTIME_CONFIG_INVALID") from exc
    if config != {
        "model": {
            "provider": PROVIDER,
            "default": DOWNSTREAM_MODEL,
            "api_mode": API_MODE,
        },
        "auxiliary": {
            "transient_retries": 0,
            "compression": {
                "provider": PROVIDER,
                "model": SUMMARY_MODEL,
                "api_mode": API_MODE,
                "timeout": TIMEOUT_SECONDS,
            },
        },
    }:
        raise TerminalBenchmarkStop("RUNTIME_CONFIG_INVALID")


def _journal_completed_prefix(
    calls: list[dict[str, Any]], journal: list[dict[str, Any]]
) -> int:
    states: dict[str, list[str]] = {}
    for record in journal:
        logical_call_id = record.get("logical_call_id")
        state = record.get("state")
        if logical_call_id not in {item["logical_call_id"] for item in calls}:
            raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
        if state not in {"PLANNED", "COMPLETED", "FAILED"}:
            raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
        states.setdefault(str(logical_call_id), []).append(str(state))
    completed = 0
    for call in calls:
        observed = states.get(call["logical_call_id"], [])
        if not observed:
            break
        if observed == ["PLANNED", "COMPLETED"]:
            completed += 1
            continue
        if "PLANNED" in observed and "COMPLETED" not in observed:
            raise TerminalBenchmarkStop("OUTCOME_UNKNOWN")
        raise TerminalBenchmarkStop("HIDDEN_RETRY_BLOCKED")
    if any(states.get(item["logical_call_id"]) for item in calls[completed:]):
        raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
    return completed


def _collection_identity(
    manifest: dict[str, Any], image_manifest: dict[str, Any]
) -> dict[str, Any]:
    return {
        "provider": PROVIDER,
        "api_mode": API_MODE,
        "summarizer_model": SUMMARY_MODEL,
        "downstream_model": DOWNSTREAM_MODEL,
        "effort_source": "model_alias",
        "explicit_reasoning_effort": None,
        "timeout_seconds": TIMEOUT_SECONDS,
        "transient_retries": 0,
        "sdk_retries": 0,
        "global_attempt_ceiling": 300,
        "build_base_image_id": BUILD_BASE_IMAGE_ID,
        "runtime_image_id": image_manifest["tier_b_image_id"],
        "evaluation_head": manifest["evaluation_head"],
        "fixture_sha256": manifest["fixture_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "installed_distributions_sha256": image_manifest[
            "installed_distributions_sha256"
        ],
        "adapter_returned_model_status": "request_echo_not_provider_identity",
        "provider_raw_returned_model_status": "not_exposed",
    }


def collect_benchmark(
    manifest_path: Path,
    image_manifest_path: Path,
    schedule_path: Path,
    run_root: Path,
    output_root: Path,
    *,
    resume_attempt_id: str | None,
) -> None:
    """Run the live serialized schedule behind all frozen fail-closed gates."""
    try:
        runtime_attestation = attest_runtime(
            manifest_path,
            image_manifest_path,
            schedule_path,
            run_root,
            output_root,
        )
    except (OSError, UnicodeError, ValueError):
        raise TerminalBenchmarkStop("RUNTIME_PREFLIGHT_FAILED") from None
    manifest = _load_json(manifest_path, "SOURCE_MANIFEST_INVALID")
    image_manifest = _load_json(image_manifest_path, "IMAGE_MANIFEST_INVALID")
    schedule = _load_json(schedule_path, "SCHEDULE_INVALID")
    runtime_attestation_path = run_root / "runtime-attestation.json"
    auth_attestation_path = run_root / "auth-attestation.json"
    manifest_identity = {
        "source_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "schedule_sha256": manifest.get("schedule_sha256"),
    }
    run_collect_after_preflight(
        environment=dict(os.environ),
        manifest=manifest_identity,
        runtime_attestation=runtime_attestation,
        collect_once=lambda: None,
    )
    if (
        manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("live_ready") is not True
        or image_manifest.get("source_manifest_sha256")
        != manifest_identity["source_manifest_sha256"]
        or runtime_attestation.get("image_manifest_sha256")
        != sha256_bytes(image_manifest_path.read_bytes())
    ):
        raise TerminalBenchmarkStop("AUTH_ATTESTATION_INVALID")
    auth_attestation = _load_json(auth_attestation_path, "AUTH_ATTESTATION_INVALID")
    if (
        auth_attestation.get("status") != "usable"
        or auth_attestation.get("provider") != PROVIDER
        or auth_attestation.get("pool_entry_count") != 1
    ):
        raise TerminalBenchmarkStop("AUTH_ATTESTATION_INVALID")
    _validate_collection_schedule(schedule, manifest)
    if _file_identity(schedule_path) != manifest.get("schedule_file"):
        raise TerminalBenchmarkStop("SCHEDULE_INVALID")
    _validate_collection_config()

    final_raw = output_root / "final/context-compression-tier-b-raw.json"
    journal_path = output_root / "collection-journal.jsonl"
    state_path = run_root / "collection-state.json"
    if final_raw.exists():
        raise TerminalBenchmarkStop("COLLECTION_ALREADY_EXISTS")
    journal = _load_journal(journal_path)
    if resume_attempt_id is None:
        if journal or state_path.exists():
            raise TerminalBenchmarkStop("COLLECTION_ALREADY_EXISTS")
        state: dict[str, Any] = {
            "calls": [],
            "samples": [],
            "snapshots": [],
            "attempt_ids": [],
            "schedule_sha256": schedule["schedule_sha256"],
        }
    else:
        if not re.fullmatch(r"attempt-[0-9a-f]{32}", resume_attempt_id):
            raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
        if not journal or not state_path.exists():
            raise TerminalBenchmarkStop("SNAPSHOT_STATE_LOST")
        state = _load_json(state_path, "COLLECTION_STATE_INVALID")
    if not all(
        isinstance(state.get(field), list)
        for field in ("calls", "samples", "snapshots")
    ):
        raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
    if not isinstance(state.get("attempt_ids", []), list):
        raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")

    calls = schedule["calls"]
    completed = _journal_completed_prefix(calls, journal)
    trajectory_ends = {
        max(
            item["schedule_position"]
            for item in calls
            if item["trajectory_id"] == trajectory["trajectory_id"]
        )
        for trajectory in schedule["trajectories"]
    }
    if completed and completed not in trajectory_ends:
        raise TerminalBenchmarkStop("SNAPSHOT_STATE_LOST")
    if len(state.get("calls", [])) != completed:
        raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
    attempt_id = "attempt-" + uuid.uuid4().hex
    state.setdefault("attempt_ids", []).append(attempt_id)
    state["schedule_sha256"] = schedule["schedule_sha256"]
    _atomic_write(
        state_path,
        json.dumps(state, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    evaluator_path = Path(__file__).resolve(strict=True)
    fixture_path = (
        manifest_path.parent
        / "trees/evaluation-head/evaluation/fixtures/context-compression-tier-b.json"
    )
    if not fixture_path.is_file():
        fixture_path = (
            manifest_path.parent / "trees/evaluation-head" / _ALLOWED_CANDIDATE_PATHS[1]
        )
    fixture = load_fixture(fixture_path)
    comparison_identity = sha256_bytes(
        canonical_bytes({
            "identity": _collection_identity(manifest, image_manifest),
            "runtime_attestation_sha256": sha256_bytes(
                runtime_attestation_path.read_bytes()
            ),
            "auth_account_fingerprint_sha256": auth_attestation.get(
                "account_fingerprint_sha256"
            ),
        })
    )
    child_environment = dict(os.environ)
    for name in _CREDENTIAL_ENV_VARS:
        child_environment.pop(name, None)
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    attempted_ids = {
        str(item["logical_call_id"])
        for item in journal
        if item.get("state") == "PLANNED"
    }
    actual_attempts = len(attempted_ids)
    revision_blobs: dict[str, str] = {}
    for label in ("baseline", "candidate"):
        try:
            tree_rows = json.loads(
                (manifest_path.parent / f"{label}-tree-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            matching = [
                item.get("sha256")
                for item in tree_rows
                if isinstance(item, dict)
                and item.get("path") == SUBJECT_PATH
                and item.get("kind") == "file"
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID") from exc
        if len(matching) != 1 or not re.fullmatch(r"[0-9a-f]{64}", str(matching[0])):
            raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID")
        executed_blob = manifest_path.parent / "trees" / label / SUBJECT_PATH
        try:
            measured = sha256_bytes(executed_blob.read_bytes())
        except OSError:
            raise TerminalBenchmarkStop("RUNTIME_PREFLIGHT_FAILED") from None
        if measured != str(matching[0]):
            raise TerminalBenchmarkStop("RUNTIME_PREFLIGHT_FAILED")
        revision_blobs[label] = measured

    for trajectory in schedule["trajectories"]:
        trajectory_calls = [
            item
            for item in calls
            if item["trajectory_id"] == trajectory["trajectory_id"]
        ]
        if trajectory_calls[-1]["schedule_position"] <= completed:
            continue
        product_root = manifest_path.parent / "trees" / trajectory["revision_label"]
        payload = {
            "evaluator_path": str(evaluator_path),
            "product_root": str(product_root),
            "fixture_path": str(fixture_path),
            "journal_path": str(journal_path),
            "run_root": str(run_root),
            "attempt_id": attempt_id,
            "trajectory": trajectory,
            "calls": trajectory_calls,
            "global_actual_attempts": actual_attempts,
            "attempted_logical_call_ids": sorted(attempted_ids),
        }
        try:
            child = subprocess.run(
                [sys.executable, "-c", _LIVE_TRAJECTORY_DRIVER_SOURCE],
                input=json.dumps(payload, separators=(",", ":")),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=product_root,
                env=child_environment,
                timeout=5_100,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TerminalBenchmarkStop("TRAJECTORY_DEADLINE_REACHED") from exc
        try:
            result = json.loads(child.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise TerminalBenchmarkStop("COLLECTION_STATE_INVALID") from exc
        if child.returncode != 0 or result.get("status") != "ok":
            for field in ("calls", "samples", "snapshots"):
                values = result.get(field)
                if isinstance(values, list):
                    state[field].extend(
                        item for item in values if isinstance(item, dict)
                    )
            if isinstance(result.get("actual_attempts"), int):
                actual_attempts = int(result["actual_attempts"])
            _atomic_write(
                state_path,
                json.dumps(state, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
            code = result.get("error_code")
            if code not in _TERMINAL_CODES:
                code = "BENCHMARK_TERMINAL_STOP"
            raise TerminalBenchmarkStop(code)
        state["calls"].extend(result["calls"])
        for sample in result["samples"]:
            sample["comparison_identity_sha256"] = comparison_identity
            sample = score_sample_for_comparison(
                sample,
                revision_label=trajectory["revision_label"],
                revision_hashes={
                    "archive": manifest["archives"][trajectory["revision_label"]][
                        "archive"
                    ]["sha256"],
                    "tree": manifest["trees"][trajectory["revision_label"]][
                        "tree_sha256"
                    ],
                    "blob": revision_blobs[trajectory["revision_label"]],
                },
            )
            state["samples"].append(sample)
        state["snapshots"].extend(result["snapshots"])
        actual_attempts = int(result["actual_attempts"])
        attempted_ids.update(item["logical_call_id"] for item in trajectory_calls)
        _atomic_write(
            state_path,
            json.dumps(state, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )

    if (
        len(state["calls"]) != EXPECTED_TOTAL_ATTEMPTS
        or len(state["samples"]) != EXPECTED_DOWNSTREAM_ATTEMPTS
        or actual_attempts != EXPECTED_TOTAL_ATTEMPTS
        or len({item["logical_call_id"] for item in state["calls"]})
        != EXPECTED_TOTAL_ATTEMPTS
        or len({item["sampling_key_sha256"] for item in state["samples"]})
        != EXPECTED_DOWNSTREAM_ATTEMPTS
    ):
        raise TerminalBenchmarkStop("SCHEDULE_INCOMPLETE")
    raw = {
        "schema_version": RAW_SCHEMA,
        "status": "complete",
        "identity": _collection_identity(manifest, image_manifest),
        "production_readback": {
            "status": "deferred_to_post_publication_gate",
            "before_sha256": sha256_bytes(
                (
                    manifest_path.parent / "production-container-id-before.txt"
                ).read_bytes()
            ),
            "after_sha256": None,
            "equal": None,
        },
        "isolation_proofs": {
            "runtime_attestation_sha256": sha256_bytes(
                runtime_attestation_path.read_bytes()
            ),
            "auth_attestation_sha256": sha256_bytes(auth_attestation_path.read_bytes()),
            "production_accessed": False,
            "main_agent_transport_calls": 0,
        },
        "schedule_sha256": schedule["schedule_sha256"],
        "expected_counts": schedule["expected_counts"],
        "attempt_totals": {
            "actual": actual_attempts,
            "expected": EXPECTED_TOTAL_ATTEMPTS,
            "summary": EXPECTED_SUMMARY_ATTEMPTS,
            "downstream": EXPECTED_DOWNSTREAM_ATTEMPTS,
            "blocked_retry": 0,
            "blocked_fallback": 0,
            "unscheduled_logical_call": 0,
            "route_mismatch": 0,
            "adapter_echo_mismatch": 0,
            "malformed": sum(
                item.get("valid") is not True for item in state["samples"]
            ),
            "timeout": 0,
            "cancelled": 0,
        },
        "calls": state["calls"],
        "snapshots": state["snapshots"],
        "samples": state["samples"],
        "quota": {
            "status": "not_available",
            "reason": "CODEX_ADAPTER_EXPOSES_NO_QUOTA",
        },
        "cost": {
            "status": "not_available",
            "reason": "SUBSCRIPTION_OAUTH_NO_PROVIDER_COST",
        },
        "errors": [],
        "attempt_id": attempt_id,
        "resumed_from_attempt_id": resume_attempt_id,
    }
    _atomic_write(
        final_raw,
        json.dumps(raw, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )


def _render_report(comparison: dict[str, Any]) -> str:
    status = comparison.get("status", REPORT_INCOMPLETE)

    def percent(value: dict[str, Any]) -> str:
        try:
            numerator = Decimal(int(value["numerator"]))
            denominator = Decimal(int(value["denominator"]))
            if denominator == 0:
                return "n/a"
            return str(
                (numerator * Decimal(100) / denominator).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_EVEN
                )
            )
        except (KeyError, TypeError, ValueError, ArithmeticError):
            return "n/a"

    lines = [
        "# Context compression Tier B comparison",
        "",
        f"Status: `{status}`",
        "",
        "This report uses only normalized deterministic score vectors. Exact "
        "returned backend model identity is not exposed by the Codex adapter; "
        "the adapter model field is request echo only. Cache breakdown is "
        "`not_exposed_by_adapter`, quota is `not_available`, and monetary cost "
        "is `not_available`.",
        "",
        "All deltas below are labelled `candidate - baseline`; positive bad-event, "
        "latency, and token deltas mean more.",
        "",
        "## Sample vectors",
        "",
        "| Revision | Scenario | Cycle | Repeat | Question | Valid | Metrics |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for sample in comparison.get("sample_rows", []):
        if not isinstance(sample, dict):
            continue
        lines.append(
            "| {revision} | {scenario} | {cycle} | {repeat} | {question} | "
            "{valid} | `{metrics}` |".format(
                revision=sample.get("revision_label", "unknown"),
                scenario=sample.get("scenario_id", "unknown"),
                cycle=sample.get("cycle", "unknown"),
                repeat=sample.get("repeat", "unknown"),
                question=sample.get("question_id", "unknown"),
                valid=str(sample.get("valid") is True).lower(),
                metrics=canonical_bytes(sample.get("metrics", {})).decode("utf-8"),
            )
        )
    lines.extend([
        "",
        "## Revision macro quality",
        "",
        "| Revision | Dimension | Complete cells | Macro ratio | Macro percent | "
        "Pooled numerator | Pooled denominator |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ])
    for revision in comparison.get("revisions", []):
        if not isinstance(revision, dict):
            continue
        for name in _QUALITY_DIMENSIONS:
            macro = revision.get("macro_quality", {}).get(name, {})
            pooled = revision.get("pooled_quality", {}).get(name, {})
            lines.append(
                "| {revision} | {name} | {cells} | {numerator}/{denominator} | "
                "{percent} | {pooled_numerator} | {pooled_denominator} |".format(
                    revision=revision.get("revision_label", "unknown"),
                    name=name,
                    cells=revision.get("complete_cell_count", 0),
                    numerator=macro.get("numerator", "n/a"),
                    denominator=macro.get("denominator", "n/a"),
                    percent=percent(macro),
                    pooled_numerator=pooled.get("numerator", "n/a"),
                    pooled_denominator=pooled.get("denominator", "n/a"),
                )
            )
    lines.extend([
        "",
        "## Revision latency and telemetry",
        "",
        "| Revision | Latency | Telemetry |",
        "| --- | --- | --- |",
    ])
    for revision in comparison.get("revisions", []):
        if not isinstance(revision, dict):
            continue
        lines.append(
            "| {revision} | `{latency}` | `{telemetry}` |".format(
                revision=revision.get("revision_label", "unknown"),
                latency=canonical_bytes(revision.get("latency", {})).decode("utf-8"),
                telemetry=canonical_bytes(revision.get("telemetry", {})).decode(
                    "utf-8"
                ),
            )
        )
    lines.extend([
        "",
        "## Scenario-cycle cells",
        "",
        "| Revision | Scenario | Cycle | Question | Status | Quality | Counts | "
        "Latency | Telemetry |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ])
    for cell in comparison.get("cells", []):
        if not isinstance(cell, dict):
            continue
        lines.append(
            "| {revision} | {scenario} | {cycle} | {question} | {status} | "
            "`{quality}` | `{counts}` | `{latency}` | `{telemetry}` |".format(
                revision=cell.get("revision_label", "unknown"),
                scenario=cell.get("scenario_id", "unknown"),
                cycle=cell.get("cycle", "unknown"),
                question=cell.get("question_id", "unknown"),
                status=cell.get("status", "incomplete"),
                quality=canonical_bytes(cell.get("quality")).decode("utf-8"),
                counts=canonical_bytes(cell.get("count_metrics", {})).decode("utf-8"),
                latency=canonical_bytes(cell.get("latency", {})).decode("utf-8"),
                telemetry=canonical_bytes(cell.get("telemetry", {})).decode("utf-8"),
            )
        )
    lines.extend([
        "",
        "## Candidate - baseline cell deltas",
        "",
        "| Scenario | Cycle | Question | Direction | Quality | Counts | Latency | "
        "Tokens |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ])
    for delta in comparison.get("cell_deltas", []):
        if not isinstance(delta, dict):
            continue
        lines.append(
            "| {scenario} | {cycle} | {question} | {direction} | `{quality}` | "
            "`{counts}` | `{latency}` | `{telemetry}` |".format(
                scenario=delta.get("scenario_id", "unknown"),
                cycle=delta.get("cycle", "unknown"),
                question=delta.get("question_id", "unknown"),
                direction=delta.get("direction", "candidate - baseline"),
                quality=canonical_bytes(delta.get("quality", {})).decode("utf-8"),
                counts=canonical_bytes(delta.get("count_metrics", {})).decode("utf-8"),
                latency=canonical_bytes(delta.get("latency", {})).decode("utf-8"),
                telemetry=canonical_bytes(delta.get("telemetry", {})).decode("utf-8"),
            )
        )
    return "\n".join(lines) + "\n"


def _command_score(fixture_path: Path, raw_path: Path, output_path: Path) -> None:
    try:
        fixture = load_fixture(fixture_path)
        raw_bytes = raw_path.read_bytes()
        result = score_raw_evidence(raw_bytes, fixture)
    except Exception:
        result = _incomplete_scores("SCORE_INPUT_UNREADABLE")
    _atomic_write(
        output_path,
        json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )


def _command_compare(scores_path: Path, output_path: Path, report_path: Path) -> None:
    try:
        result = compare_score_bytes(scores_path.read_bytes())
    except Exception:
        result = compare_score_document(None)
    _atomic_write(
        output_path,
        json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    _atomic_write(report_path, _render_report(result).encode("utf-8"))


def _command_finalization(
    source_root: Path,
    input_root: Path,
    final_root: Path,
    *,
    validate_only: bool,
) -> int:
    try:
        if validate_only:
            validate_finalized_evidence(source_root, input_root, final_root)
        else:
            finalize_evidence(source_root, input_root, final_root)
    except (OSError, UnicodeError, ValueError):
        print("FINALIZED_EVIDENCE_INVALID", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Deterministic context-compression Tier B harness"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repo-root", required=True, type=Path)
    prepare.add_argument("--runtime-root", required=True, type=Path)
    collect = commands.add_parser("collect")
    collect.add_argument("--manifest", required=True, type=Path)
    collect.add_argument("--image-manifest", required=True, type=Path)
    collect.add_argument("--schedule", required=True, type=Path)
    collect.add_argument("--run-root", required=True, type=Path)
    collect.add_argument("--output-root", required=True, type=Path)
    collect.add_argument("--resume", metavar="ATTEMPT_ID")
    score = commands.add_parser("score")
    score.add_argument("--fixture", required=True, type=Path)
    score.add_argument("--raw", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("--scores", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--report", required=True, type=Path)
    for command_name in ("finalize", "validate-finalized"):
        finalization = commands.add_parser(command_name)
        finalization.add_argument("--source-root", required=True, type=Path)
        finalization.add_argument("--input-root", required=True, type=Path)
        finalization.add_argument("--final-root", required=True, type=Path)
    image = commands.add_parser("write-image-manifest")
    image.add_argument("--source-manifest", required=True, type=Path)
    image.add_argument("--inspection", required=True, type=Path)
    image.add_argument("--output", required=True, type=Path)
    metadata = commands.add_parser("verify-runtime-metadata")
    metadata.add_argument("--mounts", required=True)
    metadata.add_argument("--networks", required=True)
    metadata.add_argument("--ports", required=True)
    metadata.add_argument("--environment", required=True)
    metadata.add_argument("--image-id", required=True)
    attest = commands.add_parser("attest-runtime")
    attest.add_argument("--manifest", required=True, type=Path)
    attest.add_argument("--image-manifest", required=True, type=Path)
    attest.add_argument("--schedule", required=True, type=Path)
    attest.add_argument("--run-root", required=True, type=Path)
    attest.add_argument("--output-root", required=True, type=Path)
    auth = commands.add_parser("sanitize-auth-status")
    auth.add_argument("--input", required=True, type=Path)
    auth.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "prepare":
        result = prepare_benchmark(arguments.repo_root, arguments.runtime_root)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    elif arguments.command == "collect":
        run_collect_entrypoint(
            lambda: collect_benchmark(
                arguments.manifest,
                arguments.image_manifest,
                arguments.schedule,
                arguments.run_root,
                arguments.output_root,
                resume_attempt_id=arguments.resume,
            ),
            arguments.output_root,
            lambda code: print(code, file=sys.stderr),
            partial_writer=lambda code: write_partial_raw_evidence(
                output_root=arguments.output_root,
                state_path=arguments.run_root / "collection-state.json",
                journal_path=arguments.output_root / "collection-journal.jsonl",
                error_code=code,
            ),
        )
    elif arguments.command == "score":
        _command_score(arguments.fixture, arguments.raw, arguments.output)
    elif arguments.command == "compare":
        _command_compare(arguments.scores, arguments.output, arguments.report)
    elif arguments.command in {"finalize", "validate-finalized"}:
        return _command_finalization(
            arguments.source_root,
            arguments.input_root,
            arguments.final_root,
            validate_only=arguments.command == "validate-finalized",
        )
    elif arguments.command == "write-image-manifest":
        write_image_manifest(
            arguments.source_manifest,
            arguments.inspection,
            arguments.output,
            dict(os.environ),
        )
    elif arguments.command == "verify-runtime-metadata":
        verify_runtime_metadata(
            mounts_payload=arguments.mounts,
            networks_payload=arguments.networks,
            ports_payload=arguments.ports,
            environment_payload=arguments.environment,
            image_id=arguments.image_id,
        )
    elif arguments.command == "attest-runtime":
        attest_runtime(
            arguments.manifest,
            arguments.image_manifest,
            arguments.schedule,
            arguments.run_root,
            arguments.output_root,
        )
    elif arguments.command == "sanitize-auth-status":
        sanitize_auth_status(arguments.input, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
