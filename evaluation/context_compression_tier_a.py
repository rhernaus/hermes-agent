#!/usr/bin/env python3
"""Tier A deterministic comparison for the bounded context-compaction change.

Disposable evaluation-only harness. It compares two frozen revisions of
``agent/context_compressor.py`` under one sanitized synthetic fixture, a fake
provider with a fixed artificial delay, a frozen temporal anchor, and a frozen
deterministic failure/cancellation schedule. No network, no provider spend, no
product source is touched.

Revision isolation: each revision's tree is extracted from git into its own
temporary directory and driven by a separate subprocess whose working
directory is that tree. The two revisions are never imported into one Python
process. The child proves this by reporting the resolved module path and its
content hash, which the parent checks against the git blob for that exact
revision.

Evidence boundary: only equality booleans, hashes, counts, safe marker and
field names, allowlisted error codes, and aggregate numbers are persisted.
Raw prompts, transcript text, compressor state values, exception strings, and
absolute paths never reach the output. A final recursive validation pass
enforces this and fails closed.

Publication boundary: this process never writes into the pre-seeded FAIL
evidence directory. It stages the finalized JSON, the finalized Markdown, a
checksum file, and — last — an explicit finalization marker, all in its own
staging directory. The workflow promotes that whole directory by a single
atomic rename, and only after this process exits 0 AND the complete pair
validates against the checksums; the FAIL directory is write-once and is
selected whenever promotion did not fully succeed. The exit code reports
whether a pair was finalized, NOT the gate verdict: a legitimate gate FAIL
exits 0 so its own evidence is published, while a crash or a staging failure
exits non-zero and leaves the FAIL directory selected.

Scope limits (do not over-read the output):
  * This is Tier A only — deterministic correctness plus fake-delay call,
    token, and latency accounting. It says nothing about summary quality, real
    provider latency or cost, or production readiness.
  * Baseline coverage loss is control evidence about the pre-change behavior,
    never a candidate gate failure.
  * When the candidate is technically correct but issues more auxiliary calls,
    sends more prompt characters or estimated tokens, or takes longer under
    the fixed fake delay, the status is ``TECHNICAL_PASS_TRADEOFF_PENDING``
    and the process still succeeds: this harness deliberately does not invent
    a materiality threshold for that trade-off. The exact deltas are reported
    for a maintainer decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Frozen identities ────────────────────────────────────────────────────
SCHEMA_VERSION = "context-compression-tier-a/2"
BASELINE_SHA = "d5e135a51353c2dbc489d5c2583158b22d8efd7b"
CANDIDATE_SHA = "f07664bb9a19788ec426db2eb8b8ec8d9572b21d"
EVAL_BRANCH = "eval/compaction-tier-a"
PR_BASE_BRANCH = "fix/compaction-complete-coverage"
SUBJECT_PATH = "agent/context_compressor.py"
FIXTURE_RELPATH = "evaluation/fixtures/context-compression-tier-a.json"
FIXTURE_SHA256 = (
    "5d1fc699d6343540c16db612d96d17215986d1a36bcc22d94d99cd1d62bdb4c7"
)
# Fixed relative artifact names. Absolute paths are never printed or persisted.
EVIDENCE_JSON_NAME = "context-compression-tier-a.json"
EVIDENCE_SUMMARY_NAME = "context-compression-tier-a.md"
# Written last, into the staging directory only, once the complete finalized
# pair is on disk. Its presence plus a matching checksum file is the sole
# signal that lets the workflow select the finalized pair for upload.
FINALIZATION_MARKER_NAME = "FINALIZED"
FINALIZATION_CHECKSUM_NAME = "FINALIZED.sha256"

# ── Frozen Tier A contract parameters (identical for both revisions) ──────
CYCLES: Tuple[int, ...] = (0, 1, 2, 4, 8)
TRIALS = 3
FAKE_DELAY_SECONDS = 0.02
SPLIT_LIMIT_CHARS = 12_000
FRAGMENT_LIMIT_CHARS = 600
BELOW_CAP_LIMIT_CHARS = 160_000
CONTEXT_LENGTH = 128_000
TAIL_TOKEN_BUDGET = 600
CURRENT_TOKENS = 100_000
MAIN_MODEL = "tier-a/main-model"
AUX_MODEL = "tier-a/aux-model"
# The single request component an auxiliary-to-main retry is EXPECTED to
# change: the first request carries the configured auxiliary model, the retry
# drops the override so the main model is used. Every other component must be
# byte-identical, so this one is compared on its own rather than folded into
# the payload hash.
ROUTING_DISCRIMINATOR = "model"
# Frozen focus/memory inputs, applied to every multi-pass and cycle call on
# both revisions. Provider memory is interpolated immediately after the source
# content inside the same prompt delimiters, so the driver splits it back out
# on its own header rather than leaving it to contaminate source-block parsing.
FOCUS_TOPIC = "tier-a synthetic focus topic"
MEMORY_CONTEXT = "tier-a synthetic memory context"
# Frozen temporal anchor. The summarizer prompt carries a date line; freezing
# the clock at its source (hermes_time.now) keeps prompt bytes comparable
# byte-for-byte instead of normalizing them away after the fact.
FROZEN_CLOCK_UTC = "2026-01-02T03:04:05+00:00"
# Frozen deterministic failure/cancellation schedule, applied identically to
# both revisions. Keys are 1-based auxiliary call ordinals within a scenario.
FAILURE_SCHEDULE = {
    "rollback": {"2": "fail"},
    "cancellation": {"2": "cancel"},
    "retry_first_pass": {"1": "timeout"},
    "retry_later_pass": {"2": "timeout"},
}
# Frozen exclusions from the exact post-call state and telemetry comparisons.
# Both carry wall-clock or identity values that cannot be deterministic; every
# other key and value must match exactly, and an unnamed new key fails.
STATE_EXCLUSIONS = ("_active_compression_telemetry", "_last_compression_telemetry")
TELEMETRY_EXCLUSIONS = ("attempt_id", "aux_call_duration_ms", "total_duration_ms")

# Provider credential variables cleared before any child runs. The evaluator
# never reads them; clearing them makes an accidental real call impossible.
CREDENTIAL_ENV_VARS = (
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
)

# Allowlisted error codes. Nothing else is ever persisted for a failure: no
# exception text, no traceback, no path.
ERROR_CODES = frozenset({
    "FIXTURE_MISSING",
    "FIXTURE_HASH_MISMATCH",
    "FIXTURE_MALFORMED",
    "REVISION_UNRESOLVED",
    "REVISION_BLOB_UNREADABLE",
    "REVISION_BLOBS_IDENTICAL",
    "REVISION_EXTRACTION_FAILED",
    "DRIVER_NO_EVIDENCE",
    "DRIVER_REPORTED_ERROR",
    "DRIVER_IDENTITY_MISMATCH",
    "DRIVER_EVIDENCE_MALFORMED",
    "DRIVER_METRIC_INVALID",
    "OUTPUT_VALIDATION_FAILED",
    "UNEXPECTED_INTERNAL_ERROR",
})

# Allowlisted codes the child may report. Anything else is recorded as
# DRIVER_UNKNOWN_CODE so an unexpected string can never reach the evidence.
DRIVER_ERROR_CODES = frozenset({
    "DRIVER_IMPORT_FAILED",
    "DRIVER_ISOLATION_VIOLATED",
    "DRIVER_SENTINEL_DRIFT",
    "DRIVER_CLOCK_SEAM_MISSING",
    "DRIVER_PROMPT_SHAPE_UNRECOGNIZED",
    "DRIVER_SCENARIO_NOT_APPLICABLE",
    "DRIVER_NONDETERMINISTIC",
    "DRIVER_EMPTY_SAMPLES",
    "DRIVER_METRIC_INVALID",
    "DRIVER_UNEXPECTED_ERROR",
})
# Exception class names the child may report alongside DRIVER_UNEXPECTED_ERROR.
# A class name is a bare identifier, but pinning the shape keeps the boundary
# mechanical rather than trusting the child.
DRIVER_ERROR_CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# Hard gates. Every entry maps to one clause of the frozen Tier A contract and
# is decided from a named check the driver reports for the candidate revision.
GATE_SPECS: Tuple[Tuple[str, Optional[str], str], ...] = (
    (
        "g1_source_coverage_order_provenance",
        "aggregate_coverage",
        "Zero source-turn or critical-marker omission after per-message "
        "serialization/redaction; exact order and role provenance; frozen "
        "focus and memory reach every bounded pass; the final summary retains "
        "every marker carried through the earlier tentative summaries.",
    ),
    (
        "g1b_full_window_budget_and_provenance",
        "full_window_derivation",
        "With a final bounded pass that holds no user row, the summary budget "
        "and user provenance are still derived from the complete window, not "
        "from the final chunk.",
    ),
    (
        "g2a_completed_tool_group_grouping",
        "tool_group",
        "A completed tool call and its matching result stay in one bounded "
        "pass when the group fits, call before result, each exactly once.",
    ),
    (
        "g2b_oversized_row_fragment_reconstruction",
        "fragment_reconstruction",
        "Fragments of an oversized serialized row reconstruct the "
        "unfragmented payload byte-for-byte.",
    ),
    (
        "g3a_later_pass_failure_rollback",
        "rollback",
        "A later-pass failure restores the summary entry to exactly the "
        "normalized pre-pass seed, records the expected diagnostic and "
        "cooldown transition, and leaves source turns untouched.",
    ),
    (
        "g3b_later_pass_cancellation",
        "cancellation",
        "A later-pass cancellation returns no summary, restores exactly the "
        "normalized seed, and leaves diagnostic, cooldown, auth, and network "
        "state exactly as they were.",
    ),
    (
        "g3c_fallback_and_configured_abort",
        "fallback_and_abort",
        "Configured abort returns a canonically unchanged transcript; the "
        "continue path publishes a deterministic fallback handoff carrying "
        "the full-window task anchor and pruned-skill marker.",
    ),
    (
        "g3d_aux_to_main_retry_placement",
        "retry_placement",
        "The auxiliary-to-main retry re-issues the same pass rather than "
        "restarting the sequence: every non-routing request-payload component "
        "is identical, the source block is identical, and routing changes "
        "exactly from the configured auxiliary model to the main model with "
        "no auxiliary override.",
    ),
    (
        "g3e_restart_and_cross_session_isolation",
        "restart_cross_session",
        "Restart rehydrates exactly one handoff and preserves the critical "
        "marker; an unrelated session never sees it.",
    ),
    (
        "g4_window_integrity_and_publication",
        "window_integrity",
        "Input/window hashes recorded; exact handoff framing, role, and "
        "persistence metadata; exactly one publication; no partial or "
        "prefix-only publication; no marker, critical-marker, or drift loss "
        "across the complete 0/1/2/4/8 progression.",
    ),
    (
        "g5_redaction_and_pruned_skill_marker",
        "redaction_and_marker",
        "Redaction holds on every prompt and output; the exact pruned-skill "
        "marker survives to the committed summary.",
    ),
    (
        "g6_below_cap_exact_parity",
        None,
        "Below-cap one-pass complete request payload, result, complete stable "
        "post-call state, and complete telemetry mapping are byte-for-byte "
        "equal between baseline and candidate; any unnamed delta fails.",
    ),
    (
        "g7_reusable_prefix_byte_identity",
        None,
        "The reusable assembled prefix ahead of a compaction checkpoint is "
        "byte-identical before and after the compaction, past the one "
        "sanctioned first-boundary shift, and deterministic across trials.",
    ),
)

# Numeric metrics the decision depends on. Every one is validated finite and
# non-negative, over a non-empty sample set, before any status is derived.
DECISION_METRICS = (
    ("volume", "aux_calls"),
    ("volume", "prompt_chars"),
    ("volume", "estimated_input_tokens"),
    ("volume", "estimated_output_tokens"),
    ("latency", "fake_call_seconds_total"),
    ("latency", "fake_call_seconds_p50"),
    ("latency", "fake_call_seconds_p95"),
    ("latency", "transaction_seconds_total"),
    ("latency", "transaction_seconds_p50"),
    ("latency", "transaction_seconds_p95"),
    ("latency", "wall_clock_seconds_total"),
)
DECISION_SAMPLE_COUNTS = (
    ("latency", "fake_call_samples"),
    ("latency", "transaction_samples"),
    ("latency", "wall_clock_samples"),
)


# ═════════════════════════════════════════════════════════════════════════
# Child driver. Executed by `python -` inside an extracted revision tree, so
# `import agent.context_compressor` resolves from that exact revision.
# ═════════════════════════════════════════════════════════════════════════
DRIVER_SOURCE = r'''
import copy
import datetime as _datetime
import hashlib
import json
import os
import re
import sys
import time
from asyncio import CancelledError
from types import SimpleNamespace
from unittest.mock import patch

CFG = json.loads(sys.argv[1])
OUT_PATH = sys.argv[2]
FIXTURE = CFG['fixture']
SECRET = FIXTURE['fake_secret']
DELAY = CFG['fake_delay_seconds']
FOCUS = CFG['focus_topic']
MEMORY = CFG['memory_context']

NO_USER_SENTINEL = 'None. This session contains no user-authored turns.'
MEMORY_SECTION_HEADER = '\n\nMEMORY PROVIDER CONTEXT:\n'
SCHEDULED_FAILURE_MESSAGE = 'tier-a scheduled pass failure'
SCHEDULED_TIMEOUT_MESSAGE = 'tier-a scheduled request timed out'
PRE_EXISTING_DIAGNOSTIC = 'tier-a pre-existing diagnostic'
SEED_PREFIX = 'tier-a seed '

LABEL_RE = re.compile(
    r'^\[(SYSTEM|USER|ASSISTANT|TOOL RESULT [^\]]*)\]: ', re.MULTILINE
)
FRAGMENT_RE = re.compile(r'^\[ASSISTANT FRAGMENT \d+/\d+\]: ')
BUDGET_RE = re.compile(r'Target ~(\d+) tokens')

EXPECTED_LABEL = {'user': 'USER', 'assistant': 'ASSISTANT', 'system': 'SYSTEM'}


class DriverError(Exception):
    def __init__(self, code):
        Exception.__init__(self, code)
        self.code = code


def emit(payload):
    with open(OUT_PATH, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, sort_keys=True)


def canonical(value):
    # Deterministic byte form. Non-JSON values collapse to their type name so
    # the canonical form can never embed a memory address.
    return json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        default=lambda obj: '<' + type(obj).__name__ + '>',
    )


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_file(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def est_tokens(text):
    return (len(text) + 3) // 4


# ── Per-trial accumulators. All timing lives here and never in `checks`, so
# `checks` stays byte-comparable across the repeated trials.
TRIAL = {}


def reset_trial():
    TRIAL.clear()
    TRIAL['fake_seconds'] = {}
    TRIAL['transaction_seconds'] = {}
    TRIAL['aux_calls'] = {}
    TRIAL['prompt_chars'] = {}
    TRIAL['estimated_input_tokens'] = {}
    TRIAL['estimated_output_tokens'] = {}


def bump(bucket, label, value):
    TRIAL[bucket][label] = TRIAL[bucket].get(label, 0) + value


def push(bucket, label, value):
    TRIAL[bucket].setdefault(label, []).append(value)


class transaction(object):
    'Time one complete blocking summary-entry or compaction transaction.'

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        push('transaction_seconds', self.label, time.perf_counter() - self.start)
        return False


def build_row(spec):
    parts = []
    if spec.get('marker'):
        parts.append(spec['marker'])
    parts.append(spec['text'])
    if spec.get('include_secret'):
        parts.append(SECRET)
    if spec.get('include_pruned_skill'):
        parts.append(FIXTURE['pruned_skill_marker'])
    if spec.get('include_critical'):
        parts.append(FIXTURE['critical_marker'])
    if spec.get('pad_count'):
        parts.append(spec['pad_char'] * spec['pad_count'])
    row = {'role': spec['role'], 'content': ' '.join(parts)}
    if spec.get('tool_calls'):
        row['tool_calls'] = copy.deepcopy(spec['tool_calls'])
    if spec.get('tool_call_id'):
        row['tool_call_id'] = spec['tool_call_id']
    return row


def build_rows(specs):
    return [build_row(spec) for spec in specs]


def growth_rows(cycle):
    growth = FIXTURE['growth']
    rows = []
    for index in range(growth['rows_per_cycle']):
        rows.append({
            'role': 'user' if index % 2 == 0 else 'assistant',
            'content': '%s%d_%d %s %s' % (
                growth['marker_prefix'],
                cycle,
                index,
                growth['text'],
                growth['pad_char'] * growth['pad_count'],
            ),
        })
    return rows


def rows_text(rows):
    return '\n'.join(str(row.get('content') or '') for row in rows)


def rows_canonical(rows):
    return canonical(rows)


TREE = os.path.realpath(os.getcwd())
cc = None
ContextCompressor = None
SUMMARY_PREFIX = None
SUMMARY_END_MARKER = None
HISTORICAL_TASK_HEADING = None
COMPRESSED_SUMMARY_METADATA_KEY = None
COMPRESSED_SUMMARY_HAS_USER_TURN_KEY = None
MODULE_PATH = None
NORMALIZE_TEXT = None


def bootstrap():
    # Imported inside the guarded region so an import failure is reported as
    # an allowlisted code instead of an opaque non-zero exit.
    global cc, ContextCompressor, SUMMARY_PREFIX, SUMMARY_END_MARKER
    global HISTORICAL_TASK_HEADING, COMPRESSED_SUMMARY_METADATA_KEY
    global COMPRESSED_SUMMARY_HAS_USER_TURN_KEY, MODULE_PATH, NORMALIZE_TEXT
    try:
        import agent.context_compressor as module
    except BaseException:
        raise DriverError('DRIVER_IMPORT_FAILED')
    cc = module
    ContextCompressor = module.ContextCompressor
    SUMMARY_PREFIX = module.SUMMARY_PREFIX
    SUMMARY_END_MARKER = module._SUMMARY_END_MARKER
    HISTORICAL_TASK_HEADING = module.HISTORICAL_TASK_HEADING
    COMPRESSED_SUMMARY_METADATA_KEY = module.COMPRESSED_SUMMARY_METADATA_KEY
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY = (
        module.COMPRESSED_SUMMARY_HAS_USER_TURN_KEY
    )
    # The compaction redactor defines what "normalized" means for a summary
    # entry, so it is the correct oracle for the exact expected rollback seed.
    # It is untouched by the revision under evaluation.
    NORMALIZE_TEXT = module._redact_compaction_text
    MODULE_PATH = os.path.realpath(module.__file__)
    if not MODULE_PATH.startswith(TREE + os.sep):
        raise DriverError('DRIVER_ISOLATION_VIOLATED')
    if module._NO_USER_TASK_SENTINEL != NO_USER_SENTINEL:
        raise DriverError('DRIVER_SENTINEL_DRIFT')


def freeze_clock():
    # Freeze the temporal anchor at its source. The summarizer resolves the
    # date through a function-local `from hermes_time import now`, so patching
    # the module attribute reaches every call site and keeps prompt bytes
    # directly comparable without post-hoc normalization.
    try:
        import hermes_time
    except BaseException:
        raise DriverError('DRIVER_CLOCK_SEAM_MISSING')
    frozen = _datetime.datetime.fromisoformat(CFG['frozen_clock_utc'])
    patcher = patch.object(hermes_time, 'now', lambda *a, **k: frozen)
    patcher.start()
    probe = hermes_time.now().strftime('%Y-%m-%d')
    if probe != frozen.strftime('%Y-%m-%d'):
        raise DriverError('DRIVER_CLOCK_SEAM_MISSING')
    return probe


def source_block(prompt):
    'Return the serialized source section, with the memory section split off.'
    if 'NEW TURNS TO INCORPORATE:\n' in prompt:
        head = prompt.split('NEW TURNS TO INCORPORATE:\n', 1)[1]
        block = head.split(
            '\n\nUpdate the summary using this exact structure.', 1
        )[0]
    elif 'TURNS TO SUMMARIZE:\n' in prompt:
        head = prompt.split('TURNS TO SUMMARIZE:\n', 1)[1]
        block = head.split('\n\nUse this exact structure:', 1)[0]
    else:
        raise DriverError('DRIVER_PROMPT_SHAPE_UNRECOGNIZED')
    # Provider memory is interpolated directly after the source content and
    # inside the same delimiters; split it back out on its own header so it
    # cannot contaminate marker counting or fragment reconstruction.
    return block.split(MEMORY_SECTION_HEADER, 1)[0]


def carries_memory(prompt):
    return MEMORY_SECTION_HEADER in prompt and MEMORY in prompt


def prompt_budget(prompt):
    match = BUDGET_RE.search(prompt)
    if match is None:
        raise DriverError('DRIVER_PROMPT_SHAPE_UNRECOGNIZED')
    return int(match.group(1))


def task_snapshot(summary):
    match = re.search(
        r'(?ms)^' + re.escape(HISTORICAL_TASK_HEADING) + r'\s*\n(.*?)(?=\n##\s|\Z)',
        summary or '',
    )
    return match.group(1).strip() if match else ''


class Fake(object):
    'Deterministic fake provider with a fixed artificial blocking delay.'

    def __init__(self, label, markers=(), no_user=False, schedule=None,
                 default_action=None, leak_secret=False):
        self.label = label
        self.markers = list(markers)
        self.no_user = no_user
        self.schedule = dict(schedule or {})
        self.default_action = default_action
        self.leak_secret = leak_secret
        self.prompts = []
        self.payload_hashes = []
        self.payload_hashes_excluding_routing = []
        self.payload_routing = []
        self.payload_keys = []
        self.bodies = []

    def __call__(self, **kwargs):
        ordinal = len(self.prompts) + 1
        prompt = kwargs['messages'][0]['content']
        self.prompts.append(prompt)
        self.payload_hashes.append(sha(canonical(kwargs)))
        # The routing discriminator is the ONE component an auxiliary-to-main
        # retry is expected to change: the first request carries the
        # configured auxiliary model, the retry drops the override entirely.
        # Everything else must be byte-identical, so it is hashed separately
        # and the discriminator is asserted on its own.
        routing = CFG['routing_discriminator']
        self.payload_hashes_excluding_routing.append(
            sha(canonical(
                dict((k, v) for k, v in kwargs.items() if k != routing)
            ))
        )
        self.payload_routing.append(kwargs.get(routing))
        self.payload_keys.append(sorted(kwargs.keys()))
        bump('aux_calls', self.label, 1)
        bump('prompt_chars', self.label, len(prompt))
        bump('estimated_input_tokens', self.label, est_tokens(prompt))
        start = time.perf_counter()
        time.sleep(DELAY)
        push('fake_seconds', self.label, time.perf_counter() - start)
        action = self.schedule.get(str(ordinal), self.default_action)
        if action == 'fail':
            raise RuntimeError(SCHEDULED_FAILURE_MESSAGE)
        if action == 'cancel':
            raise CancelledError
        if action == 'timeout':
            raise RuntimeError(SCHEDULED_TIMEOUT_MESSAGE)
        body = self.body(prompt, ordinal)
        self.bodies.append(body)
        bump('estimated_output_tokens', self.label, est_tokens(body))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))]
        )

    def body(self, prompt, ordinal):
        task = NO_USER_SENTINEL if self.no_user else 'tier-a synthetic task'
        seen = [marker for marker in self.markers if marker in prompt]
        body = (
            HISTORICAL_TASK_HEADING + '\n' + task
            + '\n\n## Goal\ntier-a deterministic summary pass '
            + str(ordinal)
            + '\n\n## Retained Markers\n' + ' '.join(seen)
        )
        # A model that echoes a credential back into its summary is the case
        # the outbound redaction boundary exists for. Without this the
        # summary-side redaction assertion would be vacuous.
        if self.leak_secret:
            body = body + '\n' + SECRET
        return body


class class_limit(object):
    'Override the aggregate summarizer input cap on the CLASS.'

    # The baseline reads this constant through a classmethod, the candidate
    # through the instance. Only a class-level override reaches both, so both
    # revisions run under exactly the same bound.

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.previous = ContextCompressor._SUMMARY_INPUT_MAX_CHARS
        ContextCompressor._SUMMARY_INPUT_MAX_CHARS = self.value
        return self

    def __exit__(self, *exc):
        ContextCompressor._SUMMARY_INPUT_MAX_CHARS = self.previous
        return False


def make(abort=False, aux_model=''):
    compressor = ContextCompressor(
        model=CFG['main_model'],
        threshold_percent=0.85,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
        config_context_length=CFG['context_length'],
        abort_on_summary_failure=abort,
        summary_model_override=aux_model or None,
    )
    compressor._tail_token_budget = CFG['tail_token_budget']
    compressor.last_prompt_tokens = CFG['current_tokens']
    return compressor


def state_map(compressor):
    'Complete stable post-call state, one canonical value per attribute.'
    mapping = {}
    for name, value in vars(compressor).items():
        if name in CFG['state_exclusions']:
            continue
        mapping[name] = sha(canonical(value))
    return mapping


def telemetry_map(compressor):
    'Complete telemetry mapping minus the frozen non-deterministic keys.'
    telemetry = getattr(compressor, '_active_compression_telemetry', None)
    if not isinstance(telemetry, dict):
        return None
    # Hashed, not raw: a differing key is named, its value never persisted.
    return dict(
        (name, sha(canonical(value)))
        for name, value in telemetry.items()
        if name not in CFG['telemetry_exclusions']
    )


def compact_fallback_turn(value):
    'Independent derivation of the deterministic fallback turn rendering.'
    # Mirrors the product's documented fallback normalization so the exact
    # expected anchor can be derived from the INPUT rather than read back out
    # of the fallback and compared with itself. The character bound is read
    # from the module so the derivation cannot drift from the contract.
    text = NORMALIZE_TEXT(value)
    text = re.sub(r'\bgh[pousr]_[A-Za-z0-9_]{8,}\b', '[REDACTED]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    limit = cc._FALLBACK_TURN_MAX_CHARS
    if len(text) > limit:
        text = text[:limit - 15].rstrip() + ' ...[truncated]'
    return re.sub(r'\bgh[pousr]_[A-Za-z0-9_.-]+', '[REDACTED]', text)


def observed_window(prompts, markers):
    'The exact selected compaction window, read off the summarizer prompts.'
    # The window the compressor selected is observable in the serialized
    # source it sent. Deriving the expected anchors from THIS (rather than
    # guessing head/tail boundaries) keeps the expectation exact while still
    # deriving it from the input rather than from the fallback under test.
    combined = '\n'.join(source_block(prompt) for prompt in prompts)
    positioned = [
        (combined.find(marker), marker)
        for marker in markers
        if marker in combined
    ]
    positioned.sort()
    return [marker for _, marker in positioned], len(
        LABEL_RE.findall(combined)
    )


def label_provenance(combined, marker):
    'Return the serialization label governing the block that holds *marker*.'
    position = combined.find(marker)
    if position < 0:
        return None
    governing = None
    for match in LABEL_RE.finditer(combined):
        if match.start() > position:
            break
        governing = match.group(1)
    return governing


def publication_rows(rows):
    return [
        (index, row) for index, row in enumerate(rows)
        if isinstance(row, dict) and row.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]


def publication_shape(rows, expect_has_user_turn):
    'Exact framing, role, and persistence metadata of the single handoff.'
    published = publication_rows(rows)
    joined = rows_text(rows)
    shape = {
        'publication_count': len(published),
        'summary_prefix_occurrences': joined.count(SUMMARY_PREFIX),
        'end_marker_occurrences': joined.count(SUMMARY_END_MARKER),
    }
    if len(published) != 1:
        shape.update({
            'role': None,
            'has_user_turn_value': None,
            'metadata_value_is_true': False,
            'prefix_then_end_marker': False,
            'body_beyond_prefix_chars': 0,
            'prefix_before_publication': None,
            'exact_framing': False,
        })
        return shape
    index, row = published[0]
    content = str(row.get('content') or '')
    prefix_at = content.find(SUMMARY_PREFIX)
    end_at = content.find(SUMMARY_END_MARKER)
    body = content.replace(SUMMARY_PREFIX, '', 1).replace(
        SUMMARY_END_MARKER, '', 1
    ).strip()
    shape.update({
        'role': row.get('role'),
        'has_user_turn_value': row.get(COMPRESSED_SUMMARY_HAS_USER_TURN_KEY),
        'metadata_value_is_true': row.get(
            COMPRESSED_SUMMARY_METADATA_KEY
        ) is True,
        'prefix_then_end_marker': prefix_at >= 0 and end_at > prefix_at,
        'body_beyond_prefix_chars': len(body),
        'prefix_before_publication': SUMMARY_PREFIX in rows_text(rows[:index]),
        'index': index,
    })
    shape['exact_framing'] = bool(
        shape['publication_count'] == 1
        and shape['summary_prefix_occurrences'] == 1
        and shape['end_marker_occurrences'] == 1
        and shape['metadata_value_is_true']
        and shape['has_user_turn_value'] is expect_has_user_turn
        and shape['prefix_then_end_marker']
        and shape['body_beyond_prefix_chars'] > 0
        and not shape['prefix_before_publication']
    )
    return shape


# ── Scenarios ────────────────────────────────────────────────────────────

def reference_budget(label, turns, no_user=False):
    'Full-window summary budget, observed from one below-cap single pass.'
    compressor = make()
    if no_user:
        compressor._summary_has_user_turn = False
    fake = Fake(label, no_user=no_user)
    with class_limit(CFG['below_cap_limit_chars']):
        with patch.object(cc, 'call_llm', fake):
            with transaction(label):
                compressor._generate_summary(
                    turns, focus_topic=FOCUS, memory_context=MEMORY,
                )
    if len(fake.prompts) != 1:
        raise DriverError('DRIVER_SCENARIO_NOT_APPLICABLE')
    return prompt_budget(fake.prompts[0]), source_block(fake.prompts[0])


def scenario_coverage(checks, metrics):
    label = 'coverage'
    specs = FIXTURE['main_session']['rows'][1:]
    markers = FIXTURE['main_session']['marker_order']
    critical = FIXTURE['critical_marker']
    turns = build_rows(specs)
    budget_ref, _ = reference_budget('coverage_reference', turns)

    compressor = make()
    fake = Fake(label, markers=markers + [critical], leak_secret=True)
    with class_limit(CFG['split_limit_chars']):
        with patch.object(cc, 'call_llm', fake):
            with transaction(label):
                summary = compressor._generate_summary(
                    turns, focus_topic=FOCUS, memory_context=MEMORY,
                )

    sources = [source_block(prompt) for prompt in fake.prompts]
    # Joined on a newline so every block label stays line-anchored for the
    # provenance scan below.
    combined = '\n'.join(sources)
    counts = dict((marker, combined.count(marker)) for marker in markers)
    positions = [combined.find(marker) for marker in markers]
    provenance = {}
    provenance_ok = True
    for spec, marker in zip(specs, markers):
        governing = label_provenance(combined, marker)
        provenance[marker] = governing
        if spec['role'] == 'tool':
            good = bool(governing) and governing.startswith('TOOL RESULT ')
        else:
            good = governing == EXPECTED_LABEL.get(spec['role'])
        provenance_ok = provenance_ok and good

    focus_every_pass = bool(fake.prompts) and all(
        FOCUS in prompt for prompt in fake.prompts
    )
    memory_every_pass = bool(fake.prompts) and all(
        carries_memory(prompt) for prompt in fake.prompts
    )
    budgets = [prompt_budget(prompt) for prompt in fake.prompts]
    budget_ok = bool(budgets) and all(value == budget_ref for value in budgets)
    # Chaining: markers first seen by the fake in an EARLY pass survive only
    # if each tentative summary is carried into the next request and into the
    # committed result. Requiring every marker in the final summary is the
    # observable form of that.
    retained_in_summary = [
        marker for marker in markers if marker in (summary or '')
    ]
    chained = len(retained_in_summary) == len(markers)
    multi_pass = len(sources) > 1
    within_bound = all(
        len(source) <= CFG['split_limit_chars'] for source in sources
    )
    complete = all(counts[marker] == 1 for marker in markers)
    ordered = all(pos >= 0 for pos in positions) and positions == sorted(
        positions
    )
    checks['aggregate_coverage'] = {
        'applicable': multi_pass,
        'passed': bool(
            summary
            and multi_pass
            and complete
            and ordered
            and provenance_ok
            and within_bound
            and focus_every_pass
            and memory_every_pass
            and budget_ok
            and chained
            and combined.count(critical) == 1
            and critical in (summary or '')
        ),
        'evidence': {
            'passes': len(sources),
            'markers_expected': len(markers),
            'markers_present_exactly_once': sum(
                1 for marker in markers if counts[marker] == 1
            ),
            'markers_missing': [
                marker for marker in markers if counts[marker] == 0
            ],
            'markers_duplicated': [
                marker for marker in markers if counts[marker] > 1
            ],
            'order_preserved': ordered,
            'role_provenance_preserved': provenance_ok,
            'role_provenance': provenance,
            'focus_reached_every_pass': focus_every_pass,
            'memory_reached_every_pass': memory_every_pass,
            'full_window_budget_tokens': budget_ref,
            'every_pass_used_full_window_budget': budget_ok,
            'distinct_pass_budgets': sorted(set(budgets)),
            'markers_retained_in_final_summary': len(retained_in_summary),
            'tentative_summary_chaining_complete': chained,
            'markers_absent_from_final_summary': [
                marker for marker in markers if marker not in (summary or '')
            ],
            'critical_marker_source_occurrences': combined.count(critical),
            'critical_marker_in_final_summary': critical in (summary or ''),
            'every_pass_within_bound': within_bound,
            'max_pass_source_chars': max(len(s) for s in sources) if sources
            else 0,
            'summary_produced': bool(summary),
        },
    }

    # Completed tool call/result grouping, observed on the same run.
    call_token = FIXTURE['tool_name'] + '('
    result_token = '[TOOL RESULT ' + FIXTURE['tool_call_id'] + ']'
    holders = [
        index for index, source in enumerate(sources) if call_token in source
    ]
    grouped = (
        len(holders) == 1
        and result_token in sources[holders[0]]
        and sources[holders[0]].index(call_token)
        < sources[holders[0]].index(result_token)
    )
    checks['tool_group'] = {
        'applicable': combined.count(call_token) > 0 and multi_pass,
        'passed': bool(
            grouped
            and combined.count(call_token) == 1
            and combined.count(result_token) == 1
        ),
        'evidence': {
            'passes_containing_tool_call': len(holders),
            'tool_call_occurrences': combined.count(call_token),
            'tool_result_occurrences': combined.count(result_token),
            'call_precedes_result_in_same_pass': grouped,
        },
    }

    # Redaction and pruned-skill survival, observed on the same run. The
    # scenario is only meaningful when the source really carries the secret
    # and the fake provider really tries to echo it back.
    marker_text = FIXTURE['pruned_skill_marker']
    secret_in_prompts = any(SECRET in prompt for prompt in fake.prompts)
    source_carries_secret = any(SECRET in row['content'] for row in turns)
    provider_echoed_secret = any(SECRET in body for body in fake.bodies)
    checks['redaction_and_marker'] = {
        'applicable': bool(
            source_carries_secret and provider_echoed_secret and fake.prompts
        ),
        'passed': bool(
            summary
            and source_carries_secret
            and provider_echoed_secret
            and not secret_in_prompts
            and SECRET not in (summary or '')
            and SECRET not in (compressor._previous_summary or '')
            and marker_text in (summary or '')
        ),
        'evidence': {
            'source_carries_secret': source_carries_secret,
            'provider_echoed_secret': provider_echoed_secret,
            'secret_present_in_any_prompt': secret_in_prompts,
            'secret_present_in_summary': SECRET in (summary or ''),
            'secret_present_in_stored_state': SECRET in (
                compressor._previous_summary or ''
            ),
            'pruned_skill_marker_survived': marker_text in (summary or ''),
            'prompts_scanned': len(fake.prompts),
        },
    }


def scenario_full_window_derivation(checks, metrics):
    'Final bounded pass with no user row must not narrow budget/provenance.'
    label = 'full_window_derivation'
    section = FIXTURE['final_assistant_only']
    markers = section['marker_order']
    turns = build_rows(section['rows'])
    budget_ref, _ = reference_budget('full_window_reference', turns)

    compressor = make()
    fake = Fake(label, markers=markers)
    with class_limit(CFG['split_limit_chars']):
        with patch.object(cc, 'call_llm', fake):
            with transaction(label):
                summary = compressor._generate_summary(
                    turns, focus_topic=FOCUS, memory_context=MEMORY,
                )

    sources = [source_block(prompt) for prompt in fake.prompts]
    multi_pass = len(sources) > 1
    final_has_no_user = bool(sources) and '[USER]:' not in sources[-1]
    budgets = [prompt_budget(prompt) for prompt in fake.prompts]
    budget_ok = bool(budgets) and all(value == budget_ref for value in budgets)
    snapshot = task_snapshot(summary or '')
    # Provenance derived from the complete window, not the final chunk: the
    # window HAS a user row, so the committed summary must carry the real task
    # anchor rather than the no-user sentinel.
    provenance_ok = bool(
        summary
        and snapshot
        and snapshot != NO_USER_SENTINEL
        and markers[0] in snapshot
    )
    chained = all(marker in (summary or '') for marker in markers)
    checks['full_window_derivation'] = {
        'applicable': bool(multi_pass and final_has_no_user),
        'passed': bool(
            summary
            and multi_pass
            and final_has_no_user
            and budget_ok
            and provenance_ok
            and chained
            and all(FOCUS in prompt for prompt in fake.prompts)
            and all(carries_memory(prompt) for prompt in fake.prompts)
        ),
        'evidence': {
            'passes': len(sources),
            'final_pass_has_no_user_row': final_has_no_user,
            'full_window_budget_tokens': budget_ref,
            'every_pass_used_full_window_budget': budget_ok,
            'distinct_pass_budgets': sorted(set(budgets)),
            'task_snapshot_is_no_user_sentinel': snapshot == NO_USER_SENTINEL,
            'task_snapshot_carries_window_user_marker': bool(
                snapshot and markers[0] in snapshot
            ),
            'markers_retained_in_final_summary': sum(
                1 for marker in markers if marker in (summary or '')
            ),
            'markers_expected': len(markers),
            'tentative_summary_chaining_complete': chained,
        },
    }


def scenario_fragments(checks, metrics):
    label = 'fragment_reconstruction'
    row = build_row(FIXTURE['oversized_row'])
    _, reference = reference_budget(
        'fragment_reference', [row], no_user=True,
    )

    fragment_compressor = make()
    fragment_compressor._summary_has_user_turn = False
    fragment_fake = Fake(label, no_user=True)
    with class_limit(CFG['fragment_limit_chars']):
        with patch.object(cc, 'call_llm', fragment_fake):
            with transaction(label):
                fragment_summary = fragment_compressor._generate_summary(
                    [row], focus_topic=FOCUS, memory_context=MEMORY,
                )

    blocks = [source_block(prompt) for prompt in fragment_fake.prompts]
    labelled = [block for block in blocks if FRAGMENT_RE.match(block)]
    fragmented = len(blocks) > 1 and len(labelled) == len(blocks)
    recovered = ''.join(
        FRAGMENT_RE.sub('', block, count=1) for block in blocks
    ) if fragmented else ''
    # Applicability: the reference row must genuinely exceed the bound, or the
    # scenario proves nothing and the run must not silently report success.
    applicable_bound = len(reference) > CFG['fragment_limit_chars']
    checks['fragment_reconstruction'] = {
        'applicable': bool(applicable_bound and fragmented),
        'passed': bool(
            applicable_bound
            and fragmented
            and fragment_summary
            and recovered == reference
            and all(
                len(block) <= CFG['fragment_limit_chars'] for block in blocks
            )
        ),
        'evidence': {
            'reference_source_chars': len(reference),
            'reference_exceeds_bound': applicable_bound,
            'fragment_blocks': len(blocks),
            'all_blocks_labelled': len(labelled) == len(blocks),
            'reconstruction_byte_identical': recovered == reference,
            'reference_source_sha256': sha(reference),
            'reconstructed_source_sha256': sha(recovered) if recovered
            else None,
            'max_block_chars': max((len(b) for b in blocks), default=0),
        },
    }


def transactional_run(label, schedule_name):
    specs = FIXTURE['main_session']['rows'][1:]
    turns = build_rows(specs)
    before = copy.deepcopy(turns)
    member_ids = [id(turn) for turn in turns]
    compressor = make()
    seed = SEED_PREFIX + SECRET
    compressor._previous_summary = seed
    compressor._last_summary_error = PRE_EXISTING_DIAGNOSTIC
    expected_seed = NORMALIZE_TEXT(seed)
    pre = {
        'cooldown': compressor._summary_failure_cooldown_until,
        'error': compressor._last_summary_error,
        'auth': bool(compressor._last_summary_auth_failure),
        'network': bool(compressor._last_summary_network_failure),
        'fallen_back': bool(
            getattr(compressor, '_summary_model_fallen_back', False)
        ),
    }
    fake = Fake(
        label,
        markers=FIXTURE['main_session']['marker_order'],
        schedule=CFG['failure_schedule'][schedule_name],
    )
    with class_limit(CFG['split_limit_chars']):
        with patch.object(cc, 'call_llm', fake):
            with transaction(label):
                summary = compressor._generate_summary(
                    turns, focus_topic=FOCUS, memory_context=MEMORY,
                )
    stored = compressor._previous_summary or ''
    return {
        'calls': len(fake.prompts),
        'summary_is_none': summary is None,
        'turns_unmodified': turns == before,
        'member_identity_preserved': [id(t) for t in turns] == member_ids,
        # Exact normalized rollback, not a prefix match: a tentative summary
        # appended to the seed would fail this.
        'stored_equals_expected_normalized_seed': stored == expected_seed,
        'stored_contains_secret': SECRET in stored,
        'expected_seed_sha256': sha(expected_seed),
        'stored_sha256': sha(stored),
        'error_equals_scheduled_failure': (
            compressor._last_summary_error == SCHEDULED_FAILURE_MESSAGE
        ),
        'error_unchanged': compressor._last_summary_error == pre['error'],
        'cooldown_unchanged': (
            compressor._summary_failure_cooldown_until == pre['cooldown']
        ),
        'cooldown_advanced': (
            compressor._summary_failure_cooldown_until > pre['cooldown']
        ),
        'auth_unchanged': (
            bool(compressor._last_summary_auth_failure) == pre['auth']
        ),
        'network_unchanged': (
            bool(compressor._last_summary_network_failure) == pre['network']
        ),
        'fallen_back_unchanged': bool(
            getattr(compressor, '_summary_model_fallen_back', False)
        ) == pre['fallen_back'],
    }


def scenario_rollback(checks, metrics):
    evidence = transactional_run('rollback', 'rollback')
    checks['rollback'] = {
        'applicable': evidence['calls'] >= 2,
        'passed': bool(
            evidence['calls'] >= 2
            and evidence['summary_is_none']
            and evidence['turns_unmodified']
            and evidence['member_identity_preserved']
            and evidence['stored_equals_expected_normalized_seed']
            and not evidence['stored_contains_secret']
            and evidence['error_equals_scheduled_failure']
            and evidence['cooldown_advanced']
            and evidence['auth_unchanged']
            and evidence['network_unchanged']
        ),
        'evidence': evidence,
    }


def scenario_cancellation(checks, metrics):
    evidence = transactional_run('cancellation', 'cancellation')
    checks['cancellation'] = {
        'applicable': evidence['calls'] >= 2,
        'passed': bool(
            evidence['calls'] >= 2
            and evidence['summary_is_none']
            and evidence['turns_unmodified']
            and evidence['member_identity_preserved']
            and evidence['stored_equals_expected_normalized_seed']
            and not evidence['stored_contains_secret']
            and evidence['error_unchanged']
            and evidence['cooldown_unchanged']
            and evidence['auth_unchanged']
            and evidence['network_unchanged']
            and evidence['fallen_back_unchanged']
        ),
        'evidence': evidence,
    }


def retry_run(label, schedule_name):
    specs = FIXTURE['main_session']['rows'][1:]
    turns = build_rows(specs)
    compressor = make(aux_model=CFG['aux_model'])
    fake = Fake(
        label,
        markers=FIXTURE['main_session']['marker_order'],
        schedule=CFG['failure_schedule'][schedule_name],
    )
    with class_limit(CFG['split_limit_chars']):
        with patch.object(cc, 'call_llm', fake):
            with transaction(label):
                summary = compressor._generate_summary(
                    turns, focus_topic=FOCUS, memory_context=MEMORY,
                )
    failing = int(list(CFG['failure_schedule'][schedule_name].keys())[0])
    sources = [source_block(prompt) for prompt in fake.prompts]
    has_retry = len(fake.prompts) > failing
    return {
        'failing_call_ordinal': failing,
        'calls': len(fake.prompts),
        'summary_produced': bool(summary),
        'retry_observed': has_retry,
        # Exact complete-request equality apart from the routing
        # discriminator, not merely the same source subsection: an altered
        # inherited summary, budget, focus, or memory would change this hash
        # while leaving the source block equal.
        'retry_payload_identical_excluding_routing': bool(
            has_retry
            and fake.payload_hashes_excluding_routing[failing]
            == fake.payload_hashes_excluding_routing[failing - 1]
        ),
        # The discriminator itself, proved in both directions.
        'first_request_selected_auxiliary_model': bool(
            has_retry and fake.payload_routing[failing - 1] == CFG['aux_model']
        ),
        'retry_selected_main_model_no_override': bool(
            has_retry and fake.payload_routing[failing] is None
        ),
        'routing_discriminator': CFG['routing_discriminator'],
        'retry_source_block_identical': bool(
            has_retry and sources[failing] == sources[failing - 1]
        ),
        'complete_payload_hashes_differ_by_routing_only': bool(
            has_retry
            and fake.payload_hashes[failing] != fake.payload_hashes[failing - 1]
            and fake.payload_hashes_excluding_routing[failing]
            == fake.payload_hashes_excluding_routing[failing - 1]
        ),
        'fell_back_to_main_model': compressor.summary_model == '',
        'fallback_flag': bool(
            getattr(compressor, '_summary_model_fallen_back', False)
        ),
    }


def scenario_retry(checks, metrics):
    first = retry_run('retry_first_pass', 'retry_first_pass')
    later = retry_run('retry_later_pass', 'retry_later_pass')
    later_applicable = later['retry_observed']
    checks['retry_placement'] = {
        'applicable': first['retry_observed'],
        'passed': bool(
            first['retry_observed']
            and first['summary_produced']
            and first['retry_payload_identical_excluding_routing']
            and first['first_request_selected_auxiliary_model']
            and first['retry_selected_main_model_no_override']
            and first['retry_source_block_identical']
            and first['fell_back_to_main_model']
            and (
                not later_applicable
                or (
                    later['summary_produced']
                    and later['retry_payload_identical_excluding_routing']
                    and later['first_request_selected_auxiliary_model']
                    and later['retry_selected_main_model_no_override']
                    and later['retry_source_block_identical']
                )
            )
        ),
        'evidence': {
            'first_pass_schedule': first,
            'later_pass_schedule': later,
            'later_pass_schedule_applicable': later_applicable,
        },
    }


def scenario_fallback_and_abort(checks, metrics):
    markers = FIXTURE['main_session']['marker_order']
    marker_text = FIXTURE['pruned_skill_marker']

    # Abort input deliberately carries no tool result and no blank echo, so
    # the cheap pre-pass has nothing to rewrite and "returned unchanged" is an
    # unambiguous canonical comparison rather than a count heuristic.
    abort_specs = FIXTURE['cross_session']['rows']
    abort_rows = build_rows(abort_specs)
    abort_input_canonical = rows_canonical(abort_rows)
    abort_compressor = make(abort=True)
    abort_fake = Fake('abort', default_action='fail')
    with class_limit(CFG['split_limit_chars']):
        with patch.object(cc, 'call_llm', abort_fake):
            with transaction('abort'):
                abort_result = abort_compressor.compress(
                    abort_rows,
                    current_tokens=CFG['current_tokens'],
                    focus_topic=FOCUS,
                    force=True,
                    memory_context=MEMORY,
                )
    abort_evidence = {
        'aux_calls': len(abort_fake.prompts),
        'canonically_unchanged': rows_canonical(abort_result)
        == abort_input_canonical,
        'input_canonical_sha256': sha(abort_input_canonical),
        'output_canonical_sha256': sha(rows_canonical(abort_result)),
        'aborted_flag': bool(abort_compressor._last_compress_aborted),
        'fallback_used_flag': bool(
            abort_compressor._last_summary_fallback_used
        ),
        'publication_count': len(publication_rows(abort_result)),
        # Member identity is NOT observable on this seam: the cheap pre-pass
        # copies every row unconditionally on both revisions, so identity is
        # recorded for completeness and the canonical comparison above is the
        # gated invariant.
        'member_identity_observable': any(
            row is original
            for row, original in zip(abort_result, abort_rows)
        ),
    }

    fallback_specs = FIXTURE['main_session']['rows']
    spec_by_marker = dict(
        (spec['marker'], spec) for spec in fallback_specs if spec.get('marker')
    )

    # Observe the exact window this input selects, using a run whose summary
    # SUCCEEDS. Window selection happens before the summary call, so the
    # failing run below selects the same window; the dropped-count
    # cross-check further down proves it did.
    probe_compressor = make(abort=False)
    probe_fake = Fake('fallback_window_probe', markers=markers)
    with class_limit(CFG['split_limit_chars']):
        with patch.object(cc, 'call_llm', probe_fake):
            with transaction('fallback_window_probe'):
                probe_compressor.compress(
                    build_rows(fallback_specs),
                    current_tokens=CFG['current_tokens'],
                    focus_topic=FOCUS,
                    force=True,
                    memory_context=MEMORY,
                )
    window_markers, window_row_count = observed_window(
        probe_fake.prompts, markers
    )
    user_markers = [
        marker for marker in window_markers
        if spec_by_marker[marker]['role'] == 'user'
    ]
    tool_markers = [
        marker for marker in window_markers
        if spec_by_marker[marker]['role'] == 'tool'
    ]
    # Exact anchor 1: the latest selected-window user task, derived from the
    # input row and rendered exactly as the deterministic fallback renders it.
    expected_task_anchor = None
    if user_markers:
        expected_task_anchor = 'User asked: ' + repr(
            compact_fallback_turn(
                build_row(spec_by_marker[user_markers[-1]])['content']
            )
        )
    # Exact anchor 2: the tool-call continuity event the fallback is
    # contractually required to carry into Completed Actions.
    expected_tool_event_anchor = 'Called tool(s): ' + FIXTURE['tool_name']

    fallback_rows = build_rows(fallback_specs)
    fallback_compressor = make(abort=False)
    fallback_fake = Fake('fallback', default_action='fail')
    with class_limit(CFG['split_limit_chars']):
        with patch.object(cc, 'call_llm', fallback_fake):
            with transaction('fallback'):
                fallback_result = fallback_compressor.compress(
                    fallback_rows,
                    current_tokens=CFG['current_tokens'],
                    focus_topic=FOCUS,
                    force=True,
                    memory_context=MEMORY,
                )
    shape = publication_shape(fallback_result, True)
    published = publication_rows(fallback_result)
    content = str(published[0][1].get('content') or '') if published else ''
    snapshot = task_snapshot(content)
    dropped_count = int(fallback_compressor._last_summary_dropped_count or 0)
    fallback_evidence = {
        'aux_calls': len(fallback_fake.prompts),
        'aborted_flag': bool(fallback_compressor._last_compress_aborted),
        'fallback_used_flag': bool(
            fallback_compressor._last_summary_fallback_used
        ),
        'dropped_count_recorded': dropped_count,
        'publication_shape': shape,
        # Exact full-window anchors, independently derived from the input.
        # Only hashes, counts, and safe marker names are persisted.
        'window_row_count_observed': window_row_count,
        'window_markers_observed': window_markers,
        'latest_window_user_marker': user_markers[-1] if user_markers else None,
        'window_carries_tool_result': bool(tool_markers),
        'dropped_count_matches_observed_window': (
            dropped_count == window_row_count
        ),
        'expected_task_anchor_sha256': sha(expected_task_anchor)
        if expected_task_anchor else None,
        'expected_task_anchor_present': bool(
            expected_task_anchor and expected_task_anchor in content
        ),
        'expected_tool_event_anchor_sha256': sha(expected_tool_event_anchor),
        'expected_tool_event_anchor_present': (
            expected_tool_event_anchor in content
        ),
        # Retained as supporting evidence only; neither is gated on its own.
        'task_snapshot_is_no_user_sentinel': snapshot == NO_USER_SENTINEL,
        'pruned_skill_marker_present': marker_text in content,
        'last_dropped_turns_section_present': '## Last Dropped Turns' in content,
        'window_markers_present_in_publication': sum(
            1 for marker in window_markers if marker in content
        ),
        'secret_in_publication': SECRET in content,
    }
    checks['fallback_and_abort'] = {
        # Not applicable unless the observed window really carries both a user
        # turn and the completed tool group the exact anchors are drawn from.
        'applicable': bool(
            abort_evidence['aux_calls'] > 0
            and fallback_evidence['aux_calls'] > 0
            and expected_task_anchor
            and tool_markers
            and window_row_count > 0
        ),
        'passed': bool(
            abort_evidence['aborted_flag']
            and abort_evidence['canonically_unchanged']
            and abort_evidence['publication_count'] == 0
            and not abort_evidence['fallback_used_flag']
            and not fallback_evidence['aborted_flag']
            and fallback_evidence['fallback_used_flag']
            and fallback_evidence['dropped_count_recorded'] > 0
            and fallback_evidence['dropped_count_matches_observed_window']
            and shape['exact_framing']
            # Exact anchors, not heading presence or a non-empty snapshot.
            and fallback_evidence['expected_task_anchor_present']
            and fallback_evidence['expected_tool_event_anchor_present']
            and fallback_evidence['pruned_skill_marker_present']
            and not fallback_evidence['secret_in_publication']
        ),
        'evidence': {
            'abort_on_summary_failure_true': abort_evidence,
            'abort_on_summary_failure_false': fallback_evidence,
        },
    }


def scenario_restart_cross_session(checks, metrics):
    critical = FIXTURE['critical_marker']
    main_specs = FIXTURE['main_session']['rows']
    cross_specs = FIXTURE['cross_session']['rows']
    markers = (
        FIXTURE['main_session']['marker_order']
        + FIXTURE['cross_session']['marker_order']
    )

    def run(compressor, rows, fake, label):
        with class_limit(CFG['split_limit_chars']):
            with patch.object(cc, 'call_llm', fake):
                with transaction(label):
                    return compressor.compress(
                        rows,
                        current_tokens=CFG['current_tokens'],
                        focus_topic=FOCUS,
                        force=True,
                        memory_context=MEMORY,
                    )

    fake = Fake('restart', markers=markers + [critical])
    first = make()
    compacted = run(first, build_rows(main_specs), fake, 'restart')
    first_shape = publication_shape(compacted, True)

    restarted = make()
    # The continuation deliberately carries none of the critical marker, so
    # the marker can only reach the restarted window through the rehydrated
    # handoff.
    resumed = compacted + growth_rows(91) + growth_rows(92)
    recompressed = run(restarted, resumed, fake, 'restart')
    restart_shape = publication_shape(recompressed, True)
    joined = rows_text(recompressed)

    unrelated_fake = Fake(
        'cross_session', markers=FIXTURE['cross_session']['marker_order'],
    )
    unrelated = run(
        restarted, build_rows(cross_specs), unrelated_fake, 'cross_session',
    )
    unrelated_text = rows_text(unrelated)

    checks['restart_cross_session'] = {
        'applicable': bool(fake.prompts and unrelated_fake.prompts),
        'passed': bool(
            first_shape['exact_framing']
            and restart_shape['exact_framing']
            and critical in joined
            and all(
                critical not in prompt for prompt in unrelated_fake.prompts
            )
            and critical not in unrelated_text
        ),
        'evidence': {
            'first_publication_shape': first_shape,
            'restart_publication_shape': restart_shape,
            'critical_marker_survived_restart': critical in joined,
            'critical_marker_in_unrelated_prompts': any(
                critical in prompt for prompt in unrelated_fake.prompts
            ),
            'critical_marker_in_unrelated_output': critical in unrelated_text,
            'unrelated_aux_calls': len(unrelated_fake.prompts),
        },
    }


def scenario_below_cap(checks, metrics):
    label = 'below_cap'
    turns = [
        {'role': 'user', 'content': 'tier-a below-cap request'},
        {'role': 'assistant', 'content': 'tier-a below-cap response'},
    ]
    compressor = make()
    # Seed the telemetry through the product's own initializer rather than a
    # bare dict: that yields the COMPLETE key set (so a candidate adding an
    # unnamed key is caught) and matches what the aux-call recorder expects to
    # find already present.
    compressor._begin_compression_telemetry(
        current_tokens=CFG['current_tokens'],
        session_id='tier-a',
        trigger_source='tier-a',
    )
    compressor._previous_summary = 'tier-a prior summary'
    compressor._last_summary_error = 'tier-a stale error'
    fake = Fake(label)

    with class_limit(CFG['below_cap_limit_chars']):
        with patch.object(cc, 'call_llm', fake):
            with transaction(label):
                summary = compressor._generate_summary(
                    turns, focus_topic=FOCUS, memory_context=MEMORY,
                )

    state = state_map(compressor)
    telemetry = telemetry_map(compressor)
    parity = {
        'aux_calls': len(fake.prompts),
        # Complete request payload, every kwarg key and value included.
        'request_payload_sha256': fake.payload_hashes[0]
        if fake.payload_hashes else None,
        'request_payload_keys': fake.payload_keys[0]
        if fake.payload_keys else None,
        'request_carries_focus_topic': bool(
            fake.prompts and FOCUS in fake.prompts[0]
        ),
        'request_carries_memory_context': bool(
            fake.prompts and carries_memory(fake.prompts[0])
        ),
        'result_summary_sha256': sha(summary) if summary else None,
        'result_has_summary_prefix': bool(
            summary and summary.startswith(SUMMARY_PREFIX)
        ),
        # Complete stable post-call state and complete telemetry mapping. The
        # per-field maps let the parent name a differing field without ever
        # persisting a value.
        'state_field_hashes': state,
        'state_map_sha256': sha(canonical(state)),
        'telemetry_field_values': telemetry,
        'telemetry_map_sha256': sha(canonical(telemetry)),
    }
    checks['below_cap_parity'] = {
        'applicable': bool(fake.prompts),
        'passed': bool(summary and len(fake.prompts) == 1),
        'evidence': {
            'aux_calls': parity['aux_calls'],
            'state_map_sha256': parity['state_map_sha256'],
            'telemetry_map_sha256': parity['telemetry_map_sha256'],
        },
    }
    metrics['below_cap_parity'] = parity


def scenario_cycles(checks, metrics):
    main_specs = FIXTURE['main_session']['rows']
    markers = FIXTURE['main_session']['marker_order']
    critical = FIXTURE['critical_marker']
    per_cycle = {}

    for cycles in CFG['cycles']:
        label = 'cycles_%d' % cycles
        rows = build_rows(main_specs)
        initial_hash = sha(rows_canonical(rows))
        compressor = make()
        fake = Fake(label, markers=markers + [critical])
        prefix_hashes = []
        prefix_stable = []
        publication_shapes = []
        telemetry_chunks = []
        per_compaction_calls = []
        pre_compression_hashes = []
        compression_count_deltas = []
        transaction_samples = []
        calls_before = 0

        with class_limit(CFG['split_limit_chars']):
            with patch.object(cc, 'call_llm', fake):
                for cycle in range(1, cycles + 1):
                    rows = rows + growth_rows(cycle)
                    # Exact pre-compression transcript hash, taken AFTER this
                    # cycle's growth append, so each entry identifies the real
                    # distinct input the forced compaction consumed.
                    pre_compression_hashes.append(sha(rows_canonical(rows)))
                    # Snapshot the pre-call bytes so the reusable prefix can be
                    # compared across the checkpoint. Taken before the call
                    # because the cheap pre-pass rewrites rows in place.
                    pre_row_bytes = [canonical(row) for row in rows]
                    count_before = compressor.compression_count
                    transactions_before = len(
                        TRIAL['transaction_seconds'].get(label, [])
                    )
                    with transaction(label):
                        rows = compressor.compress(
                            rows,
                            current_tokens=CFG['current_tokens'],
                            focus_topic=FOCUS,
                            force=True,
                            memory_context=MEMORY,
                        )
                    # A real compaction is an externally observable state
                    # transition, not a loop iteration: compression_count is
                    # incremented once, at the end of a completed compaction.
                    # A no-op call that reuses an existing handoff leaves it
                    # unchanged and is caught here.
                    compression_count_deltas.append(
                        compressor.compression_count - count_before
                    )
                    transaction_samples.append(
                        len(TRIAL['transaction_seconds'].get(label, []))
                        - transactions_before
                    )
                    per_compaction_calls.append(
                        len(fake.prompts) - calls_before
                    )
                    calls_before = len(fake.prompts)
                    shape = publication_shape(rows, True)
                    publication_shapes.append(shape)
                    index = shape.get('index')
                    if index is None:
                        prefix_hashes.append(None)
                        prefix_stable.append(False)
                    else:
                        prefix_hashes.append(sha(canonical(rows[:index])))
                        # Prompt-cache claim: rows the provider has already
                        # seen ahead of the checkpoint must come back
                        # byte-identical. The first compaction is the one
                        # sanctioned boundary shift (it appends the compaction
                        # note to the system prompt), recorded not gated.
                        prefix_stable.append(
                            index >= 1
                            and [canonical(row) for row in rows[:index]]
                            == pre_row_bytes[:index]
                        )
                    telemetry = getattr(
                        compressor, '_last_compression_telemetry', {}
                    ) or {}
                    telemetry_chunks.append({
                        'chunking': telemetry.get('chunking'),
                        'chunk_count': telemetry.get('chunk_count'),
                    })

        joined = rows_text(rows)
        lost = [marker for marker in markers if marker not in joined]
        per_cycle[str(cycles)] = {
            'cycles': cycles,
            # Derived from the observed compression-state increment, never
            # from the loop count or from an already-present handoff.
            'compactions_observed': sum(
                1 for delta in compression_count_deltas if delta == 1
            ),
            'compression_count_deltas': compression_count_deltas,
            'compression_count_final': compressor.compression_count,
            'every_cycle_incremented_once': all(
                delta == 1 for delta in compression_count_deltas
            ),
            'every_cycle_issued_calls': all(
                value > 0 for value in per_compaction_calls
            ),
            'every_cycle_timed_one_transaction': all(
                value == 1 for value in transaction_samples
            ),
            'pre_compression_hashes': pre_compression_hashes,
            'pre_compression_hashes_distinct': len(
                set(pre_compression_hashes)
            ) == len(pre_compression_hashes),
            'initial_transcript_sha256': initial_hash,
            'aux_calls_total': len(fake.prompts),
            'aux_calls_per_cycle': per_compaction_calls,
            'prompt_chars': sum(len(p) for p in fake.prompts),
            'estimated_input_tokens': sum(
                est_tokens(p) for p in fake.prompts
            ),
            'estimated_output_tokens': sum(
                est_tokens(b) for b in fake.bodies
            ),
            'window_transcript_sha256': sha(rows_canonical(rows)),
            'prefix_hashes': prefix_hashes,
            'prefix_stable_at_boundary': prefix_stable,
            'prefix_stable_after_first_boundary': all(prefix_stable[1:]),
            'first_boundary_prefix_changed': bool(
                prefix_stable and not prefix_stable[0]
            ),
            'publication_shapes': publication_shapes,
            'every_publication_exact': all(
                shape['exact_framing'] for shape in publication_shapes
            ),
            'markers_retained': len(markers) - len(lost),
            'markers_expected': len(markers),
            'markers_lost': lost,
            'critical_marker_retained': critical in joined,
            'secret_in_output': SECRET in joined,
            'telemetry_per_cycle': telemetry_chunks,
            'rows_final': len(rows),
            'focus_reached_every_call': all(
                FOCUS in prompt for prompt in fake.prompts
            ),
            'memory_reached_every_call': all(
                carries_memory(prompt) for prompt in fake.prompts
            ),
        }

    zero = per_cycle[str(CFG['cycles'][0])]
    # Progression is real only when every forced cycle produced an observed
    # compression-state increment AND the compressor's own final count agrees.
    progression_complete = all(
        per_cycle[str(count)]['compactions_observed'] == count
        and per_cycle[str(count)]['compression_count_final'] == count
        and per_cycle[str(count)]['every_cycle_incremented_once']
        for count in CFG['cycles']
    )
    integrity_ok = progression_complete
    drift = {}
    for count in CFG['cycles']:
        record = per_cycle[str(count)]
        drift[str(count)] = (
            zero['markers_retained'] - record['markers_retained']
        )
        if count == 0:
            continue
        integrity_ok = integrity_ok and (
            record['every_publication_exact']
            # Every forced cycle must show real work: a state increment, a
            # positive auxiliary-call delta, one complete blocking
            # transaction, and a distinct pre-compression input.
            and record['every_cycle_issued_calls']
            and record['every_cycle_timed_one_transaction']
            and record['pre_compression_hashes_distinct']
            and len(record['pre_compression_hashes']) == count
            and not record['markers_lost']
            and record['critical_marker_retained']
            and drift[str(count)] == 0
            and not record['secret_in_output']
            and all(value for value in record['prefix_hashes'])
            and record['focus_reached_every_call']
            and record['memory_reached_every_call']
        )

    checks['window_integrity'] = {
        'applicable': len(CFG['cycles']) > 1 and progression_complete,
        'passed': bool(integrity_ok),
        'evidence': {
            'cycles_evaluated': list(CFG['cycles']),
            'progression_complete': progression_complete,
            'compactions_observed_by_cycle': dict(
                (key, value['compactions_observed'])
                for key, value in per_cycle.items()
            ),
            'compaction_observation_source': (
                'compressor.compression_count increment per forced cycle'
            ),
            'deterministic_drift_by_cycle': drift,
            'per_cycle': dict(
                (key, {
                    'compactions_observed': value['compactions_observed'],
                    'every_publication_exact': value[
                        'every_publication_exact'
                    ],
                    'publication_shapes': value['publication_shapes'],
                    'markers_retained': value['markers_retained'],
                    'markers_expected': value['markers_expected'],
                    'markers_lost': value['markers_lost'],
                    'critical_marker_retained': value[
                        'critical_marker_retained'
                    ],
                    'compression_count_deltas': value[
                        'compression_count_deltas'
                    ],
                    'every_cycle_incremented_once': value[
                        'every_cycle_incremented_once'
                    ],
                    'every_cycle_issued_calls': value[
                        'every_cycle_issued_calls'
                    ],
                    'every_cycle_timed_one_transaction': value[
                        'every_cycle_timed_one_transaction'
                    ],
                    'aux_calls_per_cycle': value['aux_calls_per_cycle'],
                    'initial_transcript_sha256': value[
                        'initial_transcript_sha256'
                    ],
                    'pre_compression_hashes': value['pre_compression_hashes'],
                    'pre_compression_hashes_distinct': value[
                        'pre_compression_hashes_distinct'
                    ],
                    'window_transcript_sha256': value[
                        'window_transcript_sha256'
                    ],
                    'prefix_hashes': value['prefix_hashes'],
                    'prefix_stable_at_boundary': value[
                        'prefix_stable_at_boundary'
                    ],
                })
                for key, value in per_cycle.items()
            ),
        },
    }
    metrics['cycles'] = per_cycle


SCENARIOS = (
    scenario_coverage,
    scenario_full_window_derivation,
    scenario_fragments,
    scenario_rollback,
    scenario_cancellation,
    scenario_retry,
    scenario_fallback_and_abort,
    scenario_restart_cross_session,
    scenario_below_cap,
    scenario_cycles,
)


def percentile(ordered, fraction):
    if not ordered:
        raise DriverError('DRIVER_EMPTY_SAMPLES')
    index = min(
        len(ordered) - 1,
        max(0, int(round(fraction * (len(ordered) - 1)))),
    )
    return ordered[index]


def finite_non_negative(values):
    for value in values:
        if not isinstance(value, (int, float)):
            return False
        if isinstance(value, bool):
            return False
        if value != value or value in (float('inf'), float('-inf')):
            return False
        if value < 0:
            return False
    return True


def main():
    bootstrap()
    frozen_date = freeze_clock()

    checks = None
    metrics = None
    volume_per_trial = []
    wall_clock = []
    fake_seconds = []
    transaction_seconds = []
    per_scenario_volume = None
    structure_identical = True
    volume_identical = True

    for trial in range(CFG['trials']):
        reset_trial()
        trial_checks = {}
        trial_metrics = {}
        start = time.perf_counter()
        for scenario in SCENARIOS:
            scenario(trial_checks, trial_metrics)
        wall_clock.append(time.perf_counter() - start)

        volume = dict(
            (bucket, sum(TRIAL[bucket].values()))
            for bucket in (
                'aux_calls',
                'prompt_chars',
                'estimated_input_tokens',
                'estimated_output_tokens',
            )
        )
        volume_per_trial.append(volume)
        for values in TRIAL['fake_seconds'].values():
            fake_seconds.extend(values)
        for values in TRIAL['transaction_seconds'].values():
            transaction_seconds.extend(values)

        if checks is None:
            checks = trial_checks
            metrics = trial_metrics
            per_scenario_volume = dict(
                (bucket, dict(TRIAL[bucket]))
                for bucket in (
                    'aux_calls',
                    'prompt_chars',
                    'estimated_input_tokens',
                    'estimated_output_tokens',
                )
            )
        else:
            # All timing lives in the accumulators, so `checks` and the
            # structural metrics must be byte-identical across trials.
            if canonical(trial_checks) != canonical(checks):
                structure_identical = False
            if canonical(trial_metrics) != canonical(metrics):
                structure_identical = False
            if canonical(volume) != canonical(volume_per_trial[0]):
                volume_identical = False

    if not fake_seconds or not transaction_seconds or not wall_clock:
        raise DriverError('DRIVER_EMPTY_SAMPLES')
    if not finite_non_negative(
        fake_seconds + transaction_seconds + wall_clock
    ):
        raise DriverError('DRIVER_METRIC_INVALID')
    if not structure_identical or not volume_identical:
        raise DriverError('DRIVER_NONDETERMINISTIC')

    ordered_fake = sorted(fake_seconds)
    ordered_tx = sorted(transaction_seconds)
    volume = volume_per_trial[0]

    emit({
        'driver_status': 'OK',
        'revision_label': CFG['revision_label'],
        'revision_sha': CFG['revision_sha'],
        'module_path_within_tree': True,
        'module_source_sha256': sha_file(MODULE_PATH),
        'module_relpath': os.path.relpath(MODULE_PATH, TREE),
        'python_version': sys.version.split()[0],
        'frozen_prompt_date': frozen_date,
        'checks': checks,
        'metrics': metrics,
        'volume': {
            'aux_calls': volume['aux_calls'],
            'prompt_chars': volume['prompt_chars'],
            'estimated_input_tokens': volume['estimated_input_tokens'],
            'estimated_output_tokens': volume['estimated_output_tokens'],
            'per_scenario': per_scenario_volume,
        },
        'latency': {
            'fake_call_samples': len(ordered_fake),
            'fake_call_seconds_total': sum(ordered_fake),
            'fake_call_seconds_p50': percentile(ordered_fake, 0.50),
            'fake_call_seconds_p95': percentile(ordered_fake, 0.95),
            'transaction_samples': len(ordered_tx),
            'transaction_seconds_total': sum(ordered_tx),
            'transaction_seconds_p50': percentile(ordered_tx, 0.50),
            'transaction_seconds_p95': percentile(ordered_tx, 0.95),
            'wall_clock_samples': len(wall_clock),
            'wall_clock_seconds_total': sum(wall_clock),
            'wall_clock_seconds_per_trial': wall_clock,
        },
        'determinism': {
            'trials': CFG['trials'],
            'structure_identical_across_trials': structure_identical,
            'volume_identical_across_trials': volume_identical,
        },
    })


try:
    main()
except DriverError as exc:
    emit({
        'driver_status': 'ERROR',
        'revision_label': CFG.get('revision_label'),
        'revision_sha': CFG.get('revision_sha'),
        'error_code': exc.code,
        'error_class': 'DriverError',
    })
    sys.exit(3)
except BaseException as exc:
    # Only the exception CLASS NAME leaves this process. No message, no
    # traceback, no path, no transcript text.
    emit({
        'driver_status': 'ERROR',
        'revision_label': CFG.get('revision_label'),
        'revision_sha': CFG.get('revision_sha'),
        'error_code': 'DRIVER_UNEXPECTED_ERROR',
        'error_class': type(exc).__name__,
    })
    sys.exit(3)
'''


# ═════════════════════════════════════════════════════════════════════════
# Parent
# ═════════════════════════════════════════════════════════════════════════


class EvaluationError(RuntimeError):
    """Fail-closed condition carrying an allowlisted code and safe detail."""

    def __init__(self, code: str, detail: Optional[Dict[str, Any]] = None):
        RuntimeError.__init__(self, code)
        self.code = code if code in ERROR_CODES else "UNEXPECTED_INTERNAL_ERROR"
        self.detail = detail or {}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise EvaluationError("REVISION_UNRESOLVED", {"git_args": args[0]})
    return result.stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def resolve_identities(repo: Path) -> Dict[str, Any]:
    identities: Dict[str, Any] = {}
    for label, sha in (("baseline", BASELINE_SHA), ("candidate", CANDIDATE_SHA)):
        resolved = git(repo, "rev-parse", "--verify", "%s^{commit}" % sha).strip()
        if resolved != sha:
            raise EvaluationError("REVISION_UNRESOLVED", {"revision": label})
        blob = git(repo, "rev-parse", "%s:%s" % (sha, SUBJECT_PATH)).strip()
        content = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "blob", blob],
            capture_output=True,
            check=False,
        )
        if content.returncode != 0:
            raise EvaluationError("REVISION_BLOB_UNREADABLE", {"revision": label})
        identities[label] = {
            "commit_sha": sha,
            "subject_path": SUBJECT_PATH,
            "subject_blob_sha1": blob,
            "subject_content_sha256": sha256_bytes(content.stdout),
        }
    if (
        identities["baseline"]["subject_blob_sha1"]
        == identities["candidate"]["subject_blob_sha1"]
    ):
        raise EvaluationError("REVISION_BLOBS_IDENTICAL")
    return identities


def extract_revision(repo: Path, sha: str, label: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", sha],
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        raise EvaluationError("REVISION_EXTRACTION_FAILED", {"revision": label})
    untar = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive.stdout,
        capture_output=True,
        check=False,
    )
    if untar.returncode != 0:
        raise EvaluationError("REVISION_EXTRACTION_FAILED", {"revision": label})


def child_env(home: Path) -> Dict[str, str]:
    env = dict(os.environ)
    for name in CREDENTIAL_ENV_VARS:
        env[name] = ""
    env["HERMES_HOME"] = str(home)
    env["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["NO_COLOR"] = "1"
    env.pop("PYTHONPATH", None)
    return env


def run_driver(
    tree: Path,
    home: Path,
    out_path: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    # Child stdout/stderr are captured and discarded: they can contain
    # transcript text and runner-local paths, and nothing from them is ever
    # persisted. Structured evidence travels only through the JSON file.
    subprocess.run(
        [sys.executable, "-", json.dumps(config), str(out_path)],
        input=DRIVER_SOURCE,
        text=True,
        capture_output=True,
        cwd=str(tree),
        env=child_env(home),
        check=False,
    )
    if not out_path.exists():
        raise EvaluationError(
            "DRIVER_NO_EVIDENCE", {"revision": config["revision_label"]}
        )
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        raise EvaluationError(
            "DRIVER_EVIDENCE_MALFORMED", {"revision": config["revision_label"]}
        )
    return payload


def _numeric_ok(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def validate_driver_payload(payload: Dict[str, Any], identity: Dict[str, Any]) -> None:
    label = payload.get("revision_label") or "unknown"
    if payload.get("driver_status") != "OK":
        code = payload.get("error_code")
        klass = payload.get("error_class")
        raise EvaluationError("DRIVER_REPORTED_ERROR", {
            "revision": label,
            "driver_error_code": (
                code if code in DRIVER_ERROR_CODES else "DRIVER_UNKNOWN_CODE"
            ),
            "driver_error_class": (
                klass
                if isinstance(klass, str) and DRIVER_ERROR_CLASS_RE.match(klass)
                else "UnknownErrorClass"
            ),
        })
    if payload.get("revision_sha") != identity["commit_sha"]:
        raise EvaluationError("DRIVER_IDENTITY_MISMATCH", {"revision": label})
    if payload.get("module_source_sha256") != identity["subject_content_sha256"]:
        raise EvaluationError("DRIVER_IDENTITY_MISMATCH", {"revision": label})
    if payload.get("module_relpath") != SUBJECT_PATH:
        raise EvaluationError("DRIVER_IDENTITY_MISMATCH", {"revision": label})
    if not payload.get("module_path_within_tree"):
        raise EvaluationError("DRIVER_IDENTITY_MISMATCH", {"revision": label})
    for key in ("checks", "metrics", "volume", "latency", "determinism"):
        if not isinstance(payload.get(key), dict) or not payload[key]:
            raise EvaluationError(
                "DRIVER_EVIDENCE_MALFORMED", {"revision": label, "field": key}
            )
    determinism = payload["determinism"]
    if not determinism.get("structure_identical_across_trials") or not determinism.get(
        "volume_identical_across_trials"
    ):
        raise EvaluationError(
            "DRIVER_EVIDENCE_MALFORMED",
            {"revision": label, "field": "determinism"},
        )
    # Every numeric the decision depends on: finite, non-negative, non-empty.
    for section, field in DECISION_METRICS:
        if not _numeric_ok(payload[section].get(field)):
            raise EvaluationError(
                "DRIVER_METRIC_INVALID", {"revision": label, "field": field}
            )
    for section, field in DECISION_SAMPLE_COUNTS:
        count = payload[section].get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise EvaluationError(
                "DRIVER_METRIC_INVALID", {"revision": label, "field": field}
            )


def build_gates(
    candidate: Dict[str, Any],
    baseline: Dict[str, Any],
) -> List[Dict[str, Any]]:
    contracts = dict((spec[0], spec[2]) for spec in GATE_SPECS)
    gates: List[Dict[str, Any]] = []
    checks = candidate["checks"]
    for gate_id, check_name, contract in GATE_SPECS:
        if check_name is None:
            continue
        check = checks.get(check_name)
        if check is None:
            gates.append({
                "gate_id": gate_id,
                "contract": contract,
                "status": "FAIL",
                "reason": "candidate reported no evidence for this contract",
                "evidence": {},
            })
            continue
        if not check.get("applicable"):
            gates.append({
                "gate_id": gate_id,
                "contract": contract,
                "status": "FAIL",
                "reason": (
                    "the candidate scenario was not applicable, so the "
                    "contract was never exercised"
                ),
                "evidence": check.get("evidence", {}),
            })
            continue
        gates.append({
            "gate_id": gate_id,
            "contract": contract,
            "status": "PASS" if check.get("passed") else "FAIL",
            "reason": "",
            "evidence": check.get("evidence", {}),
        })

    # G4 also carries a cross-revision publication-shape comparison: the
    # candidate must not change the role or framing the handoff publishes
    # under. Folded into the existing gate rather than adding a new one.
    for gate in gates:
        if gate["gate_id"] != "g4_window_integrity_and_publication":
            continue
        base_cycles = baseline["metrics"].get("cycles", {})
        cand_cycles = candidate["metrics"].get("cycles", {})
        shape_diffs = []
        for key in sorted(cand_cycles, key=int):
            if key == "0":
                continue
            base_shapes = [
                {k: v for k, v in shape.items() if k in ("role", "exact_framing")}
                for shape in base_cycles.get(key, {}).get(
                    "publication_shapes", []
                )
            ]
            cand_shapes = [
                {k: v for k, v in shape.items() if k in ("role", "exact_framing")}
                for shape in cand_cycles[key].get("publication_shapes", [])
            ]
            if base_shapes != cand_shapes:
                shape_diffs.append({"cycles": key})
        gate["evidence"] = dict(gate["evidence"])
        gate["evidence"]["cross_revision_publication_shape_diffs"] = shape_diffs
        if shape_diffs and gate["status"] == "PASS":
            gate["status"] = "FAIL"
            gate["reason"] = (
                "the candidate publishes the handoff under a different role "
                "or framing than the baseline"
            )

    # G6 — exact below-cap parity across the complete request payload, the
    # complete stable post-call state, and the complete telemetry mapping.
    base_parity = baseline["metrics"].get("below_cap_parity") or {}
    cand_parity = candidate["metrics"].get("below_cap_parity") or {}
    scalar_fields = (
        "aux_calls",
        "request_payload_sha256",
        "request_payload_keys",
        "request_carries_focus_topic",
        "request_carries_memory_context",
        "result_summary_sha256",
        "result_has_summary_prefix",
        "state_map_sha256",
        "telemetry_map_sha256",
    )
    mismatched = [
        field
        for field in scalar_fields
        if base_parity.get(field) != cand_parity.get(field)
    ]
    base_state = base_parity.get("state_field_hashes") or {}
    cand_state = cand_parity.get("state_field_hashes") or {}
    state_diff = sorted(
        set(base_state) ^ set(cand_state)
        | {
            name
            for name in set(base_state) & set(cand_state)
            if base_state[name] != cand_state[name]
        }
    )
    base_telemetry = base_parity.get("telemetry_field_values") or {}
    cand_telemetry = cand_parity.get("telemetry_field_values") or {}
    telemetry_diff = sorted(
        set(base_telemetry) ^ set(cand_telemetry)
        | {
            name
            for name in set(base_telemetry) & set(cand_telemetry)
            if base_telemetry[name] != cand_telemetry[name]
        }
    )
    parity_applicable = bool(
        base_parity and cand_parity and cand_parity.get("aux_calls") == 1
    )
    gates.append({
        "gate_id": "g6_below_cap_exact_parity",
        "contract": contracts["g6_below_cap_exact_parity"],
        "status": (
            "PASS"
            if (
                parity_applicable
                and not mismatched
                and not state_diff
                and not telemetry_diff
                # A vacuous parity pass is not evidence: the frozen focus and
                # memory inputs must actually have reached the request.
                and cand_parity.get("request_carries_focus_topic")
                and cand_parity.get("request_carries_memory_context")
            )
            else "FAIL"
        ),
        "reason": (
            ""
            if parity_applicable
            else "the below-cap single-pass scenario was not exercised"
        ),
        "evidence": {
            "mismatched_fields": mismatched,
            "differing_state_fields": state_diff,
            "differing_telemetry_keys": telemetry_diff,
            "state_fields_compared": len(cand_state),
            "telemetry_keys_compared": len(cand_telemetry),
            "state_exclusions": list(STATE_EXCLUSIONS),
            "telemetry_exclusions": list(TELEMETRY_EXCLUSIONS),
            "baseline_state_map_sha256": base_parity.get("state_map_sha256"),
            "candidate_state_map_sha256": cand_parity.get("state_map_sha256"),
            "baseline_request_payload_sha256": base_parity.get(
                "request_payload_sha256"
            ),
            "candidate_request_payload_sha256": cand_parity.get(
                "request_payload_sha256"
            ),
        },
    })

    # G7 — reusable assembled prefix byte identity across the checkpoint.
    #
    # The claim is a WITHIN-revision one, because that is what prompt caching
    # actually depends on: the rows the provider has already seen ahead of the
    # compaction checkpoint must come back byte-identical after the
    # compaction. The very first compaction is the one sanctioned boundary
    # shift (it appends the compaction note to the system prompt), so it is
    # reported and excluded from the gate.
    #
    # A CROSS-revision byte comparison of the prefix would be the wrong gate:
    # from the second cycle onward the prefix descends from a handoff whose
    # summary text legitimately differs between the revisions — that
    # difference is the change under evaluation, not a regression. The
    # cross-revision hashes are therefore recorded as evidence only.
    base_cycles = baseline["metrics"].get("cycles", {})
    cand_cycles = candidate["metrics"].get("cycles", {})
    checkpoints = 0
    unstable = []
    cross_revision_prefix_hashes = {}
    for key in sorted(cand_cycles, key=int):
        if key == "0":
            continue
        cand_cycle = cand_cycles[key]
        stability = cand_cycle.get("prefix_stable_at_boundary") or []
        checkpoints += len(stability)
        if not cand_cycle.get("prefix_stable_after_first_boundary"):
            unstable.append({"cycles": key, "stability": stability})
        cross_revision_prefix_hashes[key] = {
            "baseline": base_cycles.get(key, {}).get("prefix_hashes"),
            "candidate": cand_cycle.get("prefix_hashes"),
        }
    prefix_complete = all(
        all(value for value in cand_cycles[key].get("prefix_hashes") or [])
        for key in cand_cycles
        if key != "0"
    )
    observed_post_first = any(
        len(cand_cycles[key].get("prefix_stable_at_boundary") or []) > 1
        for key in cand_cycles
        if key != "0"
    )
    deterministic = bool(
        candidate["determinism"].get("structure_identical_across_trials")
    )
    gates.append({
        "gate_id": "g7_reusable_prefix_byte_identity",
        "contract": contracts["g7_reusable_prefix_byte_identity"],
        "status": (
            "PASS"
            if (
                checkpoints
                and observed_post_first
                and prefix_complete
                and deterministic
                and not unstable
            )
            else "FAIL"
        ),
        "reason": (
            ""
            if (checkpoints and observed_post_first)
            else "no compaction checkpoint past the first boundary was observed"
        ),
        "evidence": {
            "checkpoints_observed": checkpoints,
            "post_first_boundary_checkpoint_observed": observed_post_first,
            "unstable_prefixes": unstable,
            "every_checkpoint_published_a_prefix": prefix_complete,
            "deterministic_across_trials": deterministic,
            "cross_revision_prefix_hashes": cross_revision_prefix_hashes,
            "cross_revision_difference_is_expected": (
                "Prefix bytes from the second cycle onward descend from a "
                "handoff whose summary text legitimately differs between the "
                "revisions; this is recorded, not gated."
            ),
        },
    })

    order = [spec[0] for spec in GATE_SPECS]
    gates.sort(key=lambda gate: order.index(gate["gate_id"]))
    return gates


# Every directional measure. A candidate increase in any of these sets
# TECHNICAL_PASS_TRADEOFF_PENDING; no materiality threshold is invented.
DIRECTIONAL_MEASURES = (
    ("volume", "aux_calls", "auxiliary calls"),
    ("volume", "prompt_chars", "prompt characters"),
    ("volume", "estimated_input_tokens", "estimated input tokens"),
    ("volume", "estimated_output_tokens", "estimated output tokens"),
    ("latency", "fake_call_seconds_total", "fixed-delay blocking total"),
    ("latency", "fake_call_seconds_p50", "fake-call p50"),
    ("latency", "fake_call_seconds_p95", "fake-call p95"),
    ("latency", "transaction_seconds_total", "transaction total"),
    ("latency", "transaction_seconds_p50", "transaction p50"),
    ("latency", "transaction_seconds_p95", "transaction p95"),
    ("latency", "wall_clock_seconds_total", "wall clock"),
)


def build_comparison(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    measures = {}
    for section, field, human in DIRECTIONAL_MEASURES:
        base_value = baseline[section][field]
        cand_value = candidate[section][field]
        measures[field] = {
            "label": human,
            "baseline": base_value,
            "candidate": cand_value,
            "delta": cand_value - base_value,
            "candidate_increased": cand_value > base_value,
        }

    base_cycles = baseline["metrics"].get("cycles", {})
    cand_cycles = candidate["metrics"].get("cycles", {})
    per_cycle = {}
    for key in sorted(cand_cycles, key=int):
        base_cycle = base_cycles.get(key, {})
        cand_cycle = cand_cycles[key]
        per_cycle[key] = {
            "baseline_aux_calls": base_cycle.get("aux_calls_total"),
            "candidate_aux_calls": cand_cycle.get("aux_calls_total"),
            "aux_call_delta": (
                (cand_cycle.get("aux_calls_total") or 0)
                - (base_cycle.get("aux_calls_total") or 0)
            ),
            "baseline_aux_calls_per_cycle": base_cycle.get(
                "aux_calls_per_cycle"
            ),
            "candidate_aux_calls_per_cycle": cand_cycle.get(
                "aux_calls_per_cycle"
            ),
            "baseline_prompt_chars": base_cycle.get("prompt_chars"),
            "candidate_prompt_chars": cand_cycle.get("prompt_chars"),
            "prompt_chars_delta": (
                (cand_cycle.get("prompt_chars") or 0)
                - (base_cycle.get("prompt_chars") or 0)
            ),
            "baseline_estimated_input_tokens": base_cycle.get(
                "estimated_input_tokens"
            ),
            "candidate_estimated_input_tokens": cand_cycle.get(
                "estimated_input_tokens"
            ),
            "estimated_input_tokens_delta": (
                (cand_cycle.get("estimated_input_tokens") or 0)
                - (base_cycle.get("estimated_input_tokens") or 0)
            ),
            "baseline_estimated_output_tokens": base_cycle.get(
                "estimated_output_tokens"
            ),
            "candidate_estimated_output_tokens": cand_cycle.get(
                "estimated_output_tokens"
            ),
            "baseline_markers_retained": base_cycle.get("markers_retained"),
            "candidate_markers_retained": cand_cycle.get("markers_retained"),
            "baseline_markers_lost": base_cycle.get("markers_lost"),
            "candidate_markers_lost": cand_cycle.get("markers_lost"),
            "baseline_critical_marker_retained": base_cycle.get(
                "critical_marker_retained"
            ),
            "candidate_critical_marker_retained": cand_cycle.get(
                "critical_marker_retained"
            ),
        }

    return {
        "directional_measures": measures,
        "per_cycle": per_cycle,
        "per_scenario_volume": {
            "baseline": baseline["volume"].get("per_scenario"),
            "candidate": candidate["volume"].get("per_scenario"),
        },
        "sample_counts": {
            "baseline": {
                "fake_call_samples": baseline["latency"]["fake_call_samples"],
                "transaction_samples": baseline["latency"][
                    "transaction_samples"
                ],
                "wall_clock_samples": baseline["latency"][
                    "wall_clock_samples"
                ],
            },
            "candidate": {
                "fake_call_samples": candidate["latency"]["fake_call_samples"],
                "transaction_samples": candidate["latency"][
                    "transaction_samples"
                ],
                "wall_clock_samples": candidate["latency"][
                    "wall_clock_samples"
                ],
            },
        },
        "baseline_control_checks": dict(
            (name, {
                "applicable": check.get("applicable"),
                "passed": check.get("passed"),
            })
            for name, check in baseline["checks"].items()
        ),
        "baseline_coverage_loss_is_control_evidence": True,
    }


def decide_status(
    gates: List[Dict[str, Any]],
    comparison: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    failed = [gate["gate_id"] for gate in gates if gate["status"] != "PASS"]
    if failed:
        return "FAIL", {
            "failed_gates": failed,
            "increased_measures": [],
            "rationale": "at least one candidate correctness gate failed",
        }
    increased = [
        name
        for name, record in sorted(comparison["directional_measures"].items())
        if record["candidate_increased"]
    ]
    if increased:
        return "TECHNICAL_PASS_TRADEOFF_PENDING", {
            "failed_gates": [],
            "increased_measures": increased,
            "rationale": (
                "candidate correctness gates all pass, but at least one "
                "directional cost or latency measure increased. This harness "
                "deliberately does not invent a materiality threshold; the "
                "trade-off is left for a maintainer decision."
            ),
        }
    return "PASS", {
        "failed_gates": [],
        "increased_measures": [],
        "rationale": (
            "candidate correctness gates all pass with no directional "
            "increase in any measured cost or latency dimension"
        ),
    }


RESIDUAL_LIMITATIONS = [
    "Tier A only: deterministic correctness plus fake-provider call, token, "
    "and latency accounting. No claim is made about summary quality, real "
    "provider latency, token cost, or production readiness.",
    "The auxiliary provider is a fake with a fixed artificial delay, so "
    "latency figures measure harness and compressor overhead around a "
    "constant blocking cost, not provider behavior.",
    "Token counts are locally estimated at four characters per token, not "
    "produced by a real tokenizer.",
    "Baseline coverage loss is control evidence about pre-change behavior "
    "and is never treated as a candidate gate failure.",
    "Aggregate-bound scenarios run under reduced summarizer input caps so "
    "the bounded-pass path is reachable within a CI budget; production "
    "defaults are larger.",
    "Message member-dict identity is not observable across the compaction "
    "entry point, because the cheap pre-pass copies every row on both "
    "revisions; canonical content equality is gated there instead, and "
    "identity is gated only on the summary-entry seam where it is real.",
    "A TECHNICAL_PASS_TRADEOFF_PENDING result is not an approval: the "
    "auxiliary-call, token, and latency trade-off is reported, not resolved.",
]


# ── Output boundary ──────────────────────────────────────────────────────
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s\"']*)|(?:(?<![\w.])/[\w.\-]+(?:/[\w.\-]+)+)")
# Well above every string this harness legitimately persists (the longest are
# gate contract sentences, ~300 chars) and far below any fixture transcript
# row, so an over-long string is itself evidence that something leaked.
MAX_PERSISTED_STRING = 600


def validate_and_scrub(document: Any, secret: str) -> Tuple[Any, List[str]]:
    """Recursively enforce the persisted-output boundary; fail closed.

    Returns the scrubbed document and a list of violation kinds. Any violation
    forces the overall status to FAIL — a leak means the evidence cannot be
    trusted, not that it should be quietly cleaned up.
    """
    violations: List[str] = []

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            text = value
            if secret and secret in text:
                text = text.replace(secret, "<redacted>")
                violations.append("fake_secret_in_output")
            scrubbed = ABSOLUTE_PATH_RE.sub("<path-redacted>", text)
            if scrubbed != text:
                violations.append("absolute_path_in_output")
                text = scrubbed
            if len(text) > MAX_PERSISTED_STRING:
                text = text[:MAX_PERSISTED_STRING] + "<truncated>"
                violations.append("oversized_string_in_output")
            return text
        if isinstance(value, dict):
            return dict((walk(k), walk(v)) for k, v in value.items())
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            violations.append("non_finite_number_in_output")
            return None
        return value

    return walk(document), sorted(set(violations))


def write_atomic(path: Path, text: str) -> None:
    """Replace *path* in one step so a partial write can never be uploaded."""
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def stage_finalized_pair(
    staging: Path,
    document: Dict[str, Any],
    rendered: str,
) -> None:
    """Stage the finalized pair, then publish an explicit finalization marker.

    Staging is a directory of its own, separate from the pre-seeded FAIL
    evidence directory, so nothing written here is uploadable until the
    workflow validates the complete pair and promotes the whole directory by a
    single atomic rename. The marker is written last and the checksum file
    second-to-last, so a termination anywhere in this sequence leaves the pair
    unselectable and the FAIL directory selected — a partially written pair can
    never surface as a selected PASS document.
    """
    json_path = staging / EVIDENCE_JSON_NAME
    summary_path = staging / EVIDENCE_SUMMARY_NAME
    json_text = json.dumps(document, indent=2, sort_keys=True) + "\n"

    write_atomic(json_path, json_text)
    write_atomic(summary_path, rendered)
    # `sha256sum -c` format, so the workflow can validate the complete pair
    # without any dependency.
    write_atomic(
        staging / FINALIZATION_CHECKSUM_NAME,
        "%s  %s\n%s  %s\n"
        % (
            sha256_bytes(json_text.encode("utf-8")),
            EVIDENCE_JSON_NAME,
            sha256_bytes(rendered.encode("utf-8")),
            EVIDENCE_SUMMARY_NAME,
        ),
    )
    write_atomic(
        staging / FINALIZATION_MARKER_NAME,
        "%s\n" % document["overall_status"],
    )


def render_summary(document: Dict[str, Any]) -> str:
    status = document["overall_status"]
    decision = document["decision"]
    comparison = document.get("comparison") or {}
    measures = comparison.get("directional_measures") or {}
    lines = [
        "# Tier A — context compaction comparison",
        "",
        "**Overall status: `%s`**" % status,
        "",
        "Disposable evaluation harness. Fake provider, fixed artificial "
        "delay, frozen clock, no network and no provider spend.",
        "",
        "| Identity | Value |",
        "| --- | --- |",
    ]
    revisions = document.get("revisions") or {}
    for label in ("baseline", "candidate"):
        record = revisions.get(label) or {}
        lines.append(
            "| %s commit | `%s` |" % (label.title(), record.get("commit_sha", "n/a"))
        )
        lines.append(
            "| %s compressor blob | `%s` |"
            % (label.title(), record.get("subject_blob_sha1", "n/a"))
        )
    lines += [
        "| Fixture sha256 | `%s` |"
        % (document.get("fixture") or {}).get("sha256", "n/a"),
        "",
        "## Technical correctness gates",
        "",
        "| Gate | Result | Contract |",
        "| --- | --- | --- |",
    ]
    for gate in document.get("gates") or []:
        lines.append(
            "| `%s` | %s | %s |"
            % (
                gate["gate_id"],
                "PASS" if gate["status"] == "PASS" else "**FAIL**",
                gate["contract"].replace("|", "/"),
            )
        )
    lines += [
        "",
        "## Unresolved cost / latency trade-off",
        "",
        "Technical correctness above is a separate question from the cost and "
        "latency numbers below. This harness does not define a materiality "
        "threshold for the trade-off, so every directional measure is "
        "reported and the decision is left open for a maintainer.",
        "",
        "| Measure | Baseline | Candidate | Delta | Increased |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _section, field, human in DIRECTIONAL_MEASURES:
        record = measures.get(field)
        if not record:
            continue
        base_value = record["baseline"]
        cand_value = record["candidate"]
        delta = record["delta"]
        if isinstance(base_value, float) or isinstance(cand_value, float):
            cells = ("%.4f" % base_value, "%.4f" % cand_value, "%+.4f" % delta)
        else:
            cells = (str(base_value), str(cand_value), "%+d" % delta)
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (human, cells[0], cells[1], cells[2],
               "yes" if record["candidate_increased"] else "no")
        )
    lines += [
        "",
        "### Per compaction-cycle progression",
        "",
        "| Cycles | Base calls | Cand calls | Call delta | Base prompt chars | "
        "Cand prompt chars | Base markers | Cand markers |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    marker_total = (document.get("fixture") or {}).get("marker_count", "?")
    for key in sorted(comparison.get("per_cycle") or {}, key=int):
        row = comparison["per_cycle"][key]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s/%s | %s/%s |"
            % (
                key,
                row["baseline_aux_calls"],
                row["candidate_aux_calls"],
                row["aux_call_delta"],
                row["baseline_prompt_chars"],
                row["candidate_prompt_chars"],
                row["baseline_markers_retained"],
                marker_total,
                row["candidate_markers_retained"],
                marker_total,
            )
        )
    lines += [
        "",
        "Baseline marker loss is control evidence about the pre-change "
        "behavior, not a candidate failure.",
        "",
        "## Decision",
        "",
        decision.get("rationale", ""),
        "",
    ]
    if decision.get("failed_gates"):
        lines += [
            "Failed gates: %s" % ", ".join(
                "`%s`" % item for item in decision["failed_gates"]
            ),
            "",
        ]
    if decision.get("increased_measures"):
        lines += [
            "Increased measures: %s" % ", ".join(
                "`%s`" % item for item in decision["increased_measures"]
            ),
            "",
        ]
    lines += ["## Residual limitations", ""]
    lines += ["- %s" % item for item in document.get("residual_limitations", [])]
    lines.append("")
    return "\n".join(lines)


def render_failure_summary(document: Dict[str, Any]) -> str:
    decision = document.get("decision") or {}
    return "\n".join([
        "# Tier A — context compaction comparison",
        "",
        "**Overall status: `%s`**" % document.get("overall_status", "FAIL"),
        "",
        "The evaluation did not produce a complete comparison, so only the "
        "machine-readable evidence is available.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Error code | `%s` |" % decision.get("error_code", "UNKNOWN"),
        "| Rationale | %s |" % decision.get("rationale", ""),
        "",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tier A comparison")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--staging-dir", default="staging")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument(
        "--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    # This process NEVER writes into the upload directory. It stages a
    # finalized pair here; the workflow decides whether that pair replaces the
    # pre-seeded FAIL placeholders.
    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    document: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "run": {
            "run_id": str(args.run_id),
            "run_attempt": str(args.run_attempt),
            "eval_branch": EVAL_BRANCH,
            "pr_base_branch": PR_BASE_BRANCH,
            "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "contract_parameters": {
            "cycles": list(CYCLES),
            "trials": TRIALS,
            "fake_delay_seconds": FAKE_DELAY_SECONDS,
            "split_limit_chars": SPLIT_LIMIT_CHARS,
            "fragment_limit_chars": FRAGMENT_LIMIT_CHARS,
            "below_cap_limit_chars": BELOW_CAP_LIMIT_CHARS,
            "context_length": CONTEXT_LENGTH,
            "tail_token_budget": TAIL_TOKEN_BUDGET,
            "current_tokens": CURRENT_TOKENS,
            "failure_schedule": FAILURE_SCHEDULE,
            "main_model": MAIN_MODEL,
            "aux_model": AUX_MODEL,
            "routing_discriminator": ROUTING_DISCRIMINATOR,
            "focus_topic": FOCUS_TOPIC,
            "memory_context": MEMORY_CONTEXT,
            "frozen_clock_utc": FROZEN_CLOCK_UTC,
            "state_exclusions": list(STATE_EXCLUSIONS),
            "telemetry_exclusions": list(TELEMETRY_EXCLUSIONS),
            "provider": "fake, in-process; no network and no provider spend",
        },
        "residual_limitations": RESIDUAL_LIMITATIONS,
        "overall_status": "FAIL",
        "decision": {
            "failed_gates": [],
            "increased_measures": [],
            "error_code": "UNEXPECTED_INTERNAL_ERROR",
            "rationale": "evaluation did not complete",
        },
        "gates": [],
        "revisions": {},
        "fixture": {},
        "per_revision": {},
        "comparison": {},
        "output_boundary": {},
    }
    secret = ""

    try:
        fixture_path = repo / FIXTURE_RELPATH
        try:
            fixture_bytes = fixture_path.read_bytes()
        except Exception:
            raise EvaluationError("FIXTURE_MISSING")
        fixture_hash = sha256_bytes(fixture_bytes)
        if fixture_hash != FIXTURE_SHA256:
            raise EvaluationError("FIXTURE_HASH_MISMATCH", {
                "expected_sha256": FIXTURE_SHA256,
                "observed_sha256": fixture_hash,
            })
        try:
            fixture = json.loads(fixture_bytes.decode("utf-8"))
            secret = fixture["fake_secret"]
            marker_count = len(fixture["main_session"]["marker_order"])
        except Exception:
            raise EvaluationError("FIXTURE_MALFORMED")
        document["fixture"] = {
            "path": FIXTURE_RELPATH,
            "sha256": fixture_hash,
            "schema_version": fixture["schema_version"],
            "marker_count": marker_count,
            "cross_session_marker_count": len(
                fixture["cross_session"]["marker_order"]
            ),
            "final_assistant_only_marker_count": len(
                fixture["final_assistant_only"]["marker_order"]
            ),
        }

        identities = resolve_identities(repo)
        document["revisions"] = identities

        payloads: Dict[str, Dict[str, Any]] = {}
        with tempfile.TemporaryDirectory(prefix="tier-a-") as workspace:
            root = Path(workspace)
            for label, sha in (
                ("baseline", BASELINE_SHA),
                ("candidate", CANDIDATE_SHA),
            ):
                tree = root / label
                home = root / ("%s-home" % label)
                home.mkdir(parents=True, exist_ok=True)
                extract_revision(repo, sha, label, tree)
                config = {
                    "revision_label": label,
                    "revision_sha": sha,
                    "fixture": fixture,
                    "cycles": list(CYCLES),
                    "trials": TRIALS,
                    "fake_delay_seconds": FAKE_DELAY_SECONDS,
                    "split_limit_chars": SPLIT_LIMIT_CHARS,
                    "fragment_limit_chars": FRAGMENT_LIMIT_CHARS,
                    "below_cap_limit_chars": BELOW_CAP_LIMIT_CHARS,
                    "context_length": CONTEXT_LENGTH,
                    "tail_token_budget": TAIL_TOKEN_BUDGET,
                    "current_tokens": CURRENT_TOKENS,
                    "failure_schedule": FAILURE_SCHEDULE,
                    "main_model": MAIN_MODEL,
                    "aux_model": AUX_MODEL,
            "routing_discriminator": ROUTING_DISCRIMINATOR,
                    "focus_topic": FOCUS_TOPIC,
                    "memory_context": MEMORY_CONTEXT,
                    "frozen_clock_utc": FROZEN_CLOCK_UTC,
                    "state_exclusions": list(STATE_EXCLUSIONS),
                    "telemetry_exclusions": list(TELEMETRY_EXCLUSIONS),
                }
                payload = run_driver(
                    tree, home, root / ("%s.json" % label), config
                )
                validate_driver_payload(payload, identities[label])
                payloads[label] = payload

        document["per_revision"] = {
            label: {
                "commit_sha": payload["revision_sha"],
                "module_source_sha256": payload["module_source_sha256"],
                "python_version": payload["python_version"],
                "frozen_prompt_date": payload["frozen_prompt_date"],
                "checks": payload["checks"],
                "metrics": payload["metrics"],
                "volume": payload["volume"],
                "latency": payload["latency"],
                "determinism": payload["determinism"],
            }
            for label, payload in payloads.items()
        }
        document["comparison"] = build_comparison(
            payloads["baseline"], payloads["candidate"]
        )
        document["gates"] = build_gates(
            payloads["candidate"], payloads["baseline"]
        )
        status, decision = decide_status(
            document["gates"], document["comparison"]
        )
        decision["error_code"] = None
        document["overall_status"] = status
        document["decision"] = decision
    except EvaluationError as exc:
        document["overall_status"] = "FAIL"
        document["decision"] = {
            "failed_gates": [],
            "increased_measures": [],
            "error_code": exc.code,
            "error_detail": exc.detail,
            "rationale": "evaluation failed closed with code %s" % exc.code,
        }
    except Exception as exc:
        document["overall_status"] = "FAIL"
        document["decision"] = {
            "failed_gates": [],
            "increased_measures": [],
            "error_code": "UNEXPECTED_INTERNAL_ERROR",
            "error_class": type(exc).__name__,
            "rationale": (
                "evaluation failed closed with an unexpected internal error"
            ),
        }

    # Final recursive validation and scrubbing, applied unconditionally before
    # anything is written. A boundary violation is itself a failure.
    document, violations = validate_and_scrub(document, secret)
    document["output_boundary"] = {
        "validated": True,
        "violations": violations,
        "max_persisted_string_chars": MAX_PERSISTED_STRING,
    }
    if violations:
        document["overall_status"] = "FAIL"
        document["decision"] = {
            "failed_gates": document["decision"].get("failed_gates", []),
            "increased_measures": [],
            "error_code": "OUTPUT_VALIDATION_FAILED",
            "rationale": (
                "the persisted evidence violated the output boundary; the "
                "run is failed closed rather than published"
            ),
        }

    try:
        rendered = render_summary(document)
    except Exception:
        rendered = render_failure_summary(document)

    # Exit-code contract, relied on by the workflow's selection step:
    #   0        -> a complete finalized evidence pair was staged and marked.
    #               The gate VERDICT is carried inside the document, not in
    #               this exit code, so a legitimate gate FAIL still publishes
    #               its own evidence instead of falling back to a placeholder.
    #   non-zero -> the pair could not be finalized. Nothing is selectable and
    #               the pre-seeded FAIL placeholders stay in force.
    try:
        stage_finalized_pair(staging, document, rendered)
    except Exception:
        print("tier-a staging=failed status=%s" % document["overall_status"])
        return 2

    print(
        "tier-a status=%s staged=%s,%s marker=%s"
        % (
            document["overall_status"],
            EVIDENCE_JSON_NAME,
            EVIDENCE_SUMMARY_NAME,
            FINALIZATION_MARKER_NAME,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
