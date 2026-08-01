"""Observable contracts for bounded, transactional context-summary passes."""

from asyncio import CancelledError
from copy import deepcopy
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    ContextCompressor,
)


def _make(limit: int = 1_000) -> ContextCompressor:
    compressor = ContextCompressor(
        model="test/model",
        threshold_percent=0.85,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
        config_context_length=128_000,
    )
    compressor._SUMMARY_INPUT_MAX_CHARS = limit
    return compressor


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _summary_body(label: str, *, no_user: bool = False) -> str:
    task = (
        "None. This session contains no user-authored turns."
        if no_user
        else label
    )
    return f"{HISTORICAL_TASK_HEADING}\n{task}\n\n## Goal\n{label}"


def _source_block(prompt: str) -> str:
    if "NEW TURNS TO INCORPORATE:\n" in prompt:
        return prompt.split("NEW TURNS TO INCORPORATE:\n", 1)[1].split(
            "\n\nUpdate the summary using this exact structure.", 1,
        )[0]
    return prompt.split("TURNS TO SUMMARIZE:\n", 1)[1].split(
        "\n\nUse this exact structure:", 1,
    )[0]


def _turns(count: int = 6, size: int = 360) -> list[dict]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"TURN_{index} " + (chr(65 + index) * size),
        }
        for index in range(count)
    ]


def test_ordered_passes_feed_each_tentative_summary_into_the_next_request():
    compressor = _make()
    compressor._active_compression_telemetry = {
        "chunking": False,
        "chunk_count": 1,
    }
    turns = _turns()
    prompts: list[str] = []
    returned_bodies: list[str] = []

    def call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        body = _summary_body(f"tentative-{len(prompts)}")
        returned_bodies.append(body)
        return _response(body)

    with patch("agent.context_compressor.call_llm", side_effect=call_llm):
        summary = compressor._generate_summary(turns)

    assert summary is not None
    assert len(prompts) > 1
    sources = [_source_block(prompt) for prompt in prompts]
    combined = "".join(sources)
    positions = [combined.index(f"TURN_{index}") for index in range(len(turns))]
    assert positions == sorted(positions)
    assert all(combined.count(f"TURN_{index}") == 1 for index in range(len(turns)))
    assert all(len(source) <= compressor._SUMMARY_INPUT_MAX_CHARS for source in sources)
    for index, prompt in enumerate(prompts[1:], start=1):
        assert returned_bodies[index - 1] in prompt
    assert compressor._active_compression_telemetry["chunking"] is True
    assert compressor._active_compression_telemetry["chunk_count"] == len(prompts)


def test_final_assistant_only_pass_uses_complete_window_facts():
    compressor = _make(limit=6_000)
    marker = (
        "[SKILL_PRUNED: content lost in compression; "
        "reload with skill_view(name='window-skill')]"
    )
    turns = [
        {"role": "user", "content": "LATEST_FULL_WINDOW_USER " + ("u" * 7_000)},
        {"role": "assistant", "content": marker + ("a" * 7_000)},
        *[
            {"role": "assistant", "content": f"assistant-{index} " + ("z" * 7_000)}
            for index in range(4)
        ],
    ]
    prompts: list[str] = []

    def call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _response(_summary_body(f"pass-{len(prompts)}"))

    with patch("agent.context_compressor.call_llm", side_effect=call_llm):
        summary = compressor._generate_summary(turns)

    assert summary is not None
    sources = [_source_block(prompt) for prompt in prompts]
    assert len(sources) > 1
    assert "[USER]:" not in sources[-1]
    budgets = []
    for prompt in prompts:
        match = re.search(r"Target ~(\d+) tokens", prompt)
        assert match is not None
        budgets.append(int(match.group(1)))
    assert len(set(budgets)) == 1
    assert budgets[0] > 2_000
    assert "LATEST_FULL_WINDOW_USER" in summary
    assert marker in summary


def test_later_pass_failure_rolls_back_normalized_state_and_source_objects():
    compressor = _make()
    secret = "sk-proj-" + ("a" * 40)
    compressor._previous_summary = f"seed {secret}"
    turns = _turns()
    before = deepcopy(turns)
    member_ids = [id(turn) for turn in turns]
    calls = 0

    def call_llm(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second pass failed")
        return _response(_summary_body("tentative partial"))

    with patch("agent.context_compressor.call_llm", side_effect=call_llm):
        summary = compressor._generate_summary(turns)

    assert summary is None
    assert calls == 2
    assert compressor._previous_summary.startswith("seed ")
    assert secret not in compressor._previous_summary
    assert compressor._last_summary_error == "second pass failed"
    assert compressor._summary_failure_cooldown_until > 0
    assert turns == before
    assert [id(turn) for turn in turns] == member_ids


def test_later_pass_cancellation_propagates_and_rolls_back_tentative_state():
    compressor = _make()
    compressor._previous_summary = "normalized seed"
    compressor._last_summary_error = "pre-existing diagnostic"
    turns = _turns()
    before = deepcopy(turns)
    member_ids = [id(turn) for turn in turns]
    calls = 0

    def call_llm(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CancelledError
        return _response(_summary_body("tentative partial"))

    with (
        patch("agent.context_compressor.call_llm", side_effect=call_llm),
        pytest.raises(CancelledError),
    ):
        compressor._generate_summary(turns)

    assert calls == 2
    assert compressor._previous_summary == "normalized seed"
    assert compressor._last_summary_error == "pre-existing diagnostic"
    assert turns == before
    assert [id(turn) for turn in turns] == member_ids


def test_full_compress_cancellation_restores_rehydrated_handoff_state():
    def conversation(prefix: str) -> list[dict]:
        rows = [{"role": "system", "content": f"{prefix} system"}]
        for index in range(12):
            rows.append({
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"{prefix}-{index} " + ("x" * 900),
            })
        return rows

    def configure(compressor: ContextCompressor) -> None:
        compressor._tail_token_budget = 600
        compressor.last_prompt_tokens = 100_000

    producer = _make(limit=1_000)
    configure(producer)
    with patch(
        "agent.context_compressor.call_llm",
        side_effect=lambda **_kwargs: _response(_summary_body("seed handoff")),
    ):
        compacted = producer.compress(
            conversation("first"),
            current_tokens=100_000,
            force=True,
        )
    assert any(row.get(COMPRESSED_SUMMARY_METADATA_KEY) for row in compacted)

    restarted = _make(limit=1_000)
    configure(restarted)
    restarted._previous_summary = None
    restarted._summary_has_user_turn = None
    resumed = compacted + conversation("resumed")[1:]
    before = deepcopy(resumed)
    calls = 0

    def cancel_later_pass(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CancelledError
        return _response(_summary_body("tentative resumed state"))

    with (
        patch("agent.context_compressor.call_llm", side_effect=cancel_later_pass),
        pytest.raises(CancelledError),
    ):
        restarted.compress(
            resumed,
            current_tokens=100_000,
            force=True,
        )

    assert calls == 2
    assert resumed == before
    assert restarted._previous_summary is None
    assert restarted._summary_has_user_turn is None


def test_fitting_completed_tool_group_stays_together_at_a_pass_boundary():
    compressor = _make()
    turns = [
        {"role": "user", "content": "before-tools " + ("u" * 700)},
        {
            "role": "assistant",
            "content": "calling terminal",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "terminal", "arguments": '{"cmd":"pwd"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "/workspace\n" + ("t" * 300)},
        {"role": "assistant", "content": "tool completed"},
    ]
    prompts: list[str] = []

    def call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _response(_summary_body(f"pass-{len(prompts)}"))

    with patch("agent.context_compressor.call_llm", side_effect=call_llm):
        assert compressor._generate_summary(turns) is not None

    sources = [_source_block(prompt) for prompt in prompts]
    tool_sources = [source for source in sources if "terminal(" in source]
    assert len(tool_sources) == 1
    assert "[TOOL RESULT call-1]" in tool_sources[0]
    combined = "".join(sources)
    assert combined.count("terminal(") == 1
    assert combined.count("[TOOL RESULT call-1]") == 1
    assert combined.index("terminal(") < combined.index("[TOOL RESULT call-1]")


def test_oversized_assistant_row_fragments_reconstruct_provider_source_bytes():
    turn = {
        "role": "assistant",
        "content": (
            "before <think>discard scratch reasoning</think> after "
            + ("x" * 7_000)
        ),
    }

    baseline = _make(limit=10_000)
    baseline._summary_has_user_turn = False
    baseline_prompts: list[str] = []

    def baseline_call(**kwargs):
        baseline_prompts.append(kwargs["messages"][0]["content"])
        return _response(_summary_body("baseline", no_user=True))

    with patch("agent.context_compressor.call_llm", side_effect=baseline_call):
        assert baseline._generate_summary([turn]) is not None
    expected = _source_block(baseline_prompts[0])

    fragmented = _make(limit=600)
    fragmented._summary_has_user_turn = False
    fragment_prompts: list[str] = []

    def fragment_call(**kwargs):
        fragment_prompts.append(kwargs["messages"][0]["content"])
        return _response(_summary_body("fragmented", no_user=True))

    with patch("agent.context_compressor.call_llm", side_effect=fragment_call):
        assert fragmented._generate_summary([turn]) is not None

    blocks = [_source_block(prompt) for prompt in fragment_prompts]
    label = re.compile(r"^\[ASSISTANT FRAGMENT \d+/\d+\]: ")
    assert len(blocks) > 1
    assert all(label.match(block) for block in blocks)
    recovered = "".join(label.sub("", block, count=1) for block in blocks)
    assert recovered == expected
    assert all(len(block) <= fragmented._SUMMARY_INPUT_MAX_CHARS for block in blocks)


def test_non_edge_pruned_skill_survives_while_every_pass_stays_redacted():
    compressor = _make(limit=700)
    secret = "sk-proj-" + ("s" * 40)
    marker = (
        "[SKILL_PRUNED: content lost in compression; "
        "reload with skill_view(name='middle-skill')]"
    )
    turns = [
        {"role": "user", "content": f"first {secret} " + ("a" * 500)},
        {"role": "assistant", "content": f"middle {marker} " + ("b" * 500)},
        {"role": "user", "content": f"last {secret} " + ("c" * 500)},
    ]
    prompts: list[str] = []

    def call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _response(_summary_body(f"pass-{len(prompts)}") + f"\n{secret}")

    with patch("agent.context_compressor.call_llm", side_effect=call_llm):
        summary = compressor._generate_summary(turns)

    assert summary is not None
    sources = [_source_block(prompt) for prompt in prompts]
    marker_pass = next(index for index, source in enumerate(sources) if marker in source)
    assert 0 < marker_pass < len(sources) - 1
    assert all(secret not in prompt for prompt in prompts)
    assert secret not in summary
    assert marker in summary
    assert all(marker in prompt for prompt in prompts[marker_pass + 1:])


def test_restart_rehydrates_one_handoff_and_cross_session_guard_clears_it():
    critical = "RESTART_CRITICAL_MARKER"

    def conversation(prefix: str, *, include_critical: bool) -> list[dict]:
        rows = [{"role": "system", "content": f"{prefix} system"}]
        for index in range(12):
            marker = f" {critical}" if include_critical and index == 4 else ""
            rows.append({
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"{prefix}-{index}{marker} " + ("x" * 900),
            })
        return rows

    def configure(compressor: ContextCompressor) -> None:
        compressor._tail_token_budget = 600
        compressor.last_prompt_tokens = 100_000

    prompts: list[str] = []

    def call_llm(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        prompts.append(prompt)
        carried = f" {critical}" if critical in prompt else ""
        return _response(_summary_body(f"cycle-{len(prompts)}{carried}"))

    first = _make(limit=1_000)
    configure(first)
    with patch("agent.context_compressor.call_llm", side_effect=call_llm):
        compacted = first.compress(
            conversation("first", include_critical=True),
            current_tokens=100_000,
            force=True,
        )
    assert sum(bool(row.get(COMPRESSED_SUMMARY_METADATA_KEY)) for row in compacted) == 1
    assert critical in "\n".join(str(row.get("content") or "") for row in compacted)

    restarted = _make(limit=1_000)
    configure(restarted)
    resumed = compacted + conversation("resumed", include_critical=False)[1:]
    with patch("agent.context_compressor.call_llm", side_effect=call_llm):
        recompressed = restarted.compress(
            resumed,
            current_tokens=100_000,
            force=True,
        )
    joined = "\n".join(str(row.get("content") or "") for row in recompressed)
    assert sum(bool(row.get(COMPRESSED_SUMMARY_METADATA_KEY)) for row in recompressed) == 1
    assert joined.count(SUMMARY_PREFIX) == 1
    assert critical in joined

    unrelated_prompts: list[str] = []

    def unrelated_call(**kwargs):
        unrelated_prompts.append(kwargs["messages"][0]["content"])
        return _response(_summary_body("unrelated session"))

    with patch("agent.context_compressor.call_llm", side_effect=unrelated_call):
        unrelated_result = restarted.compress(
            conversation("unrelated", include_critical=False),
            current_tokens=100_000,
            force=True,
        )
    assert all(critical not in prompt for prompt in unrelated_prompts)
    assert critical not in "\n".join(
        str(row.get("content") or "") for row in unrelated_result
    )


def test_below_cap_path_keeps_single_request_result_state_and_wire_contract():
    compressor = _make(limit=10_000)
    compressor._active_compression_telemetry = {}
    compressor._previous_summary = "prior summary"
    compressor._last_summary_error = "old error"
    turns = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first response"},
    ]
    calls: list[dict] = []
    response_body = _summary_body("updated result")

    def call_llm(**kwargs):
        calls.append(kwargs)
        return _response(response_body)

    with patch("agent.context_compressor.call_llm", side_effect=call_llm):
        summary = compressor._generate_summary(turns)

    assert len(calls) == 1
    assert set(calls[0]) == {"task", "main_runtime", "messages"}
    assert "max_tokens" not in calls[0]
    prompt = calls[0]["messages"][0]["content"]
    assert _source_block(prompt) == "[USER]: first request\n\n[ASSISTANT]: first response"
    assert prompt.count("prior summary") == 1
    assert summary is not None and summary.startswith(SUMMARY_PREFIX)
    assert compressor._previous_summary and summary.endswith(compressor._previous_summary)
    assert compressor._last_summary_error is None
    assert "chunking" not in compressor._active_compression_telemetry
    assert "chunk_count" not in compressor._active_compression_telemetry
