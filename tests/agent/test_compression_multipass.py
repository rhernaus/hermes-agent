"""Whole-turn, transactional context-compression pass coverage."""

from types import SimpleNamespace
from unittest.mock import patch

import agent.context_compressor as cc
import pytest
from agent.context_compressor import ContextCompressor


def _make(limit: int = 1_000) -> ContextCompressor:
    with patch.object(cc, "get_model_context_length", return_value=128_000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            quiet_mode=True,
        )
    compressor._SUMMARY_INPUT_MAX_CHARS = limit
    return compressor


def _turns(count: int = 6) -> list[dict]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"turn-{index}-" + ("x" * 360),
        }
        for index in range(count)
    ]


def test_splitter_keeps_regular_turns_intact_and_in_order():
    compressor = _make()
    turns = _turns()

    chunks = compressor._split_turns_for_summary(turns)

    assert len(chunks) > 1
    assert [turn for chunk in chunks for turn in chunk] == turns
    assert all(
        len(compressor._serialize_for_summary(chunk))
        <= compressor._SUMMARY_INPUT_MAX_CHARS
        for chunk in chunks
    )


def test_splitter_preserves_oversized_serialized_turn_without_user_attribution():
    compressor = _make()
    turn = {
        "role": "assistant",
        "content": "completed calls",
        "tool_calls": [
            {
                "id": f"call-{index}",
                "function": {
                    "name": "terminal",
                    "arguments": "x" * 900,
                },
            }
            for index in range(12)
        ],
    }
    rendered = compressor._serialize_for_summary([turn])
    assert len(rendered) > compressor._SUMMARY_INPUT_MAX_CHARS

    chunks = compressor._split_turns_for_summary([turn])
    fragments = [fragment for chunk in chunks for fragment in chunk]

    assert len(fragments) > 1
    assert all(fragment["role"] == "assistant" for fragment in fragments)
    recovered = "".join(
        fragment["content"].split("source material only]\n", 1)[1]
        for fragment in fragments
    )
    assert recovered == rendered
    assert all(
        len(compressor._serialize_for_summary(chunk))
        <= compressor._SUMMARY_INPUT_MAX_CHARS
        for chunk in chunks
    )


def test_multipass_commits_only_the_final_summary():
    compressor = _make()
    compressor._previous_summary = "seed"
    compressor._active_compression_telemetry = {}
    calls: list[list[dict]] = []

    def generate(chunk, **_kwargs):
        calls.append(chunk)
        compressor._previous_summary = f"pass-{len(calls)}"
        return f"summary-{len(calls)}"

    with patch.object(
        compressor,
        "_generate_summary",
        side_effect=generate,
    ):
        summary = compressor._generate_summary_in_passes(_turns())

    assert len(calls) > 1
    assert [turn for chunk in calls for turn in chunk] == _turns()
    assert summary == f"summary-{len(calls)}"
    assert compressor._previous_summary == f"pass-{len(calls)}"
    assert compressor._active_compression_telemetry["chunking"] is True
    assert compressor._active_compression_telemetry["chunk_count"] == len(calls)


def test_multipass_rolls_back_partial_summary_but_keeps_failure_state():
    compressor = _make()
    secret = "sk-proj-" + ("a" * 40)
    compressor._previous_summary = f"seed {secret}"
    calls = 0

    def generate(_chunk, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            compressor._previous_summary = "partial"
            return "summary-1"
        compressor._last_summary_error = "second pass failed"
        return None

    with patch.object(
        compressor,
        "_generate_summary",
        side_effect=generate,
    ):
        assert compressor._generate_summary_in_passes(_turns()) is None
    assert calls == 2
    assert compressor._previous_summary.startswith("seed ")
    assert secret not in compressor._previous_summary
    assert compressor._last_summary_error == "second pass failed"


def test_multipass_rolls_back_partial_summary_on_interruption():
    compressor = _make()
    compressor._previous_summary = "seed"
    calls = 0

    def generate(_chunk, **_kwargs):
        nonlocal calls
        calls += 1
        compressor._previous_summary = "partial"
        if calls == 2:
            raise KeyboardInterrupt
        return "summary-1"

    with (
        patch.object(
            compressor,
            "_generate_summary",
            side_effect=generate,
        ),
        patch.object(cc, "_redact_compaction_text", side_effect=lambda text: text),
        pytest.raises(KeyboardInterrupt),
    ):
        compressor._generate_summary_in_passes(_turns())

    assert calls == 2
    assert compressor._previous_summary == "seed"


def test_single_pass_keeps_existing_summary_path():
    compressor = _make()
    calls = []

    def generate(chunk, **_kwargs):
        calls.append(chunk)
        return "summary"

    turns = _turns(2)

    with patch.object(
        compressor,
        "_generate_summary",
        side_effect=generate,
    ):
        assert compressor._generate_summary_in_passes(turns) == "summary"
    assert calls == [turns]


def test_real_summary_path_sends_every_turn_without_aggregate_omission():
    compressor = _make()
    compressor._summary_has_user_turn = True
    prompts: list[str] = []

    def call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "## Active Task\nContinue.\n\n"
                            "## Goal\nPreserve the historical source."
                        )
                    )
                )
            ]
        )

    with (
        patch.object(
            ContextCompressor,
            "_SUMMARY_INPUT_MAX_CHARS",
            compressor._SUMMARY_INPUT_MAX_CHARS,
        ),
        patch.object(cc, "call_llm", side_effect=call_llm),
    ):
        bounded_once = compressor._bound_summary_input(
            compressor._serialize_for_summary(_turns())
        )
        assert "summary input truncated" in bounded_once
        assert compressor._generate_summary_in_passes(_turns()) is not None

    assert len(prompts) > 1
    current_source_sections: list[str] = []
    for prompt in prompts:
        if "NEW TURNS TO INCORPORATE:" in prompt:
            source = prompt.split("NEW TURNS TO INCORPORATE:", 1)[1]
            source = source.split("\n\nUpdate the summary", 1)[0]
        else:
            source = prompt.split("TURNS TO SUMMARIZE:", 1)[1]
            source = source.split("\n\nUse this exact structure:", 1)[0]
        current_source_sections.append(source)

    current_source = "\n".join(current_source_sections)
    assert "summary input truncated" not in current_source
    for index in range(6):
        assert current_source.count(f"turn-{index}-") == 1
    assert all(
        "PREVIOUS SUMMARY:" in prompt
        for prompt in prompts[1:]
    )
