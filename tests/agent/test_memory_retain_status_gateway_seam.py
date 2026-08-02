"""Behavioral RED coverage for gateway-generic retain lifecycle feedback.

The provider client is deterministic and local, but every Hermes lifecycle
edge in these tests is real: AIAgent construction, Hindsight's writer,
MemoryManager session-end handling, gateway cache eviction, turn setup,
SessionStore continuity/closure, and AIAgent._emit_status.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _RetainClient:
    """Deterministic stand-in for the external server at its API boundary."""

    def __init__(self, outcomes=None, gates=None):
        self._outcomes = list(outcomes or [None])
        self._gates = list(gates or [])
        self._condition = threading.Condition()
        self._calls = 0

    async def aretain_batch(self, **_kwargs):
        with self._condition:
            call_index = self._calls
            self._calls += 1
            self._condition.notify_all()
        if call_index < len(self._gates):
            assert await asyncio.to_thread(
                self._gates[call_index].wait, 5
            ), "retain gate timed out"
        outcome = self._outcomes[call_index]
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(ok=True)

    async def aclose(self):
        return None

    def wait_for_calls(self, count: int) -> None:
        deadline = time.monotonic() + 5
        with self._condition:
            while self._calls < count:
                remaining = deadline - time.monotonic()
                assert remaining > 0, f"only {self._calls} retain call(s) started"
                self._condition.wait(timeout=remaining)

    def release_all(self) -> None:
        for gate in self._gates:
            gate.set()


def _lifecycle_messages(statuses):
    return [message for kind, message in statuses if kind == "lifecycle"]


def _retain_state(message: str) -> str:
    text = message.lower()
    if "sending to memory" in text:
        return "started"
    if "accepted by the memory server for processing" in text:
        return "accepted"
    if "could not be saved to memory" in text:
        return "failed"
    if "saved to memory" in text:
        return "succeeded"
    return "other"


def _start_real_turn(agent) -> None:
    """Reach the real per-turn status point without making a model call."""
    from agent.turn_context import build_turn_context

    build_turn_context(
        agent=agent,
        user_message="next turn",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *_a, **_k: "SYSTEM",
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda value: value,
        summarize_user_message_for_log=lambda value: value,
        set_session_context=lambda _session_id: None,
        set_current_write_origin=lambda _origin: None,
        ra=lambda: SimpleNamespace(_set_interrupt=lambda *_a, **_k: None),
    )


@pytest.fixture
def agent_factory(tmp_path: Path, monkeypatch):
    """Construct real gateway agents with real Hindsight providers."""
    import agent.auxiliary_client
    import agent.model_metadata
    import hermes_cli.config
    import plugins.memory
    import plugins.memory.hindsight as hindsight
    import run_agent
    from plugins.memory.hindsight import HindsightMemoryProvider
    from run_agent import AIAgent

    hermes_home = tmp_path / "hermes-home"
    config_path = hermes_home / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True)
    agent_config = {"memory": {"provider": "hindsight"}, "agent": {}}

    providers = []
    agents = []
    clients = []

    def load_provider(_name):
        provider = HindsightMemoryProvider()
        providers.append(provider)
        return provider

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(hindsight, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(
        hindsight, "_check_api_supports_update_mode_append", lambda *_a, **_k: False
    )
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: agent_config)
    monkeypatch.setattr(
        hermes_cli.config, "load_config_readonly", lambda: agent_config
    )
    monkeypatch.setattr(plugins.memory, "load_memory_provider", load_provider)
    monkeypatch.setattr(
        agent.model_metadata, "get_model_context_length", lambda *_a, **_k: 204_800
    )
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda *_a, **_k: [])
    monkeypatch.setattr(
        run_agent, "check_toolset_requirements", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock())
    monkeypatch.setattr(
        agent.auxiliary_client, "set_runtime_main", lambda *_a, **_k: None
    )

    def make_agent(
        gateway_session_key: str,
        session_id: str,
        *,
        retain_async: bool = True,
        client: _RetainClient | None = None,
        session_db=None,
    ):
        config_path.write_text(
            json.dumps(
                {
                    "mode": "cloud",
                    "apiKey": "test-key",
                    "api_url": "https://memory.invalid",
                    "bank_id": "test-bank",
                    "auto_recall": False,
                    "retain_async": retain_async,
                }
            )
        )
        statuses = []
        before = len(providers)
        agent = AIAgent(
            api_key="test-model-key",
            base_url="https://model.invalid/v1",
            provider="openrouter",
            model="test/model",
            max_iterations=2,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id=session_id,
            platform="gateway-test",
            gateway_session_key=gateway_session_key,
            session_db=session_db,
            status_callback=lambda kind, message: statuses.append((kind, message)),
        )
        assert len(providers) == before + 1
        provider = providers[-1]
        client = client or _RetainClient()
        provider._client = client
        clients.append(client)
        agents.append(agent)
        return SimpleNamespace(agent=agent, provider=provider, statuses=statuses)

    yield make_agent

    for client in clients:
        client.release_all()
    for agent in agents:
        manager = getattr(agent, "_memory_manager", None)
        if manager is not None:
            manager.shutdown_all()


def _real_session_store(tmp_path: Path):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore

    store = SessionStore(tmp_path / "gateway-sessions", GatewayConfig())
    source = SessionSource(
        platform=Platform.RELAY,
        chat_id=f"shared-{tmp_path.name}",
        chat_type="dm",
        user_id="user",
    )
    entry = store.get_or_create_session(source)
    return store, source, entry


def test_provider_knowledge_wording_and_none_use_real_agent_status_seam(
    agent_factory,
):
    cases = (
        (True, None, "accepted"),
        (False, None, "succeeded"),
        (True, RuntimeError("server rejected retain"), "failed"),
    )

    for index, (retain_async, outcome, expected_state) in enumerate(cases):
        gateway_key = f"knowledge-{index}"
        session_id = f"knowledge-session-{index}"
        client = _RetainClient([outcome])
        producer = agent_factory(
            gateway_key,
            session_id,
            retain_async=retain_async,
            client=client,
        )

        producer.provider.sync_turn("remember this", "ack", session_id=session_id)
        producer.provider._retain_queue.join()

        started = _lifecycle_messages(producer.statuses)
        assert started == ["Sending to memory…"]
        assert [_retain_state(message) for message in started] == ["started"]
        assert all("saving" not in message.lower() for message in started)
        assert all("saved" not in message.lower() for message in started)

        consumer = agent_factory(gateway_key, session_id)
        _start_real_turn(consumer.agent)
        terminal = _lifecycle_messages(consumer.statuses)
        assert [_retain_state(message) for message in terminal] == [expected_state]
        if expected_state == "accepted":
            assert all("saved" not in message.lower() for message in terminal)
        if expected_state == "failed":
            assert all("accepted" not in message.lower() for message in terminal)

    empty = agent_factory("knowledge-none", "knowledge-none-session")
    _start_real_turn(empty.agent)
    assert _lifecycle_messages(empty.statuses) == []


def test_delayed_fifo_survives_real_cache_commit_and_fresh_reconstruction(
    tmp_path: Path, monkeypatch, agent_factory
):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    store, _source, entry = _real_session_store(tmp_path)
    store.config.default_reset_policy.mode = "idle"
    gate = threading.Event()
    client = _RetainClient([None, None], [gate, gate])
    producer = agent_factory(entry.session_key, entry.session_id, client=client)
    producer.provider.sync_turn("first", "ack", session_id=entry.session_id)
    producer.provider._retain_async = False
    producer.provider.sync_turn("second", "ack", session_id=entry.session_id)
    client.wait_for_calls(1)

    commit_seen = threading.Event()
    real_on_session_end = producer.agent._memory_manager.on_session_end

    def observe_real_session_end(messages):
        commit_seen.set()
        real_on_session_end(messages)

    monkeypatch.setattr(
        producer.agent._memory_manager, "on_session_end", observe_real_session_end
    )
    monkeypatch.setattr(gateway_run, "_AGENT_CACHE_MAX_SIZE", 1)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner.session_store = store
    producer.agent._session_messages = [{"role": "user", "content": "live"}]

    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (producer.agent, "old")
        runner._agent_cache["cache-pressure"] = (object(), "new")
        runner._enforce_agent_cache_cap()

    assert commit_seen.wait(timeout=5), "cache cap did not commit memory"
    deadline = time.monotonic() + 5
    while producer.agent._session_messages and time.monotonic() < deadline:
        time.sleep(0.01)
    assert producer.agent._session_messages == []

    isolated = agent_factory(entry.session_key, f"{entry.session_id}-other")
    _start_real_turn(isolated.agent)
    assert _lifecycle_messages(isolated.statuses) == []

    consumer = agent_factory(entry.session_key, entry.session_id)
    gate.set()
    producer.provider._retain_queue.join()
    _start_real_turn(consumer.agent)
    assert [_retain_state(message) for message in _lifecycle_messages(consumer.statuses)] == [
        "accepted",
        "succeeded",
    ]

    delivered = list(consumer.statuses)
    _start_real_turn(consumer.agent)
    assert consumer.statuses == delivered


@pytest.mark.parametrize(
    "transition",
    ("session_switch", "compression_advance"),
    ids=(
        "resume-branch-adoption-handoff-and-binding-funnel",
        "compression-lineage-cas-and-tip-switch-funnel",
    ),
)
def test_real_continuity_funnels_preserve_pretransition_writers_under_new_identity(
    tmp_path: Path, agent_factory, transition: str
):
    store, _source, entry = _real_session_store(tmp_path)
    source_id = entry.session_id
    destination_id = f"{source_id}-continuation"
    gate = threading.Event()
    client = _RetainClient([None, None], [gate, gate])
    producer = agent_factory(entry.session_key, source_id, client=client)
    producer.provider.sync_turn("first", "ack", session_id=source_id)
    producer.provider._retain_async = False
    producer.provider.sync_turn("second", "ack", session_id=source_id)
    client.wait_for_calls(1)

    if transition == "session_switch":
        store._db.create_session(destination_id, source="gateway-test")
        switched = store.switch_session(entry.session_key, destination_id)
    else:
        switched = store.advance_compression_session(
            entry.session_key, source_id, destination_id
        )
    assert switched is not None and switched.session_id == destination_id

    old_coordinates = agent_factory(entry.session_key, source_id)
    gate.set()
    producer.provider._retain_queue.join()
    _start_real_turn(old_coordinates.agent)
    assert _lifecycle_messages(old_coordinates.statuses) == []

    unrelated = agent_factory(entry.session_key, f"{destination_id}-other")
    _start_real_turn(unrelated.agent)
    assert _lifecycle_messages(unrelated.statuses) == []

    consumer = agent_factory(entry.session_key, destination_id)
    _start_real_turn(consumer.agent)
    assert [_retain_state(message) for message in _lifecycle_messages(consumer.statuses)] == [
        "accepted",
        "succeeded",
    ]
    delivered = list(consumer.statuses)
    _start_real_turn(consumer.agent)
    assert consumer.statuses == delivered


def test_compression_child_adoption_preserves_pretransition_writer(
    tmp_path: Path, agent_factory
):
    from agent.conversation_compression import recover_rotated_compression_session
    from hermes_state import SessionDB

    gateway_key = "compression-adoption"
    parent_id = "compression-parent"
    child_id = "compression-child"
    db = SessionDB(db_path=tmp_path / "compression-state.db")
    db.create_session(parent_id, source="gateway-test")
    db.end_session(parent_id, "compression")
    db.create_session(child_id, source="gateway-test", parent_session_id=parent_id)
    db.replace_messages(
        child_id,
        [
            {"role": "user", "content": "compacted context"},
            {"role": "assistant", "content": "compacted tail"},
        ],
    )

    gate = threading.Event()
    client = _RetainClient([None, None], [gate, gate])
    producer = agent_factory(
        gateway_key, parent_id, client=client, session_db=db
    )
    producer.provider.sync_turn("before compression", "ack", session_id=parent_id)
    client.wait_for_calls(1)

    recovered = recover_rotated_compression_session(producer.agent)
    assert recovered and producer.agent.session_id == child_id
    old_coordinates = agent_factory(gateway_key, parent_id)
    gate.set()
    producer.provider._retain_queue.join()

    _start_real_turn(old_coordinates.agent)
    assert _lifecycle_messages(old_coordinates.statuses) == []
    consumer = agent_factory(gateway_key, child_id)
    _start_real_turn(consumer.agent)
    assert [_retain_state(message) for message in _lifecycle_messages(consumer.statuses)] == [
        "accepted",
        "accepted",
    ]


def test_true_closure_fences_late_writer_before_blocked_durable_finalization(
    tmp_path: Path, monkeypatch, agent_factory
):
    store, _source, entry = _real_session_store(tmp_path)
    retain_gate = threading.Event()
    client = _RetainClient([None, None, None], [retain_gate])
    producer = agent_factory(entry.session_key, entry.session_id, client=client)
    producer.provider.sync_turn("old conversation", "ack", session_id=entry.session_id)
    client.wait_for_calls(1)
    old_started = _lifecycle_messages(producer.statuses)

    finalization_entered = threading.Event()
    finalization_release = threading.Event()
    real_promote = store._db.promote_to_session_reset

    def blocked_promote(session_id, reason):
        finalization_entered.set()
        assert finalization_release.wait(timeout=15), "finalization gate timed out"
        return real_promote(session_id, reason)

    monkeypatch.setattr(store._db, "promote_to_session_reset", blocked_promote)
    reset_result = []
    reset_thread = threading.Thread(
        target=lambda: reset_result.append(store.reset_session(entry.session_key)),
        name="blocked-session-reset",
    )
    reset_thread.start()
    try:
        assert finalization_entered.wait(timeout=5)
        successor_entry = store._entries[entry.session_key]
        retain_gate.set()
        producer.provider._retain_queue.join()

        old_coordinates = agent_factory(entry.session_key, entry.session_id)
        _start_real_turn(old_coordinates.agent)
        assert _lifecycle_messages(old_coordinates.statuses) == []

        # CLI /new keeps this AIAgent alive and changes its identity before
        # notifying the existing MemoryManager about the true closure.
        producer.agent.session_id = successor_entry.session_id
        before_successor = list(producer.statuses)
        _start_real_turn(producer.agent)
        assert producer.statuses == before_successor
        assert reset_thread.is_alive(), "durable finalization was not blocked"
    finally:
        finalization_release.set()
        reset_thread.join(timeout=5)
    assert not reset_thread.is_alive()
    assert reset_result and reset_result[0].session_id == successor_entry.session_id

    producer.agent._memory_manager.on_session_switch(
        successor_entry.session_id,
        parent_session_id=entry.session_id,
        reset=True,
        reason="new_session",
    )
    producer.provider._retain_queue.join()
    _start_real_turn(producer.agent)

    successor_retain_start = len(producer.statuses)
    producer.provider.sync_turn(
        "successor turn",
        "ack",
        session_id=successor_entry.session_id,
    )
    producer.provider._retain_queue.join()
    _start_real_turn(producer.agent)
    successor_lifecycle = _lifecycle_messages(
        producer.statuses[successor_retain_start:]
    )
    assert [
        message
        for message in successor_lifecycle
        if _retain_state(message) in {"accepted", "succeeded", "failed"}
    ] == ["Conversation accepted by the memory server for processing."]
    assert successor_lifecycle == [
        "Sending to memory…",
        "Conversation accepted by the memory server for processing.",
    ]
    assert old_started == ["Sending to memory…"]
    assert [_retain_state(message) for message in old_started] == ["started"]

    delivered = list(producer.statuses)
    _start_real_turn(producer.agent)
    assert producer.statuses == delivered


def test_collision_converges_fifo_and_true_close_fences_both_writer_families(
    tmp_path: Path, agent_factory
):
    for close_after_merge in (False, True):
        case_dir = tmp_path / ("close" if close_after_merge else "drain")
        store, _source, entry = _real_session_store(case_dir)
        destination_id = f"{entry.session_id}-destination"
        store._db.create_session(destination_id, source="gateway-test")

        source_late = threading.Event()
        destination_late = threading.Event()
        source_initial = threading.Event()
        source_initial.set()
        destination_initial = threading.Event()
        destination_initial.set()
        source_client = _RetainClient(
            [None, RuntimeError("source late failure")],
            [source_initial, source_late],
        )
        destination_client = _RetainClient(
            [None, None], [destination_initial, destination_late]
        )

        source = agent_factory(
            entry.session_key, entry.session_id, client=source_client
        )
        destination = agent_factory(
            entry.session_key,
            destination_id,
            retain_async=False,
            client=destination_client,
        )
        source.provider.sync_turn("source queued", "ack", session_id=entry.session_id)
        destination.provider.sync_turn(
            "destination queued", "ack", session_id=destination_id
        )
        source.provider._retain_queue.join()
        destination.provider._retain_queue.join()

        source.provider.sync_turn("source late", "ack", session_id=entry.session_id)
        destination.provider._retain_async = True
        destination.provider.sync_turn(
            "destination late", "ack", session_id=destination_id
        )
        source_client.wait_for_calls(2)
        destination_client.wait_for_calls(2)

        switched = store.switch_session(entry.session_key, destination_id)
        assert switched is not None

        if close_after_merge:
            successor = store.reset_session(entry.session_key)
            assert successor is not None
            source_late.set()
            destination_late.set()
            source.provider._retain_queue.join()
            destination.provider._retain_queue.join()

            closed_destination = agent_factory(entry.session_key, destination_id)
            fresh_successor = agent_factory(entry.session_key, successor.session_id)
            _start_real_turn(closed_destination.agent)
            _start_real_turn(fresh_successor.agent)
            assert _lifecycle_messages(closed_destination.statuses) == []
            assert _lifecycle_messages(fresh_successor.statuses) == []
            continue

        source_late.set()
        source.provider._retain_queue.join()
        destination_late.set()
        destination.provider._retain_queue.join()

        old_coordinates = agent_factory(entry.session_key, entry.session_id)
        _start_real_turn(old_coordinates.agent)
        assert _lifecycle_messages(old_coordinates.statuses) == []

        consumer = agent_factory(entry.session_key, destination_id)
        _start_real_turn(consumer.agent)
        assert [
            _retain_state(message) for message in _lifecycle_messages(consumer.statuses)
        ] == ["accepted", "succeeded", "failed", "accepted"]
        delivered = list(consumer.statuses)
        _start_real_turn(consumer.agent)
        assert consumer.statuses == delivered
