# Owned follow-up: generic gateway retain feedback

This is the Phase 1 specification for a later, independently removable owned
entry on top of PR 74379. It is not source or test implementation and makes no
claim that the behavior is built, working, or deployed.

## Scope and shared seam

Retain lifecycle feedback must use only the existing shared Hermes status
path: provider callback to `AIAgent._emit_status`, then the gateway's shared
status preparation and delivery abstraction. Add no connector API call,
adapter-specific import or branch, capability probe, or channel-specific
fallback. No file under `gateway/platforms/` or `plugins/platforms/` may be
changed. Connectors that already consume the shared status path and do not
filter these events can receive them; identical rendering by every connector
is not promised.

Mattermost is only a later deployed acceptance surface. It is not the design
target and requires no Mattermost-specific product code or adapter test matrix.

## B1: retain started

Move the provider `status_callback` injection out of the CLI-only branch in
`agent/agent_init.py` so every platform receives the existing
`agent._emit_status` callback. The callback already no-ops when a surface has
no status sink. Leave `warning_callback` unchanged.

The dispatch-time line means only that retain has started: render **sending to
memory…**, not saved or succeeded. It originates on a background worker and
can race the gateway's run-current guard, so emission at the shared seam is
deterministic but connector delivery is best-effort and non-gating. Do not add
a post-response transport to strengthen it.

## B2: terminal outcome on a later turn

Provider-local storage is insufficient because gateway cache eviction commonly
reconstructs both agent and provider between turns. The smallest accepted owner
is a lock-protected, process-local registry in `agent/memory_manager.py`.

For gateway traffic, lookup identity is
`(gateway_session_key, session_id)`. Outside gateways the first element is
empty, reducing identity to the session ID. Coordinates are lookup keys, not
ownership tokens: one live owner holds the current coordinates and a FIFO of
terminal outcomes, and the registry exposes only its current identity.

One module-level `threading.Lock` guards every operation below. Perform no I/O
while holding it. Lock order is always SessionStore then retain owner, never the
reverse.

- **Bind:** only live construction in `agent/agent_init.py` may bind an
  identity. Reconstructing the same live identity adopts its existing owner.
- **Admit:** synchronously when a retain is queued, pass the current session ID
  to the admission callback and return an opaque owner-backed recorder. A
  refused admission returns a recorder that always fails closed.
- **Record:** a delayed writer records only through its admitted token and only
  if that token resolves to a live registered owner. Recording can never bind,
  recreate, or reopen an identity. Append; never overwrite.
- **Drain:** atomically detach the complete FIFO once for a live identity, then
  emit every detached outcome in order through `AIAgent._emit_status`. A
  repeated drain is empty.
- **Transfer:** for continuity, atomically re-key the same owner. Do not copy or
  replace the FIFO. Already-admitted tokens remain live and old coordinates
  cease to resolve.
- **Converge:** if source and destination both own state, retain one registered
  root. Merge queues deterministically as source FIFO then destination FIFO,
  preserving each internal order. Both token families resolve to the root;
  later records append in lock-acquisition order. Exactly one registry entry
  remains.
- **Close:** for true closure, atomically close the registered root, clear the
  complete possibly merged FIFO, remove the registry entry, and fence every
  token family synchronously. Late records fail closed.

Displaced-token forwarding is allowed only inside previously admitted tokens,
resolved and path-compressed under the same lock. Roots never retain displaced
tokens, so this forms a forest rather than registry history.

Ordinary reconstruction is not a boundary. Generic
`MemoryManager.on_session_end`, `commit_memory_session`, cache-cap or soft
agent eviction, `/model`, configuration changes, and idle rebuilds must not
clear pending outcomes.

## Closure, continuity, and ordering

Every durable session-ID transition is classified; there is no default:

| Class | Paths | Required effect |
| --- | --- | --- |
| True closure | Gateway and CLI `/new`; compression-exhaustion fresh reset; daily, idle, and resume-expired fresh reset; true session expiry; process or gateway shutdown | Close and fence synchronously, clear the FIFO, and remove the owner |
| Continuity | `/resume`; `/branch`; CLI adoption and CLI-to-gateway handoff; compression-lineage publication and CAS/advance; compression-tip switching including manual compression; binding healing | Re-key the same owner, preserving pending FIFO and every admitted token; converge on collision |

A true closure must fence before any slow or blocking durable finalization. Put
the gateway `/new` close in shared `SessionStore.reset_session`, not only in an
individual command. Preserve the existing lock-less cache-eviction fallback.
A continuity transfer must finish before the new durable session ID is assigned
or published. Attach operations at the authoritative funnels, including
`SessionStore.reset_session`, `_get_or_create_session_impl`, `switch_session`,
`advance_compression_session`/`_heal_compression_tip_locked`, compression
lineage publication and stale-child adoption, CLI-to-gateway handoff, binding
healing, CLI session transitions, true expiry, and shutdown.

## Provider knowledge and wording

Render no state before the provider knows it:

| Provider observation | State | Required meaning |
| --- | --- | --- |
| Job dispatched to the writer | started | sending to memory; no persistence claim |
| Client returned with `retain_async = true` | accepted | accepted by the memory server for processing; never “saved” |
| Client returned with `retain_async = false` | succeeded | saved to memory; synchronous success is confirmed |
| Client call raised | failed | could not be saved; never success wording |
| No result | none | render nothing |

Exactly-once applies at the shared `_emit_status` seam. Delivery after that seam
remains each connector's existing best-effort behavior.

## Deliberate exclusions and accepted limit

Add no persistent or durable storage, service, broker, database, new transport,
new configuration key, TTL, size cap, reaper, alias, tombstone, closed set, or
registry history. Process exit discards all pending state. A true session end
abandons pending state; if closure or process restart occurs before a later turn
drains an outcome, that outcome is never delivered. This accepted limit must
not be worked around in this entry.

## Planned file surface

Product changes are confined to:

- `agent/memory_manager.py`: owner registry and bind/admit/record/drain/
  transfer/converge/close operations;
- `agent/agent_init.py`: shared B1 callback injection and real B2 admission;
- `plugins/memory/hindsight/__init__.py`: admission at enqueue, terminal record
  at the provider boundary, and accurate started wording;
- `agent/turn_context.py`: complete FIFO drain at the existing indicator point;
- `gateway/session.py`: closure and continuity funnels with required ordering;
- `agent/conversation_compression.py`: lineage and stale-child transfers;
- `gateway/run.py`: handoff, binding-heal, true-expiry, and shutdown handling;
- `cli.py` and `hermes_cli/cli_commands_mixin.py`: CLI close and continuity
  transitions; and
- `plugins/memory/hindsight/README.md`: accurate started wording.

The behavior seam belongs in
`tests/agent/test_memory_retain_status_gateway_seam.py`, with only directly
affected existing provider, handoff, gateway, CLI, and compression tests
adjusted as necessary. Do not add a product module, dependency, helper-only
recorder, speculative test matrix, new adapter test, or platform product file.
The existing handoff coverage in `tests/gateway/test_telegram_topic_mode.py`
may cover its real upstream flow; this does not authorize adapter code.

## Direct executable T2/T5 gate

After the follow-up exists, run exactly in this order:

    uv sync --frozen --extra dev --extra hindsight
    scripts/run_tests.sh tests/agent/test_memory_recall_indicator.py tests/agent/test_memory_provider_unavailable_warning.py tests/agent/test_memory_session_switch.py tests/agent/test_turn_context.py tests/gateway/test_35994_reset_button_deadlock.py tests/gateway/test_agent_cache.py tests/gateway/test_session_boundary_hooks.py tests/gateway/test_session_model_reset.py tests/gateway/test_shutdown_cache_cleanup.py tests/cli/test_cli_new_session.py tests/cli/test_session_boundary_hooks.py tests/hermes_cli/test_memory_status_env_hint.py tests/plugins/memory/test_hindsight_provider.py tests/plugins/memory/test_hindsight_local_runtime_hint.py tests/plugins/memory/test_hindsight_templates.py tests/gateway/test_session.py tests/gateway/test_session_store_runtime_stale_guard.py tests/gateway/test_resume_command.py tests/gateway/test_telegram_topic_mode.py tests/gateway/test_async_delegation_session_binding.py tests/gateway/test_35809_auto_reset_clean_context.py tests/cli/test_cli_resume_command.py tests/cli/test_branch_command.py tests/cli/test_manual_compress.py tests/agent/test_compression_concurrent_fork.py tests/agent/test_compression_rotation_state.py tests/agent/test_memory_retain_status_gateway_seam.py

Require non-zero collected tests. The seam test must use real cache-cap
`commit_memory_session`, provider-bearing `MemoryManager.on_session_end`, real
`agent_init` admission, a real `SessionStore`, the real closure and continuity
funnels, and real `AIAgent._emit_status`. It must directly prove:

- started, accepted, succeeded, and failed knowledge-boundary wording;
- two delayed outcomes surviving soft eviction and fresh reconstruction;
- one ordered complete-FIFO drain and an empty repeated drain;
- isolation of distinct session IDs sharing a gateway key;
- synchronous late-writer rejection before blocked finalization at closure;
- preservation under every continuity class, empty old coordinates, and one
  drain under the new identity;
- collision convergence with both callback families live, source-then-
  destination FIFO order, and one remaining registry entry; and
- true closure after convergence fencing both families and abandoning the
  merged FIFO.

Entry 2's final `changed_paths` must contain no platform product file, and its
diff must contain no connector name, adapter import, or per-platform branch.
Only after implementation, tests, independent review, build/promotion approval,
and deployment may a real Mattermost smoke be used as later evidence that
recall is visible and the next turn renders the outcome. The started line is
recorded if observed but is non-gating; never induce retain failure against
production.
