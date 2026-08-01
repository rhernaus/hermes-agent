# Regenerating the Hermes integration

This is a manual, fail-closed procedure. It creates no permission to fetch,
push, tag, build, publish, or deploy. Obtain the applicable authority before
each external effect.

## Phase 2 stop

Entry 1 is materialized in Phase 2: its pending marker is removed and it now
carries reviewed identities, the complete sorted `changed_paths`, and the
immutable external patch artifact. Entry 2 remains explicitly incomplete and is
marked `materialization = "pending-phase-3"`. Phase 3 must replace every
remaining pending value with reviewed identities, record the complete sorted
`changed_paths`, and then remove that marker.

Complete two-entry integration regeneration therefore remains fail-closed until
Phase 3: there is no owned follow-up commit and no combined integration result.

Phase 2 may and must independently regenerate the entry-1 tree twice from the
stored verified patch bytes, proving deterministic tree equality between the two
results. Use the Phase-2 entry-1-only procedure below; it is executable against
the current manifest state. That entry-1 proof is not an integration result and
does not satisfy any Phase-3 gate that requires the finalized two-entry
manifest.

Never interpret a pending sentinel or an empty `changed_paths` array as a hash,
tree, file name, finalized path inventory, or successful verification.

## Phase 2 procedure — entry 1 only

This procedure applies while entry 2 is still pending. It proves a deterministic
entry-1 tree and nothing more.

### Phase 2 preconditions

- Work from a clean clone with the reviewed `stack` branch and entry 1's
  immutable patch artifact present at its recorded `patch_file`.
- Require a schema-2 manifest in exactly the Phase-2 state: entry 1 carries no
  `materialization` marker and has full-length object IDs for `head_sha`,
  `head_tree` and `patch_base`, an existing `patch_file`, a 64-hex
  `patch_sha256`, and a non-empty sorted `changed_paths`; entry 2 has
  `materialization`, `retention_tag`, `head_sha`, `head_tree` and `patch_base`
  all exactly `pending-phase-3`, and an empty `changed_paths`. Any other state
  stops the run: a remaining `pending-phase-2` value, a pending or malformed
  entry-1 field, or a finalized entry 2. Once entry 2 is finalized, use the
  Phase-3 procedure instead.
- Treat the upstream pin, the entry order, and the stored bytes as reviewed
  inputs. Do not refresh, reorder, or repair them during a regeneration.
- Use no synthetic merge ref. PR 74379 is fetched from the base repository at
  `refs/pull/74379/head`; the contributor repository and branch are
  informational only.

### Phase 2 steps

1. Read entry 1's `patch_file` bytes and compute their SHA-256. Assert exact
   equality with `patch_sha256`. Do not generate, normalize, reformat, or
   otherwise replace the stored bytes. A mismatch is a hard stop.
2. Fetch `upstream.sha` and entry 1's `patch_base` and `head_sha` by their full
   SHAs, using entry 1's write-once retention tag if `fetch_ref` no longer
   resolves. Assert the fetched head equals the recorded full `head_sha`. A
   mismatch is a hard stop and requires a reviewed manifest revision, never a
   silent re-pin.
3. Prove fidelity on entry 1's original base. In a clean temporary worktree at
   its `patch_base`, apply the verified bytes and assert that the resulting Git
   tree equals `head_tree`. A mismatch is a hard stop.
4. In a first clean temporary worktree detached at exactly `upstream.sha`, apply
   only entry 1's verified bytes using three-way application. Do not replay or
   regenerate external commits in place of the stored patch.
5. Compute the paths changed from `upstream.sha` to the resulting tree. Assert
   exact equality with entry 1's recorded `changed_paths`. Extra or missing
   paths stop the run.
6. Repeat steps 4 and 5 in a second, independent clean temporary worktree.
   Assert that both resulting trees are exactly equal. Tree identity is the
   invariant; commit IDs may differ because of committer metadata.
7. Record the proven entry-1 tree, and any temporary local commit identity, as
   Phase-2 evidence only.

### Phase 2 limits

This procedure must not fetch or materialize entry 2, or read its pending
sentinels as values; must not run the Phase-3 combined, genericity, or
source-build gates; must not claim a combined integration result; must not
create an integration branch, an integration tag, or any required commit; and
must not push anything. Its only output is a deterministic entry-1 tree proof.

## Phase 3 preconditions — finalized two-entry manifest

- Work from a clean clone with the reviewed `stack` branch and its committed
  immutable patch artifacts.
- Require a finalized schema-2 manifest: no pending marker or sentinel may
  remain.
- Require full object IDs and a complete, sorted path inventory for every
  entry.
- Treat the upstream pin and entry order as reviewed inputs. Do not refresh,
  reorder, or repair them during a regeneration.
- Use no synthetic merge ref. PR 74379 is fetched from the base repository at
  `refs/pull/74379/head`; the contributor repository and branch are
  informational only.

## Phase 3 procedure — finalized two-entry manifest

1. Hash the exact bytes of `stack.toml` with SHA-256. Record the first eight
   hexadecimal characters as `manifest-sha8`; do not canonicalize or rewrite
   the file before hashing.
2. Fetch `upstream.sha` by its full SHA. For every entry, fetch its recorded
   repository/ref if that ref still resolves, otherwise fetch the owned
   write-once retention tag. Assert that the fetched object equals the recorded
   full `head_sha`. A mismatch is a hard stop and requires a reviewed manifest
   revision, never a silent re-pin.
3. Under separate write authority, create and push any newly required retention
   tag. Retention tags are write-once and contain the full retained SHA; never
   move, replace, or force-push one.
4. For each external entry, read the committed `patch_file` bytes and compute
   their SHA-256. Assert exact equality with `patch_sha256`. Do not generate,
   normalize, reformat, or otherwise replace the stored bytes. A mismatch is a
   hard stop.
5. Prove fidelity on the entry's original base. In a clean temporary worktree
   at `patch_base`, apply the verified patch bytes and assert that the resulting
   Git tree equals `head_tree`. This proof must cover every blob, including the
   reviewed merge resolution. A mismatch is a hard stop.
6. Detach at the exact `upstream.sha`. Apply those same verified bytes in
   ascending `index` order using three-way application. Materialize the owned
   entry from its own write-once retention tag and prove its tree at its own
   `patch_base`. Do not replay or regenerate external commits in place of the
   stored patch.
7. Compute the paths changed from `upstream.sha` to the resulting tree. Assert
   exact equality with the union of every entry's recorded `changed_paths`.
   Extra or missing paths stop the run.
8. Run the required baseline, focused, combined, genericity, and source-build
   gates documented in `BUILD.md` and `FOLLOWUP.md`. A clean apply is not
   compatibility evidence.
9. Independently repeat regeneration from the same manifest and stored bytes.
   The resulting tree must be identical. Commit IDs may differ because of
   committer metadata; tree identity is the invariant.
10. Record the integration commit SHA and tree. Only after every preceding gate
    passes, and only under separate tag/push authority, create the one
    write-once integration tag and push it:
    `int/<upstream-sha>-<manifest-sha8>`. Never create an integration branch or
    move an existing integration tag.

## Stop rules and revisions

Any missing object, hash mismatch, fidelity mismatch, changed path mismatch,
conflict, or failed check stops either procedure. A pending value stops the
Phase-3 procedure; in Phase 2 only the states named in the Phase-2 preconditions
are admissible, and every other pending or malformed value stops that run. There
is no tolerance for Git-version-dependent regeneration and no local conflict
resolution inside an external entry.

The revision rules below apply to both procedures.

After a conflict or mismatch, the only allowed next specification is one of:

1. drop the entry after reviewed evidence that upstream superseded it;
2. pin a reviewed newer head produced by the original author, as a new revision
   with a new immutable artifact and fidelity proof; or
3. with separate user authority, add an owned replacement entry with its own
   base, commits, identity, and review.

Never silently reclassify external work as owned. When an upstream pin or PR
head changes, make a reviewed manifest revision and a new immutable artifact;
retain the old artifact and its write-once tags unchanged. No upstream event
automatically regenerates or promotes anything.
