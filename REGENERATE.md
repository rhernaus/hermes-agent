# Regenerating the Hermes integration

This is a manual, fail-closed procedure. It creates no permission to fetch,
push, tag, build, publish, or deploy. Obtain the applicable authority before
each external effect.

## Phase 1 stop

The Phase 1 manifest is a skeleton and cannot regenerate an integration.
Every entry marked `materialization = "pending-phase-2"` is incomplete.
Phase 2 must replace every pending value with reviewed identities, record the
complete sorted `changed_paths`, create and commit the immutable external patch
artifact, and then remove the pending marker. Until then, stop: there is no
patch file, fidelity proof, owned follow-up commit, or integration result.

Never interpret a pending sentinel or an empty Phase 1 `changed_paths` array as
a hash, tree, file name, finalized path inventory, or successful verification.

## Preconditions

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

## Procedure

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

Any missing object, pending value, hash mismatch, fidelity mismatch, changed
path mismatch, conflict, or failed check stops the run. There is no tolerance
for Git-version-dependent regeneration and no local conflict resolution inside
an external entry.

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
