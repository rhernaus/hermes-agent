# Source-build contract

This document specifies a later local build. Phase 1 does not build an image,
run source tests, publish anything, or authorize any build-host effect.

## Inputs and image identity

- Build the final reviewed integration commit with the unmodified root
  `Dockerfile` from upstream Hermes. Do not patch or replace that Dockerfile.
- Pass the full integration commit SHA as
  `HERMES_GIT_SHA=<integration-commit-sha>` and verify the resulting image's
  `/opt/hermes/.hermes_build_sha` records the same value.
- Name the local build
  `localhost/hermes-base-rhernaus:int-<upstream-sha12>-<manifest-sha8>`.
  The name is only a convenience; the immutable identity used by consumers is
  the resulting image manifest digest.
- Keep the image local. There is no registry target, credential, publication,
  or push in this stack contract.

## Required gates before a build

No build or promotion may start from a clean apply alone.

1. On the bare upstream pin, run exactly, in order:

       uv sync --locked --python 3.11 --extra all --extra dev --extra anthropic --extra mistral --extra fal --extra modal --extra daytona --extra hindsight --extra parallel-web
       scripts/run_tests.sh

   Record a non-vacuous result. This is the baseline for distinguishing an
   upstream failure from a stack regression.
2. Run the directly affected T2/T5 gate exactly as specified in `FOLLOWUP.md`,
   including its pinned setup command and non-zero collection check.
3. On the combined integration commit, repeat the same two commands from step
   1, in the same order and in full. This is the authoritative combined result.
4. Prove external-patch hash and original-base tree fidelity, same-byte ordered
   application, exact changed-path union, independent regeneration tree
   equality, and the genericity checks in `REGENERATE.md` and `FOLLOWUP.md`.

Any unresolved failure blocks the build. A later pass does not erase an
unexplained earlier failure.

## Provenance record

Record all of the following in the build log and stack run log:

- build platform and architecture;
- builder name and exact Podman/Buildah version;
- the upstream root `Dockerfile` blob SHA at the integration commit;
- every resolved `FROM` image digest as actually pulled, including stages whose
  Dockerfile input is a mutable tag;
- `upstream.sha`, each entry's `head_sha` and `head_tree`, the exact manifest
  SHA-256/`manifest-sha8`, and the integration commit SHA and tree;
- the exact `HERMES_GIT_SHA` build argument;
- the output image manifest digest and creation timestamp; and
- build wall-clock time, peak disk use, and peak memory use.

Inspect build arguments and confirm that no secret mechanism was used. Inspect
the final image's `ENV`, `LABEL`, and history, plus candidate-owned additions,
without treating public endpoints intentionally shipped by upstream as
secrets. Do not print credentials.

## Reproducibility and promotion boundary

The source input is reproducible through the manifest's exact-byte and tree
proofs, and the output is immutable by digest. The image is not claimed to be
bit-for-bit reproducible: the upstream build performs `apt-get update` and
network installs even though many inputs are pinned or checksummed. Record the
resolved inputs honestly instead of claiming binary reproducibility.

After a successful local build, later overlay tests, container smoke, no-secret
inspection, provenance review, and explicit promotion/deployment approvals are
still separate gates. A build result never authorizes publication or
deployment.
