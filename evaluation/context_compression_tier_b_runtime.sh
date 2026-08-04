#!/bin/sh
set -eu

ROOT=/home/ron/hermes-compaction-tier-b
SOURCE_REPO=$ROOT/source-repo
RUNTIME=$ROOT/runtime
INPUT=$RUNTIME/input
OUTPUT=$RUNTIME/output
BUILD_CONTEXT=$RUNTIME/build-context/evaluation-head
BUNDLE=$ROOT/evaluation.bundle
PRODUCTION_ID_FILE=$ROOT/production-container-id-before.txt
EXPECTED_HEAD_FILE=$ROOT/evaluation-head.txt
CONTAINER=hermes-compaction-tier-b-runner
NETWORK=hermes-compaction-tier-b-egress
LABEL_VALUE=context-compression-tier-b
EVALUATION_BRANCH=eval/compaction-tier-a
STARTING_HEAD=953491781ba8ec39cf4b5e15c9654eb7066b33f5
CANDIDATE_SHA=f07664bb9a19788ec426db2eb8b8ec8d9572b21d
BASELINE_SHA=d5e135a51353c2dbc489d5c2583158b22d8efd7b
BASE_IMAGE=sha256:8539546b37868ca348618a8aa147ecfb68eb0caa8e597f98e649b42ed4e5c805
EVALUATOR=$SOURCE_REPO/evaluation/context_compression_tier_b.py
SELF=$(/usr/bin/realpath "$0")

die() {
    printf '%s\n' "$1" >&2
    exit 2
}

require_hex() {
    value=$1
    width=$2
    code=$3
    [ "${#value}" -eq "$width" ] || die "$code"
    case "$value" in
        *[!0-9a-f]*) die "$code" ;;
    esac
}

require_non_root_podman() {
    [ "$(id -u)" -ne 0 ] || die ROOT_CALLER_FORBIDDEN
    [ "$(podman info --format '{{.Host.Security.Rootless}}')" = true ] \
        || die ROOTLESS_PODMAN_REQUIRED
}

discover_production_id() {
    rows=$(podman ps -a --no-trunc --filter 'name=^hermes$' \
        --format '{{.ID}}\t{{.Names}}\t{{.State}}' \
        | awk -F '\t' '$2 == "hermes" { print }')
    count=$(printf '%s\n' "$rows" | awk 'NF { count += 1 } END { print count + 0 }')
    [ "$count" -eq 1 ] || die PRODUCTION_IDENTITY_MISMATCH
    observed_id=$(printf '%s\n' "$rows" | cut -f1)
    observed_name=$(printf '%s\n' "$rows" | cut -f2)
    observed_state=$(printf '%s\n' "$rows" | cut -f3)
    require_hex "$observed_id" 64 PRODUCTION_IDENTITY_MISMATCH
    [ "$observed_name" = hermes ] || die PRODUCTION_IDENTITY_MISMATCH
    [ "$observed_state" = running ] || die PRODUCTION_IDENTITY_MISMATCH
    printf '%s\n' "$observed_id"
}

require_task_names_absent() {
    if podman container exists "$CONTAINER"; then
        die TASK_CONTAINER_NAME_COLLISION
    fi
    if podman network exists "$NETWORK"; then
        die TASK_NETWORK_NAME_COLLISION
    fi
}

require_task_root() {
    [ -d "$ROOT" ] && [ ! -L "$ROOT" ] || die TASK_ROOT_MISSING
    [ "$(/usr/bin/realpath "$ROOT")" = "$ROOT" ] || die TASK_ROOT_IDENTITY_MISMATCH
    [ "$(stat -c %u "$ROOT")" -eq "$(id -u)" ] || die TASK_ROOT_OWNER_MISMATCH
    [ -d "$SOURCE_REPO/.git" ] && [ ! -L "$SOURCE_REPO" ] \
        || die STAGED_SOURCE_MISSING
    [ -f "$PRODUCTION_ID_FILE" ] && [ ! -L "$PRODUCTION_ID_FILE" ] \
        || die PRODUCTION_IDENTITY_MISSING
    [ -f "$EXPECTED_HEAD_FILE" ] && [ ! -L "$EXPECTED_HEAD_FILE" ] \
        || die EVALUATION_HEAD_MISSING
}

require_production_continuity() {
    expected=$(tr -d '\n' <"$PRODUCTION_ID_FILE")
    require_hex "$expected" 64 PRODUCTION_IDENTITY_MISMATCH
    observed=$(discover_production_id)
    [ "$observed" = "$expected" ] || die PRODUCTION_IDENTITY_MISMATCH
}

require_staged_source() {
    require_task_root
    expected_head=$(tr -d '\n' <"$EXPECTED_HEAD_FILE")
    require_hex "$expected_head" 40 EVALUATION_HEAD_INVALID
    [ "$(git -C "$SOURCE_REPO" branch --show-current)" = "$EVALUATION_BRANCH" ] \
        || die SOURCE_BRANCH_MISMATCH
    [ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" = "$expected_head" ] \
        || die SOURCE_HEAD_MISMATCH
    [ -z "$(git -C "$SOURCE_REPO" status --porcelain --untracked-files=all)" ] \
        || die SOURCE_WORKTREE_DIRTY
    cmp -s "$SELF" "$SOURCE_REPO/evaluation/context_compression_tier_b_runtime.sh" \
        || die EXECUTING_HELPER_MISMATCH
}

require_transaction() {
    require_non_root_podman
    require_staged_source
    require_production_continuity
}

read_manifest_field() {
    /usr/bin/python3 -c \
        'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value[sys.argv[2]])' \
        "$1" "$2"
}

require_task_label() {
    object_type=$1
    object_name=$2
    [ "$(podman "$object_type" inspect --format '{{index .Labels "io.hermes.benchmark"}}' "$object_name")" = "$LABEL_VALUE" ] \
        || die TASK_LABEL_MISMATCH
}

do_stage_source() {
    expected_sha=$1
    expected_size=$2
    expected_branch=$3
    expected_head=$4
    require_hex "$expected_sha" 64 BUNDLE_SHA256_INVALID
    case "$expected_size" in
        ''|*[!0-9]*) die BUNDLE_SIZE_INVALID ;;
    esac
    [ "$expected_size" -gt 0 ] || die BUNDLE_SIZE_INVALID
    [ "$expected_branch" = "$EVALUATION_BRANCH" ] || die SOURCE_BRANCH_MISMATCH
    require_hex "$expected_head" 40 EVALUATION_HEAD_INVALID
    require_non_root_podman
    [ ! -e "$ROOT" ] || die TASK_ROOT_ALREADY_EXISTS
    require_task_names_absent
    before=$(discover_production_id)

    umask 077
    mkdir "$ROOT"
    chmod 0700 "$ROOT"
    printf '%s\n' "$before" >"$PRODUCTION_ID_FILE"

    /usr/bin/python3 -c '
import os
import sys

path = sys.argv[1]
remaining = int(sys.argv[2])
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
try:
    while remaining:
        chunk = os.read(0, min(65536, remaining))
        if not chunk:
            raise SystemExit(2)
        os.write(fd, chunk)
        remaining -= len(chunk)
    if os.read(0, 1):
        raise SystemExit(2)
    os.fsync(fd)
finally:
    os.close(fd)
' "$BUNDLE" "$expected_size" || die BUNDLE_STREAM_INVALID
    [ -f "$BUNDLE" ] && [ ! -L "$BUNDLE" ] || die BUNDLE_PATH_INVALID
    [ "$(stat -c %s "$BUNDLE")" -eq "$expected_size" ] \
        || die BUNDLE_SIZE_MISMATCH
    [ "$(sha256sum "$BUNDLE" | cut -d ' ' -f1)" = "$expected_sha" ] \
        || die BUNDLE_SHA256_MISMATCH

    git clone --quiet --no-hardlinks --single-branch --branch "$EVALUATION_BRANCH" \
        "$BUNDLE" "$SOURCE_REPO" || die SOURCE_CLONE_FAILED
    [ "$(git -C "$SOURCE_REPO" branch --show-current)" = "$EVALUATION_BRANCH" ] \
        || die SOURCE_BRANCH_MISMATCH
    [ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" = "$expected_head" ] \
        || die SOURCE_HEAD_MISMATCH
    for revision in "$STARTING_HEAD" "$BASELINE_SHA" "$CANDIDATE_SHA"; do
        [ "$(git -C "$SOURCE_REPO" rev-parse --verify "$revision^{commit}")" = "$revision" ] \
            || die REVISION_IDENTITY_INVALID
    done
    git -C "$SOURCE_REPO" merge-base --is-ancestor "$STARTING_HEAD" "$expected_head" \
        || die REVISION_ANCESTRY_INVALID
    git -C "$SOURCE_REPO" merge-base --is-ancestor "$BASELINE_SHA" "$CANDIDATE_SHA" \
        || die REVISION_ANCESTRY_INVALID
    git -C "$SOURCE_REPO" merge-base --is-ancestor "$CANDIDATE_SHA" "$expected_head" \
        || die REVISION_ANCESTRY_INVALID
    [ -z "$(git -C "$SOURCE_REPO" status --porcelain --untracked-files=all)" ] \
        || die SOURCE_WORKTREE_DIRTY
    cmp -s "$SELF" "$SOURCE_REPO/evaluation/context_compression_tier_b_runtime.sh" \
        || die EXECUTING_HELPER_MISMATCH
    printf '%s\n' "$expected_head" >"$EXPECTED_HEAD_FILE"
    require_production_continuity
    rm "$BUNDLE"
}

do_prepare() {
    require_transaction
    [ ! -e "$RUNTIME" ] || die RUNTIME_ROOT_ALREADY_EXISTS
    require_task_names_absent
    umask 077
    mkdir "$RUNTIME"
    chmod 0700 "$RUNTIME"
    env \
        -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
        -u NOUS_API_KEY -u GOOGLE_API_KEY -u GEMINI_API_KEY \
        -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
        -u GITHUB_TOKEN -u CODEX_HOME \
        /usr/bin/python3 "$EVALUATOR" prepare \
        --repo-root "$SOURCE_REPO" --runtime-root "$RUNTIME"
    cp "$PRODUCTION_ID_FILE" "$INPUT/production-container-id-before.txt"
    require_production_continuity
}

do_build_image() {
    require_transaction
    [ -f "$INPUT/source-manifest.json" ] || die SOURCE_MANIFEST_MISSING
    evaluation_head=$(read_manifest_field "$INPUT/source-manifest.json" evaluation_head)
    [ -n "$evaluation_head" ] || die EVALUATION_HEAD_MISSING
    podman build --pull=never --no-cache --network=slirp4netns \
        --label io.hermes.benchmark=context-compression-tier-b \
        --label "io.hermes.evaluation-head=$evaluation_head" \
        --file "$BUILD_CONTEXT/evaluation/Containerfile.context-compression-tier-b" \
        --tag "localhost/hermes-compaction-tier-b:$evaluation_head" \
        "$BUILD_CONTEXT"
    image_id=$(podman image inspect \
        --format '{{.Id}}' "localhost/hermes-compaction-tier-b:$evaluation_head")
    case "$image_id" in sha256:????????????????????????????????????????????????????????????????) ;;
        *) die RUNTIME_IMAGE_ID_INVALID ;;
    esac
    [ "$(podman image inspect --format '{{index .Labels "io.hermes.benchmark"}}' "$image_id")" = "$LABEL_VALUE" ] \
        || die IMAGE_LABEL_MISMATCH
    [ "$(podman image inspect --format '{{index .Labels "io.hermes.evaluation-head"}}' "$image_id")" = "$evaluation_head" ] \
        || die IMAGE_HEAD_LABEL_MISMATCH
    base_layers=$(podman image inspect --format '{{json .RootFS.Layers}}' "$BASE_IMAGE")
    image_layers=$(podman image inspect --format '{{json .RootFS.Layers}}' "$image_id")
    /usr/bin/python3 -c \
        'import json,sys; b=json.loads(sys.argv[1]); i=json.loads(sys.argv[2]); raise SystemExit(0 if i[:len(b)] == b else 2)' \
        "$base_layers" "$image_layers" || die BUILD_BASE_ANCESTRY_MISMATCH
    inspection=$INPUT/.image-inspection.json
    podman run --rm --pull=never --network=none --read-only \
        --cap-drop=ALL --security-opt=no-new-privileges \
        --tmpfs /tmp:rw,nosuid,nodev,size=67108864,mode=1777 \
        --tmpfs /run:rw,nosuid,nodev,size=16777216,mode=0755 \
        --entrypoint=/opt/hermes/.venv/bin/python "$image_id" -c \
        'import importlib.metadata,json,platform; rows=sorted({(d.metadata["Name"].lower(),d.version) for d in importlib.metadata.distributions()}); print(json.dumps({"architecture":platform.machine(),"python_version":platform.python_version(),"distributions":rows},sort_keys=True,separators=(",",":")))' \
        >"$inspection"
    TIER_B_IMAGE_ID=$image_id EVALUATION_HEAD=$evaluation_head \
        BASE_LAYERS=$base_layers IMAGE_LAYERS=$image_layers \
        /usr/bin/python3 "$EVALUATOR" write-image-manifest \
        --source-manifest "$INPUT/source-manifest.json" \
        --inspection "$inspection" --output "$INPUT/image-manifest.json"
    rm "$inspection"
}

do_verify_image() {
    require_transaction
    image_id=$(read_manifest_field "$INPUT/image-manifest.json" tier_b_image_id)
    require_task_label image "$image_id"
    common='--rm --pull=never --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges'
    # shellcheck disable=SC2086
    podman run $common --tmpfs /tmp:rw,nosuid,nodev,size=67108864,mode=1777 \
        --tmpfs /run:rw,nosuid,nodev,size=16777216,mode=0755 \
        --entrypoint=/opt/hermes/.venv/bin/hermes "$image_id" --version >/dev/null
    # shellcheck disable=SC2086
    podman run $common --tmpfs /tmp:rw,nosuid,nodev,size=67108864,mode=1777 \
        --tmpfs /run:rw,nosuid,nodev,size=16777216,mode=0755 \
        --entrypoint=/opt/hermes/.venv/bin/python "$image_id" -c \
        'import agent.auxiliary_client,agent.context_compressor,evaluation.context_compression_tier_b,openai; assert all("/opt/hermes" in str(getattr(m,"__file__","")) for m in (agent.auxiliary_client,agent.context_compressor,evaluation.context_compression_tier_b))'
}

do_start() {
    require_transaction
    image_id=$(read_manifest_field "$INPUT/image-manifest.json" tier_b_image_id)
    require_task_names_absent
    podman network create \
        --label io.hermes.benchmark=context-compression-tier-b \
        "$NETWORK" >/dev/null
    podman run -d \
        --name "$CONTAINER" \
        --label io.hermes.benchmark=context-compression-tier-b \
        --pull=never --read-only --cap-drop=ALL \
        --security-opt=no-new-privileges --pids-limit=256 --userns=keep-id \
        --network="$NETWORK" \
        --mount "type=bind,src=$INPUT,dst=/benchmark/input,ro=true" \
        --mount "type=bind,src=$OUTPUT,dst=/benchmark/output,rw=true" \
        --tmpfs /benchmark/home:rw,nosuid,nodev,size=268435456,mode=0700 \
        --tmpfs /benchmark/hermes:rw,nosuid,nodev,size=268435456,mode=0700 \
        --tmpfs /benchmark/run:rw,nosuid,nodev,size=2147483648,mode=0700 \
        --tmpfs /tmp:rw,nosuid,nodev,size=1073741824,mode=1777 \
        --tmpfs /run:rw,nosuid,nodev,size=67108864,mode=0755 \
        --env HOME=/benchmark/home --env HERMES_HOME=/benchmark/hermes \
        --env PYTHONDONTWRITEBYTECODE=1 --env HERMES_DISABLE_LAZY_INSTALLS=1 \
        --env NO_COLOR=1 \
        --env TIER_B_LIVE_ACK=CONTEXT_COMPRESSION_TIER_B_AUTHORIZED \
        --workdir /opt/hermes --entrypoint=/bin/sleep \
        "$image_id" infinity >/dev/null
    do_liveness
}

do_liveness() {
    require_transaction
    image_id=$(read_manifest_field "$INPUT/image-manifest.json" tier_b_image_id)
    require_task_label container "$CONTAINER"
    [ "$(podman container inspect --format '{{.State.Status}}' "$CONTAINER")" = running ] \
        || die TASK_CONTAINER_NOT_RUNNING
    [ "$(podman container inspect --format '{{.State.Running}}' "$CONTAINER")" = true ] \
        || die TASK_CONTAINER_NOT_RUNNING
    [ "$(podman container inspect --format '{{.State.ExitCode}}' "$CONTAINER")" = 0 ] \
        || die TASK_CONTAINER_EXIT_NONZERO
    [ "$(podman container inspect --format '{{.Image}}' "$CONTAINER")" = "$image_id" ] \
        || die TASK_CONTAINER_IMAGE_MISMATCH
    top=$(podman top "$CONTAINER" pid,args)
    [ "$(printf '%s\n' "$top" | wc -l | tr -d ' ')" -eq 2 ] \
        || die TASK_PROCESS_SHAPE_INVALID
    printf '%s\n' "$top" | tail -n 1 | grep -Eq '[[:space:]]/bin/sleep infinity$' \
        || die TASK_PROCESS_SHAPE_INVALID
}

do_preflight() {
    do_liveness
    require_task_label network "$NETWORK"
    image_id=$(read_manifest_field "$INPUT/image-manifest.json" tier_b_image_id)
    [ "$(podman container inspect --format '{{.HostConfig.ReadonlyRootfs}}' "$CONTAINER")" = true ] \
        || die READ_ONLY_ROOT_REQUIRED
    [ "$(podman container inspect --format '{{.HostConfig.PidsLimit}}' "$CONTAINER")" = 256 ] \
        || die PID_LIMIT_MISMATCH
    mounts=$(podman container inspect --format '{{json .Mounts}}' "$CONTAINER")
    networks=$(podman container inspect --format '{{json .NetworkSettings.Networks}}' "$CONTAINER")
    ports=$(podman container inspect --format '{{json .NetworkSettings.Ports}}' "$CONTAINER")
    env_json=$(podman container inspect --format '{{json .Config.Env}}' "$CONTAINER")
    /usr/bin/python3 "$EVALUATOR" verify-runtime-metadata \
        --mounts "$mounts" --networks "$networks" --ports "$ports" \
        --environment "$env_json" --image-id "$image_id"
    config='model:\n  provider: openai-codex\n  default: gpt-5.6-sol\n  api_mode: codex_responses\nauxiliary:\n  transient_retries: 0\n  compression:\n    provider: openai-codex\n    model: gpt-5.6-luna\n    api_mode: codex_responses\n    timeout: 300\n'
    if ! podman exec "$CONTAINER" test -e /benchmark/hermes/config.yaml; then
        printf '%b' "$config" | podman exec -i "$CONTAINER" /bin/sh -c \
            'umask 077; cat > /benchmark/hermes/config.yaml'
    fi
    podman exec --workdir /opt/hermes "$CONTAINER" \
        /opt/hermes/.venv/bin/python /opt/hermes/evaluation/context_compression_tier_b.py attest-runtime \
        --manifest /benchmark/input/source-manifest.json \
        --image-manifest /benchmark/input/image-manifest.json \
        --schedule /benchmark/input/execution-schedule.json \
        --run-root /benchmark/run --output-root /benchmark/output
}

do_auth_status() {
    do_liveness
    podman exec "$CONTAINER" /bin/sh -c \
        '/opt/hermes/.venv/bin/hermes auth status openai-codex >/benchmark/run/auth-status.untrusted 2>&1' \
        || die AUTH_STATUS_FAILED
    podman exec --workdir /opt/hermes "$CONTAINER" \
        /opt/hermes/.venv/bin/python /opt/hermes/evaluation/context_compression_tier_b.py sanitize-auth-status \
        --input /benchmark/run/auth-status.untrusted \
        --output /benchmark/run/auth-attestation.json
}

do_finalize() {
    do_liveness
    podman exec --workdir /opt/hermes "$CONTAINER" \
        /opt/hermes/.venv/bin/python /opt/hermes/evaluation/context_compression_tier_b.py finalize \
        --source-root /opt/hermes \
        --input-root /benchmark/input \
        --final-root /benchmark/output/final
    podman exec --workdir /opt/hermes "$CONTAINER" \
        /opt/hermes/.venv/bin/python /opt/hermes/evaluation/context_compression_tier_b.py validate-finalized \
        --source-root /opt/hermes \
        --input-root /benchmark/input \
        --final-root /benchmark/output/final
}

do_stop() {
    require_transaction
    require_task_label container "$CONTAINER"
    podman stop "$CONTAINER" >/dev/null
}

do_cleanup() {
    require_transaction
    expected=$(tr -d '\n' <"$PRODUCTION_ID_FILE")
    require_task_label container "$CONTAINER"
    require_task_label network "$NETWORK"
    image_id=$(read_manifest_field "$INPUT/image-manifest.json" tier_b_image_id)
    require_task_label image "$image_id"
    [ -f "$OUTPUT/final/FINALIZED.sha256" ] || die FINALIZED_EVIDENCE_MISSING
    podman rm "$CONTAINER" >/dev/null
    podman network rm "$NETWORK" >/dev/null
    podman image rm "$image_id" >/dev/null
    resolved_root=$(/usr/bin/realpath "$ROOT")
    [ "$resolved_root" = "$ROOT" ] || die TASK_ROOT_IDENTITY_MISMATCH
    [ "$(stat -c %u "$ROOT")" -eq "$(id -u)" ] || die TASK_ROOT_OWNER_MISMATCH
    rm -rf -- "$resolved_root"
    observed=$(discover_production_id)
    [ "$observed" = "$expected" ] || die PRODUCTION_IDENTITY_MISMATCH
}

case "${1-}" in
    stage-source)
        [ "$#" -eq 5 ] || die 'usage: context_compression_tier_b_runtime.sh stage-source BUNDLE_SHA256 BUNDLE_SIZE BRANCH EVALUATION_HEAD'
        do_stage_source "$2" "$3" "$4" "$5"
        ;;
    prepare) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_prepare ;;
    build-image) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_build_image ;;
    verify-image) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_verify_image ;;
    start) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_start ;;
    liveness) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_liveness ;;
    preflight) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_preflight ;;
    auth-status) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_auth_status ;;
    finalize) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_finalize ;;
    stop) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_stop ;;
    cleanup) [ "$#" -eq 1 ] || die INVALID_ARGUMENTS; do_cleanup ;;
    *) die 'usage: context_compression_tier_b_runtime.sh {stage-source|prepare|build-image|verify-image|start|liveness|preflight|auth-status|finalize|stop|cleanup}' ;;
esac
