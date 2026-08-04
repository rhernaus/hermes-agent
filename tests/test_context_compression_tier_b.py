import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import stat
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import context_compression_tier_b as tier_b


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "evaluation/fixtures/context-compression-tier-b.json"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def perfect_answer(fixture, scenario_id):
    scenario = next(
        item for item in fixture["scenarios"] if item["scenario_id"] == scenario_id
    )
    facts = {item["field_id"]: item for item in scenario["facts"]}

    def field(field_id):
        return {
            "applicable": True,
            "status": "known",
            "value": facts[field_id]["expected"],
        }

    return {
        "schema_version": "context-compression-tier-b-answer/1",
        "scenario_id": scenario_id,
        "question_id": scenario["question_id"],
        "required_state": {
            field_id: field(field_id) for field_id in scenario["required_state_order"]
        },
        "identifiers": {
            field_id: field(field_id) for field_id in scenario["identifiers_order"]
        },
        "recommended_next_action": field("recommended_next_action"),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def fake_live_guard_modules():
    counters = {"wire": 0, "main": 0}
    state = {
        "base_url": "https://chatgpt.com/backend-api/codex",
        "max_retries": 0,
        "fallback_kind": None,
        "request_overrides": {},
        "wrong_adapter": False,
    }

    class CodexAuxiliaryClient:
        pass

    CodexAuxiliaryClient.__module__ = "agent.auxiliary_client"

    class _CodexCompletionsAdapter:
        def __init__(self):
            self._client = SimpleNamespace(
                max_retries=state["max_retries"],
                base_url=state["base_url"],
            )

        def create(self, **kwargs):
            counters["wire"] += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                model=kwargs["model"],
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                ),
            )

    _CodexCompletionsAdapter.__module__ = "agent.auxiliary_client"

    class WrongAdapter(_CodexCompletionsAdapter):
        pass

    WrongAdapter.__module__ = "agent.auxiliary_client"

    auxiliary = SimpleNamespace()

    def relay(
        client, kwargs, *, provider=None, api_mode=None, create=None, fallback_kind=None
    ):
        del create
        adapter = (
            WrongAdapter() if state["wrong_adapter"] else _CodexCompletionsAdapter()
        )
        return adapter.create(**kwargs)

    def call_llm(**kwargs):
        request = {
            "model": kwargs["model"],
            "messages": kwargs.get("messages", []),
            "timeout": kwargs["timeout"],
            **state["request_overrides"],
        }
        return auxiliary._relay_sync_completion(
            CodexAuxiliaryClient(),
            request,
            provider=kwargs["provider"],
            api_mode=kwargs["api_mode"],
            fallback_kind=state["fallback_kind"],
        )

    def main_transport(*_args, **_kwargs):
        counters["main"] += 1

    auxiliary.CodexAuxiliaryClient = CodexAuxiliaryClient
    auxiliary._CodexCompletionsAdapter = _CodexCompletionsAdapter
    auxiliary._relay_sync_completion = relay
    auxiliary.call_llm = call_llm
    auxiliary.run_codex_stream = main_transport
    auxiliary.main_responses_create = main_transport

    class ContextCompressor:
        def _fallback_to_main_for_compression(self, *_args, **_kwargs):
            counters["main"] += 1

    compressor = SimpleNamespace(ContextCompressor=ContextCompressor)
    return auxiliary, compressor, counters, state


def make_runtime_attestation_fixture(root):
    source_root = root / "embedded"
    input_root = root / "input"
    run_root = root / "run"
    output_root = root / "output"
    hermes_home = root / "hermes"
    home = root / "home"
    source_relatives = (
        "evaluation/context_compression_tier_b.py",
        "evaluation/fixtures/context-compression-tier-b.json",
        "evaluation/Containerfile.context-compression-tier-b",
        "evaluation/context_compression_tier_b_runtime.sh",
        "pyproject.toml",
        "uv.lock",
        "agent/context_compressor.py",
    )
    for relative in source_relatives:
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)

    tree_root = input_root / "trees"
    for label in ("baseline", "candidate"):
        destination = tree_root / label / "agent/context_compressor.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"{label} compressor bytes\n".encode())
    shutil.copytree(source_root, tree_root / "evaluation-head")

    archive_root = input_root / "archives"
    archive_root.mkdir(parents=True)
    archive_identities = {}
    tree_identities = {}
    for label in ("baseline", "candidate", "evaluation-head"):
        tree = tree_root / label
        tree_rows = tier_b._tree_manifest(tree)
        tree_manifest_path = input_root / f"{label}-tree-manifest.json"
        write_json(tree_manifest_path, tree_rows)
        archive_path = archive_root / f"{label}.tar"
        with tarfile.open(archive_path, mode="w") as archive:
            for path in sorted(
                tree.rglob("*"), key=lambda item: item.relative_to(tree).as_posix()
            ):
                archive.add(
                    path,
                    arcname=path.relative_to(tree).as_posix(),
                    recursive=False,
                )
        archive_rows = tier_b._archive_manifest(archive_path)
        self_check = tier_b._tree_manifest(tree)
        if archive_rows != self_check:
            raise AssertionError("attestation fixture archive/tree disagreement")
        archive_manifest_path = archive_root / f"{label}.manifest.json"
        write_json(archive_manifest_path, archive_rows)
        archive_identities[label] = {
            "revision_sha": {
                "baseline": tier_b.BASELINE_SHA,
                "candidate": tier_b.CANDIDATE_SHA,
                "evaluation-head": tier_b.STARTING_HEAD,
            }[label],
            "archive": tier_b._file_identity(archive_path),
            "manifest": tier_b._file_identity(archive_manifest_path),
            "member_count": len(archive_rows),
        }
        tree_identities[label] = {
            "manifest": tier_b._file_identity(tree_manifest_path),
            "tree_sha256": hashlib.sha256(canonical(tree_rows).encode()).hexdigest(),
            "member_count": len(tree_rows),
        }

    fixture_path = source_root / "evaluation/fixtures/context-compression-tier-b.json"
    fixture = tier_b.load_fixture(fixture_path)
    fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    schedule = tier_b.build_execution_schedule(fixture, fixture_sha)
    schedule_path = input_root / "execution-schedule.json"
    write_json(schedule_path, schedule)
    files = {
        relative: tier_b._file_identity(source_root / relative)
        for relative in source_relatives
        if relative != "agent/context_compressor.py"
    }
    source_manifest = {
        "schema_version": "context-compression-tier-b-source-manifest/1",
        "identity": {
            "branch": tier_b.EVALUATION_BRANCH,
            "evaluation_head": tier_b.STARTING_HEAD,
            "dry_candidate_overlay": False,
            "live_ready": True,
            "product_diff_paths": list(tier_b._PRODUCT_DIFF_PATHS),
        },
        "evaluation_head": tier_b.STARTING_HEAD,
        "live_ready": True,
        "revisions": {
            "baseline": tier_b.BASELINE_SHA,
            "candidate": tier_b.CANDIDATE_SHA,
            "evaluation-head": tier_b.STARTING_HEAD,
        },
        "provider": tier_b.PROVIDER,
        "api_mode": tier_b.API_MODE,
        "summarizer_model": tier_b.SUMMARY_MODEL,
        "downstream_model": tier_b.DOWNSTREAM_MODEL,
        "build_base_image_id": tier_b.BUILD_BASE_IMAGE_ID,
        "build_base_provenance": tier_b.BUILD_BASE_PROVENANCE,
        "expected_counts": schedule["expected_counts"],
        "fixture_sha256": fixture_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file": tier_b._file_identity(schedule_path),
        "archives": archive_identities,
        "trees": tree_identities,
        "build_context": {
            "tree_sha256": tree_identities["evaluation-head"]["tree_sha256"],
            "member_count": tree_identities["evaluation-head"]["member_count"],
        },
        "files": files,
    }
    manifest_path = input_root / "source-manifest.json"
    write_json(manifest_path, source_manifest)
    distributions = sorted({
        (distribution.metadata["Name"].lower(), distribution.version)
        for distribution in importlib.metadata.distributions()
    })
    distribution_rows = [list(item) for item in distributions]
    image_manifest = {
        "schema_version": "context-compression-tier-b-image-manifest/1",
        "tier_b_image_id": "sha256:" + "9" * 64,
        "evaluation_head": tier_b.STARTING_HEAD,
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "schedule_sha256": schedule["schedule_sha256"],
        "build_base_image_id": tier_b.BUILD_BASE_IMAGE_ID,
        "build_base_provenance": tier_b.BUILD_BASE_PROVENANCE,
        "base_layers": ["sha256:" + "1" * 64],
        "image_layers": ["sha256:" + "1" * 64, "sha256:" + "2" * 64],
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "uv_version": "0.11.6",
        "sync_command": "uv sync --locked --python 3.13 --extra dev",
        "build_command": (
            "podman build --pull=never --no-cache --network=slirp4netns "
            "--label io.hermes.benchmark=context-compression-tier-b "
            f"--label io.hermes.evaluation-head={tier_b.STARTING_HEAD} "
            "--file evaluation/Containerfile.context-compression-tier-b "
            f"--tag localhost/hermes-compaction-tier-b:{tier_b.STARTING_HEAD} ."
        ),
        "source_files": files,
        "evaluation_archive": archive_identities["evaluation-head"],
        "evaluation_tree": tree_identities["evaluation-head"],
        "build_context": source_manifest["build_context"],
        "installed_distributions": distribution_rows,
        "installed_distributions_sha256": hashlib.sha256(
            canonical(distribution_rows).encode()
        ).hexdigest(),
        "registry_digest_status": "not_exposed_by_local_image",
        "registry_digests": [],
    }
    image_manifest_path = input_root / "image-manifest.json"
    write_json(image_manifest_path, image_manifest)
    hermes_home.mkdir(parents=True)
    home.mkdir(parents=True)
    run_root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  provider: openai-codex\n"
        "  default: gpt-5.6-sol\n"
        "  api_mode: codex_responses\n"
        "auxiliary:\n"
        "  transient_retries: 0\n"
        "  compression:\n"
        "    provider: openai-codex\n"
        "    model: gpt-5.6-luna\n"
        "    api_mode: codex_responses\n"
        "    timeout: 300\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "HOME": str(home),
        "HERMES_HOME": str(hermes_home),
        "TIER_B_LIVE_ACK": tier_b.LIVE_ACK,
        "CI": "",
        "GITHUB_ACTIONS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment.pop("CODEX_HOME", None)
    return SimpleNamespace(
        source_root=source_root,
        input_root=input_root,
        manifest_path=manifest_path,
        image_manifest_path=image_manifest_path,
        schedule_path=schedule_path,
        run_root=run_root,
        output_root=output_root,
        environment=environment,
    )


class RuntimeHelperSourceStagingContractTests(unittest.TestCase):
    PRODUCTION_ID = "1" * 64
    TASK_IMAGE_ID = "a" * 64

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.task_root = cls.root / "task-root"
        cls.fake_bin = cls.root / "fake-bin"
        cls.fake_bin.mkdir()
        cls.podman_rows = cls.root / "podman-rows"
        cls.podman_log = cls.root / "podman.log"

        cls._write_executable(
            cls.fake_bin / "id",
            """#!/bin/sh
[ "${1-}" = -u ] || exit 97
printf '1000\\n'
""",
        )
        cls._write_executable(
            cls.fake_bin / "stat",
            """#!/bin/sh
if [ "$#" -eq 3 ] && [ "$1" = -c ] && [ "$2" = %u ] && [ "$3" = "$FAKE_TASK_ROOT" ]; then
    printf '1000\n'
    exit 0
fi
exec /usr/bin/stat "$@"
""",
        )
        cls._write_executable(
            cls.fake_bin / "podman",
            """#!/bin/sh
printf '%s\\n' "$*" >>"$FAKE_PODMAN_LOG"
case "$1 ${2-}" in
    'info --format') printf 'true\\n' ;;
    'ps -a') /bin/cat "$FAKE_PODMAN_ROWS" ;;
    'container exists'|'network exists') exit 1 ;;
    'container inspect')
        case "$*" in
            *'.Config.Labels'*) printf 'context-compression-tier-b\n' ;;
            *'.State.Status'*) printf 'running\n' ;;
            *'.State.Running'*) printf 'true\n' ;;
            *'.State.ExitCode'*) printf '0\n' ;;
            *'{{.Image}}'*) printf '%s\n' "$FAKE_TASK_IMAGE_ID" ;;
            *) exit 97 ;;
        esac
        ;;
    'image inspect') printf '8539546b37868ca348618a8aa147ecfb68eb0caa8e597f98e649b42ed4e5c805\n' ;;
    top*) printf 'PID         COMMAND\n1           /bin/sleep infinity   \n' ;;
    'run --rm')
        case " $* " in
            *' --name hermes-compaction-tier-b-prepare-sync '*)
                case " $* " in
                    *' UV_PROJECT_ENVIRONMENT=/benchmark/prepare-venv '*) ;;
                    *) exit 96 ;;
                esac
                /bin/mkdir -p "$FAKE_TASK_ROOT/prepare-venv"
                ;;
            *' --name hermes-compaction-tier-b-prepare-run '*)
                [ -z "$(/bin/ls -A "$FAKE_TASK_ROOT/runtime")" ] || exit 95
                /bin/mkdir -p "$FAKE_TASK_ROOT/runtime/input"
                printf '{"live_ready":true}\n' >"$FAKE_TASK_ROOT/runtime/input/source-manifest.json"
                ;;
            *) exit 97 ;;
        esac
        ;;
    *) exit 97 ;;
esac
""",
        )

        source = cls.root / "bundle-source"
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Tier B test",
            "GIT_AUTHOR_EMAIL": "tier-b-test@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
            "GIT_COMMITTER_NAME": "Tier B test",
            "GIT_COMMITTER_EMAIL": "tier-b-test@example.invalid",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
        }

        def run_git(*arguments):
            completed = subprocess.run(
                ["git", "-C", str(source), *arguments],
                capture_output=True,
                text=True,
                env=environment,
            )
            if completed.returncode:
                raise AssertionError(completed.stderr)

        initialized = subprocess.run(
            [
                "git",
                "init",
                "--quiet",
                "--initial-branch=eval/compaction-tier-a",
                str(source),
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        if initialized.returncode:
            raise AssertionError(initialized.stderr)
        history = source / ".tier-b-test-history"

        def commit_history(message, content):
            history.write_text(content, encoding="utf-8")
            run_git("add", "--", history.name)
            run_git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message)
            return subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()

        starting_head = commit_history("starting", "starting\n")
        baseline_sha = commit_history("baseline", "starting\nbaseline\n")
        candidate_sha = commit_history("candidate", "starting\nbaseline\ncandidate\n")
        helper_source = REPO_ROOT / "evaluation/context_compression_tier_b_runtime.sh"
        helper_bytes = helper_source.read_bytes()
        replacements = {
            b"ROOT=/home/ron/hermes-compaction-tier-b": f"ROOT={cls.task_root}".encode(),
            b"STARTING_HEAD=953491781ba8ec39cf4b5e15c9654eb7066b33f5": (
                f"STARTING_HEAD={starting_head}".encode()
            ),
            b"CANDIDATE_SHA=f07664bb9a19788ec426db2eb8b8ec8d9572b21d": (
                f"CANDIDATE_SHA={candidate_sha}".encode()
            ),
            b"BASELINE_SHA=d5e135a51353c2dbc489d5c2583158b22d8efd7b": (
                f"BASELINE_SHA={baseline_sha}".encode()
            ),
        }
        for old, replacement in replacements.items():
            if helper_bytes.count(old) != 1:
                raise AssertionError(f"helper replacement mismatch: {old!r}")
            helper_bytes = helper_bytes.replace(old, replacement)
        cls.helper = cls.root / "context_compression_tier_b_runtime.sh"
        cls.helper.write_bytes(helper_bytes)
        cls.helper.chmod(0o700)

        approved = (
            "evaluation/context_compression_tier_b.py",
            "evaluation/fixtures/context-compression-tier-b.json",
            "tests/test_context_compression_tier_b.py",
            "evaluation/Containerfile.context-compression-tier-b",
            "evaluation/context_compression_tier_b_runtime.sh",
        )
        for relative in approved:
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, destination)
        (source / "evaluation/context_compression_tier_b_runtime.sh").write_bytes(
            helper_bytes
        )
        run_git("add", "--", *approved)
        run_git(
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "tier b test",
        )
        cls.evaluation_head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        cls.bundle = cls.root / "evaluation.bundle"
        run_git(
            "bundle",
            "create",
            str(cls.bundle),
            "refs/heads/eval/compaction-tier-a",
        )
        cls.bundle_bytes = cls.bundle.read_bytes()
        cls.bundle_sha256 = hashlib.sha256(cls.bundle_bytes).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    @staticmethod
    def _write_executable(path, body):
        path.write_text(body, encoding="utf-8")
        path.chmod(0o700)

    def setUp(self):
        self._reset_task_root()
        self._set_production_rows(f"{self.PRODUCTION_ID}\thermes\trunning\n")
        self.podman_log.write_text("", encoding="utf-8")

    def tearDown(self):
        self._reset_task_root()

    def _reset_task_root(self):
        if self.task_root.exists():
            shutil.rmtree(self.task_root)

    def _set_production_rows(self, rows):
        self.podman_rows.write_text(rows, encoding="utf-8")

    def _run_helper(self, *arguments, payload=b"", task_image=None):
        environment = {
            **os.environ,
            "PATH": f"{self.fake_bin}:{os.environ['PATH']}",
            "FAKE_PODMAN_ROWS": str(self.podman_rows),
            "FAKE_PODMAN_LOG": str(self.podman_log),
            "FAKE_TASK_ROOT": str(self.task_root),
            "FAKE_TASK_IMAGE_ID": task_image or self.TASK_IMAGE_ID,
        }
        return subprocess.run(
            [str(self.helper), *arguments],
            input=payload,
            capture_output=True,
            env=environment,
        )

    def _stage(self, payload=None, sha256=None, size=None, branch=None):
        payload = self.bundle_bytes if payload is None else payload
        sha256 = self.bundle_sha256 if sha256 is None else sha256
        size = len(self.bundle_bytes) if size is None else size
        branch = "eval/compaction-tier-a" if branch is None else branch
        return self._run_helper(
            "stage-source",
            sha256,
            str(size),
            branch,
            self.evaluation_head,
            payload=payload,
        )

    def test_helper_has_no_controller_path_or_fixed_production_id(self):
        helper = (
            REPO_ROOT / "evaluation/context_compression_tier_b_runtime.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("/opt/data/", helper)
        self.assertNotIn("\nPRODUCTION_ID=", helper)

    def _prepare_liveness_transaction(self):
        staged = self._stage()
        self.assertEqual(staged.returncode, 0, staged.stderr.decode())
        input_root = self.task_root / "runtime/input"
        input_root.mkdir(parents=True)
        (input_root / "image-manifest.json").write_text(
            json.dumps({"tier_b_image_id": f"sha256:{self.TASK_IMAGE_ID}"}) + "\n",
            encoding="utf-8",
        )

    def test_liveness_accepts_live_podman_label_and_bare_image_id(self):
        self._prepare_liveness_transaction()
        result = self._run_helper("liveness")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        log = self.podman_log.read_text(encoding="utf-8")
        self.assertIn('.Config.Labels "io.hermes.benchmark"', log)
        self.assertIn("container inspect --format {{.Image}}", log)
        self.assertIn("top hermes-compaction-tier-b-runner pid,args", log)

    def test_liveness_rejects_malformed_and_mismatched_container_image(self):
        self._prepare_liveness_transaction()
        for observed in ("not-a-hash", "b" * 64):
            with self.subTest(observed=observed):
                result = self._run_helper("liveness", task_image=observed)
                self.assertEqual(result.returncode, 2)
                self.assertIn("TASK_CONTAINER_IMAGE_MISMATCH", result.stderr.decode())

    def test_task_label_paths_match_podman_object_schemas(self):
        helper = (
            REPO_ROOT / "evaluation/context_compression_tier_b_runtime.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("container) label_template='{{index .Config.Labels", helper)
        self.assertIn("image|network) label_template='{{index .Labels", helper)

    def test_image_id_canonicalization_is_shared_by_base_and_built_images(self):
        helper = (
            REPO_ROOT / "evaluation/context_compression_tier_b_runtime.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("canonical_image_id()", helper)
        self.assertIn(
            'canonical_image_id "$observed" BASE_IMAGE_IDENTITY_MISMATCH', helper
        )
        self.assertIn('canonical_image_id "$image_id" RUNTIME_IMAGE_ID_INVALID', helper)
        self.assertIn(
            'canonical_image_id "$container_image" TASK_CONTAINER_IMAGE_MISMATCH',
            helper,
        )
        self.assertEqual(helper.count('canonical_image_id "$'), 3)

    def test_all_production_queries_use_exact_name_filter(self):
        staged = self._stage()
        self.assertEqual(staged.returncode, 0, staged.stderr.decode())
        queries = []
        for line in self.podman_log.read_text(encoding="utf-8").splitlines():
            arguments = shlex.split(line)
            if arguments[:2] == ["ps", "-a"]:
                queries.append(arguments)
        self.assertGreaterEqual(len(queries), 2)
        for arguments in queries:
            filters = [
                arguments[index + 1]
                for index, value in enumerate(arguments[:-1])
                if value == "--filter"
            ]
            self.assertIn("--no-trunc", arguments)
            self.assertEqual(filters, ["name=^hermes$"])

    def test_stage_source_reconstructs_clean_exact_branch_and_head(self):
        result = self._stage()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        staged = self.task_root / "source-repo"
        self.assertTrue((staged / ".git").is_dir())
        self.assertEqual(
            subprocess.check_output(
                ["git", "-C", str(staged), "branch", "--show-current"], text=True
            ).strip(),
            "eval/compaction-tier-a",
        )
        self.assertEqual(
            subprocess.check_output(
                ["git", "-C", str(staged), "rev-parse", "HEAD"], text=True
            ).strip(),
            self.evaluation_head,
        )
        self.assertEqual(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(staged),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                text=True,
            ),
            "",
        )
        self.assertEqual(
            (self.task_root / "production-container-id-before.txt").read_text(
                encoding="ascii"
            ),
            self.PRODUCTION_ID + "\n",
        )
        self.assertFalse((self.task_root / "evaluation.bundle").exists())
        self.assertEqual(
            (staged / "evaluation/context_compression_tier_b_runtime.sh").read_bytes(),
            self.helper.read_bytes(),
        )

    def test_stage_source_rejects_invalid_bundle_bytes(self):
        cases = (
            ("truncated", self.bundle_bytes[:-1], self.bundle_sha256),
            ("extra", self.bundle_bytes + b"x", self.bundle_sha256),
            ("hash", self.bundle_bytes, "0" * 64),
        )
        for name, payload, sha256 in cases:
            with self.subTest(name=name):
                self._reset_task_root()
                result = self._stage(payload=payload, sha256=sha256)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.task_root / "source-repo").exists())

    def test_stage_source_rejects_wrong_branch_before_root(self):
        result = self._stage(branch="eval/wrong-branch")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"SOURCE_BRANCH_MISMATCH", result.stderr)
        self.assertFalse(self.task_root.exists())

    def test_stage_source_requires_one_exact_running_production_before_root(self):
        cases = (
            (
                "ambiguous",
                f"{self.PRODUCTION_ID}\thermes\trunning\n{'2' * 64}\thermes\trunning\n",
            ),
            ("not-running", f"{self.PRODUCTION_ID}\thermes\texited\n"),
        )
        for name, rows in cases:
            with self.subTest(name=name):
                self._reset_task_root()
                self._set_production_rows(rows)
                result = self._stage()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.task_root.exists())

    def test_production_id_change_blocks_prepare_before_runtime_creation(self):
        staged = self._stage()
        self.assertEqual(staged.returncode, 0, staged.stderr.decode())
        self._set_production_rows(f"{'3' * 64}\thermes\trunning\n")
        result = self._run_helper("prepare")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"PRODUCTION_IDENTITY_MISMATCH", result.stderr)
        self.assertFalse((self.task_root / "runtime").exists())

    def test_prepare_uses_locked_sync_then_networkless_evaluator_container(self):
        staged = self._stage()
        self.assertEqual(staged.returncode, 0, staged.stderr.decode())
        result = self._run_helper("prepare")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        runs = [
            shlex.split(line)
            for line in self.podman_log.read_text(encoding="utf-8").splitlines()
            if shlex.split(line)[:2] == ["run", "--rm"]
        ]
        self.assertEqual(len(runs), 2, runs)
        sync, prepare = runs
        for arguments in runs:
            self.assertIn("--pull=never", arguments)
            self.assertIn("--read-only", arguments)
            self.assertIn("--cap-drop=ALL", arguments)
            self.assertIn("--security-opt=no-new-privileges", arguments)
            self.assertIn("--userns=keep-id", arguments)
            self.assertIn("--label", arguments)
            self.assertIn("io.hermes.benchmark=context-compression-tier-b", arguments)
        self.assertIn("--name", sync)
        self.assertIn("hermes-compaction-tier-b-prepare-sync", sync)
        self.assertIn("--network=slirp4netns:allow_host_loopback=false", sync)
        self.assertIn("UV_PROJECT_ENVIRONMENT=/benchmark/prepare-venv", sync)
        self.assertTrue(
            any(value.endswith(",target=/benchmark/prepare-venv") for value in sync)
        )
        self.assertIn("uv sync --locked --python 3.13 --extra dev", " ".join(sync))
        self.assertIn("--no-install-project", sync)
        self.assertIn("--name", prepare)
        self.assertIn("hermes-compaction-tier-b-prepare-run", prepare)
        self.assertIn("--network=none", prepare)
        self.assertIn("--entrypoint=/benchmark/prepare-venv/bin/python", prepare)
        self.assertTrue(
            any(
                value.endswith(",target=/benchmark/prepare-venv,readonly")
                for value in prepare
            )
        )
        self.assertIn(
            "/benchmark/source/evaluation/context_compression_tier_b.py", prepare
        )
        self.assertNotIn(
            "OPENAI_API_KEY", " ".join(value for run in runs for value in run)
        )
        self.assertTrue((self.task_root / "prepare-venv").is_dir())
        ignorefile = self.task_root / "runtime/.tier-b-allow-all.containerignore"
        self.assertEqual(ignorefile.read_bytes(), b"!**\n")
        self.assertEqual(stat.S_IMODE(ignorefile.stat().st_mode), 0o400)
        self.assertEqual(
            (
                self.task_root / "runtime/input/production-container-id-before.txt"
            ).read_text(encoding="ascii"),
            self.PRODUCTION_ID + "\n",
        )


class RuntimeMetadataPathContractTests(unittest.TestCase):
    def _valid_payloads(self):
        mounts = [
            {
                "Destination": "/benchmark/input",
                "Type": "bind",
                "Source": "/home/ron/hermes-compaction-tier-b/runtime/input",
                "RW": False,
            },
            {
                "Destination": "/benchmark/output",
                "Type": "bind",
                "Source": "/home/ron/hermes-compaction-tier-b/runtime/output",
                "RW": True,
            },
        ]
        tmpfs = {
            "/benchmark/home": "rw,nosuid,nodev,size=268435456,mode=1777,rprivate,tmpcopyup",
            "/benchmark/hermes": "rw,nosuid,nodev,size=268435456,mode=1777,rprivate,tmpcopyup",
            "/benchmark/run": "rw,nosuid,nodev,size=2147483648,mode=1777,rprivate,tmpcopyup",
            "/tmp": "rw,nosuid,nodev,size=1073741824,mode=1777,rprivate,tmpcopyup",
            "/run": "rw,nosuid,nodev,size=67108864,mode=0755,rprivate,tmpcopyup",
        }
        environment = [
            "HOME=/benchmark/home/user",
            "HERMES_HOME=/benchmark/hermes/user",
            "HERMES_DISABLE_LAZY_INSTALLS=1",
            "PYTHONDONTWRITEBYTECODE=1",
            "NO_COLOR=1",
            f"TIER_B_LIVE_ACK={tier_b.LIVE_ACK}",
        ]
        return mounts, tmpfs, environment

    def _verify(self, mounts, tmpfs, environment):
        tier_b.verify_runtime_metadata(
            mounts_payload=json.dumps(mounts),
            tmpfs_payload=json.dumps(tmpfs),
            networks_payload=json.dumps({"hermes-compaction-tier-b-egress": {}}),
            ports_payload=json.dumps({}),
            environment_payload=json.dumps(environment),
            image_id="sha256:" + "9" * 64,
            runtime_user="1000:1000",
        )

    def test_runtime_metadata_accepts_split_bind_and_tmpfs_schemas(self):
        mounts, tmpfs, environment = self._valid_payloads()
        self._verify(mounts, tmpfs, environment)

    def test_runtime_metadata_rejects_wrong_bind_source_and_missing_tmpfs(self):
        mounts, tmpfs, environment = self._valid_payloads()
        mounts[0]["Source"] = "/home/ron/hermes-compaction-tier-b/input"
        with self.assertRaisesRegex(ValueError, "RUNTIME_MOUNT_MISMATCH"):
            self._verify(mounts, tmpfs, environment)
        mounts, tmpfs, environment = self._valid_payloads()
        del tmpfs["/run"]
        with self.assertRaisesRegex(ValueError, "RUNTIME_TMPFS_MISMATCH"):
            self._verify(mounts, tmpfs, environment)


class ExecutableFinalizationContractTests(unittest.TestCase):
    def test_real_cli_finalizes_validates_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            input_root = root / "input"
            final_root = root / "final"
            source_payloads = {
                "evaluation/context_compression_tier_b.py": b"evaluator bytes\n",
                "evaluation/fixtures/context-compression-tier-b.json": b"{}\n",
                "evaluation/context_compression_tier_b_runtime.sh": b"helper bytes\n",
                "evaluation/Containerfile.context-compression-tier-b": b"container bytes\n",
            }
            source_files = {}
            for relative, payload in source_payloads.items():
                path = source_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                source_files[relative] = {
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }

            schedule = {
                "schema_version": "context-compression-tier-b-schedule/1",
                "schedule_sha256": "a" * 64,
            }
            write_json(input_root / "execution-schedule.json", schedule)
            source_manifest = {
                "schema_version": "context-compression-tier-b-source-manifest/1",
                "files": source_files,
                "schedule_sha256": schedule["schedule_sha256"],
                "archives": {
                    label: {
                        "archive": {"sha256": character * 64},
                        "manifest": {"sha256": str(index) * 64},
                    }
                    for index, (label, character) in enumerate(
                        (
                            ("baseline", "b"),
                            ("candidate", "c"),
                            ("evaluation-head", "d"),
                        ),
                        start=1,
                    )
                },
            }
            write_json(input_root / "source-manifest.json", source_manifest)
            image_manifest = {
                "schema_version": "context-compression-tier-b-image-manifest/1",
                "source_manifest_sha256": hashlib.sha256(
                    (input_root / "source-manifest.json").read_bytes()
                ).hexdigest(),
                "schedule_sha256": schedule["schedule_sha256"],
                "installed_distributions_sha256": "e" * 64,
            }
            write_json(input_root / "image-manifest.json", image_manifest)
            for name, document in (
                (
                    "context-compression-tier-b-raw.json",
                    {
                        "schema_version": "context-compression-tier-b-raw/1",
                        "status": "incomplete",
                    },
                ),
                (
                    "context-compression-tier-b-scores.json",
                    {
                        "schema_version": "context-compression-tier-b-scores/1",
                        "status": "TIER_B_REPORT_INCOMPLETE",
                    },
                ),
                (
                    "context-compression-tier-b-comparison.json",
                    {
                        "schema_version": "context-compression-tier-b-comparison/1",
                        "status": "TIER_B_REPORT_INCOMPLETE",
                    },
                ),
            ):
                write_json(final_root / name, document)
            report_path = final_root / "context-compression-tier-b-report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                "Status: `TIER_B_REPORT_INCOMPLETE`\n", encoding="utf-8"
            )

            command = [
                sys.executable,
                str(REPO_ROOT / "evaluation/context_compression_tier_b.py"),
                "finalize",
                "--source-root",
                str(source_root),
                "--input-root",
                str(input_root),
                "--final-root",
                str(final_root),
            ]
            finalized = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(finalized.returncode, 0, finalized.stderr)
            self.assertEqual(
                (final_root / "FINALIZED").read_text(encoding="ascii"),
                "TIER_B_REPORT_INCOMPLETE\n",
            )
            checksum_lines = (
                (final_root / "FINALIZED.sha256")
                .read_text(encoding="ascii")
                .splitlines()
            )
            names = [line.split("  ", 1)[1] for line in checksum_lines]
            self.assertEqual(names, sorted(names))
            self.assertNotIn("FINALIZED.sha256", names)
            self.assertEqual(
                set(names),
                {
                    *source_payloads,
                    "source-manifest.json",
                    "image-manifest.json",
                    "execution-schedule.json",
                    "context-compression-tier-b-raw.json",
                    "context-compression-tier-b-scores.json",
                    "context-compression-tier-b-comparison.json",
                    "context-compression-tier-b-report.md",
                    "FINALIZED",
                },
            )

            validated = subprocess.run(
                [*command[:2], "validate-finalized", *command[3:]],
                capture_output=True,
                text=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            report_path.write_text("tampered\n", encoding="utf-8")
            tampered = subprocess.run(
                [*command[:2], "validate-finalized", *command[3:]],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(tampered.returncode, 0)
            self.assertNotIn("tampered", tampered.stderr)


class RuntimeArtifactAttestationContractTests(unittest.TestCase):
    def setUp(self):
        self._original_path_read_text = Path.read_text

    def _mountinfo_reader(self):
        original = self._original_path_read_text

        def safe_read(path, *args, **kwargs):
            if Path(path) == Path("/proc/self/mountinfo"):
                return "provider-free synthetic mountinfo\n"
            return original(path, *args, **kwargs)

        return safe_read

    def _attest(self, fixture):
        with (
            patch.dict(os.environ, fixture.environment, clear=True),
            patch.object(Path, "read_text", new=self._mountinfo_reader()),
        ):
            return tier_b.attest_runtime(
                fixture.manifest_path,
                fixture.image_manifest_path,
                fixture.schedule_path,
                fixture.run_root,
                fixture.output_root,
            )

    def test_real_attestation_rejects_executed_tree_and_distribution_drift(self):
        for drift in ("executed_tree", "installed_distribution"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as tmp:
                fixture = make_runtime_attestation_fixture(Path(tmp))
                baseline = self._attest(fixture)
                self.assertTrue(baseline["preflight_passed"])
                if drift == "executed_tree":
                    path = (
                        fixture.input_root
                        / "trees/baseline/agent/context_compressor.py"
                    )
                    path.write_bytes(path.read_bytes() + b"drift\n")
                else:
                    image_manifest = json.loads(
                        fixture.image_manifest_path.read_text(encoding="utf-8")
                    )
                    image_manifest["installed_distributions"].append([
                        "synthetic-drift",
                        "1",
                    ])
                    image_manifest["installed_distributions"].sort()
                    image_manifest["installed_distributions_sha256"] = hashlib.sha256(
                        canonical(image_manifest["installed_distributions"]).encode()
                    ).hexdigest()
                    write_json(fixture.image_manifest_path, image_manifest)
                with self.assertRaises(ValueError):
                    self._attest(fixture)
                self.assertFalse((fixture.run_root / "auth-attestation.json").exists())
                self.assertFalse(
                    (fixture.output_root / "collection-journal.jsonl").exists()
                )

    def test_collect_rechecks_artifacts_before_auth_read_or_child_wire(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_runtime_attestation_fixture(Path(tmp))
            attested = self._attest(fixture)
            self.assertTrue(attested["preflight_passed"])
            write_json(
                fixture.run_root / "auth-attestation.json",
                {
                    "schema_version": "context-compression-tier-b-auth-attestation/1",
                    "status": "usable",
                    "provider": "openai-codex",
                    "pool_entry_count": 1,
                    "source_class": "manual:device_code",
                    "account_fingerprint_sha256": "8" * 64,
                },
            )
            executed = (
                fixture.input_root / "trees/candidate/agent/context_compressor.py"
            )
            executed.write_bytes(executed.read_bytes() + b"drift\n")
            original_load = tier_b._load_json
            auth_reads = 0

            def counted_load(path, error_code):
                nonlocal auth_reads
                if Path(path).name == "auth-attestation.json":
                    auth_reads += 1
                return original_load(path, error_code)

            with (
                patch.dict(os.environ, fixture.environment, clear=True),
                patch.object(Path, "read_text", new=self._mountinfo_reader()),
                patch.object(tier_b, "_load_json", side_effect=counted_load),
                patch.object(
                    tier_b.subprocess,
                    "run",
                    side_effect=AssertionError("child wire reached"),
                ) as child_wire,
                self.assertRaises(tier_b.TerminalBenchmarkStop) as stopped,
            ):
                tier_b.collect_benchmark(
                    fixture.manifest_path,
                    fixture.image_manifest_path,
                    fixture.schedule_path,
                    fixture.run_root,
                    fixture.output_root,
                    resume_attempt_id=None,
                )
            self.assertEqual(stopped.exception.code, "RUNTIME_PREFLIGHT_FAILED")
            self.assertEqual(auth_reads, 0)
            child_wire.assert_not_called()


class FixtureAndScheduleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tier_b.load_fixture(FIXTURE_PATH)
        cls.fixture_sha256 = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()

    def test_fixture_expansion_and_declared_dependency_proof(self):
        validation = tier_b.validate_fixture(self.fixture)
        self.assertEqual(validation["scenario_count"], 3)
        self.assertEqual(validation["source_row_count_per_scenario"], 12)
        self.assertEqual(validation["growth_rows_per_cycle"], 6)
        self.assertEqual(
            validation["initial_dependency_counts"], {"raw_tail": 7, "summary": 23}
        )
        for scenario_id in self.fixture["scenario_order"]:
            expanded = tier_b.expand_scenario(self.fixture, scenario_id, repeat=2)
            self.assertEqual(len(expanded["rows"]), 13)
            self.assertEqual(len(expanded["rows"][0]["content"]), 200)
            self.assertEqual(
                [row["role"] for row in expanded["rows"][1:]],
                self.fixture["corpus"]["source_roles"],
            )
            for row in expanded["rows"][1:]:
                expected = 100 if row["role"] == "tool" else 2900
                self.assertEqual(len(row["content"]), expected)
            transcript = canonical(expanded["rows"])
            self.assertEqual(transcript.count("TBNONCE-R2-9P6V3D"), 1)
            for fact in expanded["facts"]:
                row = next(
                    item
                    for item in expanded["rows"]
                    if item.get("row_id") == fact["row_id"]
                )
                self.assertIn(fact["sentence"], row["content"])
            growth = tier_b.expand_growth_rows(self.fixture, scenario_id, cycle=8)
            self.assertEqual(len(growth), 6)
            self.assertEqual([row["role"] for row in growth], ["user", "assistant"] * 3)
            self.assertTrue(all(len(row["content"]) == 900 for row in growth))

    def test_exact_interleaving_and_call_arithmetic(self):
        schedule = tier_b.build_execution_schedule(
            self.fixture, fixture_sha256=self.fixture_sha256
        )
        self.assertEqual(
            schedule["schema_version"], "context-compression-tier-b-schedule/1"
        )
        self.assertEqual(schedule["expected_counts"]["trajectories"], 18)
        self.assertEqual(schedule["expected_counts"]["summary"], 162)
        self.assertEqual(schedule["expected_counts"]["downstream"], 90)
        self.assertEqual(schedule["expected_counts"]["total"], 252)
        self.assertEqual(len(schedule["calls"]), 252)
        self.assertEqual(
            [
                (item["scenario_id"], item["repeat"], item["revision_label"])
                for item in schedule["trajectories"]
            ],
            [
                ("S1_DEPLOYMENT", 1, "baseline"),
                ("S1_DEPLOYMENT", 1, "candidate"),
                ("S2_INCIDENT", 1, "candidate"),
                ("S2_INCIDENT", 1, "baseline"),
                ("S3_REPOSITORY", 1, "baseline"),
                ("S3_REPOSITORY", 1, "candidate"),
                ("S1_DEPLOYMENT", 2, "candidate"),
                ("S1_DEPLOYMENT", 2, "baseline"),
                ("S2_INCIDENT", 2, "baseline"),
                ("S2_INCIDENT", 2, "candidate"),
                ("S3_REPOSITORY", 2, "candidate"),
                ("S3_REPOSITORY", 2, "baseline"),
                ("S1_DEPLOYMENT", 3, "baseline"),
                ("S1_DEPLOYMENT", 3, "candidate"),
                ("S2_INCIDENT", 3, "candidate"),
                ("S2_INCIDENT", 3, "baseline"),
                ("S3_REPOSITORY", 3, "baseline"),
                ("S3_REPOSITORY", 3, "candidate"),
            ],
        )
        counts = {}
        for call in schedule["calls"]:
            key = call["trajectory_id"]
            counts.setdefault(key, {"summary": 0, "downstream": 0})
            counts[key][call["call_role"]] += 1
        for trajectory in schedule["trajectories"]:
            expected_summary = 8 if trajectory["revision_label"] == "baseline" else 10
            self.assertEqual(
                counts[trajectory["trajectory_id"]],
                {"summary": expected_summary, "downstream": 5},
            )


class PureScoringContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tier_b.load_fixture(FIXTURE_PATH)

    def test_perfect_strict_json_vector(self):
        answer = perfect_answer(self.fixture, "S2_INCIDENT")
        result = tier_b.normalize_and_score_answer(
            self.fixture,
            scenario_id="S2_INCIDENT",
            repeat=1,
            response_bytes=canonical(answer).encode(),
        )
        self.assertTrue(result["valid"])
        self.assertEqual(
            result["metrics"],
            {
                "applicability_correct_numerator": 13,
                "applicability_total_denominator": 13,
                "required_state_correct_numerator": 8,
                "required_state_applicable_denominator": 8,
                "identifier_provenance_correct_numerator": 4,
                "identifier_provenance_applicable_denominator": 4,
                "stale_resurrection_count": 0,
                "contradiction_count": 0,
                "unsafe_prohibited_action_count": 0,
                "explicit_uncertainty_count": 0,
                "explicit_omission_count": 0,
                "wrong_known_value_count": 0,
                "task_format_valid": 1,
                "trial_nonce_leakage_count": 0,
            },
        )

    def test_adverse_vector_categories_are_exact(self):
        answer = perfect_answer(self.fixture, "S2_INCIDENT")
        answer["required_state"]["current_root_cause"]["value"] = (
            "DATABASE_POOL_EXHAUSTION"
        )
        answer["required_state"]["current_mitigation"] = {
            "applicable": True,
            "status": "unknown",
            "value": None,
        }
        answer["identifiers"]["incident_id"]["value"] = "TBNONCE-R1-7K4M2Q"
        answer["identifiers"]["error_id"]["value"] = "SAFE_WRONG"
        answer["identifiers"]["trace_id"] = {
            "applicable": True,
            "status": "omitted",
            "value": None,
        }
        answer["recommended_next_action"]["value"] = "RESTART_GATEWAY"
        result = tier_b.normalize_and_score_answer(
            self.fixture,
            scenario_id="S2_INCIDENT",
            repeat=1,
            response_bytes=canonical(answer).encode(),
        )
        self.assertTrue(result["valid"])
        metrics = result["metrics"]
        self.assertEqual(metrics["applicability_correct_numerator"], 11)
        self.assertEqual(metrics["applicability_total_denominator"], 13)
        self.assertEqual(metrics["required_state_correct_numerator"], 6)
        self.assertEqual(metrics["required_state_applicable_denominator"], 8)
        self.assertEqual(metrics["identifier_provenance_correct_numerator"], 1)
        self.assertEqual(metrics["identifier_provenance_applicable_denominator"], 4)
        self.assertEqual(metrics["stale_resurrection_count"], 2)
        self.assertEqual(metrics["contradiction_count"], 2)
        self.assertEqual(metrics["unsafe_prohibited_action_count"], 1)
        self.assertEqual(metrics["explicit_uncertainty_count"], 1)
        self.assertEqual(metrics["explicit_omission_count"], 1)
        self.assertEqual(metrics["wrong_known_value_count"], 1)
        self.assertEqual(metrics["task_format_valid"], 1)
        self.assertEqual(metrics["trial_nonce_leakage_count"], 1)

    def test_malformed_and_unsafe_output_is_hash_only(self):
        unsafe = b"not-json sk-proj-SYNTHETIC-DO-NOT-PERSIST TBNONCE-R1-7K4M2Q"
        result = tier_b.normalize_and_score_answer(
            self.fixture,
            scenario_id="S1_DEPLOYMENT",
            repeat=1,
            response_bytes=unsafe,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(
            set(result),
            {"error_codes", "response_bytes", "response_sha256", "valid", "metrics"},
        )
        persisted = canonical(result)
        self.assertNotIn("not-json", persisted)
        self.assertNotIn("sk-proj", persisted)
        self.assertEqual(result["response_bytes"], len(unsafe))
        self.assertEqual(result["metrics"]["trial_nonce_leakage_count"], 1)

        answer = perfect_answer(self.fixture, "S1_DEPLOYMENT")
        answer["required_state"]["deployment_state"]["value"] = (
            "unsafe value with spaces"
        )
        result = tier_b.normalize_and_score_answer(
            self.fixture,
            scenario_id="S1_DEPLOYMENT",
            repeat=1,
            response_bytes=canonical(answer).encode(),
        )
        self.assertFalse(result["valid"])
        self.assertIn("UNSAFE_OBSERVED_VALUE", result["error_codes"])
        self.assertNotIn("unsafe value with spaces", canonical(result))

    def test_score_and_compare_are_total_and_fail_closed(self):
        malformed = tier_b.score_raw_evidence(b"{", self.fixture)
        self.assertEqual(malformed["status"], "TIER_B_REPORT_INCOMPLETE")
        self.assertEqual(malformed["error_codes"], ["RAW_EVIDENCE_MALFORMED"])

        answer = perfect_answer(self.fixture, "S1_DEPLOYMENT")
        normalized = tier_b.normalize_and_score_answer(
            self.fixture,
            scenario_id="S1_DEPLOYMENT",
            repeat=1,
            response_bytes=canonical(answer).encode(),
        )
        key = {
            "fixture_sha256": "f" * 64,
            "scenario_id": "S1_DEPLOYMENT",
            "repeat": 1,
            "revision_sha": self.fixture["identity"]["baseline_revision"],
            "cycle": 0,
            "question_id": "S1_CURRENT_DEPLOYMENT_STATE",
        }
        sample = {
            "sampling_key": key,
            "sampling_key_sha256": hashlib.sha256(canonical(key).encode()).hexdigest(),
            "score_identity_sha256": "a" * 64,
            "comparison_identity_sha256": "b" * 64,
            **normalized,
        }
        raw = {
            "schema_version": "context-compression-tier-b-raw/1",
            "status": "incomplete",
            "samples": [sample, sample],
        }
        duplicate = tier_b.score_raw_evidence(canonical(raw).encode(), self.fixture)
        self.assertIn("DUPLICATE_SAMPLE_KEY", duplicate["error_codes"])
        self.assertEqual(duplicate["status"], "TIER_B_REPORT_INCOMPLETE")

        invalid = dict(sample)
        invalid["valid"] = False
        invalid["metrics"] = {
            "task_format_valid": 0,
            "trial_nonce_leakage_count": 0,
        }
        raw["samples"] = [invalid]
        scored = tier_b.score_raw_evidence(canonical(raw).encode(), self.fixture)
        self.assertEqual(scored["quality_totals"]["valid_samples"], 0)
        self.assertEqual(
            scored["quality_totals"]["required_state_applicable_denominator"], 0
        )

        baseline = tier_b.score_sample_for_comparison(
            sample,
            revision_label="baseline",
            revision_hashes={"archive": "1", "tree": "2", "blob": "3"},
        )
        candidate = tier_b.score_sample_for_comparison(
            sample,
            revision_label="candidate",
            revision_hashes={"archive": "4", "tree": "5", "blob": "6"},
        )
        candidate["comparison_identity_sha256"] = "c" * 64
        compared = tier_b.compare_score_document({
            "schema_version": "context-compression-tier-b-scores/1",
            "status": "TIER_B_REPORT_INCOMPLETE",
            "samples": [baseline, candidate],
        })
        self.assertEqual(compared["status"], "TIER_B_REPORT_INCOMPLETE")
        self.assertEqual(compared["pairs"][0]["status"], "NOT_COMPARABLE")
        self.assertIsNone(compared["pairs"][0]["delta"])

        missing = tier_b.compare_score_document({
            "schema_version": "context-compression-tier-b-scores/1",
            "status": "TIER_B_REPORT_INCOMPLETE",
            "samples": [baseline],
        })
        self.assertIn("MISSING_COMPARISON_SAMPLE", missing["error_codes"])
        self.assertTrue(all(item["delta"] is None for item in missing["pairs"]))

        for payload in (b"", b"[", b"null", b"[]"):
            result = tier_b.compare_score_bytes(payload)
            self.assertEqual(result["status"], "TIER_B_REPORT_INCOMPLETE")

        nested_bad_raw = {
            "schema_version": "context-compression-tier-b-raw/1",
            "status": "complete",
            "samples": [
                {
                    "sampling_key_sha256": "a" * 64,
                    "valid": True,
                    "metrics": {},
                }
            ],
            "calls": [0],
            "attempt_totals": {},
        }
        nested_bad_score = tier_b.score_raw_evidence(
            canonical(nested_bad_raw).encode(), self.fixture
        )
        self.assertEqual(nested_bad_score["status"], "TIER_B_REPORT_INCOMPLETE")
        nested_bad_comparison = tier_b.compare_score_document({
            "schema_version": "context-compression-tier-b-scores/1",
            "status": "TIER_B_REPORT_INCOMPLETE",
            "samples": [{"revision_label": "baseline", "sampling_key": 7}],
        })
        self.assertEqual(nested_bad_comparison["status"], "TIER_B_REPORT_INCOMPLETE")

    def test_nearest_rank_and_repeat_order(self):
        self.assertEqual(tier_b.nearest_rank([1, 2, 3, 4, 100], 50), 3)
        self.assertEqual(tier_b.nearest_rank([1, 2, 3, 4, 100], 95), 100)
        self.assertEqual(
            tier_b.integer_distribution([4, 1, 100], repeat_values=[4, 1, 100]),
            {
                "status": "available",
                "minimum": 1,
                "p50": 4,
                "p95": 100,
                "maximum": 100,
                "repeat_values": [4, 1, 100],
            },
        )
        self.assertEqual(
            tier_b.integer_distribution([], repeat_values=[]),
            {"status": "not_applicable", "repeat_values": []},
        )

    def test_required_cell_macro_latency_telemetry_and_report_aggregation(self):
        fixture_sha = "f" * 64
        schedule = tier_b.build_execution_schedule(self.fixture, fixture_sha)
        samples = []
        calls = []
        count_fields = {
            "contradiction_count": 0,
            "unsafe_prohibited_action_count": 0,
            "explicit_uncertainty_count": 0,
            "explicit_omission_count": 0,
            "wrong_known_value_count": 0,
            "task_format_valid": 1,
            "trial_nonce_leakage_count": 0,
            "blocked_retry_count": 0,
            "blocked_fallback_count": 0,
        }
        summary_counts = {}
        for call in schedule["calls"]:
            key = (
                call["scenario_id"],
                call["repeat"],
                call["revision_label"],
                call["cycle"],
            )
            if call["call_role"] == "summary":
                summary_counts[key] = summary_counts.get(key, 0) + 1
            candidate_offset = 5 if call["revision_label"] == "candidate" else 0
            ordinal = call.get("summary_pass_ordinal") or 0
            call_record = {
                **{
                    field: call[field]
                    for field in (
                        "logical_call_id",
                        "schedule_position",
                        "call_role",
                        "scenario_id",
                        "repeat",
                        "revision_sha",
                        "cycle",
                    )
                },
                "revision_label": call["revision_label"],
                "trajectory_id": call["trajectory_id"],
                "duration_ns": call["repeat"]
                * (10 if call["call_role"] == "summary" else 20)
                + candidate_offset
                + ordinal,
                "transaction_duration_ns": call["repeat"]
                * (100 if call["call_role"] == "summary" else 200)
                + candidate_offset * 2,
                "trajectory_duration_ns": call["repeat"] * 1000 + candidate_offset * 20,
                "input_tokens": call["repeat"] * 100 + ordinal + candidate_offset,
                "input_tokens_status": "available",
                "output_tokens": None,
                "output_tokens_status": "not_exposed",
                "total_tokens": call["repeat"] * 120 + ordinal + candidate_offset,
                "total_tokens_status": "available",
                "cache_read_tokens": "not_exposed_by_adapter",
                "cache_read_tokens_status": "not_exposed",
                "cache_miss_tokens": "not_exposed_by_adapter",
                "cache_miss_tokens_status": "not_exposed",
                "quota": "not_available",
                "quota_status": "not_available",
                "monetary_cost": "not_available",
                "monetary_cost_status": "not_available",
            }
            calls.append(call_record)

        revision_hashes = {
            "baseline": {"archive": "1", "tree": "2", "blob": "3"},
            "candidate": {"archive": "4", "tree": "5", "blob": "6"},
        }
        for call in schedule["calls"]:
            if call["call_role"] != "downstream":
                continue
            repeat = call["repeat"]
            label = call["revision_label"]
            metrics = {
                "applicability_correct_numerator": repeat + int(label == "candidate"),
                "applicability_total_denominator": 4,
                "required_state_correct_numerator": 2,
                "required_state_applicable_denominator": 4,
                "identifier_provenance_correct_numerator": 1,
                "identifier_provenance_applicable_denominator": 2,
                "stale_resurrection_count": repeat - 1 + int(label == "candidate"),
                "summary_provider_call_count": summary_counts.get(
                    (
                        call["scenario_id"],
                        repeat,
                        label,
                        call["cycle"],
                    ),
                    0,
                ),
                "downstream_provider_call_count": 1,
                **count_fields,
            }
            comparison_identity = hashlib.sha256(
                canonical({
                    "scenario_id": call["scenario_id"],
                    "repeat": repeat,
                    "cycle": call["cycle"],
                    "question_id": call["question_id"],
                }).encode()
            ).hexdigest()
            sample = {
                "valid": True,
                "sampling_key": call["sampling_key"],
                "sampling_key_sha256": call["sampling_key_sha256"],
                "scenario_id": call["scenario_id"],
                "repeat": repeat,
                "cycle": call["cycle"],
                "question_id": call["question_id"],
                "comparison_identity_sha256": comparison_identity,
                "metrics": metrics,
                "error_codes": [],
            }
            samples.append(
                tier_b.score_sample_for_comparison(
                    sample,
                    revision_label=label,
                    revision_hashes=revision_hashes[label],
                )
            )

        raw = {
            "schema_version": "context-compression-tier-b-raw/1",
            "status": "complete",
            "attempt_totals": {
                "actual": 252,
                "expected": 252,
                "summary": 162,
                "downstream": 90,
                "blocked_retry": 0,
                "blocked_fallback": 0,
            },
            "calls": calls,
            "samples": list(reversed(samples)),
            "quota": {
                "status": "not_available",
                "reason": "CODEX_ADAPTER_EXPOSES_NO_QUOTA",
            },
            "cost": {
                "status": "not_available",
                "reason": "SUBSCRIPTION_OAUTH_NO_PROVIDER_COST",
            },
        }
        scores = tier_b.score_raw_evidence(canonical(raw).encode(), self.fixture)
        self.assertEqual(scores["status"], "TIER_B_REPORT_COMPLETE")
        self.assertEqual(len(scores["cells"]), 30)
        self.assertEqual(len(scores["revisions"]), 2)
        first = scores["samples"][0]
        self.assertEqual(
            (
                first["revision_label"],
                first["scenario_id"],
                first["cycle"],
                first["repeat"],
                first["question_id"],
            ),
            ("baseline", "S1_DEPLOYMENT", 0, 1, "S1_CURRENT_DEPLOYMENT_STATE"),
        )
        baseline_cell = next(
            cell
            for cell in scores["cells"]
            if cell["revision_label"] == "baseline"
            and cell["scenario_id"] == "S1_DEPLOYMENT"
            and cell["cycle"] == 1
        )
        self.assertEqual(
            baseline_cell["quality"]["applicability"],
            {"numerator": 6, "denominator": 12},
        )
        self.assertEqual(
            baseline_cell["count_metrics"]["stale_resurrection_count"],
            {"sum": 3, "repeat_values": [0, 1, 2]},
        )
        self.assertEqual(
            baseline_cell["latency"]["luna_call"],
            {
                "status": "available",
                "minimum": 11,
                "p50": 21,
                "p95": 31,
                "maximum": 31,
                "repeat_values": [11, 21, 31],
            },
        )
        self.assertEqual(
            baseline_cell["latency"]["compaction_transaction"]["repeat_values"],
            [100, 200, 300],
        )
        self.assertEqual(
            baseline_cell["telemetry"]["output_tokens"],
            {
                "distribution": {
                    "status": "not_applicable",
                    "repeat_values": [],
                },
                "status_counts": {"not_exposed": 6},
            },
        )
        baseline_revision = next(
            item for item in scores["revisions"] if item["revision_label"] == "baseline"
        )
        candidate_revision = next(
            item
            for item in scores["revisions"]
            if item["revision_label"] == "candidate"
        )
        self.assertEqual(
            baseline_revision["macro_quality"]["applicability"],
            {
                "numerator": 1,
                "denominator": 2,
                "complete_cell_count": 15,
            },
        )
        self.assertEqual(
            baseline_revision["pooled_quality"]["applicability"],
            {"numerator": 90, "denominator": 180},
        )
        self.assertEqual(
            baseline_revision["latency"]["full_trajectory"],
            {
                "status": "available",
                "minimum": 1000,
                "p50": 2000,
                "p95": 3000,
                "maximum": 3000,
                "repeat_values": [1000, 2000, 3000],
            },
        )
        self.assertEqual(
            baseline_revision["telemetry"]["output_tokens"]["status_counts"],
            {"not_exposed": 117},
        )
        self.assertEqual(
            candidate_revision["macro_quality"]["applicability"]["numerator"], 3
        )
        self.assertEqual(
            candidate_revision["macro_quality"]["applicability"]["denominator"],
            4,
        )

        compared = tier_b.compare_score_document(scores)
        self.assertEqual(compared["status"], "TIER_B_REPORT_COMPLETE")
        self.assertEqual(compared["delta_direction"], "candidate - baseline")
        delta = next(
            item
            for item in compared["cell_deltas"]
            if item["scenario_id"] == "S1_DEPLOYMENT" and item["cycle"] == 1
        )
        self.assertEqual(
            delta["quality"]["applicability"]["ratio_delta"],
            {"numerator": 1, "denominator": 4},
        )
        self.assertEqual(delta["count_metrics"]["stale_resurrection_count"], 3)
        self.assertEqual(delta["latency"]["luna_call"]["p50"], 6)
        self.assertEqual(
            compared["revision_deltas"]["macro_quality"]["applicability"],
            {"numerator": 1, "denominator": 4},
        )
        self.assertEqual(
            compared["revision_deltas"]["latency"]["full_trajectory"]["p50"],
            100,
        )
        report = tier_b._render_report(compared)
        self.assertIn("## Sample vectors", report)
        self.assertIn("## Revision macro quality", report)
        self.assertIn("## Revision latency and telemetry", report)
        self.assertIn("## Candidate - baseline cell deltas", report)
        self.assertIn(
            "| baseline | S1_DEPLOYMENT | 0 | 1 | S1_CURRENT_DEPLOYMENT_STATE |",
            report,
        )


class GuardResumeAndFinalizationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tier_b.load_fixture(FIXTURE_PATH)
        cls.schedule = tier_b.build_execution_schedule(cls.fixture, "f" * 64)

    def _installed_guard(
        self,
        call,
        *,
        global_actual_attempts=0,
        attempted=(),
        now_ns=lambda: 1,
        deadline_ns=100,
    ):
        auxiliary, compressor, counters, state = fake_live_guard_modules()
        journal = []
        guard = tier_b.LiveAuxiliaryGuard(
            [call],
            auxiliary_module=auxiliary,
            compressor_module=compressor,
            global_actual_attempts=global_actual_attempts,
            attempted_logical_call_ids=set(attempted),
            append_journal=journal.append,
            trajectory_started_ns=0,
            trajectory_deadline_ns=deadline_ns,
            monotonic_ns=now_ns,
        )
        guard.install()
        return guard, auxiliary, compressor, counters, state, journal

    def test_live_guard_instance_call_reaches_wire_once_and_records_attempt(self):
        call = next(
            item for item in self.schedule["calls"] if item["call_role"] == "summary"
        )
        guard, auxiliary, _compressor, counters, _state, journal = (
            self._installed_guard(call)
        )
        with guard.activate(call, snapshot_sha256="a" * 64):
            response = auxiliary.call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.6-luna",
                api_mode="codex_responses",
                timeout=300,
                messages=[{"role": "user", "content": "synthetic"}],
            )
        self.assertEqual(response.model, "gpt-5.6-luna")
        self.assertEqual(counters, {"wire": 1, "main": 0})
        self.assertEqual(guard.actual_attempts, 1)
        self.assertEqual(guard.attempted_logical_call_ids, {call["logical_call_id"]})
        self.assertEqual(
            guard.observed_routes,
            [("openai-codex", "codex_responses", "gpt-5.6-luna")],
        )
        self.assertEqual([item["state"] for item in journal], ["PLANNED", "COMPLETED"])
        self.assertTrue(
            all(item["logical_call_id"] == call["logical_call_id"] for item in journal)
        )

    def test_live_guard_blocks_every_frozen_escape_before_fake_wires(self):
        call = next(
            item for item in self.schedule["calls"] if item["call_role"] == "summary"
        )
        cases = [
            ("unscheduled", "unscheduled", {}, "UNSCHEDULED_LOGICAL_CALL"),
            ("provider", "aux", {"provider": "auto"}, "FALLBACK_ROUTE_BLOCKED"),
            ("model", "aux", {"model": "gpt-5.6-sol"}, "FALLBACK_ROUTE_BLOCKED"),
            (
                "api_mode",
                "aux",
                {"api_mode": "chat_completions"},
                "FALLBACK_ROUTE_BLOCKED",
            ),
            (
                "payment_fallback",
                "aux_state",
                {"fallback_kind": "payment"},
                "FALLBACK_ROUTE_BLOCKED",
            ),
            (
                "provider_fallback",
                "aux_state",
                {"fallback_kind": "provider"},
                "FALLBACK_ROUTE_BLOCKED",
            ),
            (
                "sdk_retry_drift",
                "aux_state",
                {"max_retries": 1},
                "WIRE_REQUEST_IDENTITY_MISMATCH",
            ),
            (
                "wrong_adapter",
                "aux_state",
                {"wrong_adapter": True},
                "ROUTE_MISMATCH",
            ),
            (
                "wrong_base_url",
                "aux_state",
                {"base_url": ("https://example.invalid/chatgpt.com/backend-api/codex")},
                "ROUTE_MISMATCH",
            ),
            (
                "forbidden_request_knob",
                "aux_state",
                {"request_overrides": {"temperature": 0}},
                "WIRE_REQUEST_IDENTITY_MISMATCH",
            ),
            ("attempt_cap", "cap", {}, "GLOBAL_ATTEMPT_CAP_REACHED"),
            ("trajectory_deadline", "deadline", {}, "TRAJECTORY_DEADLINE_REACHED"),
            ("compressor_fallback", "compressor", {}, "FALLBACK_TO_MAIN_BLOCKED"),
            ("main_codex_stream", "main_stream", {}, "MAIN_AGENT_TRANSPORT_BLOCKED"),
            (
                "main_responses_create",
                "main_responses",
                {},
                "MAIN_AGENT_TRANSPORT_BLOCKED",
            ),
        ]
        for name, operation, overrides, code in cases:
            with self.subTest(name=name):
                guard, auxiliary, compressor, counters, state, _journal = (
                    self._installed_guard(
                        call,
                        global_actual_attempts=300 if operation == "cap" else 0,
                        now_ns=(lambda: 100)
                        if operation == "deadline"
                        else (lambda: 1),
                    )
                )
                if operation == "aux_state":
                    state.update(overrides)
                selected_call = (
                    {**call, "logical_call_id": "not-scheduled"}
                    if operation == "unscheduled"
                    else call
                )
                with self.assertRaises(tier_b.TerminalBenchmarkStop) as caught:
                    if operation == "compressor":
                        compressor.ContextCompressor()._fallback_to_main_for_compression(
                            RuntimeError("synthetic"), "synthetic"
                        )
                    elif operation == "main_stream":
                        auxiliary.run_codex_stream()
                    elif operation == "main_responses":
                        auxiliary.main_responses_create()
                    else:
                        with guard.activate(selected_call, snapshot_sha256="a" * 64):
                            auxiliary.call_llm(
                                task="compression",
                                provider=overrides.get("provider", "openai-codex"),
                                model=overrides.get("model", "gpt-5.6-luna"),
                                api_mode=overrides.get("api_mode", "codex_responses"),
                                timeout=300,
                                messages=[{"role": "user", "content": "synthetic"}],
                            )
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(counters, {"wire": 0, "main": 0})

        guard, auxiliary, _compressor, counters, _state, _journal = (
            self._installed_guard(call)
        )
        with guard.activate(call, snapshot_sha256="a" * 64):
            auxiliary.call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.6-luna",
                api_mode="codex_responses",
                timeout=300,
                messages=[],
            )
        with self.assertRaises(tier_b.TerminalBenchmarkStop) as caught:
            with guard.activate(call, snapshot_sha256="a" * 64):
                auxiliary.call_llm(
                    task="compression",
                    provider="openai-codex",
                    model="gpt-5.6-luna",
                    api_mode="codex_responses",
                    timeout=300,
                    messages=[],
                )
        self.assertEqual(caught.exception.code, "HIDDEN_RETRY_BLOCKED")
        self.assertEqual(counters, {"wire": 1, "main": 0})

    def test_resume_never_resubmits_uncertain_or_lost_state(self):
        calls = self.schedule["calls"][:4]
        completed = calls[0]
        uncertain = calls[1]
        intact_missing = calls[2]
        lost_missing = calls[3]
        journal = [
            {"logical_call_id": completed["logical_call_id"], "state": "PLANNED"},
            {"logical_call_id": completed["logical_call_id"], "state": "COMPLETED"},
            {"logical_call_id": uncertain["logical_call_id"], "state": "PLANNED"},
        ]
        resume = tier_b.plan_resume(
            calls,
            journal,
            available_snapshot_ids={intact_missing["snapshot_id"]},
        )
        self.assertEqual(
            [item["logical_call_id"] for item in resume["execute_once"]],
            [intact_missing["logical_call_id"]],
        )
        self.assertEqual(resume["skipped_completed"], [completed["logical_call_id"]])
        self.assertIn(
            {
                "logical_call_id": uncertain["logical_call_id"],
                "error_code": "OUTCOME_UNKNOWN",
            },
            resume["invalid"],
        )
        self.assertIn(
            {
                "logical_call_id": lost_missing["logical_call_id"],
                "error_code": "SNAPSHOT_STATE_LOST",
            },
            resume["invalid"],
        )

    def test_safe_observed_values_are_hash_only(self):
        self.assertRegex(
            tier_b.safe_observed_value("unknown free form sk-proj-SYNTHETIC"),
            r"^sha256:[0-9a-f]{64}$",
        )

    def test_all_terminal_handlers_write_sanitized_durable_partial_raw(self):
        cases = (
            (
                lambda: tier_b.TerminalBenchmarkStop("ROUTE_MISMATCH"),
                "ROUTE_MISMATCH",
                2,
            ),
            (KeyboardInterrupt, "COLLECT_CANCELLED", 130),
            (
                lambda: RuntimeError(
                    "sk-proj-SYNTHETIC /home/ron/private must never persist"
                ),
                "UNEXPECTED_INTERNAL_ERROR",
                2,
            ),
        )
        for exception_factory, code, exit_code in cases:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_root = root / "run"
                output_root = root / "output"
                state_path = run_root / "collection-state.json"
                journal_path = output_root / "collection-journal.jsonl"
                sampling_key = {
                    "fixture_sha256": "f" * 64,
                    "scenario_id": "S1_DEPLOYMENT",
                    "repeat": 1,
                    "revision_sha": tier_b.BASELINE_SHA,
                    "cycle": 0,
                    "question_id": "S1_CURRENT_DEPLOYMENT_STATE",
                }
                state = {
                    "attempt_ids": ["attempt-" + "a" * 32],
                    "calls": [
                        {
                            "attempt_id": "attempt-" + "a" * 32,
                            "logical_call_id": "call-" + "b" * 64,
                            "schedule_position": 1,
                            "call_role": "downstream",
                            "scenario_id": "S1_DEPLOYMENT",
                            "repeat": 1,
                            "revision_label": "baseline",
                            "revision_sha": tier_b.BASELINE_SHA,
                            "cycle": 0,
                            "question_id": "S1_CURRENT_DEPLOYMENT_STATE",
                            "duration_ns": 17,
                            "secret_extra": "sk-proj-SYNTHETIC",
                        }
                    ],
                    "samples": [
                        {
                            "sampling_key": sampling_key,
                            "sampling_key_sha256": hashlib.sha256(
                                canonical(sampling_key).encode()
                            ).hexdigest(),
                            "scenario_id": "S1_DEPLOYMENT",
                            "repeat": 1,
                            "cycle": 0,
                            "question_id": "S1_CURRENT_DEPLOYMENT_STATE",
                            "valid": False,
                            "response_bytes": 9,
                            "response_sha256": "c" * 64,
                            "error_codes": ["MALFORMED_MODEL_RESPONSE"],
                            "metrics": {
                                "task_format_valid": 0,
                                "trial_nonce_leakage_count": 0,
                            },
                            "path_extra": "/home/ron/private",
                        }
                    ],
                    "snapshots": [
                        {
                            "snapshot_id": "snapshot:synthetic:cycle:0",
                            "snapshot_sha256": "d" * 64,
                            "row_count": 13,
                            "compression_count": 0,
                            "dependency_key": "synthetic:cycle:0",
                            "storage": "container_tmpfs",
                            "secret_extra": "sk-proj-SYNTHETIC",
                        }
                    ],
                    "untrusted": "sk-proj-SYNTHETIC /home/ron/private",
                }
                write_json(state_path, state)
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                journal_path.write_text(
                    canonical({
                        "attempt_id": "attempt-" + "a" * 32,
                        "logical_call_id": "call-" + "b" * 64,
                        "schedule_position": 1,
                        "state": "PLANNED",
                    })
                    + "\n",
                    encoding="utf-8",
                )
                output = []

                def action():
                    raise exception_factory()

                with self.assertRaises(SystemExit) as stopped:
                    tier_b.run_collect_entrypoint(
                        action,
                        output_root,
                        output.append,
                        partial_writer=lambda stopped_code: (
                            tier_b.write_partial_raw_evidence(
                                output_root=output_root,
                                state_path=state_path,
                                journal_path=journal_path,
                                error_code=stopped_code,
                            )
                        ),
                    )
                self.assertEqual(stopped.exception.code, exit_code)
                raw_path = output_root / "final/context-compression-tier-b-raw.json"
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    raw["schema_version"], "context-compression-tier-b-raw/1"
                )
                self.assertEqual(raw["status"], "incomplete")
                self.assertEqual(
                    raw["identity"]["attempt_ids"], ["attempt-" + "a" * 32]
                )
                self.assertEqual(
                    [item["logical_call_id"] for item in raw["calls"]],
                    ["call-" + "b" * 64],
                )
                self.assertEqual(
                    [item["sampling_key_sha256"] for item in raw["samples"]],
                    [hashlib.sha256(canonical(sampling_key).encode()).hexdigest()],
                )
                self.assertEqual(
                    [item["snapshot_id"] for item in raw["snapshots"]],
                    ["snapshot:synthetic:cycle:0"],
                )
                self.assertEqual(raw["errors"], [{"code": code}])
                self.assertFalse((output_root / "final/FINALIZED").exists())
                persisted = "".join(
                    path.read_text(encoding="utf-8")
                    for path in output_root.rglob("*")
                    if path.is_file()
                )
                self.assertIn(code, persisted)
                self.assertNotIn("sk-proj", persisted + "".join(output))
                self.assertNotIn("/home/ron", persisted + "".join(output))
                self.assertNotIn("Traceback", persisted + "".join(output))

    def test_live_collection_preconditions_stop_before_wire(self):
        base_environment = {
            "TIER_B_LIVE_ACK": "CONTEXT_COMPRESSION_TIER_B_AUTHORIZED",
            "CI": "",
            "GITHUB_ACTIONS": "",
        }
        base_manifest = {
            "source_manifest_sha256": "a" * 64,
            "schedule_sha256": "b" * 64,
        }
        base_attestation = {
            **base_manifest,
            "preflight_passed": True,
            "runtime_image_id": "sha256:" + "c" * 64,
        }
        cases = [
            (
                {**base_environment, "CI": "true"},
                base_manifest,
                base_attestation,
                "CI_LIVE_COLLECTION_FORBIDDEN",
            ),
            (
                {**base_environment, "TIER_B_LIVE_ACK": ""},
                base_manifest,
                base_attestation,
                "LIVE_ACK_REQUIRED",
            ),
            (
                base_environment,
                {**base_manifest, "source_manifest_sha256": "d" * 64},
                base_attestation,
                "SOURCE_RUNTIME_IDENTITY_MISMATCH",
            ),
            (
                base_environment,
                base_manifest,
                {**base_attestation, "preflight_passed": False},
                "RUNTIME_PREFLIGHT_FAILED",
            ),
        ]
        for environment, manifest, attestation, code in cases:
            wire_count = 0

            def wire():
                nonlocal wire_count
                wire_count += 1

            with self.assertRaises(tier_b.TerminalBenchmarkStop) as caught:
                tier_b.run_collect_after_preflight(
                    environment=environment,
                    manifest=manifest,
                    runtime_attestation=attestation,
                    collect_once=wire,
                )
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(wire_count, 0)


if __name__ == "__main__":
    unittest.main()
