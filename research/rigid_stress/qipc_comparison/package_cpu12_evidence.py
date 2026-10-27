"""Build a <=10 MiB rigid-stress evidence packet using only the Python standard library.

Only task outputs with a task-specific returncode receipt are eligible. Pending
or cancelled tasks without that receipt retain submission/cancellation metadata,
never potentially live fields or logs. Packaging does not assess goal completion.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shlex
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LIMIT = 10 * 1024 * 1024
SMALL_BINARY = 256 * 1024
JSON_LIMIT = 16 * 1024 * 1024
PRIMARY_JSON_LIMIT = 128 * 1024 * 1024
PLANAR_REQUIRED_METADATA = (
    "primary-summary-*.json.gz",
    "final-memory-summary-*.json.gz",
    "final-profile-summary-*.json.gz",
    "verify-planar-delivery-packet-v19.py",
)
PRIMARY_EXTERNAL_FIELDS = {
    "warmup_quality",
    "trajectory_metrics",
    "native_buffers",
    "egg_position_m",
    "maximum_stress_Pa",
}
PRUNE = {"analysis-source", "cache", ".cache", "__pycache__", ".git", "node-local-scratch"}
CONFIG_KEYS = (
    "arguments",
    "config",
    "configuration",
    "options",
    "model_options",
    "settings",
    "bank",
    "tile",
    "envs",
    "environments",
    "method",
    "mother_recovery",
    "packed_solve",
    "strategy",
    "output_mode",
    "precision",
    "stage",
    "banks",
    "quadrature",
    "pressure_quadrature",
    "construction",
    "atlas",
    "solver",
)
OMIT_HEADER = {
    "files",
    "source_files",
    "rows",
    "cases",
    "case_manifest",
    "history",
    "conditions",
    "traceEvents",
    "buffers",
    "allocations",
    "frame_values",
    "field_errors",
    "events",
}
SEMANTIC_TOKENS = (
    "numerical",
    "equivalence",
    "spatial",
    "qualification",
    "budget",
    "admission",
    "complete_scene",
    "native_DPR",
    "source_unchanged",
    "whole_goal",
    "goal_complete",
    "passed_semantics",
)


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _stable_bytes(path):
    before = _signature(path)
    content = path.read_bytes()
    if _signature(path) != before:
        raise RuntimeError(f"Evidence changed while reading; leave the live source untouched: {path}")
    return content


def _identity(path):
    before = _signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if _signature(path) != before:
        raise RuntimeError(f"Evidence changed while hashing; leave the live source untouched: {path}")
    return {"path": str(path), "bytes": before[2], "sha256": digest.hexdigest()}


def _read_json(path):
    if path.stat().st_size > JSON_LIMIT:
        return None
    try:
        return json.loads(_stable_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _compact(value, depth=0):
    """Retain audit meaning while bounding large arrays and source inventories."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value), "raw_value_retained_externally": True}
    if isinstance(value, dict):
        if depth >= 6:
            return {"type": "dict", "entries": len(value)}
        keys = list(value)
        output = {str(key): _compact(value[key], depth + 1) for key in keys[:128]}
        if len(keys) > 128:
            output["_omitted_keys"] = [str(key) for key in keys[128:]]
        return output
    if isinstance(value, list):
        if len(value) <= 24:
            return [_compact(item, depth + 1) for item in value]
        return {
            "entries": len(value),
            "first": [_compact(item, depth + 1) for item in value[:2]],
            "last": _compact(value[-1], depth + 1),
            "full_list_retained_externally": True,
        }
    if isinstance(value, str) and len(value) > 4096:
        return {"characters": len(value), "prefix": value[:1024], "full_text_retained_externally": True}
    return value


def _header(document):
    if not isinstance(document, dict):
        return _compact(document)
    output = {key: _compact(value) for key, value in document.items() if key not in OMIT_HEADER}
    for key in OMIT_HEADER:
        value = document.get(key)
        if isinstance(value, (dict, list)):
            output[f"{key}_entries_retained_externally"] = len(value)
    rows = document.get("rows")
    if isinstance(rows, list) and rows:
        maximum, categories, rejected, nonfinite = {}, Counter(), Counter(), Counter()
        for row in rows:
            if not isinstance(row, dict):
                continue
            if isinstance(row.get("category"), str):
                categories[row["category"]] += 1
            for flag in ("passed", "spatial_passed", "accepted"):
                if row.get(flag) is False:
                    rejected[flag] += 1
            errors = row.get("errors")
            if isinstance(errors, dict):
                for key, value in errors.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if math.isfinite(value):
                            maximum[key] = max(value, maximum.get(key, value))
                        else:
                            nonfinite[key] += 1
        output["reported_row_summary"] = {
            "rows": len(rows),
            "maximum_already_reported_errors": maximum,
            "explicitly_false_row_flags": dict(rejected),
            "nonfinite_reported_error_rows": dict(nonfinite),
            "categories": dict(categories),
            "first_row": _compact(rows[0]),
            "last_row": _compact(rows[-1]),
            "interpretation": "Metadata aggregation of existing reports; no new numerical/spatial qualification.",
        }
    return output


def _semantics(document):
    if not isinstance(document, dict):
        return {"declared_passed": None, "interpretation": "No recognized result dictionary."}
    return {
        "declared_passed": _compact(document.get("passed")),
        "declared_complete": _compact(document.get("complete")),
        "explicit_semantics": {
            key: _compact(value) for key, value in document.items() if any(token in key for token in SEMANTIC_TOKENS)
        },
        "interpretation": "Exit zero and a generic passed flag do not establish spatial qualification, speedup or goal completion.",
    }


@dataclass(frozen=True)
class Candidate:
    data: bytes
    extension: str
    category: str
    priority: int
    required: bool = False


class Packet:
    def __init__(self, run_root, study="cpu12"):
        self.run_root = run_root
        self.study = study
        self.artifacts = {}
        self.candidates = {}
        self.associations = {}

    def artifact(self, path, *, task=None, category="external"):
        key = str(path.resolve())
        if key not in self.artifacts:
            identity = _identity(path)
            identity["relative_to_run"] = (
                str(path.relative_to(self.run_root)) if path.is_relative_to(self.run_root) else None
            )
            identity["category"] = category
            identity["terminal_tasks"] = []
            self.artifacts[key] = identity
        if task is not None and task not in self.artifacts[key]["terminal_tasks"]:
            self.artifacts[key]["terminal_tasks"].append(task)
        return key

    def payload(self, data, extension, category, priority, *, source=None, required=False):
        digest = hashlib.sha256(data).hexdigest()
        if digest not in self.candidates or priority < self.candidates[digest].priority:
            retain = required or (digest in self.candidates and self.candidates[digest].required)
            self.candidates[digest] = Candidate(data, extension, category, priority, retain)
        elif required and not self.candidates[digest].required:
            previous = self.candidates[digest]
            self.candidates[digest] = Candidate(
                previous.data, previous.extension, previous.category, previous.priority, True
            )
        if source is not None:
            self.associations.setdefault(source, []).append(digest)
        return digest

    def json_payload(self, document, category, priority, *, source=None, required=False):
        return self.payload(
            gzip.compress(_json_bytes(document), compresslevel=9, mtime=0),
            ".json.gz",
            category,
            priority,
            source=source,
            required=required,
        )

    def exact_payload(self, path, key, category, priority, *, required=False):
        content = _stable_bytes(path)
        if hashlib.sha256(content).hexdigest() != self.artifacts[key]["sha256"]:
            raise RuntimeError(f"Source changed after its recorded hash: {path}")
        if path.suffix in (".json", ".jsonl", ".xml", ".txt", ".log", ".sh", ".patch", ".py"):
            return self.payload(
                gzip.compress(content, compresslevel=9, mtime=0),
                path.suffix + ".gz",
                category,
                priority,
                source=key,
                required=required,
            )
        return self.payload(content, path.suffix or ".bin", category, priority, source=key, required=required)

    def primary_measurements(self, path, task, marker, marker_before, code):
        """Retain original repeat rows and benchmark identity from a terminal primary output."""
        size = path.stat().st_size
        if size > PRIMARY_JSON_LIMIT:
            raise ValueError(
                f"Primary benchmark JSON exceeds the explicit {PRIMARY_JSON_LIMIT}-byte read limit "
                f"({size} bytes): {path}. Preserve the original externally and review its payload size."
            )
        key = self.artifact(path, task=task)
        content = _stable_bytes(path)
        if hashlib.sha256(content).hexdigest() != self.artifacts[key]["sha256"]:
            raise RuntimeError(f"Primary benchmark changed after its recorded hash: {path}")
        try:
            document = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid terminal primary benchmark JSON: {path}") from error
        if not isinstance(document, dict) or not isinstance(document.get("repeats"), list):
            raise TypeError(f"Expected original benchmark repeats in terminal primary JSON: {path}")
        payload = {
            "schema": "genesis.planar-primary-benchmark-rows.v1",
            "kind": "original_benchmark_measurement_rows",
            "qualification_assessed": False,
            "task_id": task,
            "terminal_returncode": code,
            "source": self.artifacts[key],
            "repeats": document["repeats"],
            "benchmark_metadata": {
                name: value
                for name, value in document.items()
                if name != "repeats" and name not in PRIMARY_EXTERNAL_FIELDS
            },
            "excluded_top_level_fields_retained_externally": sorted(PRIMARY_EXTERNAL_FIELDS & document.keys()),
            "nonfinite_encoding": "Original NaN/Infinity/-Infinity values retain Python JSON tokens without replacement.",
        }
        raw = (json.dumps(payload, sort_keys=True, allow_nan=True) + "\n").encode()
        digest = self.payload(
            gzip.compress(raw, compresslevel=9, mtime=0),
            ".json.gz",
            "primary-raw-measurements",
            0,
            source=key,
            required=True,
        )
        self.artifacts[key]["primary_raw_measurements"] = {
            "payload_sha256": digest,
            "repeats": len(document["repeats"]),
        }
        if _signature(marker) != marker_before:
            raise RuntimeError(f"Primary task returncode changed during packaging: {marker}")
        return key

    def collect_file(self, path, task):
        if (
            self.study == "planar"
            and task.split("/", 1)[0].startswith("primary-paired-")
            and path == self.run_root / task
            and path.suffix == ".json"
        ):
            marker, code = _returncode(path.parent, path.name)
            if code is None:
                raise RuntimeError(f"Primary raw measurements require a task-specific terminal returncode: {path}")
            return self.primary_measurements(path, task, marker, _signature(marker), code)
        key = self.artifact(path, task=task)
        size, name = self.artifacts[key]["bytes"], path.name.lower()
        if name in ("measurements.json", "measurements.npz", "pilot_timings.json", "paired.json", "paired.npz"):
            if size <= 4 * 1024 * 1024:
                self.exact_payload(path, key, "paired-raw", 0, required=True)
            return key
        if name.endswith("junit.xml") or name.startswith("junit"):
            if size <= 1024 * 1024:
                self.exact_payload(path, key, "junit", 4)
            return key
        if name in ("checks.json", "checks-summary.json"):
            document = _read_json(path)
            if isinstance(document, list):
                failures = [row for row in document if isinstance(row, dict) and row.get("passed") is False]
                summary = {
                    "source": self.artifacts[key],
                    "checks": len(document),
                    "explicit_failed_checks": _compact(failures),
                    "unclassified_rows": sum(not isinstance(row, dict) or "passed" not in row for row in document),
                }
                self.json_payload(summary, "checks-summary", 2, source=key)
            if size <= 1024 * 1024:
                self.exact_payload(path, key, "checks-raw", 14)
            return key
        if path.suffix == ".npz":
            parts = {part.lower() for part in path.parts}
            allowed = name in {
                "cases.npz",
                "model.npz",
                "selection.npz",
                "native-shell-input.npz",
                "conditions.npz",
                "source_conditions.npz",
                "source-conditions.npz",
            }
            if allowed and "oracles" not in parts and size <= SMALL_BINARY:
                self.exact_payload(path, key, "physical-input", 5)
            return key
        if path.suffix in (".log", ".out", ".err"):
            if size <= 32 * 1024:
                self.exact_payload(path, key, "compact-log", 25)
            else:
                before = _signature(path)
                with path.open("rb") as stream:
                    first = stream.read(8192)
                    stream.seek(max(0, size - 8192))
                    last = stream.read(8192)
                if _signature(path) != before:
                    raise RuntimeError(f"Terminal task log changed while extracting: {path}")
                excerpt = (
                    first
                    + b"\n\n[... omitted; complete original has the external SHA256 in artifact index ...]\n\n"
                    + last
                )
                self.payload(gzip.compress(excerpt, mtime=0), ".log-excerpt.gz", "log-excerpt", 35, source=key)
            return key
        if name == "nvidia-process-memory.jsonl" and size <= 1024 * 1024:
            self.exact_payload(path, key, "memory-samples", 7)
            return key
        if path.suffix == ".json" and name not in ("trace.json",):
            document = _read_json(path)
            if document is not None:
                self.json_payload(
                    {"source": self.artifacts[key], "header": _header(document)},
                    "result-summary" if name == "result.json" else "metadata-summary",
                    8,
                    source=key,
                )
                rows = document.get("rows", []) if isinstance(document, dict) else []
                if size <= 256 * 1024 and (not isinstance(rows, list) or len(rows) <= 128):
                    self.exact_payload(path, key, "small-metadata", 12)
        return key


def _walk_task(folder):
    for directory, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in PRUNE and not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                yield path


def _returncode(group, name):
    for path in (group / f"{name}.returncode.txt", group / f"{name}.returncode", group / name / ".returncode"):
        if path.is_file():
            content = _stable_bytes(path)
            try:
                return path, int(content.strip())
            except ValueError:
                return path, None
    return None, None


def _source_receipts(packet, campaign):
    source, analysis = {}, []
    receipts = {}
    for name in ("tasks.json", "submission.json", "source.json", "analysis-files.json", "source.patch", "job.sh"):
        path = campaign / name
        if not path.is_file():
            continue
        key = packet.artifact(path, category="frozen-campaign-receipt")
        receipts[name] = key
        document = _read_json(path) if path.suffix == ".json" else None
        if name == "source.json" and isinstance(document, dict):
            source = document
        elif name == "analysis-files.json" and isinstance(document, list):
            analysis = document
        if path.stat().st_size <= 8 * 1024 * 1024:
            packet.exact_payload(path, key, "source-receipt", 1, required=name in ("tasks.json", "submission.json"))
    return source, analysis, receipts


def _top_metadata(packet, meta):
    documents = {}
    if meta.is_dir():
        for path in sorted(meta.iterdir()):
            if not path.is_file() or path.is_symlink():
                continue
            key = packet.artifact(path, category="run-metadata")
            documents[path.name] = _read_json(path) if path.suffix == ".json" else None
            required = packet.study == "planar" and any(path.match(pattern) for pattern in PLANAR_REQUIRED_METADATA)
            if required or path.stat().st_size <= 4 * 1024 * 1024:
                packet.exact_payload(
                    path,
                    key,
                    "run-metadata",
                    0 if required else 2,
                    required=required or "preserv" in path.name or path.name == "start-state.json",
                )
    return documents


def _input_receipts(packet, task_id, arguments):
    inputs = []
    for value in arguments:
        if not isinstance(value, str) or not value.startswith("/"):
            continue
        path = Path(value)
        if path.is_dir() and (path / "manifest.json").is_file():
            path = path / "manifest.json"
        if path.is_file() and not path.is_symlink():
            key = packet.artifact(path, task=task_id, category="task-input")
            if (
                path.name in ("conditions.npz", "source_conditions.npz", "source-conditions.npz")
                and path.stat().st_size <= SMALL_BINARY
            ):
                packet.exact_payload(path, key, "source-conditions", 2)
            inputs.append(key)
    return sorted(set(inputs))


def _preservation(start, expected_frozen_count=23):
    """Record observed original-file hashes and frozen source pins without editing either."""
    original, frozen = [], []
    for row in start.get("original_files", []):
        path = Path(row["path"])
        current = _identity(path) if path.is_file() else None
        original.append(
            {
                "expected": row,
                "observed": current,
                "matches": current is not None and current["sha256"] == row["sha256"],
            }
        )
    for row in start.get("frozen_sources", []):
        path = Path(row["path"])
        command = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
        )
        revision = command.stdout.strip() if command.returncode == 0 else None
        diff = subprocess.run(
            ["git", "-C", str(path), "diff", "--no-ext-diff", "--name-only", "HEAD", "--", "genesis", "examples"],
            text=True,
            capture_output=True,
            check=False,
        )
        frozen.append(
            {
                "expected": row,
                "observed_revision": revision,
                "matches_revision": revision == row.get("revision"),
                "engine_clean": not diff.stdout.strip() if diff.returncode == 0 else None,
                "observed_engine_changed_paths": diff.stdout.splitlines() if diff.returncode == 0 else None,
                "git_error": command.stderr.strip() or diff.stderr.strip() or None,
            }
        )
    return {
        "kind": "read_only_preservation_observation",
        "original_files": original,
        "frozen_sources": frozen,
        "expected_original_count": 12,
        "expected_frozen_count": expected_frozen_count,
        "start_receipt_counts_match_scope": len(original) == 12 and len(frozen) == expected_frozen_count,
        "goal_completion_assessed": False,
    }


def _task(packet, campaign, group_index, task, source, analysis, cancellations, repro_prefix="CPU12"):
    name = task["name"]
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Invalid task name in frozen task receipt: {name!r}")
    group = campaign / f"group{group_index}"
    task_id = f"{campaign.name}/group{group_index}/{name}"
    marker, code = _returncode(group, name)
    record = {
        "id": task_id,
        "campaign": campaign.name,
        "group": group_index,
        "name": name,
        "returncode": code,
        "engine_source_pin": source.get("revision"),
        "source_path": source.get("source"),
        "declared_task": task,
        "task_payload_collected": False,
        "cancellation_receipts": cancellations,
        "execution_status": "pending_no_task_returncode",
    }
    if code is None:
        if cancellations:
            record["execution_status"] = "cancellation_recorded_without_task_returncode"
        if marker is not None:
            record["returncode_receipt"] = packet.artifact(marker, category="unparseable-returncode")
        return record
    marker_before = _signature(marker)
    marker_key = packet.artifact(marker, task=task_id, category="terminal-returncode")
    packet.exact_payload(marker, marker_key, "terminal-returncode", 0, required=True)
    record.update(
        execution_status="exit_zero" if code == 0 else "exit_nonzero",
        returncode_receipt=marker_key,
        task_payload_collected=True,
    )
    module = task.get("module")
    arguments = task.get("args", [])
    worker_relative = module.replace(".", "/") + ".py" if isinstance(module, str) else None
    worker = campaign / "analysis-source" / worker_relative if worker_relative is not None else None
    if worker is None or not worker.is_file():
        worker = Path(source["source"]) / worker_relative if source.get("source") and worker_relative else None
    if worker is not None and worker.is_file():
        key = packet.artifact(worker, task=task_id, category="actual-worker")
        expected = next((row for row in analysis if row.get("path") == worker_relative), None)
        record["actual_worker"] = {
            **packet.artifacts[key],
            "module": module,
            "captured_inventory_entry": expected,
            "matches_capture": expected.get("sha256") == packet.artifacts[key]["sha256"]
            if expected is not None
            else None,
        }
        if worker.stat().st_size <= 256 * 1024:
            packet.exact_payload(worker, key, "actual-worker", 6)
    else:
        record["actual_worker"] = {"module": module, "available": False, "path_expected": worker_relative}
    record["input_artifacts"] = _input_receipts(packet, task_id, arguments)
    results, configs = [], {}
    folder = group / name
    record["output_kind"] = "directory" if folder.is_dir() else "file" if folder.is_file() else "missing"
    paths = _walk_task(folder) if folder.is_dir() else (folder,) if folder.is_file() else ()
    for path in paths:
        packet.collect_file(path, task_id)
        if path.name in ("result.json", "start.json", "manifest.json", "setup.json") or path == folder:
            document = _read_json(path)
            if isinstance(document, dict):
                relative = "result.json" if path == folder else str(path.relative_to(folder))
                if relative in ("result.json", "manifest.json"):
                    results.append({"path": str(path), "header": _header(document), "semantics": _semantics(document)})
                if relative in ("result.json", "start.json", "manifest.json"):
                    configs[relative] = {key: _compact(document[key]) for key in CONFIG_KEYS if key in document}
                    record.setdefault("recorded_input_hashes", {})[relative] = {
                        key: _compact(value)
                        for key, value in document.items()
                        if "sha256" in key
                        or "input" in key
                        or key in ("conditions", "asset", "basis", "mesh", "reference_manifest")
                    }
                if relative == "start.json":
                    record["actual_checkout_commit"] = document.get("checkout_commit")
    record["result_documents"] = results
    record["actual_configuration"] = configs
    for log in (group / f"{name}.log", group / f"{name}.stdout", group / f"{name}.stderr"):
        if log.is_file():
            packet.collect_file(log, task_id)
    if _signature(marker) != marker_before:
        raise RuntimeError(f"Task returncode changed during packaging: {marker}")
    record["reproduction_command"] = (
        f"PYTHONPATH={shlex.quote(str(campaign / 'analysis-source'))}:{shlex.quote(str(source.get('source', '<frozen-source>')))} "
        f'python -m {shlex.quote(module or "<worker>")} {shlex.join(arguments)} --output "${{{repro_prefix}_REPRO_OUTPUT:?set a fresh data output}}"'
    )
    return record


def _plan(packet, tasks, index_base, preservation, ledger, readme):
    selected = set(packet.candidates)
    omitted = []
    while True:
        payloads, payload_manifest = {}, {}
        for digest in sorted(selected):
            candidate = packet.candidates[digest]
            path = f"payloads/{candidate.category}/{digest}{candidate.extension}"
            payloads[path] = candidate.data
            payload_manifest[path] = {"bytes": len(candidate.data), "sha256": digest, "category": candidate.category}
        artifacts = []
        for key, value in sorted(packet.artifacts.items()):
            entry = dict(value)
            entry["retained_payloads"] = [
                f"payloads/{packet.candidates[digest].category}/{digest}{packet.candidates[digest].extension}"
                for digest in sorted(set(packet.associations.get(key, [])))
                if digest in selected
            ]
            artifacts.append(entry)
        compressed_artifacts = gzip.compress(
            b"".join(json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n" for row in artifacts),
            compresslevel=9,
            mtime=0,
        )
        fixed = {
            "external-artifacts.jsonl.gz": compressed_artifacts,
            "preservation.json": _json_bytes(preservation),
            "requirements-ledger.json": _json_bytes(ledger),
            "README.md": readme.encode(),
        }
        for path, content in fixed.items():
            payloads[path] = content
            payload_manifest[path] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "category": "packet-metadata",
            }
        index = {
            **index_base,
            "tasks": tasks,
            "artifacts": len(artifacts),
            "payloads": payload_manifest,
            "omitted_optional_payloads": omitted,
            "packet_limit_bytes": LIMIT,
            "artifact_inventory": "external-artifacts.jsonl.gz",
            "payload_hash_scope": "Every payload including generated metadata; index.json is hashed in SHA256SUMS.",
        }
        index_content = _json_bytes(index)
        payloads["index.json"] = index_content
        checksum = "".join(
            f"{hashlib.sha256(content).hexdigest()}  {path}\n" for path, content in sorted(payloads.items())
        ).encode()
        payloads["SHA256SUMS"] = checksum
        total = sum(len(content) for content in payloads.values())
        if total <= LIMIT:
            return payloads, total
        optional = [digest for digest in selected if not packet.candidates[digest].required]
        if not optional:
            raise RuntimeError(
                f"Mandatory compact metadata/paired receipts alone exceed the10MiB limit ({total} bytes)."
            )
        remove = max(
            optional, key=lambda digest: (packet.candidates[digest].priority, len(packet.candidates[digest].data))
        )
        selected.remove(remove)
        omitted.append(
            {
                "payload_sha256": remove,
                "category": packet.candidates[remove].category,
                "reason": "10MiB hard limit; original remains externally hashed",
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maintained-source", required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--requirements-ledger", type=Path)
    parser.add_argument("--study", choices=("cpu12", "planar"), default="cpu12")
    args = parser.parse_args()
    if args.study == "planar" and args.requirements_ledger is None:
        parser.error("Planar loading requires its explicit requirements ledger; CPU12 targets do not apply.")
    title = "Planar loading DPR" if args.study == "planar" else "CPU12"
    repro_prefix = "PLANAR" if args.study == "planar" else "CPU12"
    expected_frozen_count = 33 if args.study == "planar" else 23
    run_root, output = args.run_root.expanduser().resolve(), args.output.expanduser().resolve()
    if not run_root.is_dir() or output.is_relative_to(run_root):
        parser.error("Use an existing run root and a separate fresh evidence destination.")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh/empty packet directory; original evidence is never overwritten.")
    maintained = subprocess.check_output(
        ["git", "-C", str(args.git_source), "rev-parse", f"{args.maintained_source}^{{commit}}"], text=True
    ).strip()
    packet, meta, tasks, campaigns = Packet(run_root, study=args.study), run_root / "meta", [], []
    metadata_documents = _top_metadata(packet, meta)
    start = metadata_documents.get("start-state.json")
    if not isinstance(start, dict):
        raise TypeError("The original-file/frozen-source preservation start-state receipt is required.")
    for campaign in sorted(path for path in run_root.iterdir() if path.is_dir() and (path / "tasks.json").is_file()):
        source, analysis, receipts = _source_receipts(packet, campaign)
        definitions = _read_json(campaign / "tasks.json")
        if not isinstance(definitions, list) or any(not isinstance(group, list) for group in definitions):
            raise ValueError(f"Expected frozen grouped tasks.json: {campaign}")
        cancellations = [
            str(meta / name)
            for name in metadata_documents
            if campaign.name in name and ("cancel" in name or "stop" in name)
        ]
        for group_index, group in enumerate(definitions):
            for definition in group:
                tasks.append(
                    _task(packet, campaign, group_index, definition, source, analysis, cancellations, repro_prefix)
                )
            group_marker = campaign / f"group{group_index}/returncode.txt"
            if group_marker.is_file() and all(
                _returncode(campaign / f"group{group_index}", task["name"])[1] is not None for task in group
            ):
                packet.artifact(group_marker, category="terminal-group-returncode")
        campaigns.append(
            {
                "name": campaign.name,
                "source_pin": source.get("revision"),
                "receipts": receipts,
                "task_count": sum(len(group) for group in definitions),
            }
        )
    preservation = _preservation(start, expected_frozen_count)
    ledger = (
        json.loads(_stable_bytes(args.requirements_ledger))
        if args.requirements_ledger
        else {
            "goal_complete": None,
            "status": "root_review_required",
            "maintained_source": maintained,
            "targets": [
                {"id": name, "status": "root_review_required", "evidence": []}
                for name in (
                    "matched_CPU12_shell_equivalence",
                    "shell_spatial_decision",
                    "volume_spatial_decision",
                    "optimized_mother_and_module_pairing",
                    "complete_Franka_max_full",
                    "memory_and_setup",
                    "lifecycle_and_surface_identity",
                    "source_and_unfinished_file_preservation",
                )
            ],
            "note": "Fill a copy of this ledger and repackage into a new directory with --requirements-ledger; never edit hashed payloads in place.",
        }
    )
    counts = dict(Counter(task["execution_status"] for task in tasks))
    created = datetime.now(timezone.utc).isoformat()
    primary_policy = (
        "For planar primary-paired-* terminal benchmark tasks, mandatory gzip payloads preserve every original "
        "repeat field and vector plus unabridged identity/configuration metadata. Other large diagnostics remain "
        "externally hashed. These are original measurements with no new qualification assessment.\n\n"
        "Existing top-level primary/memory/profile summary gzip files and the final offline verifier are "
        "mandatory exact payloads, without the optional metadata 4MiB input cutoff. The 10MiB packet limit "
        "still applies; missing summaries are not created or certified by packaging.\n\n"
        if args.study == "planar"
        else ""
    )
    missing_receipts = (
        "Missing receipts remain pending unless a separate cancellation journal records the stopped campaign. "
        "Recorded cancellations have their own index status and task list."
        if args.study == "planar"
        else "Missing receipts remain pending, including stopped jobs with a\nseparate cancellation journal."
    )
    readme = f"""# {title} compact evidence packet

Generated {created}. Run root: `{run_root}`. Maintained source: `{maintained}`.

This packet records terminal execution receipts and declared numerical/spatial
semantics separately. It does not mark the user goal complete. Review
`requirements-ledger.json` before making any completion claim.

`index.json` maps campaign/group/task, actual worker hash, source pin, exact CLI
and available result summaries. `external-artifacts.jsonl.gz` records SHA256 and
bytes for every collected task artifact, including arrays, complete fields and
logs left on data. Retained payloads are content-addressed and deduplicated.
Gzip JSON preserves exact raw paired rows/source receipts where labelled raw;
derived summaries are labelled separately. Large600-row result histories remain
external. The packet is capped at10MiB; optional omissions are explicit.

{primary_policy}\
Only task-specific `.returncode.txt`/`.returncode` receipts permit task output
collection. {missing_receipts} Their potentially live output trees are not
read. A task RC0 is execution success, not numerical/spatial qualification.
Caches and complete analysis-source trees are excluded. Actual small worker
files may be retained individually; full engine code is reproduced by Git pin.

## Verification and reproduction

```sh
sha256sum -c SHA256SUMS
gzip -dc external-artifacts.jsonl.gz
: "${{{repro_prefix}_REPRO_SOURCE:?choose a fresh source checkout}}"
: "${{{repro_prefix}_PACKET_OUTPUT:?choose a fresh packet directory}}"
git worktree add --detach "${repro_prefix}_REPRO_SOURCE" {maintained}
python -m research.rigid_stress.qipc_comparison.package_cpu12_evidence \\
  --run-root {shlex.quote(str(run_root))} --output "${repro_prefix}_PACKET_OUTPUT" \\
  --maintained-source {maintained} --git-source "${repro_prefix}_REPRO_SOURCE" \\
  --study {args.study}{' --requirements-ledger "${PLANAR_REQUIREMENTS_LEDGER:?choose the reviewed ledger}"' if args.study == "planar" else ""}
```

Each terminal task has its original module/arguments and reproduction command in
the index. Use that campaign's recorded engine pin and captured analysis-source
inventory, an `rtx-mid` Slurm allocation, image `1_26`, private caches, and a new
data-side output. Exact captured workers remain at their external source paths.
Do not execute scientific modules on a login node or overwrite any original run.
Packet creation itself uses only the standard library and changes no run files.

`preservation.json` reports current read-only observations of the original12
files and{expected_frozen_count} frozen sources from the preserved start receipt. Unknown/missing
paths stay explicit. Device-memory point/sampled measurements, intrusive profile
results, approximate-space residuals and exact-mother residuals retain their
source declarations; packaging does not reinterpret them as new validation.

To add a completed target ledger, edit a separate copy and rerun this packager
with `--requirements-ledger <COPY>` into a fresh directory. This preserves all
payload hashes and never promotes a pending task or historical result to a pass.
"""
    index_base = {
        "schema": f"genesis.{args.study}-compact-evidence.v1",
        "created_utc": created,
        "run_root": str(run_root),
        "maintained_source": maintained,
        "goal_completion_assessed": False,
        "campaigns": campaigns,
        "task_status_counts": counts,
        "pending_tasks": [task["id"] for task in tasks if not task["task_payload_collected"]],
        "collection_policy": "Only task-specific terminal returncodes permit task payload collection; metadata snapshots remain labelled.",
        "packager": _identity(Path(__file__).resolve()),
    }
    if args.study == "planar":
        index_base["cancelled_tasks"] = [
            task["id"] for task in tasks if task["execution_status"] == "cancellation_recorded_without_task_returncode"
        ]
        index_base["pending_tasks"] = [
            task["id"] for task in tasks if task["execution_status"] == "pending_no_task_returncode"
        ]
    payloads, total = _plan(packet, tasks, index_base, preservation, ledger, readme)
    output.mkdir(parents=True, exist_ok=True)
    for relative, content in sorted(payloads.items()):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    actual_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    assert actual_bytes == total and actual_bytes <= LIMIT
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": actual_bytes,
                "limit_bytes": LIMIT,
                "campaigns": len(campaigns),
                "tasks": len(tasks),
                "statuses": counts,
                "goal_completion_assessed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
