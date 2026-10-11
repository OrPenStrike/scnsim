"""SQLite authority for one bound Plan leaf's operation evidence.

The caller retains the Plan lock before entering this store. This module then
owns the operation flock and native SQLite transaction, in that order. Canonical
payload meaning, numerical artifacts, callbacks and resumable generation policy
belong to the adapter; no computation or callback runs inside a transaction.
Readers use native mode=ro and never recover implicitly. Object digests identify
exact immutable bytes, not the mutable database file or an absolute location.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from ..canonical import canonical_json_bytes
from ..errors import (
    EvidenceIntegrityError,
    UnsupportedEvidenceVersionError,
    WorkspaceRecoveryRequiredError,
)
from .storage import _fsync_directory

_DATABASE = "operations.sqlite3"
_VERSION = 6
_SCHEMA = (
    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE objects (sha256 TEXT PRIMARY KEY, payload BLOB NOT NULL)",
    "CREATE TABLE object_roles (sha256 TEXT NOT NULL REFERENCES objects(sha256), "
    "role TEXT NOT NULL, PRIMARY KEY(sha256,role))",
    "CREATE TABLE streams (name TEXT PRIMARY KEY, revision INTEGER NOT NULL)",
    "CREATE TABLE entries (stream TEXT NOT NULL REFERENCES streams(name), "
    "sequence INTEGER NOT NULL, kind TEXT NOT NULL, sha256 TEXT NOT NULL, role TEXT NOT NULL, "
    "PRIMARY KEY(stream,sequence), FOREIGN KEY(sha256,role) REFERENCES object_roles(sha256,role))",
    "CREATE TABLE pointers (name TEXT PRIMARY KEY, revision INTEGER NOT NULL, "
    "sha256 TEXT NOT NULL, role TEXT NOT NULL, "
    "FOREIGN KEY(sha256,role) REFERENCES object_roles(sha256,role))",
    "CREATE TABLE transactions (transaction_id TEXT PRIMARY KEY)",
    "CREATE TABLE operations (operation_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, "
    "role TEXT NOT NULL, method TEXT, backend TEXT, precision TEXT, status TEXT, "
    "start_ns INTEGER, end_ns INTEGER, clock_id TEXT, "
    "FOREIGN KEY(sha256,role) REFERENCES object_roles(sha256,role))",
    "CREATE INDEX operation_filters ON operations(method,backend,precision,status)",
    "CREATE TABLE completion_generations (completion_sha TEXT NOT NULL, generation INTEGER NOT NULL, "
    "block_ref TEXT NOT NULL, first_ordinal INTEGER NOT NULL, row_count INTEGER NOT NULL, "
    "summary_json TEXT NOT NULL, PRIMARY KEY(completion_sha,generation))",
    "CREATE TABLE completion_candidates (completion_sha TEXT NOT NULL, ordinal INTEGER NOT NULL, "
    "generation INTEGER NOT NULL, row_offset INTEGER NOT NULL, block_ref TEXT NOT NULL, "
    "value_ref TEXT NOT NULL, summary_json TEXT NOT NULL, PRIMARY KEY(completion_sha,ordinal))",
    "CREATE INDEX completion_candidate_order ON completion_candidates(completion_sha,generation,ordinal)",
)


def _integrity(message: str, **evidence: object) -> EvidenceIntegrityError:
    return EvidenceIntegrityError(message, stage="operation_store", evidence=evidence)


def _unsupported_version(found: str) -> UnsupportedEvidenceVersionError:
    return UnsupportedEvidenceVersionError(
        "Operation evidence uses an unsupported version; use a new Workspace and recompute.",
        stage="operation_store",
        evidence={"found_schema_version": found, "supported_schema_version": _VERSION,
                  "action": "use a new Workspace and recompute"},
    )


def _reference(digest: str, role: str, length: int) -> dict[str, Any]:
    return {"schema_version": _VERSION, "storage": "sqlite", "database": _DATABASE,
            "role": role, "sha256": digest, "byte_length": length}


def _recovery_error(error: sqlite3.Error) -> None:
    # Native extended error codes distinguish required rollback from ordinary
    # readonly, permission, corruption and SQL failures. Journal existence alone
    # says nothing about whether recovery is required.
    if getattr(error, "sqlite_errorcode", None) in (
        getattr(sqlite3, "SQLITE_READONLY_RECOVERY", 264),
        getattr(sqlite3, "SQLITE_READONLY_ROLLBACK", 776),
    ):
        raise WorkspaceRecoveryRequiredError(
            "Operation evidence requires explicit CircuitRun.recover_workspace().",
            stage="operation_store_read", evidence={"sqlite_error": str(error)},
        ) from error


def _secondary(primary: BaseException, label: str, error: BaseException) -> None:
    details = {"phase": label, "type": type(error).__name__, "message": str(error)}
    previous = getattr(primary, "operation_secondary_errors", ())
    primary.operation_secondary_errors = (*previous, details)
    add_note = getattr(primary, "add_note", None)
    if add_note is not None:
        add_note(label + ": " + repr(error))


def _close(connection: sqlite3.Connection) -> None:
    primary = sys.exc_info()[1]
    try:
        connection.close()
    except BaseException as error:
        if primary is None:
            raise
        _secondary(primary, "SQLite connection close", error)


class OperationStore:
    """Internal mechanics for an already identified operations root.

    Construction performs no I/O. Missing databases yield None from reader();
    initialization is explicit and recovery never creates a database. All writer
    and recovery callers must already hold their bound Plan writer authority.
    """

    def __init__(self, root: Path, *, plan_sha256: str, workspace_instance_id: str, phase_scope=None):
        self._phase_scope = phase_scope or (lambda name, details: nullcontext())
        self.root = Path(root)
        self.path = self.root / _DATABASE
        self._metadata = {"schema_version": str(_VERSION), "plan_sha256": plan_sha256,
                          "workspace_instance_id": workspace_instance_id}

    @contextmanager
    def _scope(self, name: str, details: Mapping[str, Any]) -> Iterator[None]:
        observer = self._phase_scope(name, details)
        observer.__enter__()
        try:
            yield
        except BaseException as primary:
            try:
                observer.__exit__(type(primary), primary, primary.__traceback__)
            except BaseException as secondary:
                _secondary(primary, "SQLite observer exit", secondary)
            raise
        else:
            observer.__exit__(None, None, None)

    @property
    def plan_sha256(self) -> str:
        return self._metadata["plan_sha256"]

    @property
    def workspace_instance_id(self) -> str:
        return self._metadata["workspace_instance_id"]

    @classmethod
    def open_readonly(cls, root: Path) -> OperationStore | None:
        """Read the archived binding; this does not assert active Plan identity."""
        store = cls(root, plan_sha256="", workspace_instance_id="")
        if not store.path.exists() and not store.path.is_symlink():
            return None
        with store._lock(exclusive=False):
            connection = None
            try:
                connection = store._connect(mode="ro")
                connection.execute("BEGIN")
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
                if set(metadata) != {"schema_version", "plan_sha256", "workspace_instance_id"}:
                    raise _integrity("Operation database binding metadata is malformed.")
                if metadata["schema_version"] != str(_VERSION):
                    raise _unsupported_version(metadata["schema_version"])
                return cls(root, plan_sha256=metadata["plan_sha256"],
                           workspace_instance_id=metadata["workspace_instance_id"])
            except sqlite3.Error as error:
                _recovery_error(error)
                raise
            finally:
                if connection is not None:
                    _close(connection)

    def _check_paths(self) -> None:
        if self.root.is_symlink() or self.path.is_symlink():
            raise _integrity("Operation store must not traverse a symlink.", path=str(self.path))
        if not self.root.is_dir():
            raise _integrity("Operation store root is not a directory.", path=str(self.root))

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[None]:
        self._check_paths()
        path = self.root / ".scnsim.lock"
        flags = (os.O_RDWR | os.O_CREAT) if exclusive else os.O_RDONLY
        descriptor = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with self._scope("sqlite_lock_wait", {"exclusive": exclusive}):
                fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _connect(self, *, mode: str) -> sqlite3.Connection:
        self._check_paths()
        connection = sqlite3.connect(self.path.absolute().as_uri() + "?mode=" + mode,
                                     uri=True, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if mode != "ro":
                if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                    raise _integrity("Operation store journal mode must be DELETE.")
                connection.execute("PRAGMA synchronous=EXTRA")
            return connection
        except BaseException:
            _close(connection)
            raise

    def _verify_binding(self, connection: sqlite3.Connection) -> None:
        actual = dict(connection.execute("SELECT key,value FROM metadata"))
        if actual.get("schema_version") is not None and actual["schema_version"] != str(_VERSION):
            raise _unsupported_version(actual["schema_version"])
        if actual != self._metadata:
            raise _integrity("Operation database does not belong to this bound Plan leaf.",
                             expected=self._metadata, actual=actual)

    def initialize(self) -> None:
        created_root = not self.root.exists()
        self.root.mkdir(exist_ok=True)
        if created_root:
            _fsync_directory(self.root.parent)
        with self._lock(exclusive=True):
            exists = self.path.exists()
            connection = self._connect(mode="rw" if exists else "rwc")
            try:
                if exists:
                    self._verify_binding(connection)
                    return
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for statement in _SCHEMA:
                        connection.execute(statement)
                    connection.executemany("INSERT INTO metadata VALUES (?,?)", self._metadata.items())
                    connection.execute("COMMIT")
                except BaseException:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
                _fsync_directory(self.root)
            finally:
                _close(connection)

    @contextmanager
    def reader(self) -> Iterator[Snapshot | None]:
        if not self.path.exists() and not self.path.is_symlink():
            yield None
            return
        with self._lock(exclusive=False):
            connection = None
            try:
                connection = self._connect(mode="ro")
                connection.execute("BEGIN")
                self._verify_binding(connection)
                yield Snapshot(connection)
            except sqlite3.Error as error:
                _recovery_error(error)
                raise
            finally:
                if connection is not None:
                    _close(connection)

    @contextmanager
    def transaction(self, transaction_id: str, *, expected_revisions: Mapping[str, int]) -> Iterator[StoreTransaction]:
        with self._lock(exclusive=True):
            connection = self._connect(mode="rw")
            transaction = StoreTransaction(connection, transaction_id)
            details = {"transaction_id": transaction_id, **transaction.counts}
            try:
                self._verify_binding(connection)
                with self._scope("sqlite_transaction_write", details):
                    connection.execute("BEGIN IMMEDIATE")
                    if Snapshot(connection).transaction_status(transaction_id)["status"] == "committed":
                        raise _integrity("A committed transaction must not be replayed.", transaction_id=transaction_id)
                    for stream, expected in expected_revisions.items():
                        row = connection.execute("SELECT revision FROM streams WHERE name=?", (stream,)).fetchone()
                        actual = 0 if row is None else row[0]
                        if actual != expected:
                            raise _integrity("Operation stream revision changed.", stream=stream,
                                             expected=expected, actual=actual)
                    yield transaction
                    connection.execute("INSERT INTO transactions VALUES (?)", (transaction_id,))
                    details.update(transaction.counts)
                    try:
                        with self._scope("sqlite_commit", details):
                            connection.execute("COMMIT")
                    except BaseException as error:
                        transaction.outcome = {"status": "unknown", "transaction_id": transaction_id}
                        error.operation_transaction_outcome = dict(transaction.outcome)
                        # Closing releases the old native transaction before an
                        # independent connection inspects its marker under flock.
                        _close(connection)
                        try:
                            with self._scope("sqlite_reconcile", {"transaction_id": transaction_id}):
                                transaction.outcome = self._reconcile(transaction_id)
                        except BaseException as secondary:
                            _secondary(error, "SQLite commit reconciliation", secondary)
                        error.operation_transaction_outcome = dict(transaction.outcome)
                        raise
                    transaction.outcome = {"status": "committed", "transaction_id": transaction_id}
            finally:
                details.update(transaction.counts)
                transaction._verified_objects.clear()
                _close(connection)

    def _reconcile(self, transaction_id: str) -> dict[str, Any]:
        try:
            connection = self._connect(mode="ro")
            try:
                self._verify_binding(connection)
                return Snapshot(connection).transaction_status(transaction_id)
            finally:
                _close(connection)
        except Exception as error:
            return {"status": "unknown", "transaction_id": transaction_id,
                    "reconciliation_error": str(error)}

    def recover(self) -> None:
        """Perform native rollback and confirm this leaf's committed binding.

        This bounded reopen does not audit historical payloads. SQLite owns
        journal recovery; transaction-marker reconciliation remains the separate
        exact-txid path used for an unconfirmed commit, with no blind replay.
        """
        with self._lock(exclusive=True):
            connection = self._connect(mode="rw")
            try:
                # The first native read performs any required rollback. Never
                # inspect or edit SQLite's rollback journal ourselves.
                self._verify_binding(connection)
            finally:
                _close(connection)
            # A fresh native read transaction confirms the committed binding,
            # rather than treating the recovery connection as readback evidence.
            connection = self._connect(mode="ro")
            try:
                connection.execute("BEGIN")
                self._verify_binding(connection)
            finally:
                _close(connection)

    def audit(self) -> None:
        """Explicit complete structural/reference/byte audit of committed data.

        This readonly inspection never performs recovery. A hot journal retains
        the same explicit recovery-required failure as other readonly readers.
        """
        with self._lock(exclusive=False):
            connection = self._connect(mode="ro")
            try:
                connection.execute("BEGIN")
                self._verify_binding(connection)
                if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise _integrity("SQLite reported damaged operation evidence.")
                if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise _integrity("Operation evidence contains a broken object reference.")
                snapshot = Snapshot(connection)
                for digest, role, length in connection.execute(
                    "SELECT r.sha256,r.role,length(o.payload) FROM object_roles r JOIN objects o USING(sha256)"
                ):
                    snapshot.get_object(_reference(digest, role, length))
            except sqlite3.Error as error:
                _recovery_error(error)
                raise
            finally:
                _close(connection)


class Snapshot:
    """One native committed read snapshot; valid only inside reader()."""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection
        self._verified_objects = {}
        self._reuse_verified = False

    @contextmanager
    def reuse_verified_objects(self):
        """Reuse verified bytes only within one numerical projection."""
        previous = self._reuse_verified
        self._reuse_verified = True
        try:
            yield self
        finally:
            self._reuse_verified = previous
            if not previous:
                self._verified_objects.clear()

    @staticmethod
    def _object_key(reference: Mapping[str, Any]):
        if set(reference) != {"schema_version", "storage", "database", "role", "sha256", "byte_length"} or (
            reference["schema_version"] != _VERSION or reference["storage"] != "sqlite"
            or reference["database"] != _DATABASE
        ):
            raise _integrity("Invalid SQLite object reference.", reference=dict(reference))
        return (reference["sha256"], reference["role"], reference["byte_length"])

    def get_object(self, reference: Mapping[str, Any]) -> bytes:
        key = self._object_key(reference)
        if key in self._verified_objects:
            return self._verified_objects[key]
        row = self._connection.execute(
            "SELECT o.payload FROM objects o JOIN object_roles r USING(sha256) WHERE r.sha256=? AND r.role=?",
            (reference["sha256"], reference["role"]),
        ).fetchone()
        if row is None:
            raise _integrity("Referenced SQLite object is missing.", reference=dict(reference))
        payload = bytes(row[0])
        if len(payload) != reference["byte_length"] or hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise _integrity("SQLite object bytes do not match their content reference.", reference=dict(reference))
        if self._reuse_verified:
            self._verified_objects[key] = payload
        return payload

    def get_objects(self, references: Sequence[Mapping[str, Any]]):
        """JOIN one evidence group's references to bytes, preserving input order."""
        for reference in references:
            self._object_key(reference)
        cursor = self._connection.execute(
            "SELECT j.key,o.payload,r.role FROM json_each(?) j "
            "LEFT JOIN objects o ON o.sha256=json_extract(j.value,'$.sha256') "
            "LEFT JOIN object_roles r ON r.sha256=o.sha256 AND r.role=json_extract(j.value,'$.role') "
            "ORDER BY cast(j.key AS INTEGER)", (json.dumps(list(references)),))
        for index, payload, role in cursor:
            reference = references[index]
            key = self._object_key(reference)
            if payload is None or role is None:
                raise _integrity("Referenced SQLite object is missing.", reference=dict(reference))
            if key in self._verified_objects:
                yield self._verified_objects[key]
                continue
            payload = bytes(payload)
            if len(payload) != reference['byte_length'] or hashlib.sha256(payload).hexdigest() != reference['sha256']:
                raise _integrity("SQLite object bytes do not match their content reference.", reference=dict(reference))
            if self._reuse_verified:
                self._verified_objects[key] = payload
            yield payload

    def iter_task_changes(self, stream: str, *, include_evaluations: bool):
        """Locate numerical/control rows; SQL filtering is never verification.

        Old task streams have only task_change as their indexed kind. Native
        JSON inspection avoids Python materialization of diagnostic payloads.
        Malformed JSON is included so it cannot disappear as a filtered row.
        Selected payloads still pass the ordinary role/length/hash boundary.
        """
        # Preserve the task event sequence invariant without decoding or returning
        # diagnostic bodies to Python. Malformed JSON is verified below.
        count, first, last, distinct, wrong_type = self._connection.execute(
            "WITH rows AS (SELECT CASE WHEN json_valid(o.payload) THEN o.payload ELSE '{}' END AS body "
            "FROM entries e LEFT JOIN objects o ON o.sha256=e.sha256 WHERE e.stream=?) "
            "SELECT count(*),min(json_extract(body,'$.event.sequence')),"
            "max(json_extract(body,'$.event.sequence')),count(DISTINCT json_extract(body,'$.event.sequence')),"
            "sum(CASE WHEN json_type(body,'$.event.sequence')='integer' THEN 0 ELSE 1 END) "
            "FROM rows WHERE json_extract(body,'$.kind')='event'", (stream,)
        ).fetchone()
        if count and (first != 0 or last != count-1 or distinct != count or wrong_type):
            raise _integrity("Benchmark task event sequence is not contiguous.", stream=stream)
        excluded = ["operation_span", "timing", "population_observed", "progress"]
        if not include_evaluations:
            excluded.append("evaluation")
        marks = ",".join("?" for _ in excluded)
        cursor = self._connection.execute(
            "SELECT e.sequence,e.sha256,e.role,o.payload,r.role FROM entries e "
            "LEFT JOIN objects o ON o.sha256=e.sha256 "
            "LEFT JOIN object_roles r ON r.sha256=e.sha256 AND r.role=e.role "
            "WHERE e.stream=? AND CASE WHEN o.payload IS NULL OR NOT json_valid(o.payload) THEN 1 "
            "WHEN json_extract(o.payload,'$.kind')='measurement' THEN 0 "
            "WHEN json_extract(o.payload,'$.kind')='event' THEN "
            f"coalesce(json_extract(o.payload,'$.event.kind') NOT IN ({marks}),1) ELSE 1 END "
            "ORDER BY e.sequence", (stream, *excluded))
        for sequence, digest, role, payload, registered_role in cursor:
            if payload is None or registered_role is None:
                raise _integrity("Task stream references missing object bytes or role.", sequence=sequence)
            payload = bytes(payload)
            reference = _reference(digest, role, len(payload))
            if hashlib.sha256(payload).hexdigest() != digest:
                raise _integrity("Task stream object bytes do not match their content reference.", reference=reference)
            if self._reuse_verified:
                self._verified_objects[(digest, role, len(payload))] = payload
            yield sequence, reference, payload

    def _ref(self, digest: str, role: str) -> dict[str, Any]:
        row = self._connection.execute("SELECT length(payload) FROM objects WHERE sha256=?", (digest,)).fetchone()
        if row is None:
            raise _integrity("SQLite object index references missing bytes.", sha256=digest)
        return _reference(digest, role, row[0])

    def stream_revision(self, stream: str) -> int:
        """Read the selected revision without materializing historical entries."""
        row = self._connection.execute("SELECT revision FROM streams WHERE name=?", (stream,)).fetchone()
        return 0 if row is None else row[0]

    def read_stream(self, stream: str) -> dict[str, Any]:
        row = self._connection.execute("SELECT revision FROM streams WHERE name=?", (stream,)).fetchone()
        entries = []
        for sequence, kind, digest, role, length in self._connection.execute(
            "SELECT e.sequence,e.kind,e.sha256,e.role,length(o.payload) FROM entries e "
            "LEFT JOIN objects o ON o.sha256=e.sha256 WHERE e.stream=? ORDER BY e.sequence", (stream,)
        ):
            if length is None:
                raise _integrity("SQLite stream references missing bytes.", sha256=digest)
            entries.append({"sequence": sequence, "kind": kind,
                            "reference": _reference(digest, role, length)})
        return {"revision": 0 if row is None else row[0], "entries": entries}

    def read_pointer(self, name: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT revision,sha256,role FROM pointers WHERE name=?", (name,)).fetchone()
        return None if row is None else {"revision": row[0], "reference": self._ref(row[1], row[2])}

    def transaction_status(self, transaction_id: str) -> dict[str, str]:
        row = self._connection.execute("SELECT 1 FROM transactions WHERE transaction_id=?", (transaction_id,)).fetchone()
        return {"status": "not_committed" if row is None else "committed", "transaction_id": transaction_id}

    def _query(self, table: str, clauses: list[str], values: list[Any]) -> list[dict[str, Any]]:
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._connection.execute(
            f"SELECT sha256,role FROM {table}{where} ORDER BY start_ns,operation_id", values)
        result = []
        for digest, role in rows:
            reference = self._ref(digest, role)
            result.append({"reference": reference, "payload": self.get_object(reference)})
        return result

    def query_operations(self, *, operation_ids: Sequence[str] | None = None, method: str | None = None,
                         backend: str | None = None, precision: str | None = None,
                         status: str | None = None) -> list[dict[str, Any]]:
        clauses, values = [], []
        if operation_ids is not None:
            if not operation_ids:
                return []
            clauses.append("operation_id IN (" + ",".join("?" for _ in operation_ids) + ")")
            values.extend(operation_ids)
        for key, value in (("method", method), ("backend", backend), ("precision", precision), ("status", status)):
            if value is not None:
                clauses.append(key + "=?")
                values.append(value)
        return self._query("operations", clauses, values)

    @staticmethod
    def _json_reference(value: Mapping[str, Any]) -> str:
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _decoded_reference(value: str) -> dict[str, Any]:
        result = json.loads(value)
        if not isinstance(result, dict):
            raise _integrity("Completion index contains a malformed object reference.")
        return result

    def read_generation_index(self, completion_sha: str, generation: int) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT block_ref,first_ordinal,row_count,summary_json FROM completion_generations "
            "WHERE completion_sha=? AND generation=?", (completion_sha, generation)
        ).fetchone()
        if row is None:
            return None
        summary = json.loads(row[3])
        if not isinstance(summary, dict):
            raise _integrity("Generation index summary is malformed.", generation=generation)
        return {"generation": generation, "block_ref": self._decoded_reference(row[0]),
                "first_ordinal": row[1], "row_count": row[2], "summary_json": summary}

    def completion_generation_count(self, completion_sha: str) -> int:
        return self._connection.execute(
            "SELECT count(*) FROM completion_generations WHERE completion_sha=?", (completion_sha,)
        ).fetchone()[0]

    def completion_candidate_count(self, completion_sha: str) -> int:
        return self._connection.execute(
            "SELECT count(*) FROM completion_candidates WHERE completion_sha=?", (completion_sha,)
        ).fetchone()[0]

    def read_candidate_index(self, completion_sha: str, ordinal: int) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT generation,row_offset,block_ref,value_ref,summary_json FROM completion_candidates "
            "WHERE completion_sha=? AND ordinal=?", (completion_sha, ordinal)
        ).fetchone()
        if row is None:
            return None
        summary = json.loads(row[4])
        if not isinstance(summary, dict):
            raise _integrity("Candidate index summary is malformed.", ordinal=ordinal)
        return {"ordinal": ordinal, "generation": row[0], "row_offset": row[1],
                "block_ref": self._decoded_reference(row[2]),
                "value_ref": self._decoded_reference(row[3]), "summary_json": summary}

    def read_candidate_summaries(self, completion_sha: str) -> list[dict[str, Any]]:
        rows = []
        cursor = self._connection.execute(
            "SELECT ordinal,generation,row_offset,summary_json FROM completion_candidates "
            "WHERE completion_sha=? ORDER BY ordinal", (completion_sha,)
        )
        for ordinal, generation, row_offset, raw in cursor:
            summary = json.loads(raw)
            if not isinstance(summary, dict):
                raise _integrity("Candidate index summary is malformed.", ordinal=ordinal)
            rows.append(summary)
        return rows

    def read_candidate_indexes(self, completion_sha: str) -> list[dict[str, Any]]:
        rows = []
        cursor = self._connection.execute(
            "SELECT ordinal,generation,row_offset,block_ref,value_ref,summary_json "
            "FROM completion_candidates WHERE completion_sha=? ORDER BY ordinal", (completion_sha,)
        )
        for ordinal, generation, row_offset, block_ref, value_ref, raw in cursor:
            summary = json.loads(raw)
            if not isinstance(summary, dict):
                raise _integrity("Candidate index summary is malformed.", ordinal=ordinal)
            rows.append({"ordinal": ordinal, "generation": generation, "row_offset": row_offset,
                         "block_ref": self._decoded_reference(block_ref),
                         "value_ref": self._decoded_reference(value_ref), "summary_json": summary})
        return rows

class StoreTransaction(Snapshot):
    """Mutation mechanics inside one store-owned SQL transaction."""

    def __init__(self, connection: sqlite3.Connection, transaction_id: str):
        super().__init__(connection)
        # Transaction writes compare each body at insertion and do not retain
        # verified payload bytes for the lifetime of a generation publication.
        # Pointer writes perform their ordinary single-object read/verification.
        self._reuse_verified = False
        self.counts = {"put_object_calls": 0, "logical_object_bytes": 0,
                       "inserted_objects": 0, "appended_rows": 0}
        self.outcome: dict[str, Any] = {"status": "not_committed", "transaction_id": transaction_id}

    def put_object(self, payload: bytes, *, role: str) -> dict[str, Any]:
        digest = hashlib.sha256(payload).hexdigest()
        self.counts["put_object_calls"] += 1
        self.counts["logical_object_bytes"] += len(payload)
        reference = _reference(digest, role, len(payload))
        cursor = self._connection.execute("INSERT INTO objects VALUES (?,?) ON CONFLICT DO NOTHING", (digest, payload))
        self.counts["inserted_objects"] += cursor.rowcount
        if cursor.rowcount == 0:
            existing = self._connection.execute(
                "SELECT payload FROM objects WHERE sha256=?", (digest,)
            ).fetchone()[0]
            if existing != payload:
                raise _integrity("Object digest identifies different stored bytes.", sha256=digest)
        self._connection.execute("INSERT INTO object_roles VALUES (?,?) ON CONFLICT DO NOTHING", (digest, role))
        return reference

    def append_with_ref(self, stream: str, kind: str, payload: bytes) -> tuple[int, dict[str, Any]]:
        reference = self.put_object(payload, role=kind)
        self._connection.execute("INSERT INTO streams VALUES (?,0) ON CONFLICT DO NOTHING", (stream,))
        self._connection.execute("UPDATE streams SET revision=revision+1 WHERE name=?", (stream,))
        sequence = self._connection.execute("SELECT revision FROM streams WHERE name=?", (stream,)).fetchone()[0]
        self._connection.execute("INSERT INTO entries VALUES (?,?,?,?,?)",
                                 (stream, sequence, kind, reference["sha256"], reference["role"]))
        self.counts["appended_rows"] += 1
        return sequence, reference

    def append(self, stream: str, kind: str, payload: bytes) -> int:
        return self.append_with_ref(stream, kind, payload)[0]

    def set_pointer(self, name: str, object_ref: Mapping[str, Any]) -> None:
        self.get_object(object_ref)
        self._connection.execute(
            "INSERT INTO pointers VALUES (?,1,?,?) ON CONFLICT(name) DO UPDATE SET "
            "revision=revision+1,sha256=excluded.sha256,role=excluded.role",
            (name, object_ref["sha256"], object_ref["role"]))

    def select_first(self, name: str, object_ref: Mapping[str, Any]) -> dict[str, Any]:
        selected = self.read_pointer(name)
        if selected is not None:
            return selected["reference"]
        self.set_pointer(name, object_ref)
        return dict(object_ref)

    def index_operation(self, operation_id: str, object_ref: Mapping[str, Any], *, method: str | None,
                        backend: str | None, precision: str | None, status: str | None,
                        start_ns: int | None, end_ns: int | None, clock_id: str | None) -> None:
        self.get_object(object_ref)
        self._connection.execute(
            "INSERT INTO operations VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(operation_id) DO UPDATE SET "
            "sha256=excluded.sha256,role=excluded.role,method=excluded.method,backend=excluded.backend,"
            "precision=excluded.precision,status=excluded.status,start_ns=excluded.start_ns,"
            "end_ns=excluded.end_ns,clock_id=excluded.clock_id",
            (operation_id, object_ref["sha256"], object_ref["role"], method, backend, precision,
             status, start_ns, end_ns, clock_id))

    def index_completion(self, completion_sha: str, generations: Sequence[Mapping[str, Any]],
                         candidates: Sequence[Mapping[str, Any]]) -> None:
        """Publish query locators atomically with their sealed completion objects."""
        for row in generations:
            summary = canonical_json_bytes(row["summary_json"]).decode("utf-8")
            self._connection.execute(
                "INSERT INTO completion_generations VALUES (?,?,?,?,?,?)",
                (completion_sha, row["generation"], self._json_reference(row["block_ref"]),
                 row["first_ordinal"], row["row_count"], summary),
            )
        for row in candidates:
            summary = canonical_json_bytes(row["summary_json"]).decode("utf-8")
            self._connection.execute(
                "INSERT INTO completion_candidates VALUES (?,?,?,?,?,?,?)",
                (completion_sha, row["ordinal"], row["generation"], row["row_offset"],
                 self._json_reference(row["block_ref"]), self._json_reference(row["value_ref"]), summary),
            )

