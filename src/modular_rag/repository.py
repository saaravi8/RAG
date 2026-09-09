"""Safe, local-only repository discovery and indexing orchestration."""

import hashlib
import json
import os
import re
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import (
    Any,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

try:
    from pathspec.patterns.gitignore.spec import GitIgnoreSpecPattern
except ImportError:  # pathspec 0.12 compatibility
    from pathspec.patterns import GitWildMatchPattern as GitIgnoreSpecPattern

from rag_ingestion import Document, DocumentSource

from .errors import (
    RepositoryIngestionError,
    RepositoryResourceLimitError,
    UnsafeRepositoryError,
)
from .fingerprint import IndexFingerprint, describe_component
from .indexing import Indexer
from .ports import CoordinatedPreparedDocumentReplacement
from .transactions import IndexTransactionCoordinator


_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_SAFE_REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_PROMPT_INJECTION = re.compile(
    r"(?:ignore\s+(?:all\s+)?previous\s+instructions|"
    r"treat\s+this\s+(?:file|text)\s+as\s+(?:a\s+)?system\s+message|"
    r"(?:call|invoke|use)\s+(?:a\s+)?(?:shell|tool|function)|"
    r"(?:reveal|upload|print|read)\s+(?:the\s+)?(?:environment|credentials|secrets)|"
    r"change\s+(?:the\s+)?(?:tenant|access|security)\s+filter)",
    re.IGNORECASE,
)


LANGUAGE_BY_SUFFIX: Mapping[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "proto",
}

CONFIG_BY_SUFFIX: Mapping[str, str] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".tf": "terraform",
    ".hcl": "hcl",
}

DOCUMENT_BY_SUFFIX: Mapping[str, str] = {
    ".md": "md",
    ".markdown": "markdown",
    ".json": "json",
    ".html": "html",
    ".htm": "htm",
    ".csv": "csv",
    ".txt": "txt",
}

SPECIAL_FILES: Mapping[str, Tuple[str, str]] = {
    "dockerfile": ("dockerfile", "config"),
    "makefile": ("make", "config"),
    "cmakelists.txt": ("cmake", "config"),
    ".gitignore": ("gitignore", "config"),
    "package.json": ("json", "config"),
}

DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "vendor",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".next",
        ".cache",
        "coverage",
        "htmlcov",
        "dist",
        "build",
        "out",
        "target",
        "tmp",
        "temp",
        "generated",
    }
)

LOCKFILES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pipfile.lock",
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
    }
)


@dataclass(frozen=True)
class RepositoryLimits:
    """Hard bounds applied before expensive repository processing."""

    max_files: int = 10_000
    max_file_bytes: int = 2_000_000
    max_total_bytes: int = 100_000_000
    max_depth: int = 32
    max_lines_per_file: int = 50_000
    max_chunks_per_file: int = 500
    max_total_chunks: int = 50_000
    max_errors: int = 100
    max_discovered_entries: int = 100_000
    max_issues: int = 1_000
    max_metadata_length: int = 512
    max_ignore_file_bytes: int = 131_072

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("{} must be an integer.".format(name))
            if value <= 0:
                raise ValueError("{} must be a positive integer.".format(name))


@dataclass(frozen=True)
class RepositoryPolicy:
    """Explicit opt-ins layered over conservative ingestion defaults."""

    limits: RepositoryLimits = field(default_factory=RepositoryLimits)
    include_generated: bool = False
    include_lockfiles: bool = False
    include_unknown_text: bool = False


@dataclass(frozen=True)
class RepositoryIssue:
    """A content-free reason explaining why one relative path was not indexed."""

    relative_path: str
    code: str


class _BoundedDiagnostics:
    """Keep issue details bounded while retaining truthful aggregate counts."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.skips: List[RepositoryIssue] = []
        self.failures: List[RepositoryIssue] = []
        self.skipped_count = 0
        self.failed_count = 0
        self._truncated = False
        self._terminal_codes = set()

    def add_skip(
        self, relative_path: str, code: str, *, terminal: bool = False
    ) -> None:
        self.skipped_count += 1
        self._store(self.skips, RepositoryIssue(relative_path, code), terminal=terminal)

    def add_failure(self, relative_path: str, code: str) -> None:
        self.failed_count += 1
        self._store(self.failures, RepositoryIssue(relative_path, code))

    def add_skipped_count(self, count: int) -> None:
        self.skipped_count += count

    def _store(
        self,
        target: List[RepositoryIssue],
        issue: RepositoryIssue,
        *,
        terminal: bool = False,
    ) -> None:
        if len(self.skips) + len(self.failures) < self.limit:
            target.append(issue)
            if terminal:
                self._terminal_codes.add(issue.code)
            return
        self._truncated = True
        if terminal and issue.code not in self._terminal_codes:
            self._replace_last(target, issue)
            self._terminal_codes.add(issue.code)

    def _replace_last(
        self, target: List[RepositoryIssue], issue: RepositoryIssue
    ) -> None:
        if target:
            target[-1] = issue
        elif target is self.skips:
            self.failures.pop()
            self.skips.append(issue)
        else:
            self.skips.pop()
            self.failures.append(issue)

    def snapshots(
        self,
    ) -> Tuple[Tuple[RepositoryIssue, ...], Tuple[RepositoryIssue, ...]]:
        if self._truncated and not self._terminal_codes:
            marker = RepositoryIssue("", "diagnostics_truncated")
            if len(self.skips) + len(self.failures) < self.limit:
                self.skips.append(marker)
            else:
                self._replace_last(self.skips, marker)
        return tuple(self.skips), tuple(self.failures)


@dataclass(frozen=True)
class RepositoryIndexReport:
    """Immutable summary of one repository snapshot ingestion."""

    repository_id: str
    commit_sha: Optional[str]
    snapshot_kind: str
    discovered: int
    indexed: int
    skipped: int
    redacted: int
    failed: int
    deleted: int
    chunk_count: int
    secret_redaction_count: int
    complete: bool
    skips: Tuple[RepositoryIssue, ...] = ()
    failures: Tuple[RepositoryIssue, ...] = ()
    unchanged: int = 0


@dataclass(frozen=True)
class RepositoryManifestEntry:
    """Identity recorded after one repository file is indexed successfully."""

    repository_id: str
    relative_path: str
    document_id: str
    file_hash: str
    commit_sha: Optional[str]
    index_fingerprint: str = ""
    chunk_count: int = 0
    document_state_token: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_count, int) or isinstance(
            self.chunk_count, bool
        ):
            raise TypeError("chunk_count must be an integer.")
        if self.chunk_count < 0:
            raise ValueError("chunk_count must be a non-negative integer.")


def _repository_document_id(repository_id: str, relative_path: str) -> str:
    identity = "{}\x00{}".format(repository_id, relative_path)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return "repo_{}".format(digest)


def _validate_manifest_snapshot(
    repository_id: str,
    entries: Mapping[str, RepositoryManifestEntry],
    *,
    allow_legacy_state_tokens: bool,
    error_type: type = ValueError,
) -> Dict[str, RepositoryManifestEntry]:
    """Validate a manifest before it can influence reuse or stored state."""

    def reject(message: str) -> None:
        raise error_type(message)

    if not isinstance(entries, Mapping):
        reject("Repository manifest entries must be a mapping.")
    if not isinstance(repository_id, str) or not _SAFE_REPOSITORY_ID.fullmatch(
        repository_id
    ):
        reject("Repository manifest has an invalid repository ID.")

    snapshot: Dict[str, RepositoryManifestEntry] = {}
    for key, entry in entries.items():
        if not isinstance(key, str) or not isinstance(entry, RepositoryManifestEntry):
            reject("Repository manifest entries have invalid types.")
        if key != entry.relative_path:
            reject("Repository manifest keys must match entry relative paths.")
        path = PurePosixPath(key)
        if (
            not key
            or path.is_absolute()
            or path.as_posix() != key
            or "\\" in key
            or _CONTROL_CHARACTERS.search(key)
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            reject("Repository manifest contains an invalid relative path.")
        if entry.repository_id != repository_id:
            reject("Repository manifest entries must match the repository ID.")
        if entry.document_id != _repository_document_id(repository_id, key):
            reject("Repository manifest contains an invalid document ID.")
        if not isinstance(entry.file_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry.file_hash
        ):
            reject("Repository manifest contains an invalid file hash.")
        legacy = allow_legacy_state_tokens and entry.document_state_token == ""
        if not legacy and (
            not isinstance(entry.document_state_token, str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry.document_state_token)
        ):
            reject("Repository manifest contains an invalid document state token.")
        if legacy:
            if entry.index_fingerprint and not re.fullmatch(
                r"[0-9a-f]{64}", entry.index_fingerprint
            ):
                reject("Repository manifest contains an invalid index fingerprint.")
        elif not isinstance(entry.index_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry.index_fingerprint
        ):
            reject("Repository manifest contains an invalid index fingerprint.")
        if entry.commit_sha is not None and (
            not isinstance(entry.commit_sha, str)
            or not re.fullmatch(r"[0-9a-f]{40,64}", entry.commit_sha)
        ):
            reject("Repository manifest contains an invalid commit SHA.")
        minimum_chunks = 0 if legacy else 1
        if (
            not isinstance(entry.chunk_count, int)
            or isinstance(entry.chunk_count, bool)
            or entry.chunk_count < minimum_chunks
        ):
            reject("Repository manifest contains an invalid chunk count.")
        snapshot[key] = entry
    return snapshot


class RepositoryManifest(Protocol):
    """Persistence boundary used to compare successive repository snapshots."""

    def entries(self, repository_id: str) -> Mapping[str, RepositoryManifestEntry]:
        """Return a defensive mapping of the repository's indexed files."""

        ...

    def replace(
        self,
        repository_id: str,
        entries: Mapping[str, RepositoryManifestEntry],
    ) -> None:
        """Replace the complete stored manifest for ``repository_id``."""

        ...

    def prepare_replace(
        self,
        repository_id: str,
        entries: Mapping[str, RepositoryManifestEntry],
    ) -> CoordinatedPreparedDocumentReplacement:
        """Stage a complete manifest snapshot without exposing it."""

        ...

    def synchronized_scan(self, repository_id: str) -> ContextManager[None]:
        """Lease one repository identity for the duration of a refresh."""

        ...


class _PreparedManifestReplacement:
    def __init__(
        self,
        manifest: "InMemoryRepositoryManifest",
        repository_id: str,
        previous: Optional[Dict[str, RepositoryManifestEntry]],
        candidate: Dict[str, RepositoryManifestEntry],
        expected_generation: int,
    ) -> None:
        self._manifest = manifest
        self._repository_id = repository_id
        self._previous = previous
        self._candidate = candidate
        self._expected_generation = expected_generation
        self._committed_generation: Optional[int] = None
        self._committed = False
        self._rolled_back = False

    @property
    def transaction_coordinator(self) -> IndexTransactionCoordinator:
        return self._manifest.transaction_coordinator

    def commit(self) -> None:
        with self._manifest.transaction_coordinator.synchronized():
            with self._manifest._lock:
                if self._committed or self._rolled_back:
                    return
                if self._manifest._generation(self._repository_id) != (
                    self._expected_generation
                ):
                    raise RepositoryIngestionError(
                        "Prepared repository manifest replacement is stale."
                    )
                self._manifest._entries[self._repository_id] = self._candidate
                self._committed_generation = self._manifest._advance_generation(
                    self._repository_id
                )
                self._committed = True

    def rollback(self) -> None:
        with self._manifest.transaction_coordinator.synchronized():
            with self._manifest._lock:
                if self._rolled_back:
                    return
                if (
                    self._committed
                    and self._manifest._generation(self._repository_id)
                    == self._committed_generation
                ):
                    if self._previous is None:
                        self._manifest._entries.pop(self._repository_id, None)
                    else:
                        self._manifest._entries[self._repository_id] = self._previous
                    self._manifest._advance_generation(self._repository_id)
                self._rolled_back = True


class _PreparedManifestEntryChange:
    """Prevalidated in-place update for one default-manifest entry."""

    def __init__(
        self,
        manifest: "InMemoryRepositoryManifest",
        repository_id: str,
        relative_path: str,
        previous: Optional[RepositoryManifestEntry],
        candidate: Optional[RepositoryManifestEntry],
        expected_generation: int,
    ) -> None:
        self._manifest = manifest
        self._repository_id = repository_id
        self._relative_path = relative_path
        self._previous = previous
        self._candidate = candidate
        self._expected_generation = expected_generation
        self._committed_generation: Optional[int] = None
        self._committed = False
        self._rolled_back = False

    @property
    def transaction_coordinator(self) -> IndexTransactionCoordinator:
        return self._manifest.transaction_coordinator

    def commit(self) -> None:
        with self._manifest.transaction_coordinator.synchronized():
            with self._manifest._lock:
                if self._committed or self._rolled_back:
                    return
                if self._manifest._generation(self._repository_id) != (
                    self._expected_generation
                ):
                    raise RepositoryIngestionError(
                        "Prepared repository manifest entry change is stale."
                    )
                entries = self._manifest._entries.setdefault(
                    self._repository_id, {}
                )
                if self._candidate is None:
                    entries.pop(self._relative_path, None)
                else:
                    entries[self._relative_path] = self._candidate
                self._committed_generation = self._manifest._advance_generation(
                    self._repository_id
                )
                self._committed = True

    def rollback(self) -> None:
        with self._manifest.transaction_coordinator.synchronized():
            with self._manifest._lock:
                if self._rolled_back:
                    return
                if (
                    self._committed
                    and self._manifest._generation(self._repository_id)
                    == self._committed_generation
                ):
                    entries = self._manifest._entries.setdefault(
                        self._repository_id, {}
                    )
                    if self._previous is None:
                        entries.pop(self._relative_path, None)
                    else:
                        entries[self._relative_path] = self._previous
                    self._manifest._advance_generation(self._repository_id)
                self._rolled_back = True


class InMemoryRepositoryManifest:
    """Thread-safe demo manifest; production applications should persist it."""

    def __init__(
        self,
        *,
        transaction_coordinator: Optional[IndexTransactionCoordinator] = None,
    ) -> None:
        self._entries: Dict[str, Dict[str, RepositoryManifestEntry]] = {}
        self._generations: Dict[str, int] = {}
        self._scan_locks: Dict[str, RLock] = {}
        self._lock = RLock()
        self.transaction_coordinator = (
            transaction_coordinator
            if transaction_coordinator is not None
            else IndexTransactionCoordinator()
        )

    @contextmanager
    def synchronized_scan(self, repository_id: str) -> Iterator[None]:
        with self._lock:
            scan_lock = self._scan_locks.setdefault(repository_id, RLock())
        with scan_lock:
            yield

    def entries(self, repository_id: str) -> Mapping[str, RepositoryManifestEntry]:
        """Return a copy of the current in-memory manifest."""

        with self.transaction_coordinator.synchronized():
            with self._lock:
                return _validate_manifest_snapshot(
                    repository_id,
                    self._entries.get(repository_id, {}),
                    allow_legacy_state_tokens=True,
                )

    def replace(
        self,
        repository_id: str,
        entries: Mapping[str, RepositoryManifestEntry],
    ) -> None:
        """Replace one repository manifest after validating entry ownership."""

        with self.transaction_coordinator.synchronized():
            prepared = self.prepare_replace(repository_id, entries)
            prepared.commit()

    def prepare_replace(
        self,
        repository_id: str,
        entries: Mapping[str, RepositoryManifestEntry],
    ) -> "CoordinatedPreparedDocumentReplacement":
        with self.transaction_coordinator.synchronized():
            with self._lock:
                snapshot = _validate_manifest_snapshot(
                    repository_id,
                    entries,
                    allow_legacy_state_tokens=True,
                )
                previous_snapshot = _validate_manifest_snapshot(
                    repository_id,
                    self._entries.get(repository_id, {}),
                    allow_legacy_state_tokens=True,
                )
                for path, entry in snapshot.items():
                    if (
                        not entry.document_state_token
                        and previous_snapshot.get(path) != entry
                    ):
                        raise ValueError(
                            "New manifest entries require a document state token."
                        )
                previous = self._entries.get(repository_id)
                return _PreparedManifestReplacement(
                    self,
                    repository_id,
                    previous,
                    snapshot,
                    self._generation(repository_id),
                )

    def prepare_replace_entry(
        self,
        repository_id: str,
        relative_path: str,
        entry: RepositoryManifestEntry,
    ) -> "CoordinatedPreparedDocumentReplacement":
        """Stage one validated entry without copying the repository manifest."""

        with self.transaction_coordinator.synchronized():
            with self._lock:
                snapshot = _validate_manifest_snapshot(
                    repository_id,
                    {relative_path: entry},
                    allow_legacy_state_tokens=True,
                )
                previous = self._entries.get(repository_id, {}).get(relative_path)
                if not entry.document_state_token and previous != entry:
                    raise ValueError(
                        "New manifest entries require a document state token."
                    )
                return _PreparedManifestEntryChange(
                    self,
                    repository_id,
                    relative_path,
                    previous,
                    snapshot[relative_path],
                    self._generation(repository_id),
                )

    def prepare_delete_entry(
        self,
        repository_id: str,
        relative_path: str,
    ) -> "CoordinatedPreparedDocumentReplacement":
        """Stage removal of one entry without copying the repository manifest."""

        with self.transaction_coordinator.synchronized():
            with self._lock:
                _validate_manifest_snapshot(
                    repository_id,
                    {},
                    allow_legacy_state_tokens=True,
                )
                previous = self._entries.get(repository_id, {}).get(relative_path)
                if previous is not None:
                    _validate_manifest_snapshot(
                        repository_id,
                        {relative_path: previous},
                        allow_legacy_state_tokens=True,
                    )
                return _PreparedManifestEntryChange(
                    self,
                    repository_id,
                    relative_path,
                    previous,
                    None,
                    self._generation(repository_id),
                )

    def _generation(self, repository_id: str) -> int:
        return self._generations.get(repository_id, 0)

    def _advance_generation(self, repository_id: str) -> int:
        generation = self._generation(repository_id) + 1
        self._generations[repository_id] = generation
        return generation


@dataclass(frozen=True)
class SecretScanResult:
    """Sanitized text and content-free diagnostics from a secret scanner."""

    text: str
    redaction_count: int
    categories: Tuple[str, ...]


class SecretScanner(Protocol):
    """Replaceable boundary for sensitive paths and content redaction."""

    def rejects_path(self, relative_path: str) -> bool:
        """Return whether a repository-relative path must never be read."""

        ...

    def scan(self, text: str) -> SecretScanResult:
        """Return text safe to index plus redaction diagnostics."""

        ...


class DefaultSecretScanner:
    """Bounded, deterministic high-confidence secret screening."""

    _SECRET_FILENAMES = frozenset(
        {
            ".env",
            ".netrc",
            ".npmrc",
            "credentials",
            "credentials.json",
            "id_dsa",
            "id_ecdsa",
            "id_ed25519",
            "id_rsa",
        }
    )
    _SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".kdbx")
    _SAFE_ENV_SUFFIXES = (".example", ".sample", ".template")
    _ASSIGNMENT = re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"password|passwd|secret)\b(\s*[:=]\s*[\"']?)"
        r"([A-Za-z0-9_./+~$-]{8,})([\"']?)"
    )
    _AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
    _PRIVATE_KEY = re.compile(
        r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----.*?"
        r"-----END [A-Z0-9 ]{0,40}PRIVATE KEY-----",
        re.DOTALL,
    )
    _CREDENTIAL_URL = re.compile(
        r"\b([A-Za-z][A-Za-z0-9+.-]{1,20}://)"
        r"[^\s/:@]{1,128}:[^\s/@]{1,256}@"
    )

    def fingerprint_components(self):
        return {
            "algorithm": "bounded-high-confidence-secret-screening",
            "algorithm_version": 1,
            "secret_filenames": tuple(sorted(self._SECRET_FILENAMES)),
            "secret_suffixes": self._SECRET_SUFFIXES,
            "safe_env_suffixes": self._SAFE_ENV_SUFFIXES,
            "assignment_pattern": self._ASSIGNMENT.pattern,
            "aws_key_pattern": self._AWS_KEY.pattern,
            "private_key_pattern": self._PRIVATE_KEY.pattern,
            "credential_url_pattern": self._CREDENTIAL_URL.pattern,
        }

    def rejects_path(self, relative_path: str) -> bool:
        """Reject conventional credential names and sensitive directories."""

        path = PurePosixPath(relative_path)
        name = path.name.lower()
        if name.startswith(".env") and not name.endswith(self._SAFE_ENV_SUFFIXES):
            return True
        if name in self._SECRET_FILENAMES or name.endswith(self._SECRET_SUFFIXES):
            return True
        directories = {part.lower() for part in path.parts[:-1]}
        return bool(directories & {".ssh", "secrets"})

    def scan(self, text: str) -> SecretScanResult:
        """Redact bounded, high-confidence credential shapes from text."""

        categories: List[str] = []
        count = 0

        def redact_block(match: re.Match) -> str:
            nonlocal count
            count += 1
            categories.append("private_key")
            return "[REDACTED_SECRET]" + "\n" * match.group(0).count("\n")

        def redact_assignment(match: re.Match) -> str:
            nonlocal count
            count += 1
            categories.append("credential_assignment")
            return "{}{}[REDACTED_SECRET]{}".format(
                match.group(1), match.group(2), match.group(4)
            )

        def redact_aws(match: re.Match) -> str:
            nonlocal count
            count += 1
            categories.append("cloud_access_key")
            return "[REDACTED_SECRET]"

        def redact_url(match: re.Match) -> str:
            nonlocal count
            count += 1
            categories.append("credential_url")
            return "{}[REDACTED_SECRET]@".format(match.group(1))

        redacted = self._PRIVATE_KEY.sub(redact_block, text)
        redacted = self._ASSIGNMENT.sub(redact_assignment, redacted)
        redacted = self._AWS_KEY.sub(redact_aws, redacted)
        redacted = self._CREDENTIAL_URL.sub(redact_url, redacted)
        return SecretScanResult(redacted, count, tuple(sorted(set(categories))))


@dataclass(frozen=True)
class _Classification:
    document_type: str
    language: str
    content_kind: str


@dataclass(frozen=True)
class _Candidate:
    relative_path: str
    size: int


@dataclass(frozen=True)
class _DiscoveryResult:
    candidates: Tuple[_Candidate, ...]
    discovered: int
    complete: bool


class _SafeRepository:
    """Validate a local root and read relative regular files without symlinks."""

    def __init__(self, source: Union[str, Path]) -> None:
        if isinstance(source, str) and _URL_SCHEME.match(source):
            raise UnsafeRepositoryError("Repository source must be a local path.")
        if not isinstance(source, (str, Path)):
            raise TypeError("repository source must be a string or Path.")
        candidate = Path(source)
        try:
            unresolved_stat = candidate.lstat()
            if candidate.is_symlink() or not stat.S_ISDIR(unresolved_stat.st_mode):
                raise UnsafeRepositoryError("Repository root must be a real directory.")
            root = candidate.resolve(strict=True)
        except UnsafeRepositoryError:
            raise
        except (OSError, RuntimeError) as exc:
            raise UnsafeRepositoryError("Repository root is unavailable.") from exc
        try:
            root_stat = root.lstat()
        except OSError as exc:
            raise UnsafeRepositoryError("Repository root is unavailable.") from exc
        if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
            raise UnsafeRepositoryError("Repository root must be a real directory.")
        if root.parent == root or root == Path.home().resolve():
            raise UnsafeRepositoryError("Repository root is too broad.")

        git_dir = root / ".git"
        try:
            git_stat = git_dir.lstat()
        except OSError as exc:
            raise UnsafeRepositoryError(
                "A local Git working tree is required."
            ) from exc
        if git_dir.is_symlink() or not stat.S_ISDIR(git_stat.st_mode):
            raise UnsafeRepositoryError(
                "Linked Git directories and alternate worktree layouts are unsupported."
            )
        self.root = root

    def absolute(self, relative_path: str) -> Path:
        parts = self._parts(relative_path)
        return self.root.joinpath(*parts)

    def read_bytes(self, relative_path: str, max_bytes: int) -> bytes:
        parts = self._parts(relative_path)
        if not parts:
            raise UnsafeRepositoryError("A regular repository file is required.")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        cloexec = getattr(os, "O_CLOEXEC", 0)
        opened: List[int] = []
        try:
            current = os.open(str(self.root), os.O_RDONLY | directory | cloexec)
            opened.append(current)
            for component in parts[:-1]:
                current = os.open(
                    component,
                    os.O_RDONLY | directory | nofollow | cloexec,
                    dir_fd=current,
                )
                opened.append(current)
            file_descriptor = os.open(
                parts[-1], os.O_RDONLY | nofollow | cloexec, dir_fd=current
            )
            opened.append(file_descriptor)
            info = os.fstat(file_descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise UnsafeRepositoryError("Repository entry is not a regular file.")
            if info.st_size > max_bytes:
                raise RepositoryResourceLimitError("Repository file exceeds its limit.")
            chunks = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(file_descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                raise RepositoryResourceLimitError("Repository file exceeds its limit.")
            return data
        except (UnsafeRepositoryError, RepositoryResourceLimitError):
            raise
        except (OSError, TypeError, NotImplementedError) as exc:
            return self._fallback_read(relative_path, max_bytes, exc)
        finally:
            for descriptor in reversed(opened):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _fallback_read(
        self, relative_path: str, max_bytes: int, original: Exception
    ) -> bytes:
        target = self.absolute(relative_path)
        try:
            before = target.lstat()
            if target.is_symlink() or not stat.S_ISREG(before.st_mode):
                raise UnsafeRepositoryError("Repository entry is not a regular file.")
            resolved = target.resolve(strict=True)
            if os.path.commonpath((str(self.root), str(resolved))) != str(self.root):
                raise UnsafeRepositoryError(
                    "Repository path escapes the approved root."
                )
            if before.st_size > max_bytes:
                raise RepositoryResourceLimitError("Repository file exceeds its limit.")
            with resolved.open("rb") as handle:
                data = handle.read(max_bytes + 1)
            after = target.lstat()
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise UnsafeRepositoryError(
                    "Repository entry changed while being read."
                )
            if len(data) > max_bytes:
                raise RepositoryResourceLimitError("Repository file exceeds its limit.")
            return data
        except (UnsafeRepositoryError, RepositoryResourceLimitError):
            raise
        except OSError as exc:
            raise UnsafeRepositoryError(
                "Repository file could not be read safely."
            ) from (exc or original)

    @staticmethod
    def _parts(relative_path: str) -> Tuple[str, ...]:
        if not isinstance(relative_path, str) or not relative_path:
            raise UnsafeRepositoryError("Repository-relative path is required.")
        value = PurePosixPath(relative_path)
        if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
            raise UnsafeRepositoryError("Repository-relative path is unsafe.")
        return value.parts

    def commit_sha(self) -> Optional[str]:
        try:
            head = self.read_bytes(".git/HEAD", 4_096).decode("ascii").strip()
            if re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
                return head.lower()
            if not head.startswith("ref: "):
                return None
            reference = head[5:].strip()
            if not re.fullmatch(r"refs/[A-Za-z0-9._/-]{1,240}", reference):
                return None
            if ".." in PurePosixPath(reference).parts:
                return None
            try:
                value = self.read_bytes(".git/{}".format(reference), 4_096)
                sha = value.decode("ascii").strip()
                return sha.lower() if re.fullmatch(r"[0-9a-fA-F]{40,64}", sha) else None
            except (RepositoryIngestionError, UnicodeDecodeError):
                packed = self.read_bytes(".git/packed-refs", 1_000_000)
                for line in packed.decode("ascii", errors="ignore").splitlines():
                    if line.endswith(" {}".format(reference)):
                        sha = line.split(" ", 1)[0]
                        if re.fullmatch(r"[0-9a-fA-F]{40,64}", sha):
                            return sha.lower()
        except (RepositoryIngestionError, UnicodeDecodeError):
            return None
        return None


class _IgnoreRules:
    def __init__(self) -> None:
        self._rules: List[Tuple[PurePosixPath, Tuple[Any, ...]]] = []

    def add(self, base: str, lines: Iterable[str]) -> None:
        patterns = []
        for line in lines:
            try:
                pattern = GitIgnoreSpecPattern(line)
            except (TypeError, ValueError):
                continue
            if getattr(pattern, "include", None) is not None:
                patterns.append(pattern)
        root = PurePosixPath(base) if base else PurePosixPath(".")
        self._rules.append((root, tuple(patterns)))

    def ignored(self, relative_path: str, *, directory: bool = False) -> bool:
        candidate = PurePosixPath(relative_path)
        ignored = False
        for base, patterns in self._rules:
            try:
                local = candidate.relative_to(base)
            except ValueError:
                continue
            value = local.as_posix() + ("/" if directory else "")
            for pattern in patterns:
                if pattern.match_file(value) is not None:
                    ignored = bool(pattern.include)
        return ignored


class _RepositoryDiscovery:
    def __init__(
        self,
        repository: _SafeRepository,
        policy: RepositoryPolicy,
        diagnostics: _BoundedDiagnostics,
    ) -> None:
        self.repository = repository
        self.policy = policy
        self.diagnostics = diagnostics
        self.rules = _IgnoreRules()
        self.candidates: List[_Candidate] = []
        self.visited_entries = 0
        self.discovered = 0
        self.complete = True

    def discover(self) -> _DiscoveryResult:
        self._walk("", 0)
        return _DiscoveryResult(
            tuple(self.candidates),
            self.discovered,
            self.complete,
        )

    def _walk(self, relative_directory: str, depth: int) -> None:
        if not self.complete:
            return
        if depth > self.policy.limits.max_depth:
            self.complete = False
            self.diagnostics.add_skip(
                relative_directory, "maximum_depth", terminal=True
            )
            return

        absolute = (
            self.repository.root
            if not relative_directory
            else self.repository.absolute(relative_directory)
        )
        if not self._load_ignore_rules(relative_directory):
            return

        entries = []
        hit_entry_limit = False
        try:
            with os.scandir(absolute) as iterator:
                for entry in iterator:
                    if (
                        self.visited_entries
                        >= self.policy.limits.max_discovered_entries
                    ):
                        hit_entry_limit = True
                        break
                    self.visited_entries += 1
                    entries.append(entry)
            entries.sort(key=lambda item: item.name)
        except OSError:
            self.complete = False
            self.diagnostics.add_skip(
                relative_directory, "directory_unreadable"
            )
            return

        for entry in entries:
            if not self.complete:
                break
            relative = self._join(relative_directory, entry.name)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                self.complete = False
                self.diagnostics.add_skip(relative, "entry_unreadable")
                continue
            if stat.S_ISLNK(info.st_mode):
                self.diagnostics.add_skip(relative, "symlink")
                continue
            if stat.S_ISDIR(info.st_mode):
                lowered = entry.name.lower()
                if lowered in DEFAULT_EXCLUDED_DIRECTORIES:
                    continue
                if self.rules.ignored(relative, directory=True):
                    continue
                if self._contains_git_marker(relative):
                    self.diagnostics.add_skip(relative, "nested_repository")
                    continue
                self._walk(relative, depth + 1)
                continue
            if not stat.S_ISREG(info.st_mode):
                self.diagnostics.add_skip(relative, "special_file")
                continue

            self.discovered += 1
            if self.rules.ignored(relative):
                self.diagnostics.add_skip(relative, "gitignored")
                continue
            if len(self.candidates) >= self.policy.limits.max_files:
                self.complete = False
                self.diagnostics.add_skip(relative, "maximum_files", terminal=True)
                break
            self.candidates.append(_Candidate(relative, int(info.st_size)))

        if self.complete and hit_entry_limit:
            self.complete = False
            self.diagnostics.add_skip(
                relative_directory, "maximum_discovered_entries", terminal=True
            )

    def _load_ignore_rules(self, relative_directory: str) -> bool:
        relative_ignore = self._join(relative_directory, ".gitignore")
        target = self.repository.absolute(relative_ignore)
        try:
            info = target.lstat()
        except FileNotFoundError:
            return True
        except OSError:
            self.complete = False
            self.diagnostics.add_skip(relative_ignore, "gitignore_unreadable")
            return False
        if not stat.S_ISREG(info.st_mode):
            return True
        try:
            raw = self.repository.read_bytes(
                relative_ignore, self.policy.limits.max_ignore_file_bytes
            )
        except (OSError, RepositoryIngestionError):
            self.complete = False
            self.diagnostics.add_skip(relative_ignore, "gitignore_unreadable")
            return False
        self.rules.add(
            relative_directory,
            raw.decode("utf-8-sig", errors="replace").splitlines(),
        )
        return True

    def _contains_git_marker(self, relative_directory: str) -> bool:
        marker = self.repository.absolute(relative_directory) / ".git"
        try:
            marker.lstat()
            return True
        except OSError:
            return False

    @staticmethod
    def _join(parent: str, name: str) -> str:
        return "{}/{}".format(parent, name) if parent else name


class RepositoryIndexer:
    """Screen repository files and index them through the existing Indexer."""

    def __init__(
        self,
        indexer: Indexer,
        *,
        manifest: Optional[RepositoryManifest] = None,
        policy: Optional[RepositoryPolicy] = None,
        secret_scanner: Optional[SecretScanner] = None,
    ) -> None:
        self.indexer = indexer
        index_coordinator = getattr(indexer, "transaction_coordinator", None)
        self.manifest = (
            manifest
            if manifest is not None
            else InMemoryRepositoryManifest(
                transaction_coordinator=(
                    index_coordinator
                    if isinstance(index_coordinator, IndexTransactionCoordinator)
                    else None
                )
            )
        )
        self.policy = policy if policy is not None else RepositoryPolicy()
        self.secret_scanner = (
            secret_scanner if secret_scanner is not None else DefaultSecretScanner()
        )
        if not callable(getattr(self.manifest, "prepare_replace", None)):
            raise RepositoryIngestionError(
                "Repository manifests must support staged replacement."
            )
        if not callable(getattr(self.manifest, "synchronized_scan", None)):
            raise RepositoryIngestionError(
                "Repository manifests must provide a synchronized scan lease."
            )
        store = getattr(self.indexer, "store", None)
        if store is not None:
            missing = tuple(
                operation
                for operation in (
                    "prepare_replace_document",
                    "prepare_delete_document",
                )
                if not callable(getattr(store, operation, None))
            )
            if missing:
                raise RepositoryIngestionError(
                    "Repository indexing requires staged replacement and deletion "
                    "on the primary vector store; missing: {}.".format(
                        ", ".join(missing)
                    )
                )
        manifest_coordinator = getattr(
            self.manifest, "transaction_coordinator", None
        )
        if (
            isinstance(index_coordinator, IndexTransactionCoordinator)
            and manifest_coordinator is not index_coordinator
        ):
            raise RepositoryIngestionError(
                "The indexer and in-process repository manifest must share one "
                "IndexTransactionCoordinator."
            )

    def index_repository(
        self,
        source: Union[str, Path],
        *,
        repository_id: str,
        policy: Optional[RepositoryPolicy] = None,
    ) -> RepositoryIndexReport:
        repository_id = self._repository_id(repository_id)
        with self.manifest.synchronized_scan(repository_id):
            return self._index_repository_locked(
                source,
                repository_id=repository_id,
                policy=policy,
            )

    def _index_repository_locked(
        self,
        source: Union[str, Path],
        *,
        repository_id: str,
        policy: Optional[RepositoryPolicy] = None,
    ) -> RepositoryIndexReport:
        """Index one bounded snapshot of a local Git working tree.

        Args:
            source: Local repository root containing a real ``.git`` directory.
                URLs, symlinked roots, and linked worktree layouts are rejected.
            repository_id: Stable caller-supplied identity used in document IDs
                and metadata.
            policy: Optional per-call resource and inclusion policy.

        Returns:
            Counts and content-free issues describing the attempted snapshot.

        Notes:
            Source files are treated as untrusted data and are never executed.
            A complete scan removes stale documents. An incomplete scan keeps
            previously indexed entries that were not safely observed.
        """

        selected = policy or self.policy
        repository = _SafeRepository(source)
        commit_sha = repository.commit_sha()
        diagnostics = _BoundedDiagnostics(selected.limits.max_issues)
        discovery = _RepositoryDiscovery(
            repository, selected, diagnostics
        ).discover()
        try:
            previous = _validate_manifest_snapshot(
                repository_id,
                self.manifest.entries(repository_id),
                allow_legacy_state_tokens=True,
                error_type=RepositoryIngestionError,
            )
        except RepositoryIngestionError:
            raise
        except Exception as exc:
            raise RepositoryIngestionError(
                "Repository manifest snapshot could not be validated."
            ) from exc
        fingerprint = self._repository_fingerprint(selected)
        current_manifest = dict(previous)
        retained: Dict[str, RepositoryManifestEntry] = {}
        failed_paths = set()
        indexed = unchanged = redacted = chunk_count = secret_count = 0
        total_bytes = total_chunks = 0
        complete = discovery.complete

        for position, candidate in enumerate(discovery.candidates):
            if diagnostics.failed_count >= selected.limits.max_errors:
                complete = False
                diagnostics.add_skipped_count(
                    len(discovery.candidates) - position - 1
                )
                diagnostics.add_skip(
                    candidate.relative_path, "maximum_errors", terminal=True
                )
                break

            relative = candidate.relative_path
            classification = self._classify(relative, selected)
            if classification is None:
                diagnostics.add_skip(relative, "unsupported_type")
                continue
            if self.secret_scanner.rejects_path(relative):
                diagnostics.add_skip(relative, "sensitive_path")
                continue
            if candidate.size > selected.limits.max_file_bytes:
                diagnostics.add_skip(relative, "file_too_large")
                continue
            if len(relative) > selected.limits.max_metadata_length:
                diagnostics.add_skip(relative, "metadata_limit")
                continue
            if total_bytes + candidate.size > selected.limits.max_total_bytes:
                complete = False
                diagnostics.add_skipped_count(len(discovery.candidates) - position - 1)
                diagnostics.add_skip(
                    relative, "total_bytes_limit", terminal=True
                )
                break

            try:
                raw = repository.read_bytes(relative, selected.limits.max_file_bytes)
                if total_bytes + len(raw) > selected.limits.max_total_bytes:
                    complete = False
                    diagnostics.add_skipped_count(
                        len(discovery.candidates) - position - 1
                    )
                    diagnostics.add_skip(
                        relative, "total_bytes_limit", terminal=True
                    )
                    break
                total_bytes += len(raw)
                file_hash = hashlib.sha256(raw).hexdigest()
                previous_entry = previous.get(relative)
                if (
                    fingerprint.reusable
                    and previous_entry is not None
                    and previous_entry.file_hash == file_hash
                    and previous_entry.index_fingerprint == fingerprint.digest
                    and previous_entry.commit_sha == commit_sha
                    and previous_entry.chunk_count > 0
                    and previous_entry.chunk_count
                    <= selected.limits.max_chunks_per_file
                    and previous_entry.document_state_token
                    and self._document_state_matches(
                        previous_entry.document_id,
                        previous_entry.document_state_token,
                    )
                ):
                    if self._binary(raw):
                        diagnostics.add_skip(relative, "binary")
                        continue
                    text, _ = self._decode(raw)
                    if text.count("\n") + 1 > selected.limits.max_lines_per_file:
                        diagnostics.add_skip(relative, "line_limit")
                        continue
                    if total_chunks + previous_entry.chunk_count > (
                        selected.limits.max_total_chunks
                    ):
                        complete = False
                        diagnostics.add_skip(
                            relative, "total_chunks_limit", terminal=True
                        )
                        break
                    total_chunks += previous_entry.chunk_count
                    current_manifest[relative] = previous_entry
                    retained[relative] = previous_entry
                    unchanged += 1
                    continue
                if self._binary(raw):
                    diagnostics.add_skip(relative, "binary")
                    continue
                text, encoding = self._decode(raw)
                if text.count("\n") + 1 > selected.limits.max_lines_per_file:
                    diagnostics.add_skip(relative, "line_limit")
                    continue
                if not selected.include_generated and self._generated(relative, text):
                    diagnostics.add_skip(relative, "generated")
                    continue
                secret_result = self.secret_scanner.scan(text)
                if not isinstance(secret_result, SecretScanResult):
                    raise TypeError("Secret scanner returned an invalid result.")
                if not secret_result.text.strip():
                    diagnostics.add_skip(relative, "empty")
                    continue

                injection_suspected = bool(_PROMPT_INJECTION.search(secret_result.text))
                document_id = self._document_id(repository_id, relative)
                document_state_token = secrets.token_hex(32)
                metadata = self._metadata(
                    repository_id=repository_id,
                    commit_sha=commit_sha,
                    relative_path=relative,
                    classification=classification,
                    document_id=document_id,
                    file_hash=file_hash,
                    index_fingerprint=fingerprint.digest,
                    document_state_token=document_state_token,
                    encoding=encoding,
                    redactions=secret_result.redaction_count,
                    injection_suspected=injection_suspected,
                    max_length=selected.limits.max_metadata_length,
                )
                document = self._document(
                    secret_result.text, classification, relative, metadata
                )
                remaining_chunks = selected.limits.max_total_chunks - total_chunks
                if remaining_chunks <= 0:
                    complete = False
                    diagnostics.add_skip(
                        relative, "total_chunks_limit", terminal=True
                    )
                    break
                def prepare_manifest(
                    index_report,
                    selected_path=relative,
                    selected_document_id=document_id,
                    selected_file_hash=file_hash,
                    selected_state_token=document_state_token,
                ):
                    entry = RepositoryManifestEntry(
                        repository_id=repository_id,
                        relative_path=selected_path,
                        document_id=selected_document_id,
                        file_hash=selected_file_hash,
                        commit_sha=commit_sha,
                        index_fingerprint=fingerprint.digest,
                        chunk_count=index_report.chunk_count,
                        document_state_token=selected_state_token,
                    )
                    return self._prepare_manifest_entry(
                        repository_id,
                        selected_path,
                        entry,
                        current_manifest,
                    )

                report = self.indexer.index_document(
                    document,
                    max_chunks=min(
                        selected.limits.max_chunks_per_file, remaining_chunks
                    ),
                    required_chunk_metadata=metadata,
                    report_preparations=(prepare_manifest,),
                )
                manifest_entry = RepositoryManifestEntry(
                    repository_id=repository_id,
                    relative_path=relative,
                    document_id=document_id,
                    file_hash=file_hash,
                    commit_sha=commit_sha,
                    index_fingerprint=fingerprint.digest,
                    chunk_count=report.chunk_count,
                    document_state_token=document_state_token,
                )
                current_manifest[relative] = manifest_entry
                total_chunks += report.chunk_count
                chunk_count += report.chunk_count
                indexed += 1
                if secret_result.redaction_count:
                    redacted += 1
                    secret_count += secret_result.redaction_count
                retained[relative] = manifest_entry
            except RepositoryResourceLimitError:
                diagnostics.add_skip(relative, "chunk_limit")
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                del exc
                failed_paths.add(relative)
                diagnostics.add_failure(relative, "processing_failed")

        current_manifest = self._finish_manifest(
            repository_id,
            previous,
            current_manifest,
            retained,
            failed_paths,
            complete,
            diagnostics,
        )
        deleted = sum(
            1
            for path in previous
            if path not in current_manifest and path not in failed_paths
        )
        skips, failures = diagnostics.snapshots()
        return RepositoryIndexReport(
            repository_id=repository_id,
            commit_sha=commit_sha,
            snapshot_kind="working_tree",
            discovered=discovery.discovered,
            indexed=indexed,
            skipped=diagnostics.skipped_count,
            redacted=redacted,
            failed=diagnostics.failed_count,
            deleted=deleted,
            chunk_count=chunk_count,
            secret_redaction_count=secret_count,
            complete=complete,
            skips=tuple(skips),
            failures=tuple(failures),
            unchanged=unchanged,
        )

    def _finish_manifest(
        self,
        repository_id: str,
        previous: Mapping[str, RepositoryManifestEntry],
        current_manifest: Mapping[str, RepositoryManifestEntry],
        retained: Mapping[str, RepositoryManifestEntry],
        failed_paths: set,
        complete: bool,
        diagnostics: _BoundedDiagnostics,
    ) -> Mapping[str, RepositoryManifestEntry]:
        desired = dict(current_manifest)
        if complete:
            stale = sorted(
                path
                for path in previous
                if path not in retained and path not in failed_paths
            )
            for path in stale:
                try:
                    self.indexer.delete(
                        previous[path].document_id,
                        additional_preparations=(
                            lambda selected_path=path: self._prepare_manifest_delete(
                                repository_id,
                                selected_path,
                                desired,
                            ),
                        ),
                    )
                    desired.pop(path, None)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    del exc
                    diagnostics.add_failure(path, "stale_delete_failed")
        return desired

    def _prepare_manifest_snapshot(
        self,
        repository_id: str,
        entries: Mapping[str, RepositoryManifestEntry],
    ) -> CoordinatedPreparedDocumentReplacement:
        snapshot = _validate_manifest_snapshot(
            repository_id,
            entries,
            allow_legacy_state_tokens=True,
            error_type=RepositoryIngestionError,
        )
        return self.manifest.prepare_replace(repository_id, snapshot)

    def _prepare_manifest_entry(
        self,
        repository_id: str,
        relative_path: str,
        entry: RepositoryManifestEntry,
        current: Mapping[str, RepositoryManifestEntry],
    ) -> CoordinatedPreparedDocumentReplacement:
        if type(self.manifest) is InMemoryRepositoryManifest:
            return self.manifest.prepare_replace_entry(
                repository_id, relative_path, entry
            )
        snapshot = dict(current)
        snapshot[relative_path] = entry
        return self._prepare_manifest_snapshot(repository_id, snapshot)

    def _prepare_manifest_delete(
        self,
        repository_id: str,
        relative_path: str,
        current: Mapping[str, RepositoryManifestEntry],
    ) -> CoordinatedPreparedDocumentReplacement:
        if type(self.manifest) is InMemoryRepositoryManifest:
            return self.manifest.prepare_delete_entry(repository_id, relative_path)
        snapshot = dict(current)
        snapshot.pop(relative_path, None)
        return self._prepare_manifest_snapshot(repository_id, snapshot)

    def _document_state_matches(self, document_id: str, expected_token: str) -> bool:
        """Require every physical index participant to own the manifest state.

        The random token guards against accidental direct replacement or deletion;
        it is not intended to authenticate a deliberately spoofing trusted writer.
        """

        store = getattr(self.indexer, "store", None)
        document_indexes = getattr(self.indexer, "document_indexes", ())
        participants = (store,) + tuple(document_indexes)
        if store is None:
            return False
        coordinator = getattr(self.indexer, "transaction_coordinator", None)
        if not isinstance(coordinator, IndexTransactionCoordinator):
            return False
        with coordinator.synchronized():
            for participant in participants:
                read_token = getattr(participant, "document_state_token", None)
                if not callable(read_token):
                    return False
                try:
                    if read_token(document_id) != expected_token:
                        return False
                except Exception:
                    return False
        return True

    def _repository_fingerprint(
        self, policy: RepositoryPolicy
    ) -> IndexFingerprint:
        candidate = getattr(self.indexer, "index_fingerprint", None)
        if isinstance(candidate, IndexFingerprint):
            index_components = candidate.components
            index_reusable = candidate.reusable
        else:
            index_components = {
                "type": "{}.{}".format(
                    type(self.indexer).__module__, type(self.indexer).__qualname__
                ),
                "opaque": True,
            }
            index_reusable = False
        scanner, scanner_reusable = describe_component(self.secret_scanner)
        state_identity = getattr(self.indexer, "index_state_identity", None)
        state_reusable = isinstance(state_identity, tuple) and bool(state_identity)
        payload = {
            "repository_schema_version": 2,
            "index": index_components,
            "index_state": state_identity,
            "policy": {
                "include_generated": policy.include_generated,
                "include_lockfiles": policy.include_lockfiles,
                "include_unknown_text": policy.include_unknown_text,
            },
            "secret_scanner": scanner,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return IndexFingerprint(
            hashlib.sha256(canonical).hexdigest(),
            index_reusable and scanner_reusable and state_reusable,
            payload,
        )

    def _document(
        self,
        text: str,
        classification: _Classification,
        relative_path: str,
        metadata: Mapping[str, Any],
    ) -> Document:
        if classification.content_kind == "document":
            processed = self.indexer.processor.process(
                DocumentSource.from_bytes(
                    text.encode("utf-8"),
                    name=relative_path,
                    document_type=classification.document_type,
                    metadata=metadata,
                )
            )
            canonical_metadata = dict(processed.metadata)
            canonical_metadata.update(metadata)
            return Document(
                processed.text,
                classification.document_type,
                canonical_metadata,
            )
        return Document(text, classification.document_type, metadata)

    @staticmethod
    def _classify(
        relative_path: str, policy: RepositoryPolicy
    ) -> Optional[_Classification]:
        path = PurePosixPath(relative_path)
        name = path.name
        lowered = name.lower()
        if lowered in LOCKFILES and not policy.include_lockfiles:
            return None
        if lowered.endswith((".min.js", ".min.css", ".map")):
            return None
        special = SPECIAL_FILES.get(lowered)
        if special:
            language, content_kind = special
            return _Classification(language, language, content_kind)
        if lowered.startswith(".env") and lowered.endswith(
            DefaultSecretScanner._SAFE_ENV_SUFFIXES
        ):
            return _Classification("env", "env", "config")
        suffix = path.suffix.lower()
        if suffix in LANGUAGE_BY_SUFFIX:
            language = LANGUAGE_BY_SUFFIX[suffix]
            return _Classification(suffix.lstrip("."), language, "code")
        if suffix in CONFIG_BY_SUFFIX:
            language = CONFIG_BY_SUFFIX[suffix]
            return _Classification(suffix.lstrip("."), language, "config")
        if suffix in DOCUMENT_BY_SUFFIX:
            return _Classification(DOCUMENT_BY_SUFFIX[suffix], "text", "document")
        if policy.include_unknown_text:
            return _Classification("txt", "text", "config")
        return None

    @staticmethod
    def _binary(data: bytes) -> bool:
        sample = data[:8_192]
        if b"\x00" in sample:
            return True
        if not sample:
            return False
        controls = sum(byte < 9 or 13 < byte < 32 for byte in sample)
        return controls / len(sample) > 0.05

    @staticmethod
    def _decode(data: bytes) -> Tuple[str, str]:
        try:
            return data.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            try:
                return data.decode("latin-1"), "latin-1"
            except UnicodeDecodeError as exc:
                raise RepositoryIngestionError(
                    "Repository file encoding is unsupported."
                ) from exc

    @staticmethod
    def _generated(relative_path: str, text: str) -> bool:
        parts = {part.lower() for part in PurePosixPath(relative_path).parts[:-1]}
        if parts & {"generated", "gen"}:
            return True
        header = text[:8_192].lower()
        return any(
            marker in header
            for marker in (
                "code generated",
                "automatically generated",
                "auto-generated",
                "@generated",
                "do not edit this file",
            )
        )

    @staticmethod
    def _metadata(
        *,
        repository_id: str,
        commit_sha: Optional[str],
        relative_path: str,
        classification: _Classification,
        document_id: str,
        file_hash: str,
        index_fingerprint: str,
        document_state_token: str,
        encoding: str,
        redactions: int,
        injection_suspected: bool,
        max_length: int,
    ) -> Mapping[str, Any]:
        if len(relative_path) > max_length:
            raise RepositoryResourceLimitError("Repository metadata exceeds its limit.")
        parts = PurePosixPath(relative_path).parts
        lowered_parts = tuple(part.lower() for part in parts)

        def clean(value: str) -> str:
            return _CONTROL_CHARACTERS.sub("", value)[:max_length]

        return {
            "repository_id": repository_id,
            "commit_sha": commit_sha,
            "snapshot_kind": "working_tree",
            "relative_path": clean(relative_path),
            "source_name": clean(relative_path),
            "filename": clean(parts[-1]),
            "extension": PurePosixPath(relative_path).suffix.lower().lstrip("."),
            "language": classification.language,
            "content_kind": classification.content_kind,
            "file_hash": file_hash,
            "index_fingerprint": index_fingerprint,
            "document_id": document_id,
            "document_state_token": document_state_token,
            "encoding": encoding,
            "is_test": any(
                part in {"test", "tests", "spec", "specs", "__tests__"}
                for part in lowered_parts
            )
            or parts[-1].lower().startswith("test_"),
            "is_migration": any(
                part in {"migration", "migrations"} for part in lowered_parts
            ),
            "is_generated": False,
            "content_trust": "untrusted_repository",
            "secret_redaction_count": redactions,
            "prompt_injection_suspected": injection_suspected,
        }

    @staticmethod
    def _repository_id(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("repository_id must be a string.")
        if not _SAFE_REPOSITORY_ID.fullmatch(value):
            raise ValueError(
                "repository_id must contain only letters, numbers, '.', '_', ':', "
                "or '-'."
            )
        return value

    @staticmethod
    def _document_id(repository_id: str, relative_path: str) -> str:
        return _repository_document_id(repository_id, relative_path)
