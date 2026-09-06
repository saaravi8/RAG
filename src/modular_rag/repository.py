"""Safe, local-only repository discovery and indexing orchestration."""

import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import (
    Any,
    Dict,
    Iterable,
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
from .indexing import Indexer


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
    max_metadata_length: int = 512
    max_ignore_file_bytes: int = 131_072

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
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


@dataclass(frozen=True)
class RepositoryManifestEntry:
    """Identity recorded after one repository file is indexed successfully."""

    repository_id: str
    relative_path: str
    document_id: str
    file_hash: str
    commit_sha: Optional[str]


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


class InMemoryRepositoryManifest:
    """Thread-safe demo manifest; production applications should persist it."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, RepositoryManifestEntry]] = {}
        self._lock = RLock()

    def entries(self, repository_id: str) -> Mapping[str, RepositoryManifestEntry]:
        """Return a copy of the current in-memory manifest."""

        with self._lock:
            return dict(self._entries.get(repository_id, {}))

    def replace(
        self,
        repository_id: str,
        entries: Mapping[str, RepositoryManifestEntry],
    ) -> None:
        """Replace one repository manifest after validating entry ownership."""

        snapshot = dict(entries)
        if any(entry.repository_id != repository_id for entry in snapshot.values()):
            raise ValueError("Manifest entries must match the repository ID.")
        with self._lock:
            self._entries[repository_id] = snapshot


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
    issues: Tuple[RepositoryIssue, ...]
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
    def __init__(self, repository: _SafeRepository, policy: RepositoryPolicy) -> None:
        self.repository = repository
        self.policy = policy
        self.rules = _IgnoreRules()
        self.candidates: List[_Candidate] = []
        self.issues: List[RepositoryIssue] = []
        self.discovered = 0
        self.complete = True

    def discover(self) -> _DiscoveryResult:
        self._walk("", 0)
        return _DiscoveryResult(
            tuple(self.candidates),
            tuple(self.issues),
            self.discovered,
            self.complete,
        )

    def _walk(self, relative_directory: str, depth: int) -> None:
        if not self.complete:
            return
        if depth > self.policy.limits.max_depth:
            self.issues.append(RepositoryIssue(relative_directory, "maximum_depth"))
            return

        absolute = (
            self.repository.root
            if not relative_directory
            else self.repository.absolute(relative_directory)
        )
        try:
            entries = sorted(os.scandir(absolute), key=lambda item: item.name)
        except OSError:
            if relative_directory:
                self.issues.append(
                    RepositoryIssue(relative_directory, "directory_unreadable")
                )
            return

        ignore_entry = next(
            (entry for entry in entries if entry.name == ".gitignore"), None
        )
        if ignore_entry is not None and not ignore_entry.is_symlink():
            try:
                info = ignore_entry.stat(follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    relative_ignore = self._join(relative_directory, ".gitignore")
                    raw = self.repository.read_bytes(
                        relative_ignore, self.policy.limits.max_ignore_file_bytes
                    )
                    self.rules.add(
                        relative_directory,
                        raw.decode("utf-8-sig", errors="replace").splitlines(),
                    )
            except RepositoryIngestionError:
                self.issues.append(
                    RepositoryIssue(
                        self._join(relative_directory, ".gitignore"),
                        "gitignore_unreadable",
                    )
                )

        for entry in entries:
            if not self.complete:
                break
            relative = self._join(relative_directory, entry.name)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                self.issues.append(RepositoryIssue(relative, "entry_unreadable"))
                continue
            if entry.is_symlink():
                self.issues.append(RepositoryIssue(relative, "symlink"))
                continue
            if stat.S_ISDIR(info.st_mode):
                lowered = entry.name.lower()
                if lowered in DEFAULT_EXCLUDED_DIRECTORIES:
                    continue
                if self.rules.ignored(relative, directory=True):
                    continue
                if self._contains_git_marker(relative):
                    self.issues.append(RepositoryIssue(relative, "nested_repository"))
                    continue
                self._walk(relative, depth + 1)
                continue
            if not stat.S_ISREG(info.st_mode):
                self.issues.append(RepositoryIssue(relative, "special_file"))
                continue

            self.discovered += 1
            if self.rules.ignored(relative):
                self.issues.append(RepositoryIssue(relative, "gitignored"))
                continue
            if len(self.candidates) >= self.policy.limits.max_files:
                self.complete = False
                self.issues.append(RepositoryIssue(relative, "maximum_files"))
                break
            self.candidates.append(_Candidate(relative, int(info.st_size)))

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
        self.manifest = (
            manifest if manifest is not None else InMemoryRepositoryManifest()
        )
        self.policy = policy if policy is not None else RepositoryPolicy()
        self.secret_scanner = (
            secret_scanner if secret_scanner is not None else DefaultSecretScanner()
        )

    def index_repository(
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

        repository_id = self._repository_id(repository_id)
        selected = policy or self.policy
        repository = _SafeRepository(source)
        commit_sha = repository.commit_sha()
        discovery = _RepositoryDiscovery(repository, selected).discover()
        previous = dict(self.manifest.entries(repository_id))
        successful: Dict[str, RepositoryManifestEntry] = {}
        failed_paths = set()
        skips = list(discovery.issues)
        failures: List[RepositoryIssue] = []
        indexed = redacted = chunk_count = secret_count = 0
        total_bytes = total_chunks = 0
        complete = discovery.complete

        for position, candidate in enumerate(discovery.candidates):
            if len(failures) >= selected.limits.max_errors:
                complete = False
                for remaining in discovery.candidates[position:]:
                    skips.append(
                        RepositoryIssue(remaining.relative_path, "maximum_errors")
                    )
                break

            relative = candidate.relative_path
            classification = self._classify(relative, selected)
            if classification is None:
                skips.append(RepositoryIssue(relative, "unsupported_type"))
                continue
            if self.secret_scanner.rejects_path(relative):
                skips.append(RepositoryIssue(relative, "sensitive_path"))
                continue
            if candidate.size > selected.limits.max_file_bytes:
                skips.append(RepositoryIssue(relative, "file_too_large"))
                continue
            if total_bytes + candidate.size > selected.limits.max_total_bytes:
                complete = False
                for remaining in discovery.candidates[position:]:
                    skips.append(
                        RepositoryIssue(remaining.relative_path, "total_bytes_limit")
                    )
                break

            try:
                raw = repository.read_bytes(relative, selected.limits.max_file_bytes)
                if total_bytes + len(raw) > selected.limits.max_total_bytes:
                    complete = False
                    for remaining in discovery.candidates[position:]:
                        skips.append(
                            RepositoryIssue(
                                remaining.relative_path, "total_bytes_limit"
                            )
                        )
                    break
                total_bytes += len(raw)
                if self._binary(raw):
                    skips.append(RepositoryIssue(relative, "binary"))
                    continue
                text, encoding = self._decode(raw)
                if text.count("\n") + 1 > selected.limits.max_lines_per_file:
                    skips.append(RepositoryIssue(relative, "line_limit"))
                    continue
                if not selected.include_generated and self._generated(relative, text):
                    skips.append(RepositoryIssue(relative, "generated"))
                    continue
                secret_result = self.secret_scanner.scan(text)
                if not isinstance(secret_result, SecretScanResult):
                    raise TypeError("Secret scanner returned an invalid result.")
                if not secret_result.text.strip():
                    skips.append(RepositoryIssue(relative, "empty"))
                    continue

                injection_suspected = bool(_PROMPT_INJECTION.search(secret_result.text))
                file_hash = hashlib.sha256(raw).hexdigest()
                document_id = self._document_id(repository_id, relative)
                metadata = self._metadata(
                    repository_id=repository_id,
                    commit_sha=commit_sha,
                    relative_path=relative,
                    classification=classification,
                    document_id=document_id,
                    file_hash=file_hash,
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
                    skips.append(RepositoryIssue(relative, "total_chunks_limit"))
                    break
                report = self.indexer.index_document(
                    document,
                    max_chunks=min(
                        selected.limits.max_chunks_per_file, remaining_chunks
                    ),
                )
                total_chunks += report.chunk_count
                chunk_count += report.chunk_count
                indexed += 1
                if secret_result.redaction_count:
                    redacted += 1
                    secret_count += secret_result.redaction_count
                successful[relative] = RepositoryManifestEntry(
                    repository_id,
                    relative,
                    document_id,
                    file_hash,
                    commit_sha,
                )
            except RepositoryResourceLimitError:
                skips.append(RepositoryIssue(relative, "chunk_limit"))
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                del exc
                failed_paths.add(relative)
                failures.append(RepositoryIssue(relative, "processing_failed"))

        current_manifest = self._finish_manifest(
            repository_id,
            previous,
            successful,
            failed_paths,
            complete,
            failures,
        )
        deleted = sum(
            1
            for path in previous
            if path not in current_manifest and path not in failed_paths
        )
        return RepositoryIndexReport(
            repository_id=repository_id,
            commit_sha=commit_sha,
            snapshot_kind="working_tree",
            discovered=discovery.discovered,
            indexed=indexed,
            skipped=len(skips),
            redacted=redacted,
            failed=len(failures),
            deleted=deleted,
            chunk_count=chunk_count,
            secret_redaction_count=secret_count,
            complete=complete,
            skips=tuple(skips),
            failures=tuple(failures),
        )

    def _finish_manifest(
        self,
        repository_id: str,
        previous: Mapping[str, RepositoryManifestEntry],
        successful: Mapping[str, RepositoryManifestEntry],
        failed_paths: set,
        complete: bool,
        failures: List[RepositoryIssue],
    ) -> Mapping[str, RepositoryManifestEntry]:
        desired: Dict[str, RepositoryManifestEntry]
        if complete:
            desired = {
                path: entry for path, entry in previous.items() if path in failed_paths
            }
        else:
            desired = dict(previous)
        desired.update(successful)

        if complete:
            stale = sorted(path for path in previous if path not in desired)
            for path in stale:
                try:
                    self.indexer.delete(previous[path].document_id)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    del exc
                    desired[path] = previous[path]
                    failures.append(RepositoryIssue(path, "stale_delete_failed"))
        self.manifest.replace(repository_id, desired)
        return desired

    def _document(
        self,
        text: str,
        classification: _Classification,
        relative_path: str,
        metadata: Mapping[str, Any],
    ) -> Document:
        if classification.content_kind == "document":
            return self.indexer.processor.process(
                DocumentSource.from_bytes(
                    text.encode("utf-8"),
                    name=relative_path,
                    document_type=classification.document_type,
                    metadata=metadata,
                )
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
            "document_id": document_id,
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
        identity = "{}\x00{}".format(repository_id, relative_path)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return "repo_{}".format(digest)
