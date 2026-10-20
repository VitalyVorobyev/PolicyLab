"""The published Koster et al. (2025) dataset: location, checksums and verified download.

The CSV accompanying Koster et al., "Deep reinforcement learning can promote sustainable
human behaviour in a common-pool resource problem", Nature Communications 16:2824 (2025),
DOI 10.1038/s41467-025-58043-7, released under CC-BY 4.0 by Google DeepMind
(https://github.com/google-deepmind/sustainable_behavior).

The file is downloaded once into ``data/raw/`` and verified against the size and the
base64 MD5 that Google Cloud Storage publishes for the object; its SHA-256 is then
recorded in a JSON sidecar. A verified file is never downloaded again or overwritten.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

SOURCE_URL: Final = "https://storage.googleapis.com/sustainable_behavior/sustainable_behavior.csv"
SOURCE_FILENAME: Final = "sustainable_behavior.csv"
SOURCE_SIZE_BYTES: Final = 214_185_378
SOURCE_MD5_BASE64: Final = "dMl+sTp8gl3Y7gOvEE1Ymg=="
"""GCS object MD5 (base64 of the 16-byte digest), from the object's x-goog-hash header."""
SOURCE_LICENSE: Final = "CC-BY-4.0"
SOURCE_CITATION: Final = (
    "Koster, R. et al. Deep reinforcement learning can promote sustainable human behaviour "
    "in a common-pool resource problem. Nat. Commun. 16, 2824 (2025). "
    "doi:10.1038/s41467-025-58043-7"
)
SIDECAR_SUFFIX: Final = ".source.json"
_CHUNK: Final = 1 << 20


class SourceVerificationError(RuntimeError):
    """The local file does not match the published size or MD5."""


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A verified local copy of the source CSV."""

    path: Path
    size_bytes: int
    md5_base64: str
    sha256: str


def file_digests(path: Path) -> tuple[int, str, str]:
    """(size in bytes, base64 MD5, hex SHA-256) of a file's exact bytes."""
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    return size, base64.b64encode(md5.digest()).decode("ascii"), sha.hexdigest()


def verify_source(path: Path) -> SourceFile:
    """Check size and MD5 against the published values; raise SourceVerificationError."""
    size, md5_b64, sha = file_digests(path)
    if size != SOURCE_SIZE_BYTES:
        raise SourceVerificationError(
            f"{path}: {size} bytes, expected {SOURCE_SIZE_BYTES} (published size)"
        )
    if md5_b64 != SOURCE_MD5_BASE64:
        raise SourceVerificationError(
            f"{path}: MD5 {md5_b64}, expected {SOURCE_MD5_BASE64} (published GCS hash)"
        )
    return SourceFile(path=path, size_bytes=size, md5_base64=md5_b64, sha256=sha)


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _sidecar_json(source: SourceFile, retrieved_at: str) -> dict[str, Any]:
    return {
        "url": SOURCE_URL,
        "file": source.path.name,
        "size_bytes": source.size_bytes,
        "md5_base64": source.md5_base64,
        "sha256": source.sha256,
        "retrieved_at": retrieved_at,
        "license": SOURCE_LICENSE,
        "citation": SOURCE_CITATION,
    }


def download(raw_dir: Path, *, url: str = SOURCE_URL) -> SourceFile:
    """Download the CSV into `raw_dir` unless a verified copy is already there.

    An existing file is verified and kept (never re-downloaded or overwritten); if it
    fails verification, SourceVerificationError is raised and the file is left for the
    owner to inspect. A new download goes to a hidden temporary file in `raw_dir`, is
    verified, and only then renamed into place; the sidecar with the SHA-256 is written
    next to it.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / SOURCE_FILENAME
    if target.exists():
        source = verify_source(target)
        sidecar = sidecar_path(target)
        if not sidecar.exists():
            stamp = datetime.fromtimestamp(target.stat().st_mtime, UTC)
            sidecar.write_text(
                json.dumps(_sidecar_json(source, stamp.strftime("%Y-%m-%dT%H:%M:%SZ")), indent=2)
                + "\n",
                encoding="utf-8",
            )
        return source
    fd, tmp_name = tempfile.mkstemp(prefix=f".{SOURCE_FILENAME}.partial-", dir=raw_dir)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as response:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
        source = verify_source(tmp)
        if target.exists():
            raise FileExistsError(f"{target} appeared during the download; not overwritten")
        os.rename(tmp, target)
        target.chmod(0o644)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    source = SourceFile(
        path=target,
        size_bytes=source.size_bytes,
        md5_base64=source.md5_base64,
        sha256=source.sha256,
    )
    retrieved = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sidecar_path(target).write_text(
        json.dumps(_sidecar_json(source, retrieved), indent=2) + "\n", encoding="utf-8"
    )
    return source
