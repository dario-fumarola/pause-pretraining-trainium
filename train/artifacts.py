"""Small immutable artifact helpers, retained from the predecessor repository."""
import hashlib,json,os,tempfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_bytes(path: str | Path, data: bytes, *, overwrite: bool = False) -> None:
    """Publish a complete file on the same filesystem; refuse accidental overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(tmp_path, path)
        else:
            # Atomic no-clobber publication on POSIX; link fails if dest exists.
            os.link(tmp_path, path)
            tmp_path.unlink()
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n").encode(), overwrite=overwrite)

