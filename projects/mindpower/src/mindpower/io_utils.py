from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(parts: list[str]) -> str:
    joined = "||".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
