"""內容庫：校驗檔案、把已驗證的大檔登記到 blobs/sha256，並用硬連結讓多個模型共用同一份資料。"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def digest(path: Path, git: bool = False) -> str:
    """大檔算 SHA256（對應 HF LFS）；小檔算 Git blob SHA1（對應 HF blob_id）。"""
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.blobs = root / "blobs" / "sha256"
        self.log = root / ".hfmirror" / "links.jsonl"
        self._verified: set[tuple] = set()

    def blob_path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError(f"SHA256 格式錯誤：{sha256!r}")
        return self.blobs / sha256[:2] / sha256

    def valid(self, path: Path, f: dict) -> bool:
        """大小與 checksum 都相符才算有效；同一個 inode 未改動時不重算。"""
        if not path.is_file() or path.stat().st_size != f["size"]:
            return False
        before = path.stat()
        expected = f["sha256"] or f["git_sha1"]
        key = (before.st_dev, before.st_ino or str(path.resolve()), before.st_mtime_ns, before.st_size, expected)
        if key in self._verified:
            return True
        if expected is None or digest(path, git=not f["sha256"]) != expected:
            return False
        after = path.stat()
        if (after.st_mtime_ns, after.st_size) != (before.st_mtime_ns, before.st_size):
            return False
        self._verified.add(key)
        return True

    def _record(self, source: Path, target: Path, f: dict) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as out:
            out.write(json.dumps({
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "sha256": f["sha256"], "bytes": f["size"], "source": str(source), "target": str(target),
            }, ensure_ascii=False) + "\n")

    def link_from_store(self, target: Path, f: dict) -> bool:
        """target 不存在且內容庫已有相同 SHA256 時，建立硬連結；回傳是否已共用。不覆蓋既有檔案。"""
        if not f.get("sha256") or target.exists():
            return False
        blob = self.blob_path(f["sha256"])
        if not blob.is_file():
            return False
        if not self.valid(blob, f):
            raise RuntimeError(f"內容庫檔案 checksum 不符，保留並停止：{blob}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(blob, target)
        self._record(blob, target, f)
        return True

    def register(self, path: Path, f: dict) -> None:
        """把已驗證的檔案加入內容庫（硬連結，不額外占空間）。內容庫已有同 SHA256 時不動。"""
        if not f.get("sha256"):
            return
        blob = self.blob_path(f["sha256"])
        if blob.exists():
            return
        if not self.valid(path, f):
            raise RuntimeError(f"登記內容庫前 checksum 不符：{path}")
        blob.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, blob)
        self._record(path, blob, f)
