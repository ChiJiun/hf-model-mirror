import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmirror.config import load_config
from hfmirror.plan import build_lock, write_json


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_sha1(b: bytes) -> str:
    return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()


class FakeHub:
    """模擬 HF：repo -> {path: bytes}。大於 16 bytes 的檔案視為 LFS（有 SHA256）。"""

    def __init__(self, repos: dict[str, dict[str, bytes]]):
        self.repos = repos
        self.downloads: list[tuple[str, str]] = []
        self.fail_next = 0

    def info(self, repo, revision="main", files_metadata=True):
        if repo not in self.repos:
            raise PermissionError("401 gated")
        sib = []
        for p, b in self.repos[repo].items():
            lfs = {"sha256": sha(b), "size": len(b)} if len(b) > 16 else None
            sib.append(SimpleNamespace(rfilename=p, size=len(b), blob_id=git_sha1(b), lfs=lfs))
        return SimpleNamespace(sha=sha(repo.encode())[:40], siblings=sib)

    def fetch(self, repo, path, commit, model_dir: Path):
        if self.fail_next:
            self.fail_next -= 1
            raise ConnectionError("暫時性網路錯誤")
        self.downloads.append((repo, path))
        t = model_dir.joinpath(*path.split("/"))
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(self.repos[repo][path])


SHARED = b"W" * 4096  # 兩個模型共用的權重
REPOS = {
    "org/base": {"config.json": b"{}", "model.safetensors": SHARED, "extra.bin": b"E" * 100},
    "org/finetune": {"config.json": b'{"a":1}', "model.safetensors": SHARED, "lora.safetensors": b"L" * 200},
}


@pytest.fixture
def hub():
    return FakeHub({k: dict(v) for k, v in REPOS.items()})


@pytest.fixture
def make_cfg(tmp_path):
    def _make(body: str = "", models=None):
        models = models or ["org/base", "org/finetune"]
        text = f'[settings]\nroot = "{(tmp_path / "out").as_posix()}"\nreserve_gb = 0\n{body}\n'
        for r in models:
            text += f'\n[[models]]\nrepo = "{r}"\n'
        p = tmp_path / "models.toml"
        p.write_text(text, encoding="utf-8")
        return load_config(p)

    return _make


@pytest.fixture
def planned(make_cfg, hub):
    cfg = make_cfg()
    write_json(cfg.lock_file, build_lock(cfg, hub.info, log=lambda *a: None))
    return cfg
