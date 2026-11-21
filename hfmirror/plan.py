"""plan：向 Hugging Face 查詢每個模型的固定 commit 與檔案清單（大小、SHA256），寫成鎖定檔。"""
from __future__ import annotations

import datetime
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .config import Config, ModelSpec

LOCK_VERSION = 1


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    return f"{n:.2f} TB"


def select(paths: list[str], include: list[str], exclude: list[str]) -> list[str]:
    def hit(p: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(p, pat) or fnmatch.fnmatchcase(PurePosixPath(p).name, pat) for pat in patterns)

    return [p for p in paths if (not include or hit(p, include)) and not hit(p, exclude)]


def safe_relpath(p: str) -> str:
    rel = PurePosixPath(p)
    if rel.is_absolute() or ".." in rel.parts or "\\" in p or ":" in p or not rel.parts:
        raise ValueError(f"不安全的檔案路徑：{p!r}")
    return p


def folder_name(repo: str, commit: str) -> str:
    return repo.replace("/", "--") + "--" + commit


def spec_key(spec: ModelSpec) -> dict:
    """鎖定檔記錄的設定欄位；設定檔改了這些就必須重新 plan。"""
    return {"repo": spec.repo, "revision": spec.revision, "category": spec.category,
            "include": spec.include, "exclude": spec.exclude, "enabled": spec.enabled}


def fetch_model(spec: ModelSpec, info_fn: Callable[..., Any]) -> dict:
    info = info_fn(spec.repo, revision=spec.revision, files_metadata=True)
    siblings = {s.rfilename: s for s in info.siblings}
    files = []
    for path in sorted(select(list(siblings), spec.include, spec.exclude)):
        s = siblings[path]
        lfs = getattr(s, "lfs", None)
        sha256 = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        size = s.size if s.size is not None else (getattr(lfs, "size", None) if lfs else None)
        if size is None:
            raise RuntimeError(f"{spec.repo}:{path} 沒有大小資訊")
        files.append({"path": safe_relpath(path), "size": int(size), "sha256": sha256, "git_sha1": None if sha256 else s.blob_id})
    if not files:
        raise RuntimeError("include/exclude 篩選後沒有任何檔案")
    return {"commit": info.sha, "files": files}


def build_lock(cfg: Config, info_fn: Callable[..., Any] | None = None, log=print) -> dict:
    if info_fn is None:
        from huggingface_hub import HfApi

        info_fn = HfApi().model_info
    models = []
    for spec in cfg.models:
        entry = {**spec_key(spec)}
        if not spec.enabled:
            entry.update(status="disabled", files=[])
        else:
            try:
                entry.update(fetch_model(spec, info_fn), status="ready")
                total = sum(f["size"] for f in entry["files"])
                log(f"[ok] {spec.repo}@{entry['commit'][:12]}：{len(entry['files'])} 個檔案，{human(total)}")
            except Exception as e:  # 需授權、找不到 repo、網路錯誤：記錄後繼續處理其他模型
                first = (str(e).strip().splitlines() or [""])[0]
                entry.update(status="blocked", reason=f"{type(e).__name__}: {first}"[:300], files=[])
                log(f"[blocked] {spec.repo}：{entry['reason']}")
        models.append(entry)
    return {
        "version": LOCK_VERSION,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": str(cfg.path),
        "models": models,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_lock(cfg: Config) -> dict:
    if not cfg.lock_file.is_file():
        raise FileNotFoundError(f"找不到鎖定檔 {cfg.lock_file}，請先執行 hfmirror plan")
    lock = json.loads(cfg.lock_file.read_text(encoding="utf-8"))
    if lock.get("version") != LOCK_VERSION:
        raise ValueError("鎖定檔版本不符，請重新執行 hfmirror plan")
    wanted = [spec_key(m) for m in cfg.models]
    have = [{k: m.get(k) for k in wanted[0]} for m in lock["models"]] if wanted else []
    if wanted != have:
        raise ValueError("設定檔的模型清單或篩選條件已變更，請重新執行 hfmirror plan")
    for m in lock["models"]:
        for f in m["files"]:
            safe_relpath(f["path"])
    return lock
