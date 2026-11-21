"""讀取 TOML 設定檔：目標路徑、保留空間與模型清單。"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass
class ModelSpec:
    repo: str
    revision: str = "main"
    category: str = "models"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class Config:
    path: Path
    root: Path
    reserve_bytes: int
    lock_file: Path
    volume_guid: str | None
    models: list[ModelSpec]

    @property
    def meta_dir(self) -> Path:
        return self.root / ".hfmirror"


def _safe_part(value: str, what: str) -> str:
    """單一路徑片段：不可含 / \\ 或 Windows 不允許的字元，也不可是 . 或 ..。"""
    if not value or value in (".", "..") or any(c in value for c in '/\\:*?"<>|') or value != value.strip():
        raise ConfigError(f"{what} 不合法：{value!r}")
    return value


def _typed(raw: dict, key: str, kind: type, default: Any, where: str) -> Any:
    value = raw.get(key, default)
    if kind is bool:
        ok = isinstance(value, bool)
    elif kind is list:
        ok = isinstance(value, list) and all(isinstance(v, str) and v for v in value)
    else:
        ok = isinstance(value, kind) and not isinstance(value, bool)
    if not ok:
        hint = "（要寫成陣列，例如 [\"*.gguf\"]）" if kind is list else ""
        raise ConfigError(f"{where} 的 {key} 型別錯誤：應為 {kind.__name__}，實際是 {value!r}{hint}")
    return list(value) if kind is list else value


def load_config(path: str | Path, root_override: str | Path | None = None) -> Config:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"找不到設定檔：{path}（可先複製 models.example.toml）")
    with path.open("rb") as f:
        data = tomllib.load(f)
    top_unknown = set(data) - {"settings", "models"}
    if top_unknown:
        raise ConfigError(f"設定檔有不認得的區塊：{sorted(top_unknown)}（只支援 [settings] 與 [[models]]，是否少打了 s？）")
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ConfigError("[settings] 必須是表格")
    unknown = set(settings) - {"root", "reserve_gb", "lock_file", "volume_guid"}
    if unknown:
        raise ConfigError(f"[settings] 有不認得的欄位：{sorted(unknown)}")
    root = root_override or _typed(settings, "root", str, "", "[settings]")
    if not root:
        raise ConfigError("settings.root（目標路徑）未設定，也沒有用 --root 指定")
    root = Path(root).expanduser()
    if not root.is_absolute():
        # --root 的相對路徑以目前工作目錄為準；設定檔裡的相對路徑以設定檔所在資料夾為準
        root = ((Path.cwd() if root_override else path.parent) / root).resolve()
    volume_guid = settings.get("volume_guid")
    if volume_guid is not None:
        _typed(settings, "volume_guid", str, None, "[settings]")
    # 預設每個設定檔有自己的鎖定檔：models.toml → models.lock.json、llm.toml → llm.lock.json
    lock = Path(_typed(settings, "lock_file", str, f"{path.stem}.lock.json", "[settings]"))
    if not lock.is_absolute():
        lock = path.parent / lock

    models: list[ModelSpec] = []
    seen: set[str] = set()
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        raise ConfigError("models 必須寫成 [[models]] 陣列")
    for i, raw in enumerate(raw_models, 1):
        if not isinstance(raw, dict) or "repo" not in raw:
            raise ConfigError(f"第 {i} 個 [[models]] 缺少 repo")
        repo = _typed(raw, "repo", str, "", f"第 {i} 個 [[models]]")
        unknown = set(raw) - {"repo", "revision", "category", "include", "exclude", "enabled"}
        if unknown:
            raise ConfigError(f"{repo} 有不認得的欄位：{sorted(unknown)}")
        parts = repo.split("/")
        if len(parts) != 2:
            raise ConfigError(f"repo 格式應為 owner/name：{repo!r}")
        for p in parts:
            _safe_part(p, "repo")
        if repo in seen:
            raise ConfigError(f"清單中重複的 repo：{repo}")
        seen.add(repo)
        spec = ModelSpec(
            repo=repo,
            revision=_typed(raw, "revision", str, "main", repo),
            category=_safe_part(_typed(raw, "category", str, "models", repo), "category"),
            include=_typed(raw, "include", list, [], repo),
            exclude=_typed(raw, "exclude", list, [], repo),
            enabled=_typed(raw, "enabled", bool, True, repo),
        )
        if not spec.revision:
            raise ConfigError(f"{repo} 的 revision 不可為空")
        models.append(spec)
    if not models:
        raise ConfigError("設定檔沒有任何 [[models]]")
    reserve_gb = settings.get("reserve_gb", 50)
    if isinstance(reserve_gb, bool) or not isinstance(reserve_gb, (int, float)) or not math.isfinite(reserve_gb) or reserve_gb < 0:
        raise ConfigError("settings.reserve_gb 必須是 >= 0 的數字")
    return Config(
        path=path,
        root=root,
        reserve_bytes=int(reserve_gb * 1_000_000_000),
        lock_file=lock,
        volume_guid=settings.get("volume_guid"),
        models=models,
    )
