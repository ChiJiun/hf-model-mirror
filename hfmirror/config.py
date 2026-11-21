"""讀取 TOML 設定檔：目標路徑、保留空間與模型清單。"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


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
    if not value or value in (".", "..") or any(c in value for c in '\\:*?"<>|') or value.startswith("/"):
        raise ConfigError(f"{what} 不合法：{value!r}")
    return value


def load_config(path: str | Path, root_override: str | Path | None = None) -> Config:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"找不到設定檔：{path}（可先複製 models.example.toml）")
    with path.open("rb") as f:
        data = tomllib.load(f)
    settings = data.get("settings", {})
    root = root_override or settings.get("root")
    if not root:
        raise ConfigError("settings.root（目標路徑）未設定，也沒有用 --root 指定")
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    lock = Path(settings.get("lock_file", "models.lock.json"))
    if not lock.is_absolute():
        lock = path.parent / lock

    models: list[ModelSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(data.get("models", []), 1):
        if "repo" not in raw:
            raise ConfigError(f"第 {i} 個 [[models]] 缺少 repo")
        unknown = set(raw) - {"repo", "revision", "category", "include", "exclude", "enabled"}
        if unknown:
            raise ConfigError(f"{raw['repo']} 有不認得的欄位：{sorted(unknown)}")
        repo = raw["repo"]
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
            revision=str(raw.get("revision", "main")),
            category=_safe_part(str(raw.get("category", "models")), "category"),
            include=list(raw.get("include", [])),
            exclude=list(raw.get("exclude", [])),
            enabled=bool(raw.get("enabled", True)),
        )
        models.append(spec)
    if not models:
        raise ConfigError("設定檔沒有任何 [[models]]")
    reserve_gb = settings.get("reserve_gb", 50)
    if not isinstance(reserve_gb, (int, float)) or reserve_gb < 0:
        raise ConfigError("settings.reserve_gb 必須是 >= 0 的數字")
    return Config(
        path=path,
        root=root,
        reserve_bytes=int(reserve_gb * 1_000_000_000),
        lock_file=lock,
        volume_guid=settings.get("volume_guid"),
        models=models,
    )
