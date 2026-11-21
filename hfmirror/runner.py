"""download / status / verify / relink 的主要流程。"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from .config import Config
from .plan import folder_name, human, load_lock, write_json
from .store import Store

DownloadFn = Callable[[str, str, str, Path], None]


class Paused(Exception):
    pass


# huggingface_hub 的永久性錯誤（權限、找不到 repo/檔案/版本）：重試也不會成功
PERMANENT_ERRORS = {"RepositoryNotFoundError", "GatedRepoError", "EntryNotFoundError",
                    "RemoteEntryNotFoundError", "RevisionNotFoundError", "DisabledRepoError"}


def check_only(lock: dict, only: list[str] | None) -> None:
    if only:
        known = {m["repo"] for m in lock["models"]}
        missing = [r for r in only if r not in known]
        if missing:
            raise ValueError(f"--model 指定的 repo 不在清單中：{missing}")


def hf_download(repo: str, path: str, commit: str, model_dir: Path) -> None:
    """預設下載器：huggingface_hub 會把未完成檔放在 <model_dir>/.cache/huggingface/download，可續傳。"""
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=repo, filename=path, revision=commit, local_dir=model_dir)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def model_dir(cfg: Config, m: dict) -> Path:
    return cfg.root / m["category"] / folder_name(m["repo"], m["commit"])


def target_of(cfg: Config, m: dict, f: dict) -> Path:
    return model_dir(cfg, m).joinpath(*f["path"].split("/"))


def check_volume(cfg: Config) -> None:
    """設定了 volume_guid 時，確認目標路徑仍在同一顆磁碟（Windows），避免換碟或代號改變後寫錯地方。"""
    if not Path(cfg.root.anchor).exists():
        raise RuntimeError(f"目標磁碟不存在或未連接：{cfg.root.anchor}")
    if cfg.volume_guid and os.name == "nt":
        out = subprocess.check_output(["mountvol", cfg.root.anchor, "/L"], text=True)
        if cfg.volume_guid.lower() not in out.lower():
            raise RuntimeError("磁碟卷 GUID 不符，停止（換碟後請確認並更新 settings.volume_guid）")


def check_hardlinks(cfg: Config) -> None:
    """開始前確認目標檔案系統支援硬連結（exFAT/FAT32 不支援），避免下載完第一個大檔才失敗。"""
    probe = cfg.meta_dir / "hardlink-probe"
    link = cfg.meta_dir / "hardlink-probe.link"
    for p in (probe, link):
        if p.exists():
            p.unlink()  # 只刪自己建立的探測檔
    probe.write_bytes(b"probe")
    try:
        os.link(probe, link)
    except OSError as e:
        raise RuntimeError(f"目標路徑不支援硬連結（可能是 exFAT/FAT32 或跨磁碟卷），無法去重：{e}") from e
    finally:
        for p in (link, probe):
            if p.exists():
                p.unlink()


@contextlib.contextmanager
def single_instance(meta: Path):
    """同一個目標路徑只允許一個下載程序；程序結束或當機時作業系統會自動釋放鎖。"""
    meta.mkdir(parents=True, exist_ok=True)
    handle = (meta / "run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError("已有下載程序在使用這個目標路徑") from None
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError("已有下載程序在使用這個目標路徑") from None
        yield
    finally:
        handle.close()


def files_fingerprint(m: dict) -> str:
    items = sorted((f["path"], f["size"], f["sha256"] or f["git_sha1"]) for f in m["files"])
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def is_current(s: dict, m: dict) -> bool:
    return s.get("commit") == m.get("commit") and s.get("files_fingerprint") == files_fingerprint(m)


class State:
    def __init__(self, cfg: Config):
        self.path = cfg.meta_dir / "state.json"
        self.md = cfg.meta_dir / "PROGRESS.md"
        self.data: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}

    def model(self, m: dict) -> dict:
        """commit 或檔案清單（include/exclude 改變）不同時，進度歸零並重新校驗。"""
        s = self.data.get(m["repo"])
        fp = files_fingerprint(m)
        if not s or s.get("commit") != m.get("commit") or s.get("files_fingerprint") != fp:
            s = {"commit": m.get("commit"), "files_fingerprint": fp, "status": "pending",
                 "completed_files": 0, "completed_bytes": 0}
            self.data[m["repo"]] = s
        s["total_bytes"] = sum(f["size"] for f in m["files"])
        s["total_files"] = len(m["files"])
        return s

    def save(self) -> None:
        write_json(self.path, self.data)
        lines = ["# 下載進度", "", f"更新：{now()}", ""]
        for repo, s in self.data.items():
            total = s.get("total_bytes") or 0
            pct = f"{s['completed_bytes'] / total * 100:5.1f}%" if total else "  -  "
            extra = f"；目前 {s['current_file']}" if s.get("status") == "downloading" and s.get("current_file") else ""
            err = f"；{s['error']}" if s.get("error") else ""
            lines.append(f"- {repo}：{s['status']} {pct}（{human(s['completed_bytes'])}/{human(total)}，"
                         f"{s['completed_files']}/{s.get('total_files', 0)} 檔）{extra}{err}")
        self.md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pause_requested(cfg: Config) -> bool:
    return (cfg.meta_dir / "pause").exists()


def request_pause(cfg: Config) -> None:
    cfg.meta_dir.mkdir(parents=True, exist_ok=True)
    (cfg.meta_dir / "pause").touch()


def download(cfg: Config, only: list[str] | None = None, fetch: DownloadFn = hf_download,
             retries: int = 5, log=print) -> int:
    lock = load_lock(cfg)
    check_only(lock, only)
    check_volume(cfg)
    cfg.root.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.root)
    with single_instance(cfg.meta_dir):
        check_hardlinks(cfg)
        pause = cfg.meta_dir / "pause"
        if pause.exists():
            pause.unlink()  # 只刪自己的控制檔
        state = State(cfg)
        s: dict | None = None
        try:
            for m in lock["models"]:
                if only and m["repo"] not in only:
                    continue
                if m["status"] != "ready":
                    if m["status"] == "blocked":
                        log(f"[略過] {m['repo']}：{m.get('reason', '')}")
                    continue
                s = state.model(m)
                if s["status"] == "completed":
                    s = None  # 已完成的模型不要在之後的暫停／錯誤處理中被改寫
                    continue
                s.update(status="verifying", completed_files=0, completed_bytes=0, error=None)
                state.save()
                for f in m["files"]:
                    check_volume(cfg)
                    target = target_of(cfg, m, f)
                    if not target.exists() and store.link_from_store(target, f):
                        log(f"[共用] {m['repo']}:{f['path']}")
                    if not store.valid(target, f):
                        if target.exists():
                            raise RuntimeError(f"既有檔案校驗失敗，保留並停止：{target}")
                        if _pause_requested(cfg):
                            raise Paused
                        free = shutil.disk_usage(cfg.root).free
                        if free < f["size"] + cfg.reserve_bytes:
                            raise RuntimeError(f"空間不足：可用 {human(free)}，需要 {human(f['size'])}＋保留 {human(cfg.reserve_bytes)}")
                        s.update(status="downloading", current_file=f["path"])
                        state.save()
                        log(f"[下載] {m['repo']}:{f['path']}（{human(f['size'])}）")
                        for attempt in range(retries):
                            try:
                                fetch(m["repo"], f["path"], m["commit"], model_dir(cfg, m))
                                break
                            except KeyboardInterrupt:
                                raise
                            except Exception as e:
                                if type(e).__name__ in PERMANENT_ERRORS:
                                    raise RuntimeError(f"無法下載（不重試）：{type(e).__name__}: {e}") from e
                                if attempt == retries - 1:
                                    raise RuntimeError(f"重試 {retries} 次仍失敗：{e}") from e
                                wait = min(2 ** (attempt + 1), 60)
                                log(f"[重試] {e}；{wait} 秒後重試")
                                for _ in range(wait):
                                    if _pause_requested(cfg):
                                        raise Paused
                                    time.sleep(1)
                        s["status"] = "verifying"
                        state.save()
                        if not store.valid(target, f):
                            raise RuntimeError(f"下載後 checksum 不符，保留並停止：{target}")
                    store.register(target, f)
                    s["completed_files"] += 1
                    s["completed_bytes"] += f["size"]
                    s["updated"] = now()
                    s.pop("current_file", None)
                    state.save()
                    if _pause_requested(cfg):
                        raise Paused
                s.update(status="completed", verified=now())
                state.save()
                s = None
                log(f"[完成] {m['repo']}")
        except (Paused, KeyboardInterrupt):
            if s is not None:
                s["status"] = "paused"
                state.save()
            log("已暫停；之後執行 hfmirror download 會接續")
            return 130
        except Exception as e:
            if s is not None:
                s.update(status="failed", error=str(e)[:300])
                state.save()
            log(f"[錯誤] {e}")
            return 1
    return 0


def remaining_bytes(cfg: Config, lock: dict) -> int:
    """還需要下載的不重複位元組（相同 SHA256 只算一次，本機或內容庫已有的不算）。"""
    store, seen, need = Store(cfg.root), set(), 0
    for m in lock["models"]:
        if m["status"] != "ready":
            continue
        for f in m["files"]:
            key = f["sha256"] or (m["repo"], f["path"])
            if key in seen:
                continue
            seen.add(key)
            t = target_of(cfg, m, f)
            if t.is_file() and t.stat().st_size == f["size"]:
                continue
            if f["sha256"] and store.blob_path(f["sha256"]).is_file():
                continue
            need += f["size"]
    return need


def status(cfg: Config, log=print) -> int:
    lock = load_lock(cfg)
    state = State(cfg)
    for m in lock["models"]:
        s = state.data.get(m["repo"], {})
        total = sum(f["size"] for f in m["files"])
        if m["status"] != "ready":
            log(f"{m['repo']}：{m['status']} {m.get('reason', '')}")
            continue
        current = is_current(s, m)
        done = s.get("completed_bytes", 0) if current else 0
        st = s.get("status", "pending") if current else "pending"
        cur = f"，目前 {s['current_file']}" if st == "downloading" and s.get("current_file") else ""
        err = f"，錯誤：{s['error']}" if s.get("error") else ""
        log(f"{m['repo']}：{st} {human(done)}/{human(total)}{cur}{err}")
    probe = cfg.root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent  # 目標資料夾還沒建立時，用最近存在的上層查可用空間
    free = human(shutil.disk_usage(probe).free) if probe.exists() else "未知（磁碟未連接）"
    log(f"還需下載（去重後）：{human(remaining_bytes(cfg, lock))}；磁碟可用：{free}")
    return 0


def verify(cfg: Config, only: list[str] | None = None, relink: bool = False, log=print) -> int:
    """離線驗證（不下載）。relink=True 時，先從內容庫補回缺少的硬連結。"""
    lock = load_lock(cfg)
    check_only(lock, only)
    check_volume(cfg)
    if relink:  # 會建立硬連結，不可與 download 同時執行
        with single_instance(cfg.meta_dir):
            return _verify(cfg, lock, only, True, log)
    return _verify(cfg, lock, only, False, log)


def _verify(cfg: Config, lock: dict, only: list[str] | None, relink: bool, log) -> int:
    store, bad = Store(cfg.root), 0
    for m in lock["models"]:
        if m["status"] != "ready" or (only and m["repo"] not in only):
            continue
        missing, failed = [], []
        for f in m["files"]:
            t = target_of(cfg, m, f)
            if relink and not t.exists():
                store.link_from_store(t, f)
            if not t.exists():
                missing.append(f["path"])
            elif not store.valid(t, f):
                failed.append(f["path"])
        ok = not missing and not failed
        bad += not ok
        log(f"{m['repo']}：{'VERIFIED' if ok else 'INCOMPLETE'}（缺 {len(missing)}、不符 {len(failed)}）")
        for p in failed:
            log(f"  不符：{p}")
    return 1 if bad else 0
