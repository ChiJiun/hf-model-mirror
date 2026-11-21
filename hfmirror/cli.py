"""命令列入口：hfmirror plan / download / status / pause / verify。"""
from __future__ import annotations

import argparse
import sys

from . import runner
from .config import ConfigError, load_config
from .plan import build_lock, write_json


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows 主控台預設編碼可能無法輸出中文
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="hfmirror", description="依設定檔批次下載 Hugging Face 模型，SHA256 相同檔案只存一份")
    ap.add_argument("-c", "--config", default="models.toml", help="設定檔路徑（預設 models.toml）")
    ap.add_argument("--root", help="覆寫設定檔的 settings.root（目標路徑）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="查詢各模型固定 commit 與檔案清單，寫入鎖定檔")
    d = sub.add_parser("download", help="依鎖定檔下載（可中斷後重跑接續）")
    d.add_argument("--model", action="append", help="只處理指定 repo，可重複")
    sub.add_parser("status", help="顯示進度與去重後剩餘下載量")
    sub.add_parser("pause", help="要求執行中的下載在目前檔案完成後停止")
    v = sub.add_parser("verify", help="離線重新校驗已下載檔案（不下載）")
    v.add_argument("--model", action="append")
    v.add_argument("--relink", action="store_true", help="先從內容庫補回缺少的硬連結")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config, a.root)
        if a.cmd == "plan":
            lock = build_lock(cfg)
            write_json(cfg.lock_file, lock)
            print(f"已寫入 {cfg.lock_file}")
            return 0
        if a.cmd == "download":
            return runner.download(cfg, a.model)
        if a.cmd == "status":
            return runner.status(cfg)
        if a.cmd == "pause":
            runner.request_pause(cfg)
            print("已要求暫停：目前檔案下載並校驗完後停止")
            return 0
        if a.cmd == "verify":
            return runner.verify(cfg, a.model, a.relink)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 2
    return 0
