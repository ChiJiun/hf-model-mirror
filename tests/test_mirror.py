import json
import os

import pytest

from hfmirror import runner
from hfmirror.config import ConfigError, load_config
from hfmirror.plan import build_lock, load_lock, select, write_json

quiet = dict(log=lambda *a: None)


def test_select_patterns():
    paths = ["a/model.safetensors", "README.md", "onnx/model.onnx", "q/x-Q4_K_M.gguf", "q/x-Q8_0.gguf"]
    assert select(paths, ["*.safetensors", "README.md"], []) == ["a/model.safetensors", "README.md"]
    assert select(paths, [], ["onnx/*"]) == [p for p in paths if not p.startswith("onnx")]
    assert select(paths, ["*Q4_K_M.gguf"], []) == ["q/x-Q4_K_M.gguf"]


def test_config_errors(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text('[settings]\nroot="x"\n[[models]]\nrepo="bad"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)
    p.write_text('[settings]\nroot="x"\n[[models]]\nrepo="a/b"\ntypo=1\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="不認得"):
        load_config(p)
    p.write_text('[[models]]\nrepo="a/b"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="root"):
        load_config(p)
    cfg = load_config(p, root_override=tmp_path / "override")
    assert cfg.root == tmp_path / "override"


def test_download_shares_identical_files(planned, hub):
    assert runner.download(planned, fetch=hub.fetch, **quiet) == 0
    # 共用的 model.safetensors 只下載一次
    assert sum(1 for r, p in hub.downloads if p == "model.safetensors") == 1
    lock = load_lock(planned)
    a, b = (runner.target_of(planned, m, next(f for f in m["files"] if f["path"] == "model.safetensors")) for m in lock["models"])
    assert os.path.samefile(a, b)
    assert runner.verify(planned, **quiet) == 0
    state = json.loads((planned.meta_dir / "state.json").read_text(encoding="utf-8"))
    assert all(s["status"] == "completed" for s in state.values())


def test_rerun_downloads_nothing(planned, hub):
    runner.download(planned, fetch=hub.fetch, **quiet)
    n = len(hub.downloads)
    # 重置狀態後再跑：只校驗、不重新下載
    (planned.meta_dir / "state.json").unlink()
    assert runner.download(planned, fetch=hub.fetch, **quiet) == 0
    assert len(hub.downloads) == n


def test_corrupt_existing_file_is_kept_and_stops(planned, hub):
    runner.download(planned, fetch=hub.fetch, **quiet)
    lock = load_lock(planned)
    m = lock["models"][0]
    f = next(f for f in m["files"] if f["path"] == "extra.bin")
    t = runner.target_of(planned, m, f)
    t.write_bytes(b"X" * 100)
    (planned.meta_dir / "state.json").unlink()
    assert runner.download(planned, fetch=hub.fetch, **quiet) == 1
    assert t.read_bytes() == b"X" * 100  # 不覆蓋、不刪除
    assert runner.verify(planned, **quiet) == 1


def test_pause_then_resume(planned, hub):
    calls = []

    def fetch_then_pause(repo, path, commit, d):
        hub.fetch(repo, path, commit, d)
        calls.append(path)
        runner.request_pause(planned)

    assert runner.download(planned, fetch=fetch_then_pause, **quiet) == 130
    assert len(calls) == 1
    assert runner.download(planned, fetch=hub.fetch, **quiet) == 0
    assert runner.verify(planned, **quiet) == 0


def test_retry_on_transient_error(planned, hub, monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    hub.fail_next = 2
    assert runner.download(planned, fetch=hub.fetch, **quiet) == 0


def test_blocked_repo_is_skipped(make_cfg, hub):
    cfg = make_cfg(models=["org/base", "gated/model"])
    lock = build_lock(cfg, hub.info, **quiet)
    assert [m["status"] for m in lock["models"]] == ["ready", "blocked"]
    write_json(cfg.lock_file, lock)
    assert runner.download(cfg, fetch=hub.fetch, **quiet) == 0


def test_config_change_requires_replan(planned, tmp_path):
    p = planned.path
    p.write_text(p.read_text(encoding="utf-8").replace('repo = "org/finetune"', 'repo = "org/finetune"\nexclude = ["lora*"]'), encoding="utf-8")
    with pytest.raises(ValueError, match="plan"):
        load_lock(load_config(p))


def test_relink_restores_deleted_name(planned, hub):
    runner.download(planned, fetch=hub.fetch, **quiet)
    m = load_lock(planned)["models"][1]
    f = next(f for f in m["files"] if f["path"] == "model.safetensors")
    t = runner.target_of(planned, m, f)
    t.unlink()
    assert runner.verify(planned, **quiet) == 1
    assert runner.verify(planned, relink=True, **quiet) == 0
    assert len(hub.downloads) == 5  # 補回不需要重新下載


def test_insufficient_space_stops(make_cfg, hub):
    cfg = make_cfg(body="")
    cfg.reserve_bytes = 10**18
    write_json(cfg.lock_file, build_lock(cfg, hub.info, **quiet))
    assert runner.download(cfg, fetch=hub.fetch, **quiet) == 1
    assert hub.downloads == []


def test_status_reports_remaining(planned, hub, capsys):
    runner.status(planned)
    out = capsys.readouterr().out
    # 不重複內容：2 個 config + 共用權重 1 份 + extra + lora
    expected = 2 + 7 + 4096 + 100 + 200
    lock = load_lock(planned)
    assert runner.remaining_bytes(planned, lock) == expected
    assert "還需下載" in out
