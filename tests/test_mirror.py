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


@pytest.mark.parametrize("line", [
    'include = "*.gguf"',          # 字串而非陣列：以前會被拆成單字元、含 "*" → 下載整個 repo
    'exclude = "onnx/*"',
    'enabled = "false"',
    'revision = 123',
    'category = "x/../../escape"',  # 不可跳出 root
    'category = ".."',
    'include = [1]',
])
def test_config_type_errors(tmp_path, line):
    p = tmp_path / "m.toml"
    p.write_text(f'[settings]\nroot="out"\n[[models]]\nrepo="a/b"\n{line}\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_settings_type_errors(tmp_path):
    p = tmp_path / "m.toml"
    for bad in ['reserve_gb = true', 'reserve_gb = "50"', 'root = 1', 'rooot = "x"']:
        p.write_text(f'[settings]\n{bad}\n[[models]]\nrepo="a/b"\n', encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p, root_override=None if "root" in bad and "rooot" not in bad else tmp_path)


def test_plan_reuses_pinned_commit(make_cfg, hub):
    cfg = make_cfg()
    first = build_lock(cfg, hub.info, **quiet)
    calls = []

    def info(repo, **kw):
        calls.append(repo)
        r = hub.info(repo, **kw)
        r.sha = "f" * 40  # 上游有新 commit
        return r

    again = build_lock(cfg, info, previous=first, **quiet)
    assert calls == [] and again["models"] == first["models"]
    updated = build_lock(cfg, info, previous=first, update=True, **quiet)
    assert len(calls) == 2 and all(m["commit"] == "f" * 40 for m in updated["models"])


def test_completed_model_not_marked_paused(planned, hub):
    assert runner.download(planned, fetch=hub.fetch, only=["org/base"], **quiet) == 0
    runner.request_pause(planned)

    def fetch_pause(*a):
        runner.request_pause(planned)
        hub.fetch(*a)

    # 第二次執行：base 已完成會跳過；finetune 下載第一個檔案後暫停
    (planned.meta_dir / "pause").unlink()
    assert runner.download(planned, fetch=fetch_pause, **quiet) == 130
    state = json.loads((planned.meta_dir / "state.json").read_text(encoding="utf-8"))
    assert state["org/base"]["status"] == "completed"
    assert state["org/finetune"]["status"] == "paused"


def test_hardlink_unsupported_stops_early(planned, hub, monkeypatch):
    def no_link(*a, **k):
        raise OSError("不支援硬連結")

    monkeypatch.setattr(runner.os, "link", no_link)
    with pytest.raises(RuntimeError, match="硬連結"):
        runner.download(planned, fetch=hub.fetch, **quiet)
    assert hub.downloads == []


@pytest.mark.parametrize("text", [
    '[settings]\nroot="out"\n[[model]]\nrepo="a/b"\n[[models]]\nrepo="c/d"\n',  # [[model]] 少 s
    '[setting]\nreserve_gb=999\n[[models]]\nrepo="a/b"\n',
    '[settings]\nroot="out"\nreserve_gb=inf\n[[models]]\nrepo="a/b"\n',
])
def test_top_level_typos_and_inf(tmp_path, text):
    p = tmp_path / "m.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p, root_override=tmp_path)


def test_each_config_has_own_lock(tmp_path, hub):
    from hfmirror import cli

    for name, repo in (("a", "org/base"), ("b", "org/finetune")):
        (tmp_path / f"{name}.toml").write_text(f'[settings]\nroot="out"\nreserve_gb=0\n[[models]]\nrepo="{repo}"\n', encoding="utf-8")
    a, b = load_config(tmp_path / "a.toml"), load_config(tmp_path / "b.toml")
    assert a.lock_file.name == "a.lock.json" and b.lock_file.name == "b.lock.json"
    for c in (a, b):
        write_json(c.lock_file, build_lock(c, hub.info, **quiet))
    assert load_lock(a)["models"][0]["repo"] == "org/base"
    # 兩份設定檔明確指定同一個 lock_file 時，plan 不覆蓋別人的鎖定檔
    (tmp_path / "c.toml").write_text('[settings]\nroot="out"\nlock_file="a.lock.json"\n[[models]]\nrepo="org/base"\n', encoding="utf-8")
    assert cli.main(["-c", str(tmp_path / "c.toml"), "plan"]) == 2


def test_root_override_relative_to_cwd(tmp_path, monkeypatch):
    p = tmp_path / "cfg" / "m.toml"
    p.parent.mkdir()
    p.write_text('[settings]\nroot="out"\n[[models]]\nrepo="a/b"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_config(p).root == (tmp_path / "cfg" / "out").resolve()
    assert load_config(p, root_override="dest").root == (tmp_path / "dest").resolve()


def test_unknown_model_flag_errors(planned, hub):
    with pytest.raises(ValueError, match="不在清單"):
        runner.download(planned, only=["org/typo"], fetch=hub.fetch, **quiet)
    with pytest.raises(ValueError, match="不在清單"):
        runner.verify(planned, only=["org/typo"], **quiet)


def test_permanent_error_not_retried(planned, monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("不應重試")))

    class GatedRepoError(Exception):
        pass

    calls = []

    def fetch(*a):
        calls.append(a)
        raise GatedRepoError("401")

    assert runner.download(planned, fetch=fetch, **quiet) == 1
    assert len(calls) == 1


def test_include_change_downloads_new_files(tmp_path, hub):
    """改了 include 之後重跑 plan + download，必須下載新增的檔案（不能因為 state 是 completed 就跳過）。"""
    p = tmp_path / "models.toml"
    body = '[settings]\nroot="out"\nreserve_gb=0\n[[models]]\nrepo="org/base"\ninclude=[{inc}]\n'
    p.write_text(body.format(inc='"config.json"'), encoding="utf-8")
    cfg = load_config(p)
    write_json(cfg.lock_file, build_lock(cfg, hub.info, **quiet))
    assert runner.download(cfg, fetch=hub.fetch, **quiet) == 0
    assert hub.downloads == [("org/base", "config.json")]
    p.write_text(body.format(inc='"config.json", "*.safetensors"'), encoding="utf-8")
    cfg = load_config(p)
    prev = json.loads(cfg.lock_file.read_text(encoding="utf-8"))
    write_json(cfg.lock_file, build_lock(cfg, hub.info, previous=prev, **quiet))
    assert runner.download(cfg, fetch=hub.fetch, **quiet) == 0
    assert ("org/base", "model.safetensors") in hub.downloads
    assert runner.verify(cfg, **quiet) == 0


@pytest.mark.parametrize("line", ['root = "  "', 'volume_guid = ""', 'lock_file = ""', 'lock_file = "sub/"'])
def test_empty_settings_rejected(tmp_path, line):
    p = tmp_path / "m.toml"
    extra = '' if line.startswith("root") else 'root = "out"\n'
    p.write_text(f'[settings]\n{extra}{line}\n[[models]]\nrepo="a/b"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)
