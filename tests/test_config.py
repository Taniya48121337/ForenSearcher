"""configs/*.toml 是"同一份代码、两份配置"的载体：解析、默认值、错误提示都要可靠。"""

from pathlib import Path

import pytest

from foren_searcher.config import Config, ConfigError, load_config, parse_config

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["local", "server"])
def test_shipped_configs_parse(name):
    cfg = load_config(REPO / "configs" / f"{name}.toml")
    assert cfg.profile == name
    assert cfg.project_root == REPO
    assert cfg.models.student.startswith("Qwen3.5-")
    assert cfg.harness_root is None


def test_missing_sections_fall_back_to_defaults(tmp_path):
    cfg = parse_config('profile = "x"\n[models]\nstudent = "Qwen3.5-2B"\n', source=tmp_path / "configs" / "x.toml")
    assert cfg.models.student == "Qwen3.5-2B"
    assert cfg.models.model_root == ".models"
    assert cfg.training.lora_rank == Config().training.lora_rank
    assert cfg.student_model_dir == (tmp_path / ".models" / "Qwen3.5-2B").resolve()


@pytest.mark.parametrize("text, message", [
    ("[models]\nstudnt = 'x'\n", "未知字段"),
    ("[training]\nlora_rank = 'big'\n", "应为整数"),
    ("[training]\nbf16 = 1\n", "true/false"),
    ("[nope]\nx = 1\n", "未知段"),
    ("profile = [\n", "不是合法 TOML"),
])
def test_bad_configs_explain_what_is_wrong(text, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(text)


def test_env_var_selects_config(tmp_path, monkeypatch):
    path = tmp_path / "configs" / "server.toml"
    path.parent.mkdir()
    path.write_text('[runtime]\nmax_rounds = 9\n', encoding="utf-8")
    monkeypatch.setenv("FOREN_CONFIG", str(path))
    assert load_config().runtime.max_rounds == 9
    monkeypatch.setenv("FOREN_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(ConfigError, match="不存在"):
        load_config()


def test_no_config_means_builtin_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("FOREN_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.profile == "builtin-default"
    assert cfg.source is None
    assert cfg.model_root == (tmp_path / ".models").resolve()
