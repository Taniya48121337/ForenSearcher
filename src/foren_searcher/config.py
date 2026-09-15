"""运行配置：同一份代码，两份配置（本地调试 / 服务器训练）。

设计决定（DESIGN.md 第 7 节）：模型名、路径、批大小、LoRA 秩这些环境相关的东西全部
放进配置文件，代码本身不区分"我在 5090 上还是在 8×A100 上"。

用法：
    from foren_searcher.config import load_config
    cfg = load_config()                 # 按 FOREN_CONFIG 环境变量 → configs/local.toml → 内置默认 的顺序找
    cfg = load_config("configs/server.toml")

文件格式是 TOML（Python 3.11 标准库自带解析器，不引入新依赖）。示例见仓库根目录 configs/。
路径字段若是相对路径，相对于配置文件所在目录的上一级（即项目根目录）解析。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ENV_VAR = "FOREN_CONFIG"
DEFAULT_CONFIG_PATH = Path("configs") / "local.toml"


@dataclass(frozen=True)
class ModelsConfig:
    """模型口径：学生（GIA 与 HI 共用底座）、教师、本地权重目录。"""

    model_root: str = ".models"
    student: str = "Qwen3.5-4B"
    teacher: str = "api"
    teacher_endpoint: str = ""
    max_input_tokens: int = 24_000
    max_new_tokens: int = 4_096


@dataclass(frozen=True)
class PathsConfig:
    """数据与输出位置。harness_root 仅为兼容旧命令保留，留空即不使用 Harness-1。"""

    outputs: str = "outputs"
    sft_data: str = "experiments/sft_data"
    datasets: str = "external"
    harness_root: str = ""


@dataclass(frozen=True)
class RuntimeConfig:
    """一次调查的运行限制。"""

    max_rounds: int = 3
    max_gia_turns: int = 8
    gia_turn_timeout_s: int = 300
    max_workspace_turns: int = 12
    workspace_timeout_s: int = 3600
    max_concurrency: int = 4
    max_workspaces_per_round: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    """LoRA SFT 的关键超参。本地只验证流程，数字以服务器配置为准。"""

    lora_rank: int = 32
    lora_alpha: int = 64
    batch_size: int = 1
    gradient_accumulation: int = 8
    max_seq_len: int = 4_096
    learning_rate: float = 1e-4
    max_steps: int = 20
    device_map: str = "auto"
    bf16: bool = True


@dataclass(frozen=True)
class Config:
    profile: str = "builtin-default"
    source: Path | None = None
    models: ModelsConfig = field(default_factory=ModelsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def project_root(self) -> Path:
        """配置文件所在目录的上一级；没有配置文件时是当前目录。"""

        if self.source is None:
            return Path.cwd()
        return self.source.resolve().parent.parent

    def resolve(self, value: str | Path) -> Path:
        """把配置里的相对路径解析成绝对路径。"""

        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def model_root(self) -> Path:
        return self.resolve(self.models.model_root)

    @property
    def student_model_dir(self) -> Path:
        name = self.models.student
        return Path(name) if Path(name).is_absolute() else self.model_root / name

    @property
    def harness_root(self) -> Path | None:
        return self.resolve(self.paths.harness_root) if self.paths.harness_root else None


class ConfigError(ValueError):
    """配置文件缺失字段类型不对、或含未知字段。"""


def _build(section_type: type, data: dict[str, Any], section: str) -> Any:
    known = {f.name: f for f in fields(section_type)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"[{section}] 含未知字段: {', '.join(unknown)}；允许的字段: {', '.join(sorted(known))}")
    values: dict[str, Any] = {}
    for name, value in data.items():
        expected = known[name].type
        if expected == "int" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigError(f"[{section}].{name} 应为整数，得到 {value!r}")
        if expected == "float" and not isinstance(value, (int, float)):
            raise ConfigError(f"[{section}].{name} 应为数字，得到 {value!r}")
        if expected == "bool" and not isinstance(value, bool):
            raise ConfigError(f"[{section}].{name} 应为 true/false，得到 {value!r}")
        if expected == "str" and not isinstance(value, str):
            raise ConfigError(f"[{section}].{name} 应为字符串，得到 {value!r}")
        values[name] = value
    return section_type(**values)


def parse_config(text: str, *, source: Path | None = None) -> Config:
    """解析 TOML 文本。缺省的段和字段用内置默认值补齐。"""

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"配置文件不是合法 TOML: {exc}") from exc
    sections = {"models": ModelsConfig, "paths": PathsConfig, "runtime": RuntimeConfig, "training": TrainingConfig}
    unknown = sorted(set(data) - set(sections) - {"profile"})
    if unknown:
        raise ConfigError(f"配置文件含未知段: {', '.join(unknown)}；允许的段: profile, {', '.join(sections)}")
    parsed = {name: _build(kind, data.get(name, {}) or {}, name) for name, kind in sections.items()}
    profile = data.get("profile", source.stem if source else "inline")
    if not isinstance(profile, str):
        raise ConfigError("profile 应为字符串")
    return Config(profile=profile, source=source, **parsed)


def find_config_path(explicit: str | Path | None = None) -> Path | None:
    """定位配置文件：显式参数 → 环境变量 FOREN_CONFIG → 当前目录下 configs/local.toml → 无。"""

    for candidate in (explicit, os.environ.get(ENV_VAR), DEFAULT_CONFIG_PATH):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
        if candidate is explicit or candidate == os.environ.get(ENV_VAR):
            raise ConfigError(f"配置文件不存在: {path}")
    return None


def load_config(explicit: str | Path | None = None) -> Config:
    """读取配置；找不到任何配置文件时返回内置默认值（相当于本地调试口径）。"""

    path = find_config_path(explicit)
    if path is None:
        return Config()
    return parse_config(path.read_text(encoding="utf-8"), source=path)


__all__ = [
    "Config", "ConfigError", "DEFAULT_CONFIG_PATH", "ENV_VAR", "ModelsConfig", "PathsConfig", "RuntimeConfig",
    "TrainingConfig", "find_config_path", "load_config", "parse_config",
]
