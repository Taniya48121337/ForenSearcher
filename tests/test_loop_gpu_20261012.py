"""Loop GPU plan on 8×A100 (10-12): per-role model names, single-GPU training guard, server scripts, packing."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOOP_SCRIPTS = ["loop_env.sh", "10_loop_setup.sh", "11_loop_layout.sh", "12_add_lora.sh", "13_train_queue.sh", "14_run_cases.sh"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_role_models_plain_name_serves_both_roles():
    from foren_searcher.datagen.teacher import parse_role_models

    assert parse_role_models("teacher") == {"HI": "teacher", "GIA": "teacher"}
    assert parse_role_models("HI=hi_r1, gia=gia_r1") == {"HI": "hi_r1", "GIA": "gia_r1"}


@pytest.mark.parametrize("bad", ["HI=hi_r1", "HI=hi_r1,GIA=", "HI=a,XX=b"])
def test_role_models_reject_partial_or_unknown_roles(bad):
    from foren_searcher.datagen.teacher import parse_role_models

    with pytest.raises(ValueError):
        parse_role_models(bad)


def test_endpoint_sends_the_role_s_model(tmp_path):
    from foren_searcher.datagen.teacher import EndpointTeacher

    teacher = EndpointTeacher(store=None, output_dir=tmp_path, endpoint="http://127.0.0.1:8002/v1", model="HI=hi_r1,GIA=gia_r1")
    try:
        assert teacher.model_for({"workspace_id": "W1-1"}) == "hi_r1"
        assert teacher.model_for({}) == "gia_r1"
        assert teacher.model_config["model_by_role"] == {"HI": "hi_r1", "GIA": "gia_r1"}
    finally:
        teacher.close()


def test_training_refuses_auto_device_map_over_several_gpus():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    S = _load("sft_lora", ROOT / "training" / "sft_lora.py")
    cfg = S.TrainConfig(role="hi", export_dir="x")
    assert cfg.gradient_checkpointing and not cfg.allow_multi_gpu
    S.check_visible_gpus(cfg, count=1)
    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        S.check_visible_gpus(cfg, count=8)
    S.check_visible_gpus(S.TrainConfig(role="hi", export_dir="x", allow_multi_gpu=True), count=2)
    S.check_visible_gpus(S.TrainConfig(role="hi", export_dir="x", device_map="cuda:0"), count=8)


def _defaults() -> dict[str, str]:
    text = (ROOT / "server" / "loop_env.sh").read_text(encoding="utf-8")
    return dict(re.findall(r'export (FS_\w+)="\$\{\1:-([^}]*)\}"', text))


def test_layout_a_gpus_do_not_overlap_and_training_reuses_teacher_gpus():
    d = _defaults()
    ids = lambda v: {g for g in v.replace(";", ",").split(",") if g}   # noqa: E731
    teacher, student, verifier = ids(d["FS_TEACHER_GPUS"]), ids(d["FS_STUDENT_GPUS"]), ids(d["FS_VERIFIER_GPUS"])
    assert not (teacher & student or teacher & verifier or student & verifier)
    assert teacher | student | verifier | ids(d["FS_TOOL_GPUS"]) == {str(i) for i in range(8)}
    assert ids(d["FS_TRAIN_SLOTS"]) <= teacher
    assert d["FS_LORA_RANK"] == "32"


def test_loop_ports_differ_from_each_other():
    text = (ROOT / "server" / "loop_env.sh").read_text(encoding="utf-8")
    ports = re.findall(r'FS_\w+_PORT="\$\{FS_\w+_PORT:-(\d+)\}"', text)
    assert sorted(ports) == ["8000", "8001", "8002", "8003"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("name", LOOP_SCRIPTS)
def test_loop_scripts_are_valid_bash(name):
    subprocess.run(["bash", "-n", str(ROOT / "server" / name)], check=True)


def test_pack_ships_training_and_loop_scripts():
    K = _load("_pack_server", ROOT / "scripts" / "_pack_server.py")
    files = {p.relative_to(ROOT).as_posix() for p in K.collect()}
    for need in ["training/sft_lora.py", "configs/loop_server.toml", *(f"server/{n}" for n in LOOP_SCRIPTS)]:
        assert need in files, need
