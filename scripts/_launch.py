"""双击启动脚本的实际执行者（2026-09-30）。

为什么有它：.cmd 里写中文，cmd.exe 在 chcp 65001 前后按字节偏移续读批处理，含中文的行会被从中间切开
（"The system cannot find the path specified." / "'��（只是没有语音转写）。' is not recognized"）。
现在 .cmd 只剩纯 ASCII 的几行：设环境变量、调用本脚本、pause；中文提示、日志、步骤都在这里。

    python scripts\\_launch.py install_asr | r0c_reverify | mech_ab | new10 | calib_precision | gold_eval_v1p2 | diffmedia_ab | rules_preview | verifier_candidates | evid_smoke | evid_full | evid_score | reverify_036 | r0e | r0e_followup

每一步子进程的输出写进日志文件（窗口里只显示步骤和结果）；stdin 一律为空，不会卡在确认提示上
（install_asr 的安装确认例外，那是在本脚本里问的）。
"""

from __future__ import annotations

import glob
import re
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = Path("experiments") / "sft_data"
RUNS = SFT / "runs"
PY = sys.executable
CODEX_MODEL = os.environ.get("FOREN_CODEX_MODEL") or "gpt-6-astra"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Log:
    def __init__(self, name: str, title: str) -> None:
        self.path = ROOT / name
        self.path.write_text(f"==== {title} {now()}\n", encoding="utf-8")
        self.failed: list[str] = []

    def write(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def say(self, text: str) -> None:
        print(text, flush=True)
        self.write(text)

    def run(self, label: str, argv: list[str], *, show: bool = False) -> int:
        """跑一步：输出进日志（show=True 时同时打印到窗口），返回退出码。"""

        self.say(f"{label}（{now()}）")
        self.write("  > " + " ".join(argv))
        with self.path.open("a", encoding="utf-8") as stream:
            try:
                proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            except OSError as exc:
                self.say(f"  启动失败：{exc}")
                return 127
            assert proc.stdout is not None
            for line in proc.stdout:
                stream.write(line)
                stream.flush()
                if show:
                    print(line, end="", flush=True)
            code = proc.wait()
        self.say(f"  退出码 {code}")
        if code != 0:
            self.failed.append(label.split("（")[0].strip())
        return code

    def finish(self, text: str) -> None:
        if self.failed:
            self.say(f"有 {len(self.failed)} 步没成功：{'；'.join(self.failed)}。看 {self.path.name} 里对应步骤的报错。")
        self.say(("结束。" if self.failed else "完成。") + text)


def env_setup() -> None:
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["FOREN_CODEX_MODEL"] = CODEX_MODEL


def codex_preflight(log: Log) -> bool:
    codex = shutil.which("codex")
    if not codex:
        log.say("找不到 codex 命令（PATH 里没有），没有开跑。")
        return False
    code = log.run(f"[0] 检查 Codex 能否用 {CODEX_MODEL}",
                   [codex, "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "-m", CODEX_MODEL, "只回复 ok"])
    if code != 0:
        log.say(f"Codex 用不了 {CODEX_MODEL}，没有开跑。看 {log.path.name} 末尾的报错。")
        log.write("CODEX_PREFLIGHT_FAILED")
        return False
    return True


# ---------------------------------------------------------------------------- 任务

def install_asr() -> int:
    log = Log("_安装语音转写日志.txt", "安装语音转写依赖 faster-whisper（09-30）")
    print("这一步会用 pip 安装 faster-whisper（约 100–200 MB，含 ctranslate2 / onnxruntime），装进下面这个 python：")
    print(f"  {PY}")
    print("装好后，run_cases 默认（--inspect-ops auto）会给 HI 开“语音转写”工具；权重用 .models\\inspect\\faster-whisper-small，已在本机。")
    print("不想装就直接关掉窗口，其他流程照常能跑（只是没有语音转写）。")
    try:
        answer = input("确认安装请输入 y 回车：").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        log.say("没有安装（未确认）。")
        return 1
    log.run("[1/3] pip install faster-whisper", [PY, "-m", "pip", "install", "faster-whisper>=1.1,<2"], show=True)
    log.run("[2/3] 检查能否导入", [PY, "-c", "import faster_whisper, ctranslate2; print('faster_whisper', faster_whisper.__version__, "
                                          "'ctranslate2', ctranslate2.__version__)"], show=True)
    log.run("[3/3] 本机 INSPECT 会开哪些操作", [PY, "-c",
            "from foren_searcher.config import load_config\n"
            "from foren_searcher.datagen.run_cases import resolve_inspect_operations\n"
            "ops, d = resolve_inspect_operations('auto', load_config().model_root)\n"
            "print('INSPECT 操作（auto）：' + ', '.join(ops))\n"
            "[print('  未开', k, '：', v['reason']) for k, v in d['model_ops'].items() if not v['enabled']]"], show=True)
    log.finish(f"完整输出在 {log.path.name}")
    return 0


def r0c_reverify() -> int:
    log = Log("_r0c重核日志.txt", "r0c 用核验器 0.3.3 重核（09-30）")
    print("正在用 Claude CLI（Sonnet）按新核验器重核 r0c 的 11 个 run，不重跑教师。")
    print("文本对命中缓存；视频帧对、多图汇总、答案标签要重判，约 400 次调用，预计 1–2 小时，走订阅额度。")
    print(f"中途关窗口也没关系：已判过的在缓存里，再双击会接着判。详细输出在 {log.path.name}")
    old = SFT / "_verifier032_r0c"
    exports = sorted(glob.glob(str(SFT / "export" / "r0c_*claude-cli*")))
    marker = old / "_已备份.txt"          # 只备份一次：再双击时不能把 0.3.3 的新结果盖到备份上
    if not marker.exists():
        old.mkdir(parents=True, exist_ok=True)
        copied = []
        for pattern in ("verdicts_r0c_judged.claude-cli*.jsonl", "verdicts_units_r0c_judged.claude-cli*.jsonl"):
            for path in glob.glob(str(SFT / pattern)):
                shutil.copy2(path, old / Path(path).name)
                copied.append(Path(path).name)
        if exports and not (old / "export").exists():
            shutil.copytree(exports[0], old / "export")
            copied.append(Path(exports[0]).name + "\\")
        marker.write_text(f"{now()} 备份自 verifier@0.3.2 的结果：{', '.join(copied)}\n", encoding="utf-8")
        log.say(f"[0/3] 0.3.2 的结果已复制到 {old}（只复制不删除）：{', '.join(copied) or '没有找到'}")
    else:
        log.say(f"[0/3] {old} 已有备份，跳过")
    log.run("[1/3] 核验 + 导出（跳过第 4 步召回校准）",
            [PY, "scripts/complete_verify_local.py", "--judge", "claude-cli", "--runs-glob", "r0c", "--round", "r0c",
             "--hi-spec", "hi@0.4.0", "--gia-spec", "gia@0.5.0", "--skip", "4"])
    out = SFT / "verifier032_vs_033_r0c.md"
    log.run("[2/3] 0.3.2 → 0.3.3 对照",
            [PY, "scripts/_verifier_diff.py",
             "--old", str(old / "verdicts_r0c_judged.claude-cli*.jsonl"), str(old / "verdicts_units_r0c_judged.claude-cli*.jsonl"),
             "--new", str(SFT / "verdicts_r0c_judged.claude-cli*.jsonl"), str(SFT / "verdicts_units_r0c_judged.claude-cli*.jsonl"),
             "--out", str(out)])
    if exports:
        log.run("[3/3] 样例页", [PY, "scripts/_peek_export.py", exports[0], "--hi", "16"])
    log.finish(f"对照在 {out}；日志 {log.path.name}")
    return 0


TRUE6 = ["true:4124903", "true:17838107", "true:20205316753904", "true:20361105", "true:24171541709824", "true:39132300"]


def mech_ab() -> int:
    log = Log("_机制对照日志.txt", "机制对照（09-30，要在 r0c 之后跑）：两项结构改动各开一组，和 r0c 同案例比")
    print("正在运行：TRUE 6 案例开“搜索附摘录”、VideoSham 4126 开“附图上限 16”，约 1 小时。")
    print("检索用 r0c 录下的快照回放（未命中才联网补录）；规格与 INSPECT 操作集也与 r0c 相同（hi@0.4.0、只开基础五项），只差被测的那一项。")
    print(f"详细输出在 {log.path.name}")
    if not codex_preflight(log):
        return 1
    common = ["--teacher", "codex", "--retry-failed", "--blocklist", "exclude", "--search-mode", "replay_or_record",
              "--hi-spec", "hi@0.4.0", "--gia-spec", "gia@0.5.0", "--inspect-ops", "base"]
    log.run("[1/4] A 组：SEARCH 打开网页后附 600 字相关摘录（search_excerpt_chars=600），TRUE 6 个案例，轮次 r0cx",
            [PY, "-m", "foren_searcher.datagen.run_cases", "--round", "r0cx", "--ids", *TRUE6, *common,
             "--harness-override", "search_excerpt_chars=600"])
    log.run("[2/4] B 组：每次最多附图 16 张（image_input_limit=16），videosham:4126，轮次 r0ci",
            [PY, "-m", "foren_searcher.datagen.run_cases", "--round", "r0ci", "--ids", "videosham:4126", *common,
             "--harness-override", "image_input_limit=16"])
    before = [str(RUNS / f"true_{c.split(':')[1]}_r0c_1") for c in TRUE6]
    log.run("[3/4] A 组对照：r0c 的 TRUE 6 → r0cx",
            [PY, "scripts/compare_rounds.py", "--before-runs", *before, "--after-round", "r0cx", "--labels", "r0c", "摘录",
             "--out", str(SFT / "compare_excerpt_r0c_vs_r0cx.md")])
    log.run("[4/4] B 组对照：r0c 的 4126 → r0ci",
            [PY, "scripts/compare_rounds.py", "--before-runs", str(RUNS / "videosham_4126_r0c_1"), "--after-round", "r0ci",
             "--labels", "r0c", "附图16", "--out", str(SFT / "compare_imagelimit_r0c_vs_r0ci.md")])
    log.finish(f"对照表在 {SFT}\\compare_excerpt_r0c_vs_r0cx.md 和 compare_imagelimit_r0c_vs_r0ci.md；日志 {log.path.name}")
    return 0


NEW10 = ["averitec:1707", "averitec:1766", "averitec:392", "averimatec:423", "averimatec:660", "averimatec:699",
         "qvhighlights:9167", "vcsl:pair_1", "vcsl:pair_5", "vcsl:pair_6"]


def new10() -> int:
    log = Log("_新案例r0c日志.txt", "新增案例里属于 dev_train 的 10 个：Codex 教师首跑（hi@0.4.1 / gia@0.5.0）")
    print("正在运行：Codex 跑 10 个新案例（约 1–1.5 小时）→ Sonnet 核验与导出（约 30 分钟）。")
    print("只跑训练划分的案例；dev_select / dev_confirm / dev_eval 的案例留作评估，不产训练数据。")
    print("用 hi@0.4.1（回看图片一次带齐）；INSPECT 自动开本机有权重且装了包的模型工具。")
    print(f"详细输出在 {log.path.name}")
    if not codex_preflight(log):
        return 1
    log.run("[1/3] Codex 教师首跑（轮次 r0cn；黑名单 exclude；录下检索快照）",
            [PY, "-m", "foren_searcher.datagen.run_cases", "--teacher", "codex", "--retry-failed", "--round", "r0cn", "--ids", *NEW10,
             "--blocklist", "exclude", "--search-mode", "record", "--hi-spec", "hi@0.4.1", "--gia-spec", "gia@0.5.0"])
    log.run("[2/3] Sonnet 核验 + 导出",
            [PY, "scripts/complete_verify_local.py", "--judge", "claude-cli", "--runs-glob", "r0cn", "--round", "r0cn",
             "--hi-spec", "hi@0.4.1", "--gia-spec", "gia@0.5.0", "--skip", "4"])
    exports = sorted(glob.glob(str(SFT / "export" / "r0cn_*claude-cli*")))
    log.run("[3/3] 取证指标",
            [PY, "scripts/compare_rounds.py", "--before-runs", str(RUNS / "_none_"), "--after-round", "r0cn",
             *(["--after-export", exports[0]] if exports else []), "--labels", "无", "r0cn", "--out", str(SFT / "stats_r0cn.md")])
    if exports:
        log.run("      样例页", [PY, "scripts/_peek_export.py", exports[0], "--hi", "16"])
    log.finish(f"取证指标 {SFT}\\stats_r0cn.md；日志 {log.path.name}")
    return 0


def finish0929() -> int:
    """原 _0929本机收尾.cmd 的六步（09-29 写的，还没跑过；原 .cmd 含中文，有被 cmd 切坏的风险）。"""

    log = Log("_0929本机收尾日志.txt", "0929 本机收尾")
    print(f"正在运行，约 5-10 分钟，详细输出在 {log.path.name}")
    log.run("[1/6] 案卷护栏修补（699 / 39132300，已改过会跳过）", [PY, "scripts/_patch_cases_20260929.py", "--apply"])
    log.run("[2/6] 案卷 x 检索护栏对照", [PY, "scripts/_guard_case_audit.py", "--out",
                                          str(SFT / "cases_draft" / "guard_audit_20260929.md")])
    log.run("[3/6] verifier@0.3.1 可选证据组影响（第一层，全部本机 run）", [PY, "scripts/_optional_evidence_impact.py"])
    log.run("[4/6] 金标 v2：premise_smuggling / task_not_gap_driven 任务级批次",
            [PY, "scripts/_gold_sample.py", "--plan-tasks", "--out", str(SFT / "gold_set" / "v2_plan_tasks")])
    datagen_tests = sorted(glob.glob("tests/test_datagen_*.py")) + [
        "tests/test_search_retrieval_guard.py", "tests/test_search_recording.py", "tests/test_inspect_tool_cache.py",
        "tests/test_complete_verify_paths.py"]
    log.run("[5/6] 回归测试（数据流水线）", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *datagen_tests])
    log.run("[6/6] 回归测试（全部离线测试）", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"])
    log.finish(f"结果在 {log.path.name}，把它留着我来读。")
    return 0


def source_d(extra: list[str]) -> int:
    """原 _来源D离线换题_ollama.cmd：本机 Ollama 小模型跑来源 D 第二批（只验证链路，数字不进规划）。"""

    log = Log("_来源D离线换题日志.txt", "来源 D 离线换题：Ollama qwen3-vl 教师 + gemma3 核验")
    print(f"正在运行（Ollama 需已启动），时间较长，详细输出在 {log.path.name}")
    log.run("[1/2] 离线换题 + 核验", [PY, "-m", "foren_searcher.datagen.offline_gia", "--endpoint", "http://127.0.0.1:11434/v1",
                                    "--model", "qwen3vl-8b-32k:latest", "--then-verify", "--judge-endpoint", "http://127.0.0.1:11434/v1",
                                    "--judge-model", "gemma3-12b-16k:latest", *extra])
    log.run("[2/2] pytest 来源 D", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_datagen_offline_gia.py"])
    log.finish(f"结果在 {log.path.name}")
    return 0


def calib_precision() -> int:
    """核验器精确率：四层自动估计 + 召回守门，0.3.3 与 0.3.4 两版对比，再出一份两版共用的定向人工批次（10-02/10-03）。

    步骤：[1] 程序层预估（不调模型）→ 按回车 → [2] 只重算第二层得到 0.3.4 判定 → [3] 0.3.3 校准 → [4] 0.3.4 校准 → [5] 人工批次。
    全部调用走缓存：云端已经跑过的（verify_cache.precision.*.jsonl）这里直接命中，不花额度。
    """

    log = Log("_核验精确率日志.txt", "核验器精确率：四层自动估计 + 0.3.3/0.3.4 对比 + 定向人工批次")
    base = [p for p in sorted(glob.glob(str(SFT / "verdicts_r0c_judged.claude-cli*.jsonl")))
            if not re.search(r"\.(v0\d+|off\d+)\.jsonl$", p)]          # 不要把 reanchor 产出的 .v034.jsonl 当原始判定
    if not base:
        log.say("找不到 r0c 的 Claude CLI 核验结果（verdicts_r0c_judged.claude-cli*.jsonl），先跑 r0c 重核。")
        return 1
    rv = base[-1]
    uv = rv.replace("verdicts_r0c_judged", "verdicts_units_r0c_judged")
    rv4, uv4 = rv.replace(".jsonl", ".v034.jsonl"), uv.replace(".jsonl", ".v034.jsonl")
    cache = [p for p in sorted(glob.glob(str(SFT / "verify_cache.claude-cli-*.jsonl")))]
    second_kind = os.environ.get("FOREN_SECOND_KIND") or "claude-cli"
    second_model = os.environ.get("FOREN_SECOND_MODEL") or ("opus" if second_kind == "claude-cli" else "qwen3.5-27b")
    sonnet = str(SFT / "verify_cache.precision.sonnet.jsonl")
    common = ["--runs", str(RUNS), "--cases", str(SFT / "cases"), "--replay-cache", cache[-1] if cache else sonnet, "--cache", sonnet,
              "--second-kind", second_kind, "--second-model", second_model,
              "--second-cache", str(SFT / f"verify_cache.precision.{second_model.replace(':', '-')}.jsonl"),
              "--writer-model", "haiku", "--writer-cache", str(SFT / "verify_cache.precision.haiku.jsonl"), "--inject", "30"]
    if second_kind == "local":
        common += ["--endpoint", os.environ.get("FOREN_JUDGE_ENDPOINT") or "http://127.0.0.1:11434/v1"]
    mod = "foren_searcher.datagen.verify.calibrate_precision"
    print(f"第二评委：{second_kind}:{second_model}（换本机模型：set FOREN_SECOND_KIND=local 与 FOREN_SECOND_MODEL=模型名，Ollama 需已启动）")
    log.run("[1/5] 程序层（不调模型）：重建核对 + C1 定位 + 预计调用数",
            [PY, "-m", mod, "--report-verdicts", rv, "--unit-verdicts", uv, "--out", str(SFT / "precision_r0c_v033.json"),
             "--verifier", "0.3.3", *common, "--no-judge"], show=True)
    try:
        input("确认后按回车开始模型层（走订阅额度；云端已跑过的都在缓存里；中途关窗口再双击会接着跑）；不想跑就关窗口。")
    except EOFError:
        pass
    log.run("[2/5] 只重算第二层：r0c 的 0.3.4 判定",
            [PY, "-m", "foren_searcher.datagen.verify.reanchor", "--report-verdicts", rv, "--unit-verdicts", uv, "--runs", str(RUNS),
             "--cache", sonnet, "--out-suffix", "v034"])
    log.run("[3/5] 0.3.3 校准（四层 + 召回守门）",
            [PY, "-m", mod, "--report-verdicts", rv, "--unit-verdicts", uv, "--out", str(SFT / "precision_r0c_v033.json"), "--verifier", "0.3.3",
             "--retest-cache", str(SFT / "verify_cache.precision.sonnet_retest.jsonl"), *common])
    log.run("[4/5] 0.3.4 校准（四层 + 召回守门）",
            [PY, "-m", mod, "--report-verdicts", rv4, "--unit-verdicts", uv4, "--out", str(SFT / "precision_r0c_v034.json"), "--verifier", "0.3.4",
             "--retest-cache", str(SFT / "verify_cache.precision.sonnet_retest_v034.jsonl"), *common])
    out = SFT / "gold_set" / "v1p2_hard"
    if (out / "items.json").exists():
        log.say(f"[5/5] {out} 已有批次（可能已开始标注），不覆盖")
    else:
        log.run("[5/5] 定向人工批次（两版疑难件并集 + 对照，标一次评两版）",
                [PY, "scripts/_gold_sample.py", "--from-hard", str(SFT / "precision_r0c_v033.json"), str(SFT / "precision_r0c_v034.json"),
                 "--report-verdicts", rv, "--unit-verdicts", uv, "--out", str(out)])
    log.finish(f"汇总 {SFT / 'precision_r0c_v033.md'}、{SFT / 'precision_r0c_v034.md'}；标注页 {out / '标注.html'}")
    return 0


def gold_eval_v1p2() -> int:
    """金标 v1p2_hard：用你复核后导出的 labels_<缩写>.json 算两版核验器的精确率与召回（10-04）。

    复核页是批次目录里的 复核.html（双击打开）；导出的文件放回批次目录后运行本任务。不调模型。
    """

    log = Log("_金标复核评估日志.txt", "金标 v1p2_hard：人工复核结果 → 0.3.3 / 0.3.4 精确率与召回")
    batch = SFT / "gold_set" / "v1p2_hard"
    labels = [p for p in sorted(batch.glob("labels_*.json")) if not re.search(r"labels_(claude|tst)", p.name, re.I)]
    if not labels:
        log.say(f"{batch} 里还没有你导出的 labels_<缩写>.json：先双击 复核.html 复核并导出，再把文件放进这个目录。")
        return 1
    base = [p for p in sorted(glob.glob(str(SFT / "verdicts_r0c_judged.claude-cli*.jsonl")))
            if not re.search(r"\.(v0\d+|off\d+)\.jsonl$", p)]
    if not base:
        log.say("找不到 r0c 的 Claude CLI 核验结果（verdicts_r0c_judged.claude-cli*.jsonl）。")
        return 1
    rv = base[-1]
    uv = rv.replace("verdicts_r0c_judged", "verdicts_units_r0c_judged")
    log.say("标注文件：" + "、".join(p.name for p in labels))
    log.run("[1/1] 金标评估（两版核验器 × 分层 × 入选原因）",
            [PY, "scripts/_gold_eval.py", "--batch", str(batch), "--labels", *[p.name for p in labels], "--report-verdicts", rv, "--unit-verdicts", uv])
    log.finish(f"结果表写在 {batch} 下的 对照_*.md；日志 {log.path.name}")
    return 0


VIDEOSHAM5 = ["videosham:0069", "videosham:1007", "videosham:1028", "videosham:2010", "videosham:4126"]


def diffmedia_ab() -> int:
    """diff_media 对照（10-04）：5 个 VideoSham 案例，A 组 hi@0.4.1（无 diff_media）、B 组 hi@0.4.2（开 diff_media），其余设置全同。

    两组都新跑（不拿 r0c 当 A 组：r0c 是 hi@0.4.0、只开基础五项 INSPECT，差的不止一项）。
    """

    log = Log("_逐帧差异对照日志.txt", "diff_media 对照：VideoSham 5 案例 × hi@0.4.1 / hi@0.4.2")
    print("正在运行：Codex 跑 5 个 VideoSham 案例两遍（A 组不开、B 组开逐帧差异工具），约 2 小时；")
    print("然后 Sonnet 核验两组（约 40 分钟，走订阅额度），最后出取证指标和区间准确度对照。")
    print(f"中途关窗口没关系，再双击会跳过已跑完的 run、已判过的核验。详细输出在 {log.path.name}")
    if not codex_preflight(log):
        return 1
    common = ["--teacher", "codex", "--retry-failed", "--blocklist", "exclude", "--search-mode", "replay_or_record",
              "--gia-spec", "gia@0.5.0", "--ids", *VIDEOSHAM5]
    arms = (("r0dv1", "hi@0.4.1", "A 组：不开 diff_media"), ("r0dv2", "hi@0.4.2", "B 组：开 diff_media"))
    for i, (rnd, spec, title) in enumerate(arms, 1):
        log.run(f"[{i}/6] {title}（{spec}，轮次 {rnd}）",
                [PY, "-m", "foren_searcher.datagen.run_cases", "--round", rnd, "--hi-spec", spec, *common])
    for i, (rnd, spec, title) in enumerate(arms, 3):
        log.run(f"[{i}/6] Sonnet 核验 + 导出：{title}",
                [PY, "scripts/complete_verify_local.py", "--judge", "claude-cli", "--runs-glob", rnd, "--round", rnd,
                 "--hi-spec", spec, "--gia-spec", "gia@0.5.0", "--skip", "4"])
    before = sorted(str(p) for p in RUNS.glob("videosham_*_r0dv1_*") if p.is_dir() and ".failed" not in p.name)
    ex1 = sorted(glob.glob(str(SFT / "export" / "r0dv1_*claude-cli*")))
    ex2 = sorted(glob.glob(str(SFT / "export" / "r0dv2_*claude-cli*")))
    log.run("[5/6] 取证指标对照（步数、INSPECT 操作、核验结果、accepted 数）",
            [PY, "scripts/compare_rounds.py", "--before-runs", *(before or [str(RUNS / "_none_")]), "--after-round", "r0dv2",
             *(["--before-export", ex1[-1]] if ex1 else []), *(["--after-export", ex2[-1]] if ex2 else []),
             "--labels", "hi@0.4.1", "hi@0.4.2", "--out", str(SFT / "compare_diffmedia_r0dv1_vs_r0dv2.md")])
    log.run("[6/6] 区间准确度（与像素差分标准段逐帧比对，不依赖核验器）",
            [PY, "scripts/_diff_media_ab.py", "--rounds", "r0dv1", "r0dv2", "--out", str(SFT / "diff_media_ab.md")])
    log.finish(f"对照表 {SFT}\\compare_diffmedia_r0dv1_vs_r0dv2.md 和 {SFT}\\diff_media_ab.md；日志 {log.path.name}")
    return 0


def rules_preview() -> int:
    """候选程序规则（task_given / back_reference / verdict_sentence）在 r0c、r0cn 上的效果预览（10-04）。

    只重算第二层并把规则打开，产物另存为 *.v036r.jsonl，原结果不动；规则是纯程序判断，预计全部命中缓存、不花额度。
    """

    log = Log("_程序规则预览日志.txt", "核验器候选程序规则（默认关闭）：r0c / r0cn 预览")
    skip = re.compile(r"\.(v0\d+\w*|off\d+)\.jsonl$")
    plan = []
    for rnd, cache in (("r0c", SFT / "verify_cache.precision.sonnet.jsonl"), ("r0cn", SFT / "verify_cache.claude-cli-claude-sonnet-5-5.jsonl")):
        base = [p for p in sorted(glob.glob(str(SFT / f"verdicts_{rnd}_judged.claude-cli*.jsonl"))) if not skip.search(p)]
        if not base:
            log.say(f"找不到 {rnd} 的核验结果，跳过")
            continue
        rv = max(base, key=os.path.getmtime)
        uv = rv.replace(f"verdicts_{rnd}_judged", f"verdicts_units_{rnd}_judged")
        plan.append((rnd, rv, uv, cache))
    for i, (rnd, rv, uv, cache) in enumerate(plan, 1):
        log.run(f"[{2 * i - 1}/{len(plan) * 2}] {rnd}：第二层重算 + 程序规则（产物 .v036r.jsonl）",
                [PY, "-m", "foren_searcher.datagen.verify.reanchor", "--report-verdicts", rv, "--unit-verdicts", uv,
                 "--runs", str(RUNS), "--cache", str(cache), "--rules", "on", "--out-suffix", "v036r"], show=True)
        # r0c 的原始文件是 0.3.3，拿 0.3.4 的 .v034 当“规则关”的对照；r0cn 原始文件已是 0.3.4/0.3.5
        old = [rv.replace(".jsonl", ".v034.jsonl"), uv.replace(".jsonl", ".v034.jsonl")]
        if not Path(old[0]).is_file():
            old = [rv, uv]
        log.run(f"[{2 * i}/{len(plan) * 2}] {rnd}：规则关 → 规则开 对照",
                [PY, "scripts/_verifier_diff.py", "--old", *old,
                 "--new", rv.replace(".jsonl", ".v036r.jsonl"), uv.replace(".jsonl", ".v036r.jsonl"),
                 "--out", str(SFT / f"program_rules_{rnd}.md")])
    log.finish(f"对照在 {SFT}\\program_rules_r0c.md、program_rules_r0cn.md；日志 {log.path.name}。规则仍默认关闭，等金标复核后再定。")
    return 0


def verifier_candidates() -> int:
    """核验器候选改动（10-04，默认都关闭）：程序规则 + 推理核对 + 答案语义覆盖，在 r0c / r0cn 已有结果上叠加，不重判第二层。

    新增调用只有推理核对与语义覆盖（云端已跑过，在 verify_cache.candidates.sonnet.jsonl 里），预计全部命中缓存。
    """

    log = Log("_核验候选改动日志.txt", "核验器候选改动：程序规则 + 推理核对 + 答案语义覆盖（r0c / r0cn）")
    skip = re.compile(r"\.(v0\d+\w*|off\d+)\.jsonl$")
    cache = SFT / "verify_cache.candidates.sonnet.jsonl"
    plan = []
    for rnd in ("r0c", "r0cn"):
        base = [p for p in sorted(glob.glob(str(SFT / f"verdicts_{rnd}_judged.claude-cli*.jsonl"))) if not skip.search(p)]
        if not base:
            log.say(f"找不到 {rnd} 的核验结果，跳过")
            continue
        rv = max(base, key=os.path.getmtime)
        uv = rv.replace(f"verdicts_{rnd}_judged", f"verdicts_units_{rnd}_judged")
        if Path(rv.replace(".jsonl", ".v034.jsonl")).is_file():          # r0c 原文件是 0.3.3，以 0.3.4 的 .v034 为底
            rv, uv = rv.replace(".jsonl", ".v034.jsonl"), uv.replace(".jsonl", ".v034.jsonl")
        plan.append((rnd, rv, uv))
    for i, (rnd, rv, uv) in enumerate(plan, 1):
        log.run(f"[{i}/{len(plan)}] {rnd}：叠加候选改动（产物 .v036c.jsonl）",
                [PY, "-m", "foren_searcher.datagen.verify.candidates", "--report-verdicts", rv, "--unit-verdicts", uv,
                 "--runs", str(RUNS), "--cases", str(SFT / "cases"), "--cache", str(cache), "--out", str(SFT / f"verifier_candidates_{rnd}.md")],
                show=True)
    log.finish(f"对照在 {SFT}\\verifier_candidates_r0c.md、verifier_candidates_r0cn.md；日志 {log.path.name}。候选仍默认关闭。")
    return 0


# ---------------------------------------------------------------------------- EVID-Bench baseline（10-04）

EVID = Path("experiments") / "evid"
EVID_ROUND = os.environ.get("FOREN_EVID_ROUND") or "evidb1"   # evidb0 = 10-05 冒烟（hi@0.4.1、每轮最多 2 个工作区），留作记录
EVID_CONFIG = Path("configs") / "evid_local.toml"             # 与 local.toml 只差 max_workspaces_per_round 2 → 3
EVID_SPECS = ("hi@0.4.2", "gia@0.5.0")                         # 10-05 起与 run_cases 默认一致（hi@0.4.2 开放 diff_media）
OLLAMA = os.environ.get("FOREN_OLLAMA_ENDPOINT") or "http://127.0.0.1:11434/v1"
EVID_MODEL = os.environ.get("FOREN_EVID_MODEL") or "qwen3vl-8b-32k:latest"      # 被测：本机 Qwen3-VL-8B-Instruct（Ollama）
EVID_JUDGES = ("gemma3-12b-16k:latest", "qwen3-8b-16k:latest", "qwen3vl-8b-32k:latest")
EVID_SMOKE = ("video_001", "video_027", "video_151")    # 三大类各一个：单源剪辑 / 多源拼接 / AI 生成


def ollama_models() -> list[str] | None:
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(OLLAMA.rstrip("/") + "/models", timeout=5) as response:
            return [m["id"] for m in _json.loads(response.read().decode("utf-8")).get("data", [])]
    except Exception:  # noqa: BLE001
        return None


def start_ollama(log: Log) -> list[str] | None:
    """Ollama 没开：找到 ollama 命令就在后台起 ollama serve（独立进程，关掉本窗口也不停），最多等 60 秒。"""

    exe = shutil.which("ollama")
    if exe is None:
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        exe = str(local) if local.is_file() else None
    if exe is None:
        log.say("找不到 ollama 命令（PATH 和 %LOCALAPPDATA%\\Programs\\Ollama 里都没有）。")
        return None
    log.say(f"Ollama 没在运行，正在后台启动：{exe} serve（最多等 60 秒）")
    flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0       # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags, close_fds=True)
    except OSError as exc:
        log.say(f"  启动失败：{exc}")
        return None
    for _ in range(30):
        time.sleep(2)
        served = ollama_models()
        if served is not None:
            log.say("  Ollama 已就绪。")
            return served
    return None


def evid_preflight(log: Log) -> bool:
    """Ollama 在不在（不在就自动起）；被测模型与三个评委在不在；qwen3-8b-16k 没有就按 Modelfile 建（只是给 qwen3:8b 加 16K 上下文）。"""

    served = ollama_models()
    if served is None:
        served = start_ollama(log)
    if served is None:
        log.say(f"连不上 Ollama（{OLLAMA}），自动启动也没成功。请手动打开 Ollama（开始菜单里点 Ollama，或命令行 ollama serve），"
                "等托盘出现图标后再双击本文件。")
        log.failed.append("Ollama 未就绪")
        return False
    log.write(f"Ollama 上的模型：{served}")
    if "qwen3-8b-16k:latest" not in served and "qwen3:8b" in served:
        log.run("[0] 建评委 qwen3-8b-16k（qwen3:8b + 16K 上下文）",
                ["ollama", "create", "qwen3-8b-16k", "-f", str(Path("scripts") / "ollama" / "judge_qwen3.Modelfile")])
        served = ollama_models() or served
    missing = [m for m in (EVID_MODEL, *EVID_JUDGES) if m not in served]
    if missing:
        log.failed.append("缺模型")
        log.say(f"Ollama 上缺这些模型：{missing}。之前的 scripts\\_本地模型准备.cmd 会建 qwen3vl-8b-32k 与 gemma3-12b-16k；"
                "qwen3:8b 用 ollama pull qwen3:8b。")
        return False
    return True


def _evid_run_and_score(log: Log, ids: tuple[str, ...] | None, title: str) -> int:
    if not evid_preflight(log):
        return 1
    os.environ["FOREN_CONFIG"] = str(EVID_CONFIG)        # 子进程继承；run_config.config_source 会记下
    log.run("[1/4] EVID-Bench 标注 → 案卷（experiments\\evid\\cases，split=test）", [PY, "scripts/_evid_build_cases.py"])
    run = [PY, "-m", "foren_searcher.datagen.run_cases", "--cases", str(EVID / "cases"), "--runs-dir", str(EVID / "runs"),
           "--split", "test", "--round", EVID_ROUND, "--endpoint", OLLAMA, "--model", EVID_MODEL,
           "--instruction-mode", "student", "--retry-failed", "--blocklist", "exclude",
           "--hi-spec", EVID_SPECS[0], "--gia-spec", EVID_SPECS[1],   # 显式写出，run_config 与日志里一眼可见
           "--search-mode", "replay_or_record", "--snapshot-root", str(EVID / "search_snapshots")]
    if ids:
        run += ["--ids", *(f"evid:{i}" for i in ids)]      # run_cases 认 case_id（evid:video_001）或案卷文件名
    log.run(f"[2/4] 跑调查：{title}（{EVID_MODEL}，{EVID_SPECS[0]}，只给学生提示，轮次 {EVID_ROUND}）", run, show=True)
    score = [PY, "scripts/_evid_score.py", "--round", EVID_ROUND, "--endpoint", OLLAMA] + (["--ids", *ids] if ids else [])
    log.run("[3/4] 答案切分与完成情况（不调评委）", [*score, "--parse-only"])
    log.run("[4/4] 三评委打分（官方 EVID 评分提示词 + 多数票）", [*score, "--judges", *(f"local:{m}" for m in EVID_JUDGES)],
            show=True)
    return 0


def evid_smoke() -> int:
    """EVID-Bench baseline 冒烟：3 个样本（三大类各一），看链路、时长、答案格式和评委能不能出分。"""

    log = Log("_EVID冒烟日志.txt", "EVID-Bench baseline 冒烟：3 个样本 × 本机 Qwen3-VL-8B")
    print(f"正在运行：本机 Ollama 的 {EVID_MODEL} 跑 3 个 EVID 样本（{', '.join(EVID_SMOKE)}），再用 3 个本机评委打分。")
    print(f"中途关窗口没关系，再双击会跳过已跑完的 run。详细输出在 {log.path.name}")
    if _evid_run_and_score(log, EVID_SMOKE, "冒烟 3 个") != 0:
        log.finish("没有开跑。")
        return 1
    log.finish(f"汇总在 {EVID}\\scores\\{EVID_ROUND}.*.md；单个 run 在 {EVID}\\runs\\；日志 {log.path.name}")
    return 0


def evid_full() -> int:
    """EVID-Bench baseline 全量 222 个；已跑完的（含冒烟的 3 个）自动跳过。"""

    log = Log("_EVID全量日志.txt", "EVID-Bench baseline 全量：222 个样本 × 本机 Qwen3-VL-8B")
    print(f"正在运行：{EVID_MODEL} 跑全部 222 个 EVID 样本，再打分。时间取决于冒烟里每个样本的耗时（见 _EVID冒烟日志.txt）。")
    print(f"可以随时关窗口，再双击会从断点继续。详细输出在 {log.path.name}")
    if _evid_run_and_score(log, None, "全量 222 个") != 0:
        log.finish("没有开跑。")
        return 1
    log.finish(f"汇总在 {EVID}\\scores\\{EVID_ROUND}.*.md；日志 {log.path.name}")
    return 0


def evid_score() -> int:
    """只重新打分（例如换评委、补跑了几个 run 之后）。"""

    log = Log("_EVID评分日志.txt", f"EVID-Bench 评分：轮次 {EVID_ROUND}")
    if not evid_preflight(log):
        log.finish("没有打分。")
        return 1
    score = [PY, "scripts/_evid_score.py", "--round", EVID_ROUND, "--endpoint", OLLAMA]
    log.run("[1/2] 答案切分与完成情况", [*score, "--parse-only"])
    log.run("[2/2] 三评委打分", [*score, "--judges", *(f"local:{m}" for m in EVID_JUDGES)], show=True)
    log.finish(f"汇总在 {EVID}\\scores\\{EVID_ROUND}.*.md")
    return 0


def reverify_036() -> int:
    """核验器 0.3.6 正式启用后，r0c / r0cn 用新核验器重核并重新导出（10-05）。不重跑教师。可重复双击：备份只做一次。

    0.3.6 = 0.3.5 + 默认打开的程序规则 / 推理核对 / 语义覆盖（见 verdict.py 的版本注释）。
    先把 0.3.4 / 0.3.5 时期的判定与导出复制到 _verifier035_backup（只复制一次），再把精确率实验与候选实验的缓存并进主缓存，
    这样 0.3.6 新增的判定（云端已跑过）都能命中，预计几乎不花额度。
    """

    import json as _json

    log = Log("_核验器036重核日志.txt", "核验器 0.3.6：r0c / r0cn 重核 + 重新导出（不重跑教师）")
    backup = SFT / "_verifier035_backup"
    marker = backup / "_已备份.txt"
    skip = re.compile(r"\.(v0\d+\w*|off\d+)\.jsonl$")
    rounds = (("r0c", "hi@0.4.0"), ("r0cn", "hi@0.4.1"))
    if not marker.exists():
        backup.mkdir(parents=True, exist_ok=True)
        copied = []
        for rnd, _ in rounds:
            for pattern in (f"verdicts_{rnd}_judged.claude-cli*.jsonl", f"verdicts_units_{rnd}_judged.claude-cli*.jsonl"):
                for path in glob.glob(str(SFT / pattern)):
                    if not skip.search(path):
                        shutil.copy2(path, backup / Path(path).name)
                        copied.append(Path(path).name)
            for path in glob.glob(str(SFT / "export" / f"{rnd}_*claude-cli*")):
                if Path(path).is_dir() and not (backup / "export" / Path(path).name).exists():
                    shutil.copytree(path, backup / "export" / Path(path).name)
                    copied.append(Path(path).name + "\\")
        marker.write_text(f"{now()} 备份 0.3.4 / 0.3.5 时期的判定与导出：{', '.join(copied)}\n", encoding="utf-8")
        log.say(f"[0/4] 旧结果已复制到 {backup}（只复制不删除）：{len(copied)} 项")
    else:
        log.say(f"[0/4] {backup} 已有备份，跳过")
    main_cache = SFT / "verify_cache.claude-cli-claude-sonnet-5-5.jsonl"
    seen: set[str] = set()
    if main_cache.is_file():
        for line in main_cache.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(_json.loads(line)["key"])
            except (ValueError, KeyError, TypeError):
                continue
    added = 0
    with main_cache.open("a", encoding="utf-8") as out:
        for extra in ("verify_cache.precision.sonnet.jsonl", "verify_cache.candidates.sonnet.jsonl"):
            src = SFT / extra
            if not src.is_file():
                continue
            for line in src.read_text(encoding="utf-8").splitlines():
                try:
                    key = _json.loads(line)["key"]
                except (ValueError, KeyError, TypeError):
                    continue
                if key not in seen:
                    seen.add(key)
                    out.write(line + "\n")
                    added += 1
    log.say(f"[1/4] 精确率 / 候选实验的缓存并入主缓存：新增 {added} 条")
    for i, (rnd, spec) in enumerate(rounds, 2):
        log.run(f"[{i}/4] {rnd}：0.3.6 重核 + 导出（{spec}）",
                [PY, "scripts/complete_verify_local.py", "--judge", "claude-cli", "--runs-glob", rnd, "--round", rnd,
                 "--hi-spec", spec, "--gia-spec", "gia@0.5.0", "--skip", "4"])
    def latest(folder: Path, prefix: str) -> list[str]:
        files = [p for p in glob.glob(str(folder / f"{prefix}.claude-cli*.jsonl")) if not skip.search(p)]
        return [max(files, key=os.path.getmtime)] if files else []

    for rnd, _ in rounds:
        # 旧：r0c 用 0.3.4 的 .v034（备份里是 0.3.3 原始文件），r0cn 用备份里最新的 0.3.5；新：每类只取最新一份（同轮可能有 9runs/10runs 两份）
        v034 = [p for p in glob.glob(str(SFT / f"verdicts_{rnd}_judged.claude-cli*.v034.jsonl"))]
        old = ([v034[0], v034[0].replace(f"verdicts_{rnd}_judged", f"verdicts_units_{rnd}_judged")] if v034
               else latest(backup, f"verdicts_{rnd}_judged") + latest(backup, f"verdicts_units_{rnd}_judged"))
        new = latest(SFT, f"verdicts_{rnd}_judged") + latest(SFT, f"verdicts_units_{rnd}_judged")
        log.run(f"      {rnd}：0.3.4/0.3.5 → 0.3.6 对照", [PY, "scripts/_verifier_diff.py", "--old", *old, "--new", *new,
                                                       "--out", str(SFT / f"verifier036_{rnd}.md")])
    log.finish(f"对照在 {SFT}\\verifier036_r0c.md、verifier036_r0cn.md；新导出在 {SFT}\\export；日志 {log.path.name}")
    return 0


R0E_SPECS = ["--hi-spec", "hi@0.5.0", "--gia-spec", "gia@0.6.0"]
R0E_MARKS = {"hi_prompt": "搜索引擎看不到本次调查的",                      # hi@0.5.0 学生提示新增的一条
             "search_schema": "Image-input search is not available"}      # 10-05 晚：SEARCH 不再列以图搜图
R0E_SUPERSEDED = SFT / "runs_superseded" / "r0e"


def r0e_signature(run: Path) -> dict:
    """看一个 r0e run 实际用的是不是当前代码：HI 提示、SEARCH 工具清单、视频搜索走向。读 run 目录，不调模型。"""

    import json as _json

    cfg_path = run / "run_config.json"
    if not cfg_path.is_file():
        return {"ok": False, "why": "没有 run_config.json（半截）"}
    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
    problems = []
    if cfg.get("search"):
        if (cfg.get("video_search") or {}).get("mode") != "ytdlp_then_ddgs":
            problems.append("视频搜索未走 yt-dlp")
        if not (cfg.get("query_guard") or {}).get("enabled"):
            problems.append("查询守卫未开")
    hi_seen = False
    for req in sorted((run / "requests").glob("request_*/request.json")):
        try:
            data = _json.loads(req.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        system = str(data.get("system_prompt") or "")
        if "多模态调查代理 HI" not in system:
            continue
        hi_seen = True
        if R0E_MARKS["hi_prompt"] not in system:
            problems.append("HI 提示是 hi@0.5.0 以前的")
        payload = _json.dumps(data.get("payload"), ensure_ascii=False)
        if '"SEARCH"' in payload and R0E_MARKS["search_schema"] not in payload:
            problems.append("SEARCH 仍列以图搜图")
        break
    if not hi_seen:
        problems.append("找不到 HI 请求")
    return {"ok": not problems, "why": "；".join(problems)}


def round_r0e() -> int:
    """下一轮正式数据（10-05）：dev_train 全部 21 个案例，Codex 教师，核验器 0.3.6，规格 hi@0.5.0 / gia@0.6.0。

    案卷 10-05 补过：699 / 17838107 / 39132300 / 4124903 的必需证据组原来只剩 1 个非核查站点可取（其余被检索护栏拦），
    各补了 r0c / r0cn 已核验引用过的非核查来源（见各案卷 notes 末尾）。

    10-05 第一次双击时并行会话中途改了运行时代码（hi@0.5.0 / gia@0.6.0 提示、查询守卫、yt-dlp 视频搜索、SEARCH 不列以图搜图），
    前 5 个 run 各用了不同的旧配置。所以每次开跑前先逐个检查已完成的 r0e run（r0e_signature），
    不是当前配置的移到 runs_superseded\\r0e 后重跑；跑完再查一遍，仍有不一致就停在核验之前。
    可重复双击：配置一致且已跑完的 run、已判过的核验都会跳过。
    """

    log = Log("_r0e日志.txt", "r0e：dev_train 21 个案例，Codex 教师 + 核验器 0.3.6（hi@0.5.0 / gia@0.6.0）")
    print("正在运行：先检查已跑完的 r0e run 是否都是当前配置，不是的移走重跑；")
    print("然后 Codex 跑剩下的案例（每个约 5–15 分钟）→ Sonnet 核验与导出（约 1 小时，走订阅额度）→ 取证指标。")
    print("跑的过程中请不要在别的会话里改 src 下的代码：每个 run 都会重新加载代码，中途改动会让同一轮混用两套配置。")
    print(f"中途关窗口没关系，再双击会接着跑。详细输出在 {log.path.name}")

    def audit(stage: str) -> list[Path]:
        runs = sorted(p for p in RUNS.glob("*_r0e_*") if p.is_dir() and ".failed" not in p.name)
        stale = []
        for run in runs:
            sig = r0e_signature(run)
            if not sig["ok"]:
                stale.append(run)
                log.say(f"      {run.name}：{sig['why']}")
        log.say(f"{stage}：{len(runs)} 个 r0e run，{len(runs) - len(stale)} 个是当前配置，{len(stale)} 个不是")
        return stale

    stale = audit("[0/4] 配置检查")
    if stale:
        R0E_SUPERSEDED.mkdir(parents=True, exist_ok=True)
        for run in stale:
            target = R0E_SUPERSEDED / run.name
            if target.exists():
                target = R0E_SUPERSEDED / f"{run.name}.{time.strftime('%Y%m%d%H%M%S')}"
            shutil.move(str(run), str(target))
        log.say(f"      已移到 {R0E_SUPERSEDED}（不删除），下面重跑")
    if not codex_preflight(log):
        return 1
    log.run("[1/4] Codex 教师（轮次 r0e；split dev_train；黑名单 exclude；检索快照有则回放、没有则录）",
            [PY, "-m", "foren_searcher.datagen.run_cases", "--teacher", "codex", "--retry-failed", "--round", "r0e",
             "--split", "dev_train", "--blocklist", "exclude", "--search-mode", "replay_or_record", *R0E_SPECS])
    stale = audit("[2/4] 跑完复查")
    if stale:
        log.say("有 run 不是当前配置（多半是跑的过程中代码又被改了），没有进入核验。确认代码不再改动后再双击，会移走这些 run 重跑。")
        log.finish(f"日志 {log.path.name}")
        return 1
    log.run("[3/4] Sonnet 核验（verifier@0.3.6）+ 导出",
            [PY, "scripts/complete_verify_local.py", "--judge", "claude-cli", "--runs-glob", "r0e", "--round", "r0e", *R0E_SPECS,
             "--skip", "4"])
    before = sorted(str(p) for rnd in ("r0c", "r0cn") for p in RUNS.glob(f"*_{rnd}_*") if p.is_dir() and ".failed" not in p.name)
    exports = sorted(glob.glob(str(SFT / "export" / "r0e_*claude-cli*")), key=os.path.getmtime)
    log.run("[4/4] 取证指标：r0c + r0cn（同 21 个案例）→ r0e",
            [PY, "scripts/compare_rounds.py", "--before-runs", *(before or [str(RUNS / "_none_")]), "--after-round", "r0e",
             *(["--after-export", exports[-1]] if exports else []), "--labels", "r0c+r0cn", "r0e", "--out", str(SFT / "stats_r0e.md")])
    if exports:
        log.run("      样例页", [PY, "scripts/_peek_export.py", exports[-1], "--hi", "16"])
    log.finish(f"取证指标 {SFT}\\stats_r0e.md；导出在 {SFT}\\export；日志 {log.path.name}")
    return 0

def _latest_verdicts(rnd: str) -> tuple[str | None, str | None]:
    """某轮最新的 (报告判定, 单元判定) 文件；跳过 .v034 / .off / 候选叠加产物。"""

    skip = re.compile(r"\.(v0\d+\w*|off\d+)\.jsonl$")

    def latest(prefix: str) -> str | None:
        files = [p for p in glob.glob(str(SFT / f"{prefix}_{rnd}_judged.claude-cli*.jsonl")) if not skip.search(p)]
        return max(files, key=os.path.getmtime) if files else None

    return latest("verdicts"), latest("verdicts_units")


def _r0e_ready(log: Log) -> tuple[str, str] | None:
    """r0e 跑完且配置一致才返回 (报告判定, 单元判定)；否则说明原因返回 None。"""

    import json as _json

    rep, units = _latest_verdicts("r0e")
    if not rep or not units:
        log.say("还没有 r0e 的核验结果：先双击 _r0e.cmd 跑完。")
        return None
    runs = sorted({_json.loads(line)["subject"]["run"] for line in Path(rep).read_text(encoding="utf-8").splitlines() if line.strip()})
    bad = [r for r in runs if not (RUNS / r).is_dir() or not r0e_signature(RUNS / r)["ok"]]
    if bad:
        log.say(f"最新的 r0e 核验结果（{Path(rep).name}）里有 {len(bad)} 个 run 已被移走或不是当前配置：{', '.join(bad[:6])}。"
                "先双击 _r0e.cmd 跑完（它会重跑这些 run 并重新核验）。")
        return None
    log.say(f"r0e 核验结果：{Path(rep).name}（{len(runs)} 个 run，配置一致）")
    return rep, units


def r0e_followup() -> int:
    """r0e 之后的两件事（10-05）：①多来源合并复核的离线评估（候选，默认关闭）；②出错点重答（GIA 答案没通过的状态让教师再答）。

    ① 只在已有判定上叠加，不改原文件：verify.candidates --joint on，对 r0e、r0c、r0cn 最新的 0.3.6 判定各出一份 .v036j.jsonl，
       汇总写 experiments\\sft_data\\verifier_joint_eval.md。新增的核验调用只有"一句话引多条来源、仍判无支持"的片段，几十次。
    ② datagen.bridge：r0e 每个 negative 答案最多重答 2 次（Codex，同一状态），新答案走完整核验；
       通过的写进 experiments\\sft_data\\bridges\\r0e\\（gia_bridge_accepted.jsonl + gia_preference_pairs.jsonl），汇总 summary.md。
    可重复双击：已判过的走缓存，已重答过的不再调教师。
    """

    log = Log("_r0e后续日志.txt", "r0e 后续：多来源合并复核评估 + 出错点重答")
    os.environ.setdefault("FOREN_JUDGE_CONCURRENCY", "3")      # 与 complete_verify_local 对 Claude CLI 的设置一致
    print("正在运行：①多来源合并复核的离线评估（几分钟，走订阅额度）；②出错点重答（约 10 个答案 × 最多 2 次 Codex，约半小时）。")
    print(f"详细输出在 {log.path.name}")
    ready = _r0e_ready(log)
    if ready is None:
        log.finish(f"日志 {log.path.name}")
        return 1
    rep_e, units_e = ready
    import json as _json

    work = SFT / "_joint_eval"
    work.mkdir(parents=True, exist_ok=True)
    outs = []
    for i, rnd in enumerate(("r0e", "r0c", "r0cn"), 1):
        rep, units = (rep_e, units_e) if rnd == "r0e" else _latest_verdicts(rnd)
        if not rep:
            log.say(f"      {rnd}：没有判定文件，跳过")
            continue
        # 只留 run 目录还在的行（被移到 runs_superseded 的 run 打不开）
        copies = []
        for src in (rep, units):
            if not src:
                continue
            rows = [l for l in Path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
            keep = [l for l in rows if (RUNS / _json.loads(l)["subject"]["run"]).is_dir()]
            dst = work / Path(src).name
            dst.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
            copies.append(str(dst))
        out = work / f"joint_{rnd}.md"
        outs.append(out)
        log.run(f"[1.{i}] 多来源合并复核评估：{rnd}",
                [PY, "-m", "foren_searcher.datagen.verify.candidates", "--report-verdicts", copies[0],
                 *(["--unit-verdicts", copies[1]] if len(copies) > 1 else []), "--runs", str(RUNS),
                 "--cache", str(SFT / "verify_cache.claude-cli-claude-sonnet-5-5.jsonl"), "--judge-model", "claude-sonnet-5-5",
                 "--infer", "off", "--semcov", "off", "--joint", "on", "--out-suffix", "v036j", "--out", str(out)])
    merged = ["# 多来源合并复核（候选）离线评估", "", "只叠加多来源合并复核（程序规则照常、推理核对与语义覆盖关闭以免重复），原判定文件不动。", ""]
    for out in outs:
        if out.is_file():
            merged.append(out.read_text(encoding="utf-8"))
    (SFT / "verifier_joint_eval.md").write_text("\n".join(merged), encoding="utf-8")
    log.run("[2/2] 出错点重答（r0e 的 negative 答案，Codex 同一状态最多 2 次）",
            [PY, "-m", "foren_searcher.datagen.bridge", "--round", "r0e", "--units", units_e, "--verdicts", rep_e,
             "--gia-spec", "gia@0.6.0", "--attempts", "2", "--judge", "claude-cli", "--model", "claude-sonnet-5-5",
             "--cache", str(SFT / "verify_cache.jsonl")])
    log.finish(f"评估 {SFT}\\verifier_joint_eval.md；重答 {SFT}\\bridges\\r0e\\summary.md；日志 {log.path.name}")
    return 0



R0F_SPECS = ["--hi-spec", "hi@0.7.0", "--gia-spec", "gia@0.8.0"]     # 10-07：英文规格（= hi@0.6.0 / gia@0.7.0 + language=en）
R0F_SUPERSEDED = SFT / "runs_superseded" / "r0f"
R0F_MARKS = {"hi_role": "multimodal investigation agent",                       # hi@0.7.0 英文 HI 系统提示
             "hi_prompt": "Search engines cannot see this investigation's",    # hi@0.5.0 起的"查询不写内部编号"一条（英文）
             "search_schema": "Image-input search is not available"}


def r0f_signature(run: Path) -> dict:
    """r0f run 是不是当前配置（10-07）：英文规格 hi@0.7.0 / gia@0.8.0（输入补全 + language=en）、英文任务文字、
    yt-dlp 视频搜索与查询守卫、HI 提示是英文版、教师看到的是学生同档位（compact_16k）渲染后的状态。读 run 目录，不调模型。"""

    import json as _json
    import re as _re

    cfg_path = run / "run_config.json"
    if not cfg_path.is_file():
        return {"ok": False, "why": "没有 run_config.json（半截）"}
    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
    problems = []
    if cfg.get("search"):
        if (cfg.get("video_search") or {}).get("mode") != "ytdlp_then_ddgs":
            problems.append("视频搜索未走 yt-dlp")
        if not (cfg.get("query_guard") or {}).get("enabled"):
            problems.append("查询守卫未开")
    specs = _json.dumps(cfg.get("teacher_specs") or {})
    if "hi@0.7.0" not in specs or "gia@0.8.0" not in specs:
        problems.append(f"规格不是 hi@0.7.0 / gia@0.8.0（{specs}）")
    features = ((cfg.get("runtime_guards") or {}).get("input_features")) or {}
    if not features.get("ingest_keyframes"):
        problems.append("输入补全没打开（runtime_guards.input_features 无 ingest_keyframes）")
    if features.get("language") != "en" or cfg.get("task_language") != "en":
        problems.append("不是英文规格 / 英文任务文字")
    elif _re.search(r"[\u4e00-\u9fff]", str(cfg.get("task") or "")):
        problems.append("任务文字里有中文")
    rendered = total = 0
    hi_seen = False
    for req in sorted((run / "requests").glob("request_*/request.json")):
        try:
            data = _json.loads(req.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        total += 1
        rendered += (data.get("state_render") or {}).get("renderer") == "compact_16k"
        system = str(data.get("system_prompt") or "")
        if hi_seen or R0F_MARKS["hi_role"] not in system:
            continue
        hi_seen = True
        if R0F_MARKS["hi_prompt"] not in system:
            problems.append("HI 提示不是当前英文版")
        if _re.search(r"[\u4e00-\u9fff]", system):
            problems.append("HI 系统提示 / 教师指令里有中文")
        payload = _json.dumps(data.get("payload"), ensure_ascii=False)
        if '"SEARCH"' in payload and R0F_MARKS["search_schema"] not in payload:
            problems.append("SEARCH 仍列以图搜图")
    if not hi_seen:
        problems.append("找不到英文 HI 请求")
    if total and rendered == 0:
        problems.append("教师看的仍是原始全量视图（request.json 没有 state_render）")
    return {"ok": not problems, "why": "；".join(problems), "rendered_requests": f"{rendered}/{total}"}


def round_r0f() -> int:
    """下一轮（10-05 晚定，10-07 改为英文）：修掉"教师与学生输入不一致"等问题之后的第一轮正式数据。

    与 r0e 相同的 21 个 dev_train 案例、Codex 教师、核验器 0.3.6（程序规则 10-07 起中英双语），改动：
    - 规格 hi@0.7.0 / gia@0.8.0：模型输入输出全英文（任务文字用案卷 task_text_en），state_renderer=compact_16k，输入补全（首轮关键帧与时长、证据卡 origin、候选可执行用法、
      跨轮交接 R、HI 证据视图给原件帧留名额、GIA 附图原件帧优先、约束可见）；
    - 教师按学生同一档位看状态（Codex 与端点教师都一样；r0e 的 35 条报告监督里 17 条用了学生看不到的内容）；
    - 导出默认把仍"用了学生看不到的内容"的行降为 held（--hidden-context hold），manifest.hidden_context 记数。
    第 0 步先用同一规则把 r0e 重新导出一次（只读已有判定，不调模型），便于与 r0f 对照。
    可重复双击：配置一致且已跑完的 run、已判过的核验都会跳过；配置不一致的 r0f run 移到 runs_superseded\\r0f 重跑。
    """

    log = Log("_r0f日志.txt", "r0f：dev_train 21 个案例，英文，Codex 教师（学生同档位输入）+ 核验器 0.3.6（hi@0.7.0 / gia@0.8.0）")
    print("正在运行：r0e 按新规则重新导出（几分钟）→ 检查已跑完的 r0f run → Codex 跑剩下的案例（每个约 5–15 分钟）")
    print("→ Sonnet 核验与导出（约 1 小时，走订阅额度）→ r0e 对 r0f 的取证指标。")
    print("跑的过程中请不要在别的会话里改 src 下的代码：每个 run 都会重新加载代码。")
    print(f"中途关窗口没关系，再双击会接着跑。详细输出在 {log.path.name}")

    if _latest_verdicts("r0e")[0]:
        log.run("[0/5] r0e 重新导出（教师用了学生看不到的内容的行降为 held）",
                [PY, "scripts/complete_verify_local.py", "--judge", "claude-cli", "--runs-glob", "r0e", "--round", "r0e",
                 *R0E_SPECS, "--from", "5", "--to", "5"])
    else:
        log.say("[0/5] 没有 r0e 判定，跳过重新导出")

    def audit(stage: str) -> list[Path]:
        runs = sorted(p for p in RUNS.glob("*_r0f_*") if p.is_dir() and ".failed" not in p.name)
        stale = []
        for run in runs:
            sig = r0f_signature(run)
            if not sig["ok"]:
                stale.append(run)
                log.say(f"      {run.name}：{sig['why']}")
        log.say(f"{stage}：{len(runs)} 个 r0f run，{len(runs) - len(stale)} 个是当前配置，{len(stale)} 个不是")
        return stale

    stale = audit("[1/5] 配置检查")
    if stale:
        R0F_SUPERSEDED.mkdir(parents=True, exist_ok=True)
        for run in stale:
            target = R0F_SUPERSEDED / run.name
            if target.exists():
                target = R0F_SUPERSEDED / f"{run.name}.{time.strftime('%Y%m%d%H%M%S')}"
            shutil.move(str(run), str(target))
        log.say(f"      已移到 {R0F_SUPERSEDED}（不删除），下面重跑")
    if not codex_preflight(log):
        return 1
    log.run("[2/5] Codex 教师（轮次 r0f；split dev_train；黑名单 exclude；检索快照有则回放、没有则录）",
            [PY, "-m", "foren_searcher.datagen.run_cases", "--teacher", "codex", "--retry-failed", "--round", "r0f",
             "--split", "dev_train", "--blocklist", "exclude", "--search-mode", "replay_or_record", *R0F_SPECS])
    stale = audit("[3/5] 跑完复查")
    if stale:
        log.say("有 run 不是当前配置（多半是跑的过程中代码又被改了），没有进入核验。确认代码不再改动后再双击，会移走这些 run 重跑。")
        log.finish(f"日志 {log.path.name}")
        return 1
    log.run("[4/5] Sonnet 核验（verifier@0.3.6）+ 导出（hidden_context hold）",
            [PY, "scripts/complete_verify_local.py", "--judge", "claude-cli", "--runs-glob", "r0f", "--round", "r0f", *R0F_SPECS,
             "--skip", "4"])
    before = sorted(str(p) for p in RUNS.glob("*_r0e_*") if p.is_dir() and ".failed" not in p.name)
    exports = sorted(glob.glob(str(SFT / "export" / "r0f_*claude-cli*")), key=os.path.getmtime)
    log.run("[5/5] 取证指标：r0e → r0f（同 21 个案例）",
            [PY, "scripts/compare_rounds.py", "--before-runs", *(before or [str(RUNS / "_none_")]), "--after-round", "r0f",
             *(["--after-export", exports[-1]] if exports else []), "--labels", "r0e", "r0f", "--out", str(SFT / "stats_r0f.md")])
    if exports:
        log.run("      样例页", [PY, "scripts/_peek_export.py", exports[-1], "--hi", "16"])
    log.finish(f"取证指标 {SFT}\\stats_r0f.md；导出在 {SFT}\\export；日志 {log.path.name}")
    return 0


def traj_pilot(extra: list[str]) -> int:
    """分叉式轨迹生成试点（10-11）：自由跑 → V1 屏蔽原件 → 关键状态分叉 → 事实表打分。详见 scripts/traj_pilot.py。

    训练目标不变：GIA / HI 的逐决策策略（学生可见状态 → JSON 决定）。试点看分叉出来的做法结果是否真的不同、
    选出的那支能否从可见状态推出、V1 下教师是否克制、每支成本多少。可重复双击，做过的跳过；
    其余参数原样传给 traj_pilot.py（例如 --tag tp1 --cases averitec_392）。
    """

    log = Log("_分叉试点日志.txt", "分叉式轨迹生成试点（Codex 教师，英文规格 + 合并工具目录）")
    print("正在运行：3 个案例自由跑 → 2 个案例屏蔽原件再跑 → 每条轨迹挑 2 个关键状态让教师给别的做法并分叉续跑 → 打分。")
    print("约 3–4 小时（每支分叉都要 Codex 续跑）。中途关窗口没关系，再双击会跳过已完成的部分。")
    print(f"跑的过程中请不要在别的会话里改 src 下的代码。详细输出在 {log.path.name}")
    if not codex_preflight(log):
        return 1
    for step, label in (("free", "[1/4] 自由跑"), ("cf", "[2/4] V1：屏蔽原件再跑"), ("fork", "[3/4] 关键状态分叉"),
                        ("analyze", "[4/4] 事实表打分")):
        log.run(label, [PY, "scripts/traj_pilot.py", step, *extra])
    tag = extra[extra.index("--tag") + 1] if "--tag" in extra else "tp1"
    log.finish(f"结果 {SFT}\\traj_pilot\\{tag}\\report.md；日志 {log.path.name}")
    return 0

def traj_pilot1(extra: list[str]) -> int:
    """分叉试点第二轮 tp1（10-11 晚）：不重跑自由轨迹，在 tp0 的三条自由轨迹上重新挑关键状态分叉。
    改动：前缀按角色 + 状态指纹对号回放；step 远度（分叉动作执行完就停，不花教师调用）+ 每条轨迹 1 个 workspace 远度
    （跑到该工作区交报告，教师用 medium 档）；V1 屏蔽修好（认通配）+ 新增 V0 不让检索；打分按证据组 / 工具产出 /
    核验模型判结论（本机 Claude CLI）；写偏好对候选 pairs.jsonl。其余参数原样传给 traj_pilot.py。"""

    log = Log("_分叉试点tp1日志.txt", "分叉试点 tp1（在 tp0 自由轨迹上分叉；Codex 教师续跑用 medium 档）")
    print("正在运行：V1 屏蔽原件 + V0 不让检索（各 2 个案例）→ 每条 tp0 轨迹挑 3 个关键状态分叉 → 打分并写偏好对候选。")
    print("预计 1.5–2.5 小时：V1/V0 要整条跑；分叉的 step 支不花教师调用，workspace 支每支约 5–10 分钟。")
    print(f"中途关窗口没关系，再双击会跳过已完成的部分。跑的过程中请不要在别的会话里改 src 下的代码。详细输出在 {log.path.name}")
    if not codex_preflight(log):
        return 1
    args = ["--tag", "tp1", "--source-tag", "tp0", *extra]
    for step, label in (("cf", "[1/3] V1 屏蔽原件 + V0 不让检索"), ("fork", "[2/3] 关键状态分叉"), ("analyze", "[3/3] 打分与偏好对")):
        log.run(label, [PY, "scripts/traj_pilot.py", step, *args])
    log.finish(f"结果 {SFT}\\traj_pilot\\tp1\\report.md 与 pairs.jsonl；日志 {log.path.name}")
    return 0

TP2_NEW = ["averimatec_423", "averimatec_699", "averitec_1766", "true_17838107", "true_39132300", "qvhighlights_9167"]
TP2_CLAIMS = ["averimatec_423", "averimatec_699", "averitec_1766", "true_17838107", "true_39132300"]


def traj_pilot2(extra: list[str]) -> int:
    """分叉试点第三轮 tp2（10-12）：扩到 8 个案例，看偏好对的产出率。
    新增 6 个案例先自由跑（Codex xhigh，和 tp0 同配置）；averitec_392 / true_4124903 复用 tp0 的自由轨迹。
    修了 tp1 的三处：分叉回放按源轨迹当时的检索快照（含超时）重放、前缀走偏立即作废；Wayback 存档按原址认；
    averitec_392 原站域名拼写。step 只在取材料失败 / 原件链接可见的状态上跑，每条轨迹 2 个 workspace 远度。
    V0（不让检索）只对新增的 5 个核查题跑。其余参数原样传给 traj_pilot.py。"""

    log = Log("_分叉试点tp2日志.txt", "分叉试点 tp2（8 个案例；自由跑 xhigh，分叉续跑 medium）")
    print("正在运行：6 个新案例自由跑 → 5 个核查题 V0 不让检索 → 8 条轨迹各挑 3 个关键状态分叉 → 打分并写偏好对候选。")
    print("预计 4–5 小时（自由跑每条约 20–30 分钟；分叉的 workspace 支每支约 2–5 分钟）。适合放一夜。")
    print(f"中途关窗口没关系，再双击会跳过已完成的部分。跑的过程中请不要在别的会话里改 src 下的代码。详细输出在 {log.path.name}")
    if not codex_preflight(log):
        return 1
    common = ["--tag", "tp2", "--source-tag", "tp2", "--fallback-source", "tp0", *extra]
    steps = (("free", "[1/4] 新案例自由跑", ["--cases", *TP2_NEW]),
             ("cf", "[2/4] V0 不让检索", ["--cf", *TP2_CLAIMS, "--cf-modes", "v0"]),
             ("fork", "[3/4] 关键状态分叉", ["--cases", *TP2_NEW, "averitec_392", "true_4124903"]),
             ("analyze", "[4/4] 打分与偏好对", ["--cases", *TP2_NEW, "averitec_392", "true_4124903"]))
    for step, label, args in steps:
        log.run(label, [PY, "scripts/traj_pilot.py", step, *common, *args])
    log.finish(f"结果 {SFT}\\traj_pilot\\tp2\\report.md 与 pairs.jsonl；日志 {log.path.name}")
    return 0

JOBS = {"install_asr": install_asr, "r0c_reverify": r0c_reverify, "mech_ab": mech_ab, "new10": new10,
        "finish0929": finish0929, "source_d": source_d, "calib_precision": calib_precision, "gold_eval_v1p2": gold_eval_v1p2,
        "diffmedia_ab": diffmedia_ab, "rules_preview": rules_preview, "verifier_candidates": verifier_candidates,
        "evid_smoke": evid_smoke, "evid_full": evid_full, "evid_score": evid_score,
        "reverify_036": reverify_036, "r0e": round_r0e, "r0e_followup": r0e_followup, "r0f": round_r0f,
        "traj_pilot": traj_pilot, "traj_pilot1": traj_pilot1, "traj_pilot2": traj_pilot2}
EXTRA_ARGS = {"source_d", "traj_pilot", "traj_pilot1", "traj_pilot2"}         # 这些任务把 .cmd 的其余参数（%*）原样传下去


def main() -> int:
    job = sys.argv[1] if len(sys.argv) > 1 else ""
    if job not in JOBS or (len(sys.argv) > 2 and job not in EXTRA_ARGS):
        print("用法：python scripts\\_launch.py " + " | ".join(JOBS))
        return 2
    env_setup()
    return JOBS[job](sys.argv[2:]) if job in EXTRA_ARGS else JOBS[job]()


if __name__ == "__main__":
    raise SystemExit(main())
