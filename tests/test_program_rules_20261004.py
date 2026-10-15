"""核验器候选程序规则（默认关闭）：复述任务信息 / 回指已建立内容 / 答案总结论。用 r0c、r0cn 里的真实片段。"""

from __future__ import annotations

import pytest

from foren_searcher.datagen.errors import ErrorType, Severity, error
from foren_searcher.datagen.verify import anchored as A

TASK_392 = "核查以下声明……Budweiser employee……（声明日期：2020-07-02）"
TASK_660 = "核查以下声明，必要时检查给定媒体并搜索外部证据。（声明日期：2022-09-14）"


def test_task_given() -> None:
    assert A.is_task_given("声明日期2020年7月2日", TASK_392)
    assert A.is_task_given("声明日期2022年9月14日", TASK_660)
    assert not A.is_task_given("声明日期2022年9月15日", TASK_660)          # 日期不在任务里
    assert not A.is_task_given("警方于2022年9月14日逮捕嫌疑人", TASK_660)   # 夹带了任务以外的事实
    assert not A.is_task_given("声明日期2020年7月2日", None)


def test_back_reference() -> None:
    others = {"A1": "可确认两者存在局部相同画面，但动作的时间安排及文件音轨配置不同",
              "A2": "两者起始帧中，骑手俯身骑红色自行车的姿态、车身位置和背景构图视觉一致；A的2.40秒与B的0.24秒画面对应",
              "A3": "两组对应点在A中相隔0.40秒，在B中仅相隔0.04秒"}
    assert A.back_reference_support("已确认的局部对应", others)
    assert A.back_reference_support("现有证据能够确认局部画面对应及播放时刻差异", {"F2": "相同播放时刻的动作状态不同，局部画面对应"})
    assert not A.back_reference_support("现有证据能够确认人物被替换成另一名骑手", others)       # 回指的内容别处没有
    assert not A.back_reference_support("局部对应", others)                                    # 没有回指标记


def test_verdict_sentence() -> None:
    assert A.is_verdict_sentence("该声明错误")
    assert A.is_verdict_sentence("结论：该声明有证据支持，但应限定为某些设备可能出现短暂读数")
    assert not A.is_verdict_sentence("混淆了两根铁柱")


def _err(fid: str, span: str, severity: Severity = Severity.CRITICAL):
    return error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持这一部分", span=span, method="model",
                 severity=severity, extra={"finding": fid})


def test_apply_program_rules_off_and_on(monkeypatch) -> None:
    errs = [_err("A8", "声明日期2022年9月14日"), _err("A1", "该声明错误"), _err("A9", "即位宣告与加冕不能混为一谈")]
    statements = {"A1": "该声明错误", "A2": "配图显示……", "A3": "ITV 报道……", "A8": "加冕晚于声明日期2022年9月14日", "A9": "即位宣告与加冕不能混为一谈"}
    monkeypatch.setattr(A, "PROGRAM_RULES", {k: False for k in A.PROGRAM_RULES})
    assert A.apply_program_rules(errs, statements, TASK_660, answer=True) == errs         # 关闭：原样返回（0.3.6 起默认打开）
    monkeypatch.setattr(A, "PROGRAM_RULES", {k: True for k in A.PROGRAM_RULES})
    out = A.apply_program_rules(errs, statements, TASK_660, answer=True)
    rules = [(e.extra.get("program_rule"), e.severity) for e in out]
    assert rules[0] == ("task_given", Severity.MINOR)
    assert rules[1] == ("verdict_sentence", Severity.MINOR)
    assert rules[2] == (None, Severity.CRITICAL)                                          # 推理句不在规则范围内
    # 报告单元不用总结论规则
    out = A.apply_program_rules([_err("F1", "该声明错误")], {"F1": "x", "F2": "y", "F3": "z"}, None, answer=False)
    assert out[0].severity is Severity.CRITICAL
