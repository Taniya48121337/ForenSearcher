"""第二层的"看图复核"（2026-10-05）：图片对判成 supported 之后，再按要素逐条核一遍，并核对场景是否对得上。

大白话：替换测试把句子引用的帧换成别的案例的图，核验器仍有一部分判"支持"：
- 句子说"B 的前景黑衣人物"，图里是穿黄衣的公路骑车人，它按"前景有个人"就算支持（H042）；
- 句子说"下缘出现模糊的深色头顶状片段"，图里是戴帽子的徒步者，它也算支持（H043）；
- "A 一侧未见该人物"这类"没有某物"的说法，任何无关的图都"支持"。
单次判定是先看到句子再看图，容易被句子带着走，只要有一个宽泛特征对上就判支持。

这里对 supported 的图片对再调一次模型，换一种问法：
1. 先只看图说出实际场景（scene），再判断它和任务 / 句子说的是不是同一个场景、同一类对象（scene_match）；
2. 每条"支持的部分"只在图里有明显反证时才不确认：能看清的要素与说法明显不符（颜色、衣着、类别、数量、文字、方位），
   或场景明显不符。看不清、单帧看不出变化、涉及别的帧或另一份材料都不算反证（第一版要求"逐项看得到"，
   在 r0c 原图上把 106 个支持对里的 37 个削掉了部分，过严）。
没被确认的部分从 supported 里移到 rejected（留痕 extra.image_confirm），判定按剩下的重算；后续覆盖汇总会把它们当作没有支持。
只对图片对、且只对判成 supported 的对追加一次调用；其余对的判定和缓存都不变。

10-05 离线评估（scripts/_image_confirm_eval.py，r0c 替换测试 76 对 + 原图 132 对）：
- 第一版"逐项都要看得到"：替换后判支持 6 → 3，但原图 106 个支持对里 21 个被整对降级、37 个被削掉部分，
  多数是"单帧看不出 A/B 对比或变化"这类不该算反证的理由，过严；
- 现版"只认明显反证"：原图 0 个降级（1 处削掉部分），替换后判支持 6 → 5，只拦住 H043（E50）。
  另外 4 个是"未见某物"类说法，任何无关画面都满足；H042（E10）的"前景黑衣人物"对一张穿黑短裤的骑车人图，
  只凭句子和任务也说不上明显不符。要拦住替换图得有同一段素材的参照帧，而管线里的帧都由程序按出处生成，
  不会真的被换掉——替换测试测的盖章在真实数据里基本不会发生。
结论：默认关闭、不接入管线，留作实验。
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from .anchored import MAX_TASK_CHARS, Judgment, JsonJudge, _call_checked, _obj, _source_payload
from .context import EvidenceText

IMAGE_CONFIRM = {"enabled": os.environ.get("FOREN_IMAGE_CONFIRM", "0") in ("1", "on", "true")}   # 实验性，默认关闭，未接入管线

CONFIRM_SYSTEM_PROMPT = """你是看图复核员。另一位核对员认为下面这张图支持待核句里的若干部分（claims）。你的任务是找出**图里有明显反证**的 claim，不是重新要求更多证据。
给你：图片、来源标注（属于哪份材料、哪一时刻）、调查任务（只用来理解场景和指代）、待核句全文、claims。
1. 先只看图，用一句话写出图里实际是什么场景、主要人物和物体（scene），不要受待核句影响。
2. scene_match：这张图与任务、待核句说的是否可能是同一个场景、同一类对象。只有**明显**不是时才填 false
   （例如句子讲滑板场里的黑衣人，图里是公路上的骑车人；句子讲车辆，图里是湖面；句子讲室内讲台，图里是夜景）。拿不准填 true。
3. 对每条 claim，只在下列情况填 confirmed=false：
   - 图里**能看清**该要素，且与说法明显不符（颜色、衣着、对象类别、数量、可读文字、位置左右上下明显相反）；
   - scene_match=false。
   以下**不是**否定理由，一律 confirmed=true：看不清或太小无法确认；单帧看不出变化、先后、快慢；涉及别的帧、别的时刻或另一份材料；
   需要和其他画面对比；关于取帧时刻、第几帧、视频总长的部分（按来源标注判断）。
4. 只依据图片和来源标注，不使用外部知识。
输出 JSON（字符串里的双引号用“”）：
{"scene":"一句话","scene_match":true|false,"claims":[{"part":"原样照抄 claim","confirmed":true|false,"why":"一句话"}]}"""

CONFIRM_SCHEMA = _obj({"scene": {"type": "string"}, "scene_match": {"type": "boolean"},
                       "claims": {"type": "array", "items": _obj({"part": {"type": "string"}, "confirmed": {"type": "boolean"},
                                                                  "why": {"type": "string"}})}})


async def confirm_image_judgment(judge: JsonJudge, judgment: Judgment, text: EvidenceText, statement: str, *,
                                 task: str | None = None) -> Judgment:
    """supported 的图片对：按要素与场景复核；返回（可能被降级的）Judgment。调用失败时原样返回。"""

    if not (IMAGE_CONFIRM["enabled"] and judgment.is_image and judgment.verdict == "supported" and judgment.supported
            and judgment.image_path is not None):
        return judgment
    payload: dict[str, Any] = {"source": _source_payload(text), "statement": statement,
                               "claims": [p.part for p in judgment.supported]}
    if task and task.strip():
        payload["task"] = task.strip()[:MAX_TASK_CHARS]
    raw, _failure = await _call_checked(judge, CONFIRM_SYSTEM_PROMPT, payload, [judgment.image_path], CONFIRM_SCHEMA)
    if not isinstance(raw, dict):
        return judgment
    scene_ok = bool(raw.get("scene_match", True))
    verdicts = {str(c.get("part")): c for c in raw.get("claims") or [] if isinstance(c, dict)}
    keep, drop = [], []
    for p in judgment.supported:
        c = verdicts.get(p.part)
        ok = scene_ok and (c is None or bool(c.get("confirmed")))     # 模型漏写某条时按场景结论处理
        (keep if ok else drop).append(p)
    info = {"scene": str(raw.get("scene") or "")[:200], "scene_match": scene_ok,
            "dropped": [{"part": p.part, "why": str((verdicts.get(p.part) or {}).get("why") or ("场景不符" if not scene_ok else ""))[:160]}
                        for p in drop]}
    if not drop:
        return replace(judgment, raw={**judgment.raw, "image_confirm": info})
    verdict = "contradicted" if judgment.contradicted else "supported" if keep else "not_found"
    note = f"（看图复核未确认 {len(drop)} 处{'；场景不符' if not scene_ok else ''}）" + judgment.notes
    return replace(judgment, verdict=verdict, supported=keep, rejected=list(judgment.rejected) + drop, notes=note,
                   raw={**judgment.raw, "image_confirm": info})


__all__ = ["CONFIRM_SCHEMA", "CONFIRM_SYSTEM_PROMPT", "IMAGE_CONFIRM", "confirm_image_judgment"]
