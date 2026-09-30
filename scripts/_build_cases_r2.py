"""案卷扩充第二批（2026-09-28）：22 个新案卷，连同原有 11 个共 33 个。

    python scripts/_build_cases_r2.py [--out experiments/sft_data/cases] [--check-only]

来源与规矩（接 案卷审核记录_20260924 的审核原则）：
- 题面、声明、日期、媒体、被排除的原始核查文章都取自数据集标注（data/<dataset>/<id>/reference.json）。
- requirements 由总控按"正确的调查必须建立什么"逐案手写，2–4 条，最后一条是限制，must_cite=False。
  标注里的问答只用来决定"要建立哪些事实"，不把答案写进要求。
- required_evidence：标注给出的来源（多为存档或社交平台，可能取不到）一律 must_cite=False；
  试点 run 实际打开过、且直接支撑某条要求的非原始来源，放进 any_one_of 组并 must_cite=True。
  视频类只要求派生证据（抽帧）。
- checks 保守：label_expected 只给 Refuted（"不实"），Supported / NEI 不写（"属实"是"不属实"的子串，会误判）；
  QVHighlights 写 interval_iou（min_iou 0.5）；VCSL 标注区间两侧语义不唯一、试点答案常只取其中一段，不写区间检查，只记在 notes。
- source_policy：声明类 exclude_original（排除标注里的原始核查文章，含存档地址；AFP 文章去掉 -1 后缀也排除），
  不按日期过滤（与试点一致，notes 注明声明日期）；视频类 provided_media。
- split 全部先写 dev_train，之后由 scripts/_assign_splits.py 按 family 一次性分配。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SFT = ROOT / "experiments" / "sft_data"

from foren_searcher.datagen.cases import Case  # noqa: E402

CLAIM_TASK_MEDIA = "核查以下声明，必要时检查给定媒体并搜索外部证据。依据实际证据回答，说明结论范围与不确定性："
CLAIM_TASK_TEXT = "核查以下声明，搜索外部证据。依据实际证据回答，说明结论范围与不确定性："
QV_TASK = "在给定视频中定位符合以下描述的片段，给出秒数范围及证据；说明定位的不确定性："
VCSL_TASK = "在给定视频 A 与 B 中查找相同或经变换的对应片段，给出两侧时间范围及证据。不要把内容对应推断成原创归属。输入仅包含视频轨，不评估音频。"
REFUTED = {"kind": "label_expected", "params": {"label": "不实", "aliases": ["虚假", "Refuted", "False"]},
           "note": "数据集标签 Refuted", "requirement": None}
LIMIT = "未能确定的内容与结论的适用范围（未取得的来源、无法核实的细节）"


def ev(key: str, desc: str, pattern: str | None, must: bool, supports: list[str], acquired_by: str | None = None) -> dict:
    return {"key": key, "description": desc, "url_pattern": pattern, "acquired_by": acquired_by,
            "must_cite": must, "supports": supports}


def req(*texts: str) -> list[dict]:
    out = [{"id": f"REQ{i}", "text": t, "must_cite": True} for i, t in enumerate(texts, 1)]
    out.append({"id": f"REQ{len(texts) + 1}", "text": LIMIT, "must_cite": False})
    return out


# ---------------------------------------------------------------- 声明类（AVeriTeC 7 + AVeriMaTeC 7）
CLAIMS: dict[str, dict[str, Any]] = {
    "averitec:1707": dict(
        requirements=req("2018 年联邦预算中 Yaba 联邦神经精神病院获得的拨款数额，以及该数字的统计口径（总额、资本或经常性支出）",
                         "联邦政府 2018 年实际向该院拨付（released）的金额；取不到时须明确说明无法核实"),
        evidence=[ev("budget_2018", "2018 年联邦预算文件中该院的预算条目", "africacheck.org/wp-content/uploads/2019/11/2018-budget-final*", False, ["REQ1"])],
        rule="all", label=None),
    "averitec:1766": dict(
        requirements=req("扎伊拉·瓦西姆 2019 年 6 月宣布退出影视圈的记载，以及声明发布时是否有她复出的可靠证据",
                         "声明所依据的照片的拍摄时间或出处，是否早于她宣布退圈"),
        evidence=[ev("quit_news18", "News18 关于其宣布退圈的报道", "news18.com/news/movies/dangal-actor-zaira-wasim-quits-bollywood*", True, ["REQ1"]),
                  ev("quit_deccan", "Deccan Chronicle 关于其宣布退圈的报道（试点 run 取到）", "deccanchronicle.com/entertainment/bollywood/300619/zaira-wasim-quits-bollywood*", True, ["REQ1"]),
                  ev("photo_origin_tweet", "照片早于退圈的说明推文（存档）", "*funu365/status/1170985803410837505*", False, ["REQ2"])],
        rule="any_one_of:quit_news18,quit_deccan", label=REFUTED),
    "averitec:254": dict(
        requirements=req("肯尼亚各郡在声明所指财年（2019/20）预算中分配给农业的平均比例，及数据的出处与统计口径",
                         "该比例与声明中 7.3% 的比较是否基于同一财年、同一口径"),
        evidence=[ev("ag_allocations", "2019/20 各郡农业拨款统计表", "africacheck.org/sites/default/files/agriculture-allocations-2019-20.pdf*", False, ["REQ1", "REQ2"])],
        rule="all", label=REFUTED),
    "averitec:2694": dict(
        requirements=req("声明所说的过去五年（约 2014–2019）内，南非是否新建并开办了两所大学，分别是哪两所、何时开始招生",
                         "\"建成\"的含义与时间范围是否和记载相符（开办时间、建设阶段）"),
        evidence=[ev("sanews_named", "南非政府新闻社关于两所新大学命名的报道（试点 run 取到）", "sanews.gov.za/south-africa/new-sa-universities-named*", True, ["REQ1"]),
                  ev("spu_site", "Sol Plaatje University 官网的沿革或建设记载（试点 run 取到）", "spu.ac.za/*", True, ["REQ1", "REQ2"]),
                  ev("treasury_review", "2018 年预算审查报告（标注来源）", "treasury.gov.za/documents/national%20budget/2018/review/fullbr.pdf*", False, ["REQ1"])],
        rule="any_one_of:sanews_named,spu_site", label=None),
    "averitec:2799": dict(
        requirements=req("消防部门或消防员组织对\"林业管理不善导致加州山火\"这一说法的公开回应",
                         "2018 年 11 月加州山火的类型、位置与起因的可靠说明，能否支持\"林业管理不善\"的归因"),
        evidence=[ev("pfa_tweet", "帕萨迪纳消防员协会主席的推文（标注来源）", "*pfa809/status/1061371157583032320*", False, ["REQ1", "REQ2"]),
                  ev("newsweek", "Newsweek 关于消防员反驳的报道（试点 run 取到）", "newsweek.com/trump-wildfire-tweet-was-reckless*", True, ["REQ1"]),
                  ev("nbcnews", "NBC 关于消防员反驳的报道（试点 run 取到）", "nbcnews.com/news/us-news/trump-draws-ire-firefighters*", True, ["REQ1"]),
                  ev("cpf", "加州专业消防员协会主席的回应（试点 run 取到）", "legacy.cpf.org/go/cpf/news-and-events/news/cpf-president-brian-rice*", True, ["REQ1"])],
        rule="any_one_of:newsweek,nbcnews,cpf", label=REFUTED),
    "averitec:2872": dict(
        requirements=req("是否存在 Mel Gibson 说过这段话的可靠记录（原始采访或节目出处）",
                         "该说法最早的出处，以及它所附的节目（2017 年 The Graham Norton Show）中他实际谈了什么"),
        evidence=[ev("origin_post", "最早传播该说法的文章（存档）", "*smilecelebs.com/2017/12/01/mel-gibson-hollywood-elites*", False, ["REQ2"])],
        rule="all", label=REFUTED),
    "averitec:392": dict(
        requirements=req("这则故事最早发布在哪个网站或帖子", "该网站的性质：是否声明其内容为讽刺或虚构"),
        evidence=[ev("origin_site", "最早发布该故事的网站", "foolishhumor.com*", True, ["REQ1", "REQ2"]),
                  ev("origin_archive", "原站存档", "archive.ph/vdhyf*", True, ["REQ1", "REQ2"]),
                  ev("timesnow", "Times Now 核查（试点 run 取到）", "timesnownews.com/the-buzz/article/budweiser-employee-peeing*", True, ["REQ1", "REQ2"]),
                  ev("businesstoday", "Business Today 核查（试点 run 取到）", "businesstoday.in/latest/trends/story/did-a-budweiser-employee-really-urinate*", True, ["REQ1", "REQ2"]),
                  ev("ibtimes", "IBTimes 核查（试点 run 取到）", "ibtimes.co.in/fact-check-no-budweiser-employee*", True, ["REQ1", "REQ2"])],
        rule="any_one_of:origin_site,origin_archive,timesnow,businesstoday,ibtimes", label=REFUTED),
    "averimatec:423": dict(
        requirements=req("权威机构（如 NOAA、IPCC）对全球海平面上升速率随时间变化的测量结论",
                         "给定图片的内容，以及它能否支持\"海平面上升速率没有增加\"的说法"),
        evidence=[ev("noaa", "NOAA/climate.gov 关于全球海平面上升的说明", "climate.gov/news-features/understanding-climate/climate-change-global-sea-level*", True, ["REQ1"]),
                  ev("ipcc", "IPCC 评估报告中的海平面章节（试点 run 取到 ipcc.ch 报告页）", "ipcc.ch/report/*", True, ["REQ1"]),
                  ev("pnas", "PNAS 关于过去三千年海平面变化的论文（标注来源）", "*pnas.org/doi/10.1073/pnas.1517056113*", True, ["REQ1"])],
        rule="any_one_of:noaa,ipcc,pnas", label=REFUTED),
    "averimatec:441": dict(
        requirements=req("图中坟墓的实际所在地", "坟墓加装铁栅和锁的真实原因"),
        evidence=[ev("altnews", "Alt News 核查（试点 run 取到）", "altnews.in/media-misreport-viral-photo-of-grave*", True, ["REQ1", "REQ2"]),
                  ev("etvbharat", "ETV Bharat 核查（试点 run 取到）", "etvbharat.com/english/state/telangana/fact-check-picture-of-grill-lock-on-grave*", True, ["REQ1", "REQ2"]),
                  ev("ndtv", "NDTV 报道（试点 run 取到）", "ndtv.com/india-news/viral-picture-of-padlock-on-grave*", True, ["REQ1", "REQ2"]),
                  ev("logicalindian", "The Logical Indian 核查（试点 run 取到）", "thelogicalindian.com/pakistan-put-locks-daughters-graves*", True, ["REQ1", "REQ2"]),
                  ev("local_tweet", "当地人说明坟墓位置的推文（标注来源）", "*iamsalmansallu/status/1652680592100847617*", False, ["REQ1"]),
                  ev("owner_tweet", "说明加锁原因的推文（标注来源）", "*warispathan/status/1652654774062313472*", False, ["REQ2"])],
        rule="any_one_of:altnews,etvbharat,ndtv,logicalindian", label=REFUTED),
    "averimatec:63": dict(
        requirements=req("卫生主管机构（如斯里兰卡国家登革热控制中心）推荐的有效驱蚊方法",
                         "丁香或插丁香的青柠（原物形态）是否有驱蚊效果的科学依据"),
        evidence=[ev("ndcu_prevent", "斯里兰卡国家登革热控制中心的预防说明页（试点 run 经存档取到）", "*dengue.health.gov.lk/web/index.php/en/information/how-to-prevent*", True, ["REQ1"]),
                  ev("clove_review", "关于丁香油驱虫效果的研究（标注来源）", "sciencedirect.com/science/article/abs/pii/s0882596311001898*", False, ["REQ2"]),
                  ev("mnt_clove", "Medical News Today 关于天然驱蚊剂的文章（标注来源）", "medicalnewstoday.com/articles/325337*", False, ["REQ2"])],
        rule="all", label=REFUTED),
    "averimatec:652": dict(
        requirements=req("图片的最早发布者与发布日期", "图中这只狗的真实经历：是否被警方作为流浪狗收容并送养"),
        evidence=[ev("southlake_fb", "Southlake 公共安全部门的原始帖子（标注来源）", "facebook.com/southlakepublicsafety/*", False, ["REQ1", "REQ2"])],
        rule="all", label=REFUTED),
    "averimatec:660": dict(
        requirements=req("图片实际显示的场景（地点、时间、事件）", "图中动物（军团吉祥物山羊）的身份，以及它是否被献祭"),
        evidence=[ev("yahoo_goat", "山羊领队进入加的夫城堡的报道（标注来源）", "*band-led-goat-marches-cardiff*", True, ["REQ1"]),
                  ev("blackheritage", "Black Heritage Radio 关于宣告仪式游行的报道（试点 run 取到）", "blackheritageradio.com/uk-goat-leads-welsh-military-parade*", True, ["REQ1", "REQ2"]),
                  ev("moneycontrol", "Moneycontrol 关于山羊军衔与薪水的报道（试点 run 取到）", "moneycontrol.com/news/trends/goat-seen-at-king-charles-iiis-proclamation*", True, ["REQ1", "REQ2"]),
                  ev("mascot_history", "皇家威尔士团博物馆关于吉祥物的介绍（标注来源）", "*royalwelshmuseum.wales/a-brief-history-of-the-regimental-mascot*", True, ["REQ2"])],
        rule="any_one_of:yahoo_goat,blackheritage,moneycontrol,mascot_history", label=REFUTED),
    "averimatec:699": dict(
        requirements=req("图中铁柱的实际身份与所在地", "顾特卜塔（Qutub Minar）建筑群内铁柱铭文的实际内容"),
        evidence=[ev("altnews", "Alt News 核查（试点 run 取到）", "altnews.in/photo-of-iron-pillar-in-bharatpur-fort*", True, ["REQ1"]),
                  ev("newsmobile", "NewsMobile 核查（试点 run 取到）", "newsmobile.in/nm-fact-checker/fact-check-image-of-loha-stambh*", True, ["REQ1"]),
                  ev("telugupost", "Telugu Post 核查（试点 run 取到）", "telugupost.com/english-factcheck/iron-pillar-in-viral-image*", True, ["REQ1"]),
                  ev("toi", "Times of India 核查（试点 run 取到）", "timesofindia.indiatimes.com/times-fact-check/news/fake-alert-photo-of-devanagari-inscriptions*", True, ["REQ1"]),
                  ev("thewire", "The Wire 关于顾特卜塔铁柱的文章（标注来源）", "*thewire.in/article/history/qutb-minar-complex-iron-pillar*", False, ["REQ2"])],
        rule="any_one_of:altnews,newsmobile,telugupost,toi", label=REFUTED),
    "averimatec:98": dict(
        requirements=req("各张图片的最早出处与原始说明（尤其武器图片的原始拍摄场合）",
                         "图中人物的身份，以及其与武器说法的关系",
                         "埃塞俄比亚正教会分裂的时间，图片首次发布是否早于这一事件"),
        evidence=[ev("haqcheck", "HaqCheck 核查（试点 run 取到）", "haqcheck.org/false-the-image-doesnt-show-weapons*", True, ["REQ1", "REQ2"]),
                  ev("afp_archive", "AFP 图片档案存档（标注来源）", "perma.cc/krm4-24qe*", True, ["REQ1"]),
                  ev("image2_origin", "第 2 张图片的最早发布（标注来源）", "perma.cc/8wt4-3tvq*", False, ["REQ1", "REQ3"]),
                  ev("person_origin", "人物身份的出处（标注来源）", "perma.cc/t32w-4bt5*", False, ["REQ2"]),
                  ev("church_split", "教会分裂时间的报道（标注来源）", "perma.cc/2ms6-9r4y*", False, ["REQ3"])],
        rule="any_one_of:haqcheck,afp_archive", label=REFUTED),
}

QV = {"0": None, "1": None, "2": None, "9167": None}
VCSL = ("pair_1", "pair_4", "pair_5", "pair_6")
# pair_1/5/6 的两侧都是同一首雪人动画歌曲的不同剪辑（试点帧里都是雪人、柠檬、冰淇淋场景），按同一事件分组，避免跨划分泄漏
VCSL_FAMILY = {"pair_1": "vcsl:snowman_song", "pair_5": "vcsl:snowman_song", "pair_6": "vcsl:snowman_song", "pair_4": "vcsl:pair_4"}


def norm_key(url: str) -> str:
    value = re.sub(r"^(https?://)?web\.archive\.org/web/[^/]+/", "", url.strip(), flags=re.I)
    value = re.sub(r"^https?://", "", value, flags=re.I)
    value = re.sub(r"^www\.", "", value, flags=re.I)
    value = value.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    host, _, path = value.partition("/")
    return host.lower() + ("/" + path if path else "")


def iso(date: str) -> str:
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return date
    d, m, y = date.split("-")
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def claim_case(case_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    dataset, sid = case_id.split(":")
    ref = json.loads((SFT / "data" / dataset / sid / "reference.json").read_text(encoding="utf-8"))
    claim = (ref.get("claim") or ref.get("claim_text")).strip()
    date = iso(ref.get("claim_date") or ref.get("date"))
    article = ref.get("fact_checking_article") or ref.get("article")
    keys = [norm_key(article)]
    if keys[0].startswith("factcheck.afp.com/") and re.search(r"-\d+$", keys[0]):
        keys.append(re.sub(r"-\d+$", "", keys[0]))
    inputs = sorted((SFT / "data" / dataset / sid / "inputs").glob("image_*.jpg")) if dataset == "averimatec" else []
    media = [{"ref": f"I{i}", "path": f"../data/{dataset}/{sid}/inputs/{p.name}", "media_type": "image/jpeg"} for i, p in enumerate(inputs)]
    task = (CLAIM_TASK_MEDIA if media else CLAIM_TASK_TEXT) + claim + f"（声明日期：{date}）"
    checks = [spec["label"]] if spec["label"] else []
    label_note = "" if spec["label"] else f"数据集标签 {ref['label']}，按保守规矩不写 label_expected（子串匹配区分不了\"属实/不属实\"等）。"
    return {
        "case_id": case_id, "dataset": dataset, "source_id": sid, "split": "dev_train", "task_kind": "claim_check",
        "task_text": task, "family_key": case_id, "media": media, "claims": [claim],
        "source_policy": {"mode": "exclude_original", "excluded_keys": keys, "date_cutoff": None},
        "requirements": spec["requirements"], "required_evidence": spec["evidence"],
        "required_evidence_rule": spec["rule"], "checks": checks,
        "reference_path": f"../data/{dataset}/{sid}/reference.json", "used_for_revision": [],
        "notes": ("2026-09-28 第二批：总控依据数据集标注手写 requirements（不含答案），证据项取标注来源（must_cite=False）"
                  "与试点 run 实际打开过的非原始来源（any_one_of，must_cite=True）。"
                  f"声明日期 {date}，不按日期过滤来源（与试点一致）；排除原始核查文章 {article}。{label_note}"),
    }


def qv_case(sid: str) -> dict[str, Any]:
    ref = json.loads((SFT / "data" / "qvhighlights" / sid / "reference.json").read_text(encoding="utf-8"))
    windows = ref["relevant_windows"]
    checks = [{"kind": "interval_iou", "params": {"start": a, "end": b, "min_iou": 0.5}, "note": "QVHighlights relevant_windows（片段内秒数）",
               "requirement": "REQ1"} for a, b in windows]
    return {
        "case_id": f"qvhighlights:{sid}", "dataset": "qvhighlights", "source_id": sid, "split": "dev_train",
        "task_kind": "temporal_locate", "task_text": QV_TASK + ref["query"], "family_key": f"qvhighlights:{ref['vid'].rsplit('_', 2)[0]}",
        "media": [{"ref": "I0", "path": f"../data/qvhighlights/{sid}/inputs/A.mp4", "media_type": "video/mp4"}],
        "claims": [], "source_policy": {"mode": "provided_media", "excluded_keys": [], "date_cutoff": None},
        "requirements": [
            {"id": "REQ1", "text": "符合描述的片段在视频中的起止时间（秒）", "must_cite": True},
            {"id": "REQ2", "text": "支持定位的画面依据：片段内以及边界两侧的帧所显示的内容", "must_cite": True},
            {"id": "REQ3", "text": "定位精度与边界的不确定范围（未检查的区间不当作不符合）", "must_cite": False}],
        "required_evidence": [ev("frames", "给定视频的抽帧（片段内与边界两侧）", None, True, ["REQ1", "REQ2"], "INSPECT/get_frames"),
                              ev("media_profile", "视频时长与帧率", None, False, ["REQ3"], "INSPECT/media_profile")],
        "required_evidence_rule": "all", "checks": checks,
        "reference_path": f"../data/qvhighlights/{sid}/reference.json", "used_for_revision": [],
        "notes": f"2026-09-28 第二批：标注片段 {windows}（视频 {ref['duration']} 秒）；family 按源 YouTube 视频号分组。",
    }


def vcsl_case(pair: str) -> dict[str, Any]:
    ref = json.loads((SFT / "data" / "vcsl" / pair / "reference.json").read_text(encoding="utf-8"))
    segs = "；".join(f"A {a_s}–{a_e} 秒 ↔ B {b_s}–{b_e} 秒" for a_s, b_s, a_e, b_e in ref["annotation"])
    return {
        "case_id": f"vcsl:{pair}", "dataset": "vcsl", "source_id": pair, "split": "dev_train", "task_kind": "copy_detect",
        "task_text": VCSL_TASK, "family_key": VCSL_FAMILY[pair],
        "media": [{"ref": "I0", "path": f"../data/vcsl/{pair}/inputs/A.mp4", "media_type": "video/mp4"},
                  {"ref": "I1", "path": f"../data/vcsl/{pair}/inputs/B.mp4", "media_type": "video/mp4"}],
        "claims": [], "source_policy": {"mode": "provided_media", "excluded_keys": [], "date_cutoff": None},
        "requirements": [
            {"id": "REQ1", "text": "A 与 B 中内容对应（相同或经变换）的片段，以及两侧各自的时间范围", "must_cite": True},
            {"id": "REQ2", "text": "对应关系的画面依据：两侧对应帧的内容，以及画面包装、裁剪、叠加等变换", "must_cite": True},
            {"id": "REQ3", "text": "未能确定的内容：边界精度、未检查的区间；不把内容对应推断为原创归属或传播先后", "must_cite": False}],
        # 对齐候选只是线索；核验器会把规则里的每一项都当必需组（must_cite 只影响 completed_without_finishing），所以不列
        "required_evidence": [ev("frames", "两段视频的对应帧", None, True, ["REQ1", "REQ2"], "INSPECT/get_frames")],
        "required_evidence_rule": "all", "checks": [],
        "reference_path": f"../data/vcsl/{pair}/reference.json", "used_for_revision": [],
        "notes": (f"2026-09-28 第二批：VCSL 标注对应段 {segs}（格式按 [A起, B起, A止, B止] 解读，与两段视频时长一致）。"
                  "标注较粗，试点答案常只给出其中一部分或找到标注外的重复位置，所以不写 interval_iou，由充分性层判。"),
    }


PATCHES = SFT / "cases_draft" / "r2_20260928" / "review_patches.json"


def _set_path(data: Any, path: str, value: Any) -> None:
    """按审稿修改单的字段路径改值："a.b[2].c"；[N] 等于当前长度表示追加，value 为 None 表示删除该项。"""

    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cur = data
    for tok in tokens[:-1]:
        cur = cur[int(tok[1:-1])] if tok.startswith("[") else cur[tok]
    last = tokens[-1]
    if last.startswith("["):
        i = int(last[1:-1])
        if value is None:
            del cur[i]
        elif i == len(cur):
            cur.append(value)
        else:
            cur[i] = value
    else:
        cur[last] = value


def build(apply_review: bool = True) -> list[dict[str, Any]]:
    cases = [claim_case(cid, spec) for cid, spec in CLAIMS.items()]
    cases += [qv_case(sid) for sid in QV]
    cases += [vcsl_case(p) for p in VCSL]
    if apply_review and PATCHES.exists():
        # 独立审稿（2026-09-28，review_20260928.md）的修改单：逐项套用，改动留痕在 notes
        patches = json.loads(PATCHES.read_text(encoding="utf-8"))
        for data in cases:
            item = patches.get(data["case_id"]) or {}
            for change in item.get("changes") or []:
                _set_path(data, change["field"], change["new"])
            if item.get("changes"):
                data["notes"] += f"（已按 2026-09-28 独立审稿修改 {len(item['changes'])} 处，见 cases_draft/r2_20260928/review_20260928.md）"
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(SFT / "cases"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    cases = build()
    for data in cases:
        Case.from_dict(data)       # 校验
        for m in data["media"]:
            path = (SFT / "cases" / m["path"]).resolve()
            if not path.exists():
                raise SystemExit(f"{data['case_id']}: 媒体不存在 {path}")
    print(f"{len(cases)} 个新案卷校验通过")
    if args.check_only:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for data in cases:
        target = out / (data["case_id"].replace(":", "_") + ".json")
        if target.exists():
            raise SystemExit(f"{target} 已存在，停止（不覆盖）")
    for data in cases:
        (out / (data["case_id"].replace(":", "_") + ".json")).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
