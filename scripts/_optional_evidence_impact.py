"""verifier@0.3.1 影响估计（本机跑，只读）：对全部本机 run 的工作区报告按范围重跑第一层 exec_checks，
统计有多少份报告出现"可选证据组未引用"（0.3.1 只记 minor），以及按 0.3.0 口径（每组记 MAJOR）第一层会多判 negative 的份数。

    python scripts/_optional_evidence_impact.py

不调模型；第三层（充分性）的影响要等重跑核验后对照。
"""
import sys, collections; sys.path.insert(0,'src')
from pathlib import Path
from foren_searcher.datagen.cases import load_cases
from foren_searcher.datagen.runlog import load_run
from foren_searcher.datagen.verify import exec_checks
from foren_searcher.datagen.verify.scope import requirements_for_workspace
from foren_searcher.datagen.verify.verdict import read_verdicts
from foren_searcher.datagen.errors import Severity, ErrorType
cases = load_cases('experiments/sft_data/cases')
units = read_verdicts('experiments/sft_data/verdicts_units_judged.jsonl')
stats = collections.Counter(); per = collections.Counter()
for run in sorted(Path('experiments/sft_data/runs').iterdir()):
    if not (run/'run_config.json').is_file(): continue
    try: log = load_run(run)
    except Exception as e: continue
    try:
        case = cases.get(log.case_id)
        if case is None: continue
        for rec in log.store.workspace_report_records():
            try: text = str(log.store.workspace_task_record(rec.task_id).get("task") or "")
            except KeyError: text = ""
            req,_ = requirements_for_workspace(case, text, run=log.name, workspace_id=rec.task_id, units=units)
            errs = exec_checks(rec.to_dict(), log.lookup, case=case, requirement_ids=req)
            opt = [e for e in errs if e.extra.get('optional')]
            stats['reports']+=1
            if opt:
                stats['reports_with_optional_gap']+=1; stats['optional_notes']+=len(opt)
                firm_major = sum(1 for e in errs if e.extra.get('confidence')!='flag' and e.severity is Severity.MAJOR)
                # 旧语义：opt 各记 MAJOR（+ 不触发 completed_without）
                if firm_major < 2 <= firm_major + len(opt): stats['would_have_been_negative_by_layer1']+=1; per[log.case_id]+=1
    finally:
        log.close()
print(dict(stats)); print(dict(per))
