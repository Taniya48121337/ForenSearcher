"""datagen：自我发展循环的数据侧（DESIGN.md 第 8 节）。

子模块按循环环节命名：
- errors   错误类型清单（唯一定义处）
- cases    案例文件：任务、材料、来源规则、必需证据清单、程序化断言
- perturb  证据扰动器（抽积木）：在证据库副本上遮蔽 / 替换 / 注入，生成自带期望的对照状态
- specs    版本化规格：student_prompt / teacher_instruction / rubric / harness 参数分开存、带 hash、升号规则
- bootstrap 从运行时常量和试点指令文件生成 hi@0.1.0 / gia@0.1.0
- verify/  核验器：report 单元三层（exec_checks → anchored → sufficiency），step / plan / answer 单元（units.py，程序 + hindsight），
           calibrate.py 规则校准；verdict.py 汇总为 Verdict；judges.py 接核验模型；reverify.py 重核试点 runs
- runlog   读一个 run 目录（trajectory + execution_links + requests + store）为 Decision 列表，供核验单元与 export 共用
- export   按 v2 §3 入选规则出 hi/gia 的 accepted / held / rejected jsonl + manifest；行形态与学生推理路径一致
- check_inputs 用学生模型 processor 核对 label mask、图片展开与 token 数
后续：diagnose、revise、gate。
"""

from .errors import ErrorRecord, ErrorType, Level, Role, Severity, error, tally, types_for
from .specs import RubricItem, Spec, SpecError

__all__ = ["ErrorRecord", "ErrorType", "Level", "Role", "RubricItem", "Severity", "Spec", "SpecError", "error",
           "tally", "types_for"]
