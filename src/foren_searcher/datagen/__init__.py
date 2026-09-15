"""datagen：自我发展循环的数据侧（DESIGN.md 第 8 节）。

子模块按循环环节命名：
- errors   错误类型清单（唯一定义处）
- cases    案例文件：任务、材料、来源规则、必需证据清单、程序化断言
- perturb  证据扰动器（抽积木）：在证据库副本上遮蔽 / 替换 / 注入，生成自带期望的对照状态
后续：specs（版本化规格）、verify（核验器）、diagnose、revise、export、gate。
"""

from .errors import ErrorRecord, ErrorType, Level, Role, Severity, error, tally, types_for

__all__ = ["ErrorRecord", "ErrorType", "Level", "Role", "Severity", "error", "tally", "types_for"]
