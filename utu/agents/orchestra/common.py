from dataclasses import dataclass, field
from typing import Literal

from agents import RunResultStreaming

from ..common import DataClassWithStreamEvents, TaskRecorder


# 子代理信息类，用于规划者了解可用的代理
# 子代理信息类，用于规划者了解可用的代理
@dataclass
class AgentInfo:
    """Subagent information (for planner)"""

    # 代理名称
    name: str
    # 代理描述
    desc: str
    # 代理优势
    strengths: str
    # 代理劣势
    weaknesses: str


# 子任务类，代表由特定代理执行的任务
# 子任务类，代表由特定代理执行的任务
@dataclass
class Subtask:
    # 执行此任务的代理名称
    agent_name: str
    # 任务描述
    task: str
    # 任务是否已完成
    completed: bool | None = None


# 创建计划结果类，包含分析结果和待办任务列表
# 创建计划结果类，包含分析结果和待办任务列表
@dataclass
class CreatePlanResult(DataClassWithStreamEvents):
    # 任务分析结果
    analysis: str = ""
    # 待办子任务列表
    todo: list[Subtask] = field(default_factory=list)

    # 获取轨迹信息的属性
    # 获取轨迹信息的属性
    @property
    def trajectory(self):
        # 构建待办任务的字符串列表
        todos_str = []
        for i, subtask in enumerate(self.todo, 1):
            todos_str.append(f"{i}. {subtask.task} ({subtask.agent_name})")
        todos_str = "\n".join(todos_str)
        # 返回轨迹信息字典
        return {
            "agent": "planner",
            "trajectory": [
                {"role": "assistant", "content": self.analysis},
                {"role": "assistant", "content": todos_str},
            ],
        }


# 工作者结果类，包含任务执行结果和轨迹信息
# 工作者结果类，包含任务执行结果和轨迹信息
@dataclass
class WorkerResult(DataClassWithStreamEvents):
    # 执行的任务描述
    task: str = ""
    # 任务执行输出
    output: str = ""
    # 轨迹信息字典
    trajectory: dict = field(default_factory=dict)

    # 流式结果对象
    stream: RunResultStreaming | None = None


# 分析结果类，包含最终的分析输出
# 分析结果类，包含最终的分析输出
@dataclass
class AnalysisResult(DataClassWithStreamEvents):
    # 分析输出结果
    output: str = ""

    # 获取轨迹信息的属性
    @property
    def trajectory(self):
        return {"agent": "analysis", "trajectory": [{"role": "assistant", "content": self.output}]}


# 管弦乐队任务记录器，扩展基础任务记录器以支持管弦乐队工作流
# 管弦乐队任务记录器，扩展基础任务记录器以支持管弦乐队工作流
@dataclass
class OrchestraTaskRecorder(TaskRecorder):
    # 规划结果
    plan: CreatePlanResult = field(default=None)
    # 任务记录列表
    task_records: list[WorkerResult] = field(default_factory=list)

    # 设置规划结果
    # 设置规划结果
    def set_plan(self, plan: CreatePlanResult):
        self.plan = plan
        # 将规划结果添加到轨迹中
        self.trajectories.append(plan.trajectory)

    # 添加工作者结果
    # 添加工作者结果
    def add_worker_result(self, result: WorkerResult):
        # 添加到任务记录列表
        self.task_records.append(result)
        # 添加轨迹信息
        self.trajectories.append(result.trajectory)

    # 添加报告者结果
    # 添加报告者结果
    def add_reporter_result(self, result: AnalysisResult):
        # 添加轨迹信息
        self.trajectories.append(result.trajectory)

    # 获取规划的字符串表示
    # 获取规划的字符串表示
    def get_plan_str(self) -> str:
        # 将所有待办任务格式化为编号列表
        return "\n".join([f"{i}. {t.task}" for i, t in enumerate(self.plan.todo, 1)])

    # 获取轨迹的字符串表示
    # 获取轨迹的字符串表示
    def get_trajectory_str(self) -> str:
        # 将每个任务和对应的输出格式化为XML风格的字符串
        return "\n".join(
            [
                f"<subtask>{t.task}</subtask>\n<output>{r.output}</output>"
                for i, (r, t) in enumerate(zip(self.task_records, self.plan.todo, strict=False), 1)
            ]
        )


# 管弦乐队流式事件类，用于流式输出管弦乐队代理的事件
# 管弦乐队流式事件类，用于流式输出管弦乐队代理的事件
@dataclass
class OrchestraStreamEvent:
    # 事件名称
    name: Literal["plan_start", "plan", "worker", "report_start", "report"]
    # 事件项，可以是规划结果、工作者结果或分析结果
    item: CreatePlanResult | WorkerResult | AnalysisResult | None = None
    # 事件类型，固定为管弦乐队流式事件
    type: Literal["orchestra_stream_event"] = "orchestra_stream_event"
