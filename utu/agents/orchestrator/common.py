import json
from dataclasses import dataclass, field
from typing import Literal

from agents import TResponseInputItem

# from openai.types.responses import EasyInputMessageParam
from ..common import DataClassWithStreamEvents


# 任务类，代表计划中的一个子任务
@dataclass
class Task:
    # 执行此任务的代理名称
    agent_name: str
    # 任务的具体描述内容
    task: str
    # 任务执行后的输出结果，初始为 None
    result: str | None = None
    # 标记该任务是否为计划中的最后一个任务
    is_last_task: bool = False  # whether this task is the last task of the plan


# 计划类，包含输入任务、分析过程和分解后的子任务列表
@dataclass
class Plan:
    # 初始用户输入
    input: str = ""
    # 对任务的分析过程描述
    analysis: str = ""
    # 分解出的子任务对象列表
    tasks: list[Task] = field(default_factory=list)

    # 将任务列表格式化为 JSON 字符串
    def format_tasks(self) -> str:
        # 提取每个任务的代理名称和任务描述，构建列表
        tasks = [{"name": t.agent_name, "task": t.task} for t in self.tasks]
        # 返回格式化后的 JSON 字符串，不转义非 ASCII 字符
        return json.dumps(tasks, ensure_ascii=False)

    # 将整个计划格式化为带有分析和计划标签的 XML 风格字符串
    def format_plan(self) -> str:
        # 返回 XML 格式的分析和计划内容
        return f"<analysis>{self.analysis}</analysis>\n<plan>{self.format_tasks()}</plan>"


# 编排器流式事件类，用于在流式处理过程中传递事件信息
@dataclass
class OrchestratorStreamEvent:
    # 事件名称：规划开始、规划完成、任务开始、任务完成
    name: Literal[
        "plan.start",
        "plan.done",
        "task.start",
        "task.done",
    ]
    # 与事件关联的数据项，可以是 Plan 对象、Task 对象或 None
    item: Plan | Task | None = None
    # 事件类型标识，固定为 "orchestrator_stream_event"
    type: Literal["orchestrator_stream_event"] = "orchestrator_stream_event"


# 任务记录器类，继承流式事件支持，用于跟踪和记录整个编排任务的状态和历史
@dataclass
class Recorder(DataClassWithStreamEvents):
    # 当前主要任务信息
    # current main task
    # 当前用户的输入内容
    input: str = ""  # current user input
    # 针对输入的最终输出结果。命名与 @agents.RunResult 保持一致
    final_output: str = None  # final output to `input`. Naming consistent with @agents.RunResult
    # 与当前输入相对应的执行轨迹列表
    trajectories: list = field(default_factory=list)  # trajs corresponding to `input`
    # 运行的唯一追踪 ID
    trace_id: str = ""

    # 规划相关状态信息
    # planning
    # 计划中的子任务列表
    tasks: list[Task] = None
    # 当前正在执行的任务 ID 索引
    current_task_id: int = 0

    # 历史记录信息
    # history
    # 历史计划记录列表，用于在多轮对话中保持上下文
    history_plan: list[TResponseInputItem] = field(default_factory=list)
    # 历史对话消息列表
    history_messages: list[TResponseInputItem] = field(default_factory=list)

    # 获取计划任务的字符串表示形式（编号列表）
    def get_plan_str(self) -> str:
        # 将任务列表格式化为编号形式的字符串
        return "\n".join([f"{i}. {t.task}" for i, t in enumerate(self.tasks, 1)])

    # 获取执行轨迹的字符串表示形式
    def get_trajectory_str(self) -> str:
        # 构建执行轨迹的字符串列表
        traj = []
        for t in self.tasks:
            # 如果某个任务还没有结果，则停止记录后续轨迹
            if t.result is None:
                break
            # 将每个任务及其输出包装在 XML 风格的标签中
            traj.append(f"<task>{t.task}</task>\n<output>{t.result}</output>")
        # 返回所有轨迹项连接成的字符串
        return "\n".join(traj)

    # 向记录器中添加生成的计划，并更新计划历史
    def add_plan(self, plan: Plan) -> None:
        # 设置任务列表
        self.tasks = plan.tasks
        # 将用户问题和生成的计划助理回复添加到计划历史中
        self.history_plan.extend(
            [
                {"role": "user", "content": f"<question>{plan.input}</question>"},
                {"role": "assistant", "content": plan.format_plan()},
            ]
        )

    # 设置最终输出结果并更新消息历史
    def add_final_output(self, final_output: str) -> None:
        # 设置最终输出内容
        self.final_output = final_output
        # 将输入（占位符）和最终回复添加到消息历史中
        self.history_messages.extend(
            [
                {"role": "user", "content": "self.input"},
                {"role": "assistant", "content": final_output},
            ]
        )

    # 创建一个新的记录器实例并保留当前历史记录，用于支持多轮对话
    def new(self, input: str = None, trace_id: str = None) -> "Recorder":
        # 创建一个带有相同历史的新记录器 -- 用于多轮聊天
        """Create a new recorder with the same history -- for multi-turn chat."""
        # 初始化新的记录器对象
        new_recorder = Recorder(input=input, trace_id=trace_id)
        # 复制历史计划记录
        new_recorder.history_plan = self.history_plan.copy()
        # 复制历史对话消息记录
        new_recorder.history_messages = self.history_messages.copy()
        return new_recorder
