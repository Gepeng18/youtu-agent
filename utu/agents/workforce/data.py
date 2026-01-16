from dataclasses import dataclass, field
from typing import Literal

from ..common import TaskRecorder


# 子任务类，代表工作力工作流中的一个具体执行步骤
@dataclass
class Subtask:
    # 任务的唯一标识 ID
    task_id: int
    # 任务的简短名称
    task_name: str
    # 任务的详细描述内容，初始为 None
    task_description: str = None
    # 任务的当前状态，默认为 "not started"（未开始）
    task_status: Literal["not started", "in progress", "completed", "success", "failed", "partial success"] = (
        "not started"
    )
    # 任务执行后的输出结果
    task_result: str = None
    # 任务执行后的详细输出结果
    task_result_detailed: str = None
    # 负责执行该任务的代理名称
    assigned_agent: str = None

    # 获取包含执行结果的格式化任务信息
    @property
    def formatted_with_result(self) -> str:
        # 构建基础信息列表，包含任务 ID 标签和状态标签
        infos = [
            f"<task_id:{self.task_id}>{self.task_name}</task_id:{self.task_id}>",
            f"<task_status>{self.task_status}</task_status>",
        ]
        # 如果任务已有结果，则添加结果标签
        if self.task_result is not None:
            infos.append(f"<task_result>{self.task_result}</task_result>")
        # 返回连接后的字符串
        return "\n".join(infos)


# 工作空间任务记录器类，继承基础任务记录器，专门用于工作力代理的复杂工作流
@dataclass
class WorkspaceTaskRecorder(TaskRecorder):
    # 总体任务描述
    overall_task: str = ""
    # 执行者代理的参数配置列表
    executor_agent_kwargs_list: list[dict] = field(default_factory=list)
    # 当前的任务计划列表
    task_plan: list[Subtask] = field(default_factory=list)

    # 获取所有执行者代理的信息摘要
    @property
    def executor_agents_info(self) -> str:
        # 获取执行者代理信息
        """Get the executor agents info."""
        # TODO: 添加工具信息
        # TODO: add tool infos
        # 遍历执行者配置列表，生成名称和描述的字符串
        return "\n".join(
            f"- {agent_kwargs['name']}: {agent_kwargs['description']}"  # TODO: add tool infos
            for agent_kwargs in self.executor_agent_kwargs_list
        )

    # 获取所有执行者代理的名称列表字符串表示
    @property
    def executor_agents_names(self) -> str:
        # 提取所有执行者代理的名称并转换为字符串格式
        return str([agent_kwargs["name"] for agent_kwargs in self.executor_agent_kwargs_list])

    # -----------------------------------------------------------
    # 获取包含任务执行结果的格式化任务计划列表
    @property
    def formatted_task_plan_list_with_task_results(self) -> list[str]:
        # 为显示格式化任务计划
        """Format the task plan for display."""
        # 调用每个子任务的格式化方法
        return [task.formatted_with_result for task in self.task_plan]

    # 获取用于显示的整体任务计划字符串
    @property
    def formatted_task_plan(self) -> str:
        # 为显示格式化任务计划
        """Format the task plan for display."""
        formatted_plan_list = []
        # 遍历任务计划，构建编号、名称和状态的字符串
        for task in self.task_plan:
            formatted_plan_list.append(f"{task.task_id}. {task.task_name} - Status: {task.task_status}")
        # 返回连接后的字符串
        return "\n".join(formatted_plan_list)

    # -----------------------------------------------------------
    # 初始化任务计划列表
    def plan_init(self, plan_list: list[Subtask]) -> None:
        # 将传入的任务列表设置为当前计划
        self.task_plan = plan_list

    # 更新任务计划，通常用于重新规划
    def plan_update(self, task: Subtask, updated_plan: list[str]) -> None:
        # 获取已完成的任务（包括当前刚完成的任务）
        finished_tasks = self.task_plan[: task.task_id]
        # 创建新的待完成任务列表，ID 从当前任务 ID 之后开始递增
        new_tasks = [Subtask(task_id=task.task_id + i, task_name=t) for i, t in enumerate(updated_plan)]
        # 合并已完成任务和新生成的任务列表
        self.task_plan = finished_tasks + new_tasks

    # -----------------------------------------------------------
    # 检查是否还有未完成的任务
    @property
    def has_uncompleted_tasks(self) -> bool:
        # 如果计划为空，则返回 False
        if self.task_plan is None:
            return False
        # 遍历任务计划，查找状态为 "not started"（未开始）的任务
        for task in self.task_plan:
            if task.task_status == "not started":
                return True
        return False

    # 获取下一个待执行的任务
    def get_next_task(self) -> Subtask:
        # 确保任务计划存在
        assert self.task_plan is not None, "No task plan available."
        # 遍历任务计划，返回第一个状态为 "not started" 的任务
        for task in self.task_plan:
            if task.task_status == "not started":
                return task
        # 如果没有未完成的任务，则返回提示信息
        return "No uncompleted tasks."
