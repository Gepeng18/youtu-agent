"""
- [ ] standardize parser?
- [ ] only LLM config is needed!
"""

import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..llm_agent import LLMAgent
from .data import Subtask, WorkspaceTaskRecorder

logger = get_logger(__name__)
PROMPTS = FileUtils.load_prompts("agents/workforce/planner.yaml")


# 规划者代理类，负责任务分解和计划管理
class PlannerAgent:
    # 任务规划器，处理任务分解
    """Task planner that handles task decomposition."""

    # 初始化规划者代理
    def __init__(self, config: AgentConfig):
        # 保存代理配置
        self.config = config
        # 初始化 LLM 代理，用于生成和更新计划
        self.llm = LLMAgent(model_config=config.workforce_planner_model)

    # 根据总体任务和可用代理规划任务
    """
    构建 prompts 调用 llm
    1. planner.yaml 的 TASK_PLAN_PROMPT 作为 user prompt
    """
    async def plan_task(self, recorder: WorkspaceTaskRecorder) -> None:
        # 基于总体任务和可用代理规划任务
        """Plan tasks based on the overall task and available agents."""
        # TODO: 使用 `failure_info` 进行重新规划
        # TODO: replan with `failure_info`
        # 格式化任务规划提示词
        plan_prompt = PROMPTS["TASK_PLAN_PROMPT"].format(
            overall_task=recorder.overall_task,
            executor_agents_info=recorder.executor_agents_info,
        )
        # 运行 LLM 代理生成初始计划
        plan_recorder = await self.llm.run(plan_prompt)
        # 将规划者的执行结果添加到记录器的轨迹中
        recorder.add_run_result(plan_recorder.get_run_result(), "planner")  # add planner trajectory

        # 解析任务
        # parse tasks
        # 使用正则表达式提取 <task> 标签中的内容
        pattern = "<task>(.*?)</task>"
        tasks_content: list[str] = re.findall(pattern, plan_recorder.final_output, re.DOTALL)
        # 去除任务描述中的空白字符并过滤空字符串
        tasks_content = [task.strip() for task in tasks_content if task.strip()]
        # 将解析出的任务描述转换为 Subtask 对象列表
        tasks = [Subtask(task_id=i + 1, task_name=task) for i, task in enumerate(tasks_content)]
        # 在记录器中初始化任务计划
        recorder.plan_init(tasks)

    # 根据已完成的任务更新任务计划
    """
    构建 prompts 调用 llm
    1. planner.yaml 的 TASK_UPDATE_PLAN_PROMPT 作为 user prompt
    """
    async def plan_update(self, recorder: WorkspaceTaskRecorder, task: Subtask) -> str:
        # 根据已完成的任务更新任务计划
        """Update the task plan based on completed tasks."""
        # 获取包含任务结果的格式化任务计划列表
        task_plan_list = recorder.formatted_task_plan_list_with_task_results
        # 获取当前任务的 ID
        last_task_id = task.task_id
        # 构建已执行任务的计划字符串表示
        previous_task_plan = "\n".join(f"{task}" for task in task_plan_list[: last_task_id + 1])
        # 构建未完成任务的计划字符串表示
        unfinished_task_plan = "\n".join(f"{task}" for task in task_plan_list[last_task_id + 1 :])

        # 格式化任务更新计划的提示词
        task_update_plan_prompt = (
            PROMPTS["TASK_UPDATE_PLAN_PROMPT"]
            .strip()
            .format(
                overall_task=recorder.overall_task,
                previous_task_plan=previous_task_plan,
                unfinished_task_plan=unfinished_task_plan,
            )
        )
        # 运行 LLM 代理生成更新后的计划建议
        plan_update_recorder = await self.llm.run(task_update_plan_prompt)
        # 将规划更新的执行结果添加到记录器的轨迹中
        recorder.add_run_result(plan_update_recorder.get_run_result(), "planner")  # add planner trajectory
        # 解析更新响应，提取选择（继续、更新、停止）和可能的更新任务列表
        choice, updated_plan = self._parse_update_response(plan_update_recorder.final_output)
        # 选项：continue（继续）、update（更新）、stop（停止）
        # choice: continue, update, stop
        if choice == "update":
            # 如果选择更新，则在记录器中更新任务计划
            recorder.plan_update(task, updated_plan)
        # 返回所做的选择
        return choice

    # 解析 LLM 的更新响应文本
    def _parse_update_response(self, response: str) -> tuple[str, list[str] | None]:
        # TODO: 将 "stop" 分解为 "early_completion"（提前完成）和 "task_collapse"（任务崩溃）
        # TODO: split "stop" into "early_completion" and "task_collapse"
        # 解析选择
        # Parse choice
        # 提取 <choice> 标签中的选择项
        pattern_choice = r"<choice>(.*?)</choice>"
        match_choice = re.search(pattern_choice, response, re.DOTALL)
        if match_choice:
            # 获取选择并转为小写且去除空白
            choice = match_choice.group(1).strip().lower()
            # 验证选择是否在预定义的合法选项中
            if choice not in ["continue", "update", "stop"]:
                # 如果选择无效，记录警告并默认为 "continue"
                logger.warning(f"Unexpected choice value: {choice}. Defaulting to 'continue'.")
                choice = "continue"
        else:
            # 如果未找到选择标签，记录警告并默认为 "continue"
            logger.warning("No choice found in response. Defaulting to 'continue'.")
            choice = "continue"

        # 如果选择是 "update"，则解析更新后的计划内容
        # Parse updated plan if choice is "update"
        updated_tasks = None
        if choice == "update":
            # 提取 <updated_unfinished_task_plan> 标签中的内容
            pattern_updated_plan = r"<updated_unfinished_task_plan>(.*?)</updated_unfinished_task_plan>"
            match_updated_plan = re.search(pattern_updated_plan, response, re.DOTALL)
            if match_updated_plan:
                # 尝试两种任务格式：<task> 和 <task_id:X>
                # Try both task formats: <task> and <task_id:X>
                updated_plan_content = match_updated_plan.group(1).strip()
                # 模式 1：标准任务标签 <task>
                task_pattern = r"<task>(.*?)</task>"
                task_matches = re.findall(task_pattern, updated_plan_content, re.DOTALL)
                # 如果未找到标准标签，尝试模式 2：任务 ID 格式 <task_id:X>
                # If no standard task tags found, try task_id format
                if not task_matches:
                    task_id_pattern = r"<task_id:\d+>(.*?)</task_id:\d+>"
                    task_matches = re.findall(task_id_pattern, updated_plan_content, re.DOTALL)

                # 构建解析出的更新任务描述列表
                updated_tasks = [task.strip() for task in task_matches if task.strip()]
                # 如果最终未解析出任何任务，记录警告
                if not updated_tasks:
                    logger.warning("No tasks found in updated plan. Defaulting to None.")
                    updated_tasks = None
            else:
                # 如果未找到更新计划标签，记录警告
                logger.warning("No updated plan found in response. Defaulting to None.")
                updated_tasks = None

        # 返回解析出的选择和更新任务列表
        return choice, updated_tasks

    # 检查任务计划的执行状态
    """
    构建 prompts 调用 llm
    1. planner.yaml 的 TASK_CHECK_PROMPT 作为 user prompt
    """
    async def plan_check(self, recorder: WorkspaceTaskRecorder, task: Subtask) -> None:
        # 格式化任务检查提示词，包含总体任务、当前计划及最后一个任务的执行详情
        task_check_prompt = (
            PROMPTS["TASK_CHECK_PROMPT"]
            .strip()
            .format(
                overall_task=recorder.overall_task,
                task_plan=recorder.formatted_task_plan,
                last_completed_task=task.task_name,
                last_completed_task_id=task.task_id,
                last_completed_task_description=task.task_description,
                last_completed_task_result=task.task_result,
            )
        )
        # 运行 LLM 代理执行状态检查
        res = await self.llm.run(task_check_prompt)
        # 将任务检查的执行结果添加到记录器的轨迹中
        recorder.add_run_result(res.get_run_result(), "planner")  # add planner trajectory
        # 解析响应文本并更新任务状态
        # parse and update task status
        task_check_result = self._parse_check_response(res.final_output)
        task.task_status = task_check_result

    # 解析状态检查响应文本
    def _parse_check_response(self, response: str) -> str:
        # 提取 <task_status> 标签中的状态信息
        pattern = r"<task_status>(.*?)</task_status>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            # 获取状态信息并转为小写且去除空白
            task_status = match.group(1).strip().lower()
            # 特殊处理包含 "partial" 的情况（如 "partial_success"）
            if "partial" in task_status:  # in case that models output "partial_success"
                return "partial success"
            # 验证状态是否在预定义的合法项中
            if task_status in ["success", "failed", "partial success"]:
                return task_status
            else:
                # 如果状态无效，记录警告并默认为 "partial success"
                logger.warning(f"Unexpected task status value: {task_status}. Defaulting to 'partial success'.")
                return "partial success"
        else:
            # 如果未找到状态标签，记录警告并默认为 "partial success"
            logger.warning("No task status found in response. Defaulting to 'partial success'.")
            return "partial success"
