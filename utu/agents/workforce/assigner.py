import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..llm_agent import LLMAgent
from .data import Subtask, WorkspaceTaskRecorder

logger = get_logger(__name__)

PROMPTS: dict[str, str] = FileUtils.load_prompts("agents/workforce/assigner.yaml")


# 任务分配者代理类，负责将子任务分配给最合适的工作者
class AssignerAgent:
    # 任务分配器，处理任务分配
    """Task assigner that handles task assignment.

    使用示例::
    Usage::

        assigner_agent = AssignerAgent(assigner_agent)
        assigner_response = assigner_agent.assign_task(...)
    """

    # 初始化任务分配者代理
    def __init__(self, config: AgentConfig):
        # 保存代理配置
        self.config = config
        # 初始化 LLM 代理，用于生成任务分配决策
        self.llm = LLMAgent(model_config=config.workforce_planner_model)

    # 将任务分配给具有最佳能力的工作者节点
    """
    构建 prompts 调用 llm
    1. assigner.yaml 的 TASK_ASSIGN_SYS_PROMPT 作为 system prompt
    2. assigner.yaml 的 TASK_ASSIGN_USER_PROMPT 作为 user prompt
    """
    async def assign_task(self, recorder: WorkspaceTaskRecorder) -> Subtask:
        # 将任务分配给具有最佳能力的工作者节点
        """Assigns a task to a worker node with the best capability."""
        # 从记录器获取下一个待分配的任务
        next_task = recorder.get_next_task()

        # 格式化任务分配的系统提示词，包含总体任务、任务计划和执行者信息
        sp = PROMPTS["TASK_ASSIGN_SYS_PROMPT"].format(
            overall_task=recorder.overall_task,
            task_plan="\n".join(recorder.formatted_task_plan_list_with_task_results),
            executor_agents_info=recorder.executor_agents_info,
        )
        # 格式化任务分配的用户提示词，包含下一个任务和执行者名称列表
        up = PROMPTS["TASK_ASSIGN_USER_PROMPT"].format(
            next_task=next_task.task_name,
            executor_agents_names=recorder.executor_agents_names,
        )
        # 设置 LLM 代理的指令（系统提示词）
        self.llm.set_instructions(sp)
        # 运行 LLM 代理获取分配决策
        assign_recorder = await self.llm.run(up)
        # 将分配者的执行结果添加到记录器的轨迹中
        recorder.add_run_result(assign_recorder.get_run_result(), "assigner")  # add assigner trajectory

        # 解析分配结果
        # parse assign result
        # 调用解析方法提取选定的代理和详细任务描述
        assign_result = self._parse_assign_result(assign_recorder.final_output)
        # 更新任务对象的详细描述和选定的执行代理
        next_task.task_description = assign_result["assign_task"]
        next_task.assigned_agent = assign_result["assign_agent"]
        # 返回更新后的任务对象
        return next_task

    # 解析 LLM 返回的任务分配结果文本
    def _parse_assign_result(self, response):
        try:
            # 使用正则表达式提取 <selected_agent> 标签中的代理名称
            agent_match = re.search(r"<selected_agent>(.*?)</selected_agent>", response, re.DOTALL)
            selected_agent = agent_match.group(1).strip()
            # 使用正则表达式提取 <detailed_task_description> 标签中的详细任务描述
            task_match = re.search(r"<detailed_task_description>(.*?)</detailed_task_description>", response, re.DOTALL)
            detailed_task = task_match.group(1).strip()
            # 返回包含分配结果的字典
            return {"assign_agent": selected_agent, "assign_task": detailed_task}
        except Exception as e:
            # 记录解析失败的错误信息和原始响应内容
            logger.error(f"Failed to parse assignment result: {e}")
            logger.error(f"Response content: {response}")
            # 抛出解析异常
            raise ValueError(f"Failed to parse assignment result: {e}") from e
