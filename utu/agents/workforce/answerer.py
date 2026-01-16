import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..llm_agent import LLMAgent
from .data import WorkspaceTaskRecorder

logger = get_logger(__name__)
PROMPTS: dict[str, str] = FileUtils.load_prompts("agents/workforce/answerer.yaml")


# 回答者代理类，负责根据任务执行结果生成最终答案
class AnswererAgent:
    # 答案提取器，负责从任务执行结果中生成最终答案
    """Answer extractor that handles final answer generation from task execution results."""

    # 初始化回答者代理
    def __init__(self, config: AgentConfig):
        # 保存代理配置
        self.config = config
        # 初始化 LLM 代理，用于生成最终答案
        self.llm = LLMAgent(model_config=config.workforce_answerer_model)

    # 从格式化的任务执行结果中提取最终答案
    """
    构建 prompts 调用 llm
    1. answerer.yaml 的 FINAL_ANSWER_PROMPT 作为 user prompt
    """
    async def extract_final_answer(self, recorder: WorkspaceTaskRecorder) -> str:
        # 从格式化的任务执行结果中提取最终答案
        """Extract the final answer from formatted task execution results."""
        # 生成最终答案提示词
        # Generate final answer prompt
        final_prompt = (
            PROMPTS["FINAL_ANSWER_PROMPT"]
            .strip()
            .format(
                question=recorder.overall_task,
                task_results="\n".join(recorder.formatted_task_plan_list_with_task_results),
            )
        )
        # 运行 LLM 代理获取最终回答
        final_recorder = await self.llm.run(final_prompt)
        # 将回答者的执行结果添加到记录器的轨迹中
        recorder.add_run_result(final_recorder.get_run_result(), "answerer")  # add answerer trajectory
        # 解析响应文本提取最终答案
        final_answer = self._parse_final_response(final_recorder.final_output)
        return final_answer

    # 解析 LLM 返回的最终答案文本
    def _parse_final_response(self, response: str) -> str:
        # 使用正则表达式提取 <answer> 标签中的内容
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        # 获取匹配的内容并去除首尾空白
        final_answer = answer_match.group(1).strip()
        return final_answer

    # 使用 LLM 检查模型回答与标准答案在语义上是否等效
    """
    构建 prompts 调用 llm
    1. answerer.yaml 的 ANSWER_CHECK_PROMPT 作为 user prompt
    """
    async def answer_check(self, question: str, model_answer: str, ground_truth: str) -> bool:
        # 使用 LLM 检查模型回答和标准答案是否在语义上等效
        """Check if model answer and ground truth are semantically equivalent using LLM."""
