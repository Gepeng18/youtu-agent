from typing import Any

from agents import Agent, AgentOutputSchemaBase, Runner, RunResultStreaming, TResponseInputItem, trace

from ..config import ModelConfigs
from ..utils import AgentsUtils, get_logger
from .common import TaskRecorder

logger = get_logger(__name__)


# 简化的LLM代理类，直接包装一个模型提供基本的推理能力
class LLMAgent:
    """Minimal agent that wraps a model."""

    # 初始化LLM代理实例
    def __init__(
        self,
        # 模型配置，包含模型提供商和设置信息
        model_config: ModelConfigs,
        # 代理名称，可选，默认为"LLMAgent"
        name: str = None,
        # 代理指令，用于指导模型行为
        instructions: str = None,
        output_type: type[Any] | AgentOutputSchemaBase | None = None,
    ):
        # 保存模型配置
        self.config = model_config
        # 创建openai-agents的Agent实例
        self.agent = Agent(
            name=name or "LLMAgent",
            instructions=instructions,
            model=AgentsUtils.get_agents_model(**model_config.model_provider.model_dump()),
            model_settings=model_config.model_settings,
            output_type=output_type,
        )

    # 设置代理的指令内容
    def set_instructions(self, instructions: str):
        logger.info(f"Set instructions for LLMAgent: {instructions[:50]}...")
        self.agent.instructions = instructions

    # 运行代理处理输入，返回任务记录器
    async def run(self, input: str | list[TResponseInputItem], trace_id: str = None) -> TaskRecorder:
        # TODO: 自定义代理名称
        # TODO: customized the agent name
        # 生成或使用提供的追踪ID
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        # 创建任务记录器
        task_recorder = TaskRecorder(input, trace_id)

        # 根据是否已有追踪上下文决定是否创建新的追踪
        if AgentsUtils.get_current_trace():
            run_result = await Runner.run(self.agent, input)
        else:
            trace_id = trace_id or AgentsUtils.gen_trace_id()
            with trace(workflow_name="llm_agent", trace_id=trace_id):
                run_result = await Runner.run(self.agent, input)
        # 添加运行结果到记录器并设置最终输出
        task_recorder.add_run_result(run_result)
        task_recorder.set_final_output(run_result.final_output)
        return task_recorder

    # 流式运行代理，返回流式结果对象
    def run_streamed(self, input: str | list[TResponseInputItem], trace_id: str = None) -> RunResultStreaming:
        # 根据是否已有追踪上下文决定是否创建新的追踪
        if AgentsUtils.get_current_trace():
            return Runner.run_streamed(self.agent, input)
        else:
            trace_id = trace_id or AgentsUtils.gen_trace_id()
            with trace(workflow_name="llm_agent", trace_id=trace_id):
                return Runner.run_streamed(self.agent, input)
