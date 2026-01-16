from agents import RunContextWrapper, TContext, TResponseInputItem

from ..utils import get_logger

logger = get_logger(__name__)


# 基础上下文管理器类，定义输入预处理的接口
class BaseContextManager:
    # 初始化上下文管理器
    def __init__(self, config: dict = None):
        # 保存配置字典，默认为空
        self.config = config or {}

    # 预处理输入内容（基础实现直接返回原内容）
    def preprocess(
        self, input: str | list[TResponseInputItem], run_context: RunContextWrapper[TContext] = None
    ) -> str | list[TResponseInputItem]:
        # 默认直接返回输入，不进行修改
        return input


# 虚拟上下文管理器实现，演示如何处理最大轮数限制
class DummyContextManager(BaseContextManager):
    # 初始化虚拟上下文管理器
    def __init__(self, config: dict = None):
        # 调用父类初始化
        super().__init__(config)
        # 定义达到最大轮数时的提示消息
        self.max_turn_stop_msg = (
            "You have reached the maximum number of turns allowed. "
            "Please DO NOT use ANY tools, provide your final answer."
        )

    # 预处理输入内容，处理轮数溢出等情况
    def preprocess(
        self, input: str | list[TResponseInputItem], run_context: RunContextWrapper[TContext] = None
    ) -> str | list[TResponseInputItem]:
        # 注意：过滤类型为 "reasoning" 的项，因为 vllm 目前无法处理它们
        # NOTE: filter type="reasoning" items for vllm cannot process it for now!
        # return ChatCompletionConverter.filter_items(input)

        # 处理最大轮数超出限制的情况
        # handle MaxTurnsExceeded
        # 从运行上下文中获取 context 字典
        context: dict = run_context.context
        # 确保 context 是字典类型
        assert isinstance(context, dict), "run_context.context should be a dict"
        # 提取当前轮数和最大允许轮数
        current_turn, max_turns = context.get("current_turn"), context.get("max_turns")
        # 记录调试信息
        logger.debug(f"Current turn: {current_turn}, Max turns: {max_turns}")
        # 如果当前轮数达到了最大限制
        if current_turn is not None and max_turns is not None and current_turn == max_turns:
            # 记录警告日志并向输入列表中注入停止消息
            logger.warning(f"Max turns {max_turns} encountered! Injecting stop message.")
            # 将提示语作为用户角色消息添加到对话历史末尾
            input.append(
                {
                    "role": "user",
                    "content": self.max_turn_stop_msg,
                }
            )
        # 返回预处理后的输入
        return input
