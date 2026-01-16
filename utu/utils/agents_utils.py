import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Literal

from agents import (
    FunctionTool,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    Model,
    ModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    ReasoningItem,
    RunItem,
    RunResult,
    StreamEvent,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)
from agents.models.chatcmpl_converter import Converter
from agents.stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent, RunItemStreamEvent
from agents.tracing import Trace, gen_trace_id, get_current_trace
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.responses import ResponseFunctionToolCall

from .image import encode_image
from .openai_utils import OpenAIChatCompletionParams
from .print_utils import PrintUtils

logger = logging.getLogger(__name__)


# OpenAI 聊天补全格式转换器类，用于在不同消息格式之间进行转换
class ChatCompletionConverter(Converter):
    # 将输入项列表转换为聊天补全消息列表
    @classmethod
    def items_to_messages(cls, items: str | Iterable[TResponseInputItem]) -> list[ChatCompletionMessageParam]:
        # 跳过推理项，参见 chatcmpl_converter.Converter.items_to_messages()
        # skip reasoning, see chatcmpl_converter.Converter.items_to_messages()
        # 处理 agents.exceptions.UserError: 未处理的项类型或结构
        # agents.exceptions.UserError: Unhandled item type or structure:
        # {'id': '__fake_id__', 'summary': [{'text': '...', 'type': 'summary_text'}], 'type': 'reasoning'}
        # 如果不是字符串，则先进行过滤
        if not isinstance(items, str):  # TODO: check it!
            items = cls.filter_items(items)
        # 调用父类的转换方法
        return Converter.items_to_messages(items)

    # 过滤输入项，剔除推理类型的项
    @classmethod
    def filter_items(cls, items: str | Iterable[TResponseInputItem]) -> str | list[TResponseInputItem]:
        # 如果输入是字符串，直接返回
        if isinstance(items, str):
            return items
        filtered_items = []
        # 遍历输入项，跳过类型为 "reasoning" 的项
        for item in items:
            if item.get("type", None) == "reasoning":
                continue
            filtered_items.append(item)
        return filtered_items

    # 将输入项转换为包含 role 和 content 的字典列表
    @classmethod
    def items_to_dict(cls, items: str | Iterable[TResponseInputItem]) -> list[dict]:
        # 将项转换为包含 {"role", "content"} 的字典列表
        """convert items to a list of dict which have {"role", "content"}
        正在开发中！
        WIP!
        """
        # 如果是字符串，包装成用户角色的消息
        if isinstance(items, str):
            return [{"role": "user", "content": items}]
        result = []
        # 遍历各项并根据其类型进行转换
        for item in items:
            # 尝试各种可能的项类型并转换
            if msg := Converter.maybe_easy_input_message(item):
                result.append(msg)
            elif msg := Converter.maybe_input_message(item):
                result.append(msg)
            elif msg := Converter.maybe_response_output_message(item):
                result.append(msg)
            elif msg := Converter.maybe_file_search_call(item):
                # 处理文件搜索调用
                msg.update({"role": "tool", "content": msg["results"]})
                result.append(msg)
            elif msg := Converter.maybe_function_tool_call(item):
                # 处理函数工具调用
                msg.update({"role": "assistant", "content": f"{msg['name']}({msg['arguments']})"})
                result.append(msg)
            elif msg := Converter.maybe_function_tool_call_output(item):
                # 处理函数工具调用输出
                msg.update({"role": "tool", "content": msg["output"], "tool_call_id": msg["call_id"]})
                result.append(msg)
            elif msg := Converter.maybe_reasoning_message(item):
                # 处理推理消息
                msg.update({"role": "assistant", "content": msg["summary"]})
                result.append(msg)
            else:
                # 记录未知消息类型警告
                logger.warning(f"Unknown message type: {item}")
                result.append({"role": "assistant", "content": f"Unknown message type: {item}"})
        return result


# 针对 openai-agents SDK 的实用工具类
class AgentsUtils:
    # 针对 openai-agents SDK 的实用工具
    """Utils for openai-agents SDK"""

    # 生成唯一的组 ID（用于 OpenAI 追踪）
    @staticmethod
    def generate_group_id() -> str:
        # 生成唯一的组 ID。（在 OpenAI 追踪中使用）
        """Generate a unique group ID. (Used in OpenAI tracing)
        参考：https://openai.github.io/openai-agents-python/tracing/
        Ref: https://openai.github.io/openai-agents-python/tracing/
        """
        # 返回随机生成的 UUID 的前 16 位
        return uuid.uuid4().hex[:16]

    # 生成追踪 ID
    @staticmethod
    def gen_trace_id() -> str:
        # 调用基础库的生成方法
        return gen_trace_id()

    # 获取当前的追踪对象
    @staticmethod
    def get_current_trace() -> Trace:
        # 调用基础库的获取方法
        return get_current_trace()

    # 获取配置的代理模型实例
    @staticmethod
    def get_agents_model(
        type: Literal["responses", "chat.completions", "litellm"] = None,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
    ) -> Model:
        # 如果未指定，则从环境变量中获取类型和模型
        type = type or os.getenv("UTU_LLM_TYPE", "chat.completions")
        model = model or os.getenv("UTU_LLM_MODEL")
        # 处理 litellm 类型
        if type == "litellm":
            # Ref: https://docs.litellm.ai/docs/providers
            # NOTE: should set .evn properly! e.g. AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION for Azure
            # 参考：https://docs.litellm.ai/docs/providers
            # NOTE: 应正确设置环境变量！例如 Azure 需要 AZURE_API_KEY 等
            #   https://docs.litellm.ai/docs/providers/azure/
            from agents.extensions.models.litellm_model import LitellmModel

            return LitellmModel(model=model)

        # 获取基础 URL 和 API 密钥
        base_url = base_url or os.getenv("UTU_LLM_BASE_URL")
        api_key = api_key or os.getenv("UTU_LLM_API_KEY")
        # 确保关键配置项已设置
        if not api_key or not base_url:
            raise ValueError("UTU_LLM_API_KEY and UTU_LLM_BASE_URL must be set")
        # 初始化异步 OpenAI 客户端
        openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=100,
        )
        # 根据类型返回对应的模型对象
        if type == "chat.completions":
            return OpenAIChatCompletionsModel(model=model, openai_client=openai_client)
        elif type == "responses":
            return OpenAIResponsesModel(model=model, openai_client=openai_client)
        else:
            raise ValueError("Invalid type: " + type)

    # 从代理运行结果中提取执行轨迹
    @staticmethod
    def get_trajectory_from_agent_result(agent_result: RunResult, agent_name: str = None) -> dict:
        # 如果未指定代理名称，则从结果中获取
        if agent_name is None:
            agent_name = agent_result.last_agent.name
        # 返回包含代理名称和格式化轨迹的消息字典
        return {
            "agent": agent_name,
            "trajectory": ChatCompletionConverter.items_to_messages(agent_result.to_input_list()),
        }

    # 打印 Runner.run() 生成的新项
    @staticmethod
    def print_new_items(new_items: list[RunItem]) -> None:
        # 打印由 Runner.run() 生成的新项
        """Print new items generated by Runner.run()"""
        # 遍历新生成的项并根据类型打印
        for new_item in new_items:
            agent_name = new_item.agent.name
            if isinstance(new_item, MessageOutputItem):
                # 打印文本消息输出
                PrintUtils.print_bot(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
            elif isinstance(new_item, HandoffOutputItem):
                # 打印代理移交信息
                PrintUtils.print_info(f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}")
            elif isinstance(new_item, ToolCallItem):
                # 打印工具调用信息
                # 不要使用 OpenAI 的内置工具
                assert isinstance(new_item.raw_item, ResponseFunctionToolCall)  # DONOT use openai's built-in tools
                PrintUtils.print_info(
                    f"{agent_name}: Calling a tool: {new_item.raw_item.name}({json.loads(new_item.raw_item.arguments)})"
                )
            elif isinstance(new_item, ToolCallOutputItem):
                # 打印工具调用输出
                PrintUtils.print_tool(f"Tool call output: {new_item.output}")
            elif isinstance(new_item, ReasoningItem):
                # 打印推理过程信息
                PrintUtils.print_info(f"{agent_name}: Reasoning: {new_item.raw_item}")
            else:
                # 记录跳过项的信息
                PrintUtils.print_info(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")

    # 打印由 Runner.run_streamed() 生成的流式事件
    @staticmethod
    async def print_stream_events(result: AsyncIterator[StreamEvent]) -> None:
        # 打印由 Runner.run_streamed() 生成的流式事件
        """Print stream events generated by Runner.run_streamed()"""
        # 异步遍历流式事件
        async for event in result:
            # print(f"> [DEBUG] event: {event}")
            if isinstance(event, RawResponsesStreamEvent):
                # 处理原始响应流事件
                # event.data -- ResponseStreamEvent
                if event.data.type == "response.output_item.added":
                    # 处理新项添加事件
                    match event.data.item.type:
                        # 各种可能的项类型：message, function_call, reasoning 等
                        # computer_call, code_interpreter_call, custom_tool_call, file_search_call, function_call,
                        # we_search_call, image_generation_call, local_shell_call,
                        # mcp_call, mcp_list_tools, mcp_approval_request, message, reasoning
                        case "message":
                            pass
                        case "function_call":
                            # 打印工具调用开始标签
                            PrintUtils.print_bot(
                                f"<toolcall name={event.data.item.name}>{event.data.item.arguments}", end=""
                            )
                        case _:
                            # 打印其他类型的标签
                            PrintUtils.print_bot(f"<{event.data.item.type}>", end="")
                elif event.data.type == "response.output_item.done":
                    # 处理项完成事件
                    match event.data.item.type:
                        case "message":
                            pass
                            # PrintUtils.print_bot("")  # 增加新行？
                            # PrintUtils.print_bot("")  # add a new line?
                        case "function_call":
                            # 打印工具调用结束标签
                            PrintUtils.print_bot("</toolcall>")
                        case _:
                            # 记录项完成，vllm 的输出顺序有时可能不正确
                            # PrintUtils.print_bot(f"</{event.data.item.type}>")
                            logger.info(f"</{event.data.item.type}>")  # It seems that vllm's output order is wrong
                elif event.data.type == "response.content_part.added":
                    # 处理内容部分添加事件
                    match event.data.part.type:
                        # output_text, refusal
                        case "output_text":
                            pass
                        case "refusal":
                            # 打印拒绝标签
                            PrintUtils.print_bot(f"<refusal>{event.data.part.refusal}", end="")
                        case _:
                            # 警告未知部分类型
                            logger.warning(f"Unknown part type: {event.data.part.type}! {event}")
                elif event.data.type == "response.content_part.done":
                    # 处理内容部分完成事件
                    match event.data.part.type:
                        case "output_text":
                            pass
                        case "refusal":
                            # 打印拒绝结束标签
                            PrintUtils.print_bot("</refusal>")
                        case _:
                            logger.warning(f"Unknown part type: {event.data.part.type}! {event}")
                elif event.data.type == "response.reasoning_summary_part.added":
                    # 打印推理摘要开始
                    PrintUtils.print_info("<reasoning_summary>", end="")
                elif event.data.type == "response.reasoning_summary_part.done":
                    # 记录推理摘要结束
                    # PrintUtils.print_info("</reasoning_summary>", end="")
                    logger.info("</reasoning_summary>")  # It seems that vllm's output order is wrong
                elif event.data.type == "response.reasoning_summary_text.delta":
                    # 打印推理摘要增量
                    PrintUtils.print_info(f"{event.data.delta}", end="")
                elif event.data.type == "response.function_call_arguments.delta":
                    # 打印函数调用参数增量
                    PrintUtils.print_bot(f"{event.data.delta}", end="")
                elif event.data.type == "response.function_call_arguments.done":
                    pass
                elif event.data.type == "response.output_text.delta":
                    # 打印输出文本增量
                    PrintUtils.print_bot(f"{event.data.delta}", end="")
                elif event.data.type == "response.reasoning_text.delta":
                    # 打印推理文本增量
                    PrintUtils.print_info(f"{event.data.delta}", end="")
                elif event.data.type == "response.reasoning_text.done":
                    # 打印推理文本结束
                    PrintUtils.print_info("</reasoning_text>", end="")
                elif event.data.type in ("response.output_text.done",):
                    # 打印空行
                    PrintUtils.print_info("")
                elif event.data.type in (
                    "response.created",
                    "response.completed",
                    "response.in_progress",
                ):
                    pass
                else:
                    # 记录未知事件类型信息
                    PrintUtils.print_info(f"Unknown event type: {event.data.type}! {event}")
                    # raise ValueError(f"Unknown event type: {event.data.type}")
            elif isinstance(event, RunItemStreamEvent):
                # 处理运行项流事件
                item: RunItem = event.item
                if item.type == "message_output_item":
                    # 不要打印两次以避免重复（已经处理了 `response.output_text.delta`）
                    pass  # do not print twice to avoid duplicate! (already handled `response.output_text.delta`)
                    # PrintUtils.print_bot(f"{ItemHelpers.text_message_output(item).strip()}")
                elif item.type == "reasoning_item":
                    pass
                elif item.type == "tool_call_item":
                    pass
                    # PrintUtils.print_bot([tool_call] {item.raw_item.name}({item.raw_item.arguments})")
                elif item.type == "tool_call_output_item":
                    # 打印工具输出
                    PrintUtils.print_tool(f"[tool_output] {item.output}")  # item.raw_item
                    # 与 `ToolCallItem` 相同
                elif item.type == "handoff_call_item":  # same as `ToolCallItem`
                    # 打印移交调用信息
                    PrintUtils.print_bot(f"[handoff_call] {item.raw_item.name}({item.raw_item.arguments})")
                elif item.type == "handoff_output_item":
                    # 打印移交结果信息
                    PrintUtils.print_info(f">> Handoff from {item.source_agent.name} to {item.target_agent.name}")
                elif event.type in (
                    "mcp_list_tools_item",
                    "mcp_approval_request_item",
                    "mcp_approval_response_item",
                ):
                    # 跳过 MCP 相关的特殊项
                    PrintUtils.print_info(f"  >>> Skipping item: {event}")
                else:
                    # 跳过其他未知类别的项
                    PrintUtils.print_info(f"  >>> Skipping item: {item.__class__.__name__}")
            elif isinstance(event, AgentUpdatedStreamEvent):
                # 处理代理更新事件
                PrintUtils.print_info(f">> new agent: {event.new_agent.name}")
            # 跳过来自 youtu-agent 的自定义事件
            # skip events from youtu-agent
            elif event.type in ("orchestrator_stream_event", "orchestra_stream_event", "simple_agent_generated"):
                pass
            else:
                # 警告未知事件类型
                logger.warning(f"Unknown event type: {event.type}! {event}")
        # 流式输出后换行
        print()  # Newline after stream?

    # 将 OpenAI 聊天补全参数转换为代理的模型设置对象
    @staticmethod
    def convert_model_settings(params: OpenAIChatCompletionParams) -> ModelSettings:
        # "tools", "messages", "model"
        # 待修复：移至 extra_args
        # FIXME: move to extra_args
        for p in ("max_completion_tokens", "top_logprobs", "logprobs", "seed", "stop"):
            if p in params:
                # 记录不支持的参数警告
                logger.warning(f"Parameter `{p}` is not supported in ModelSettings")
        return ModelSettings(
            max_tokens=params.get("max_tokens", None),
            temperature=params.get("temperature", None),
            top_p=params.get("top_p", None),
            frequency_penalty=params.get("frequency_penalty", None),
            presence_penalty=params.get("presence_penalty", None),
            tool_choice=params.get("tool_choice", None),
            parallel_tool_calls=params.get("parallel_tool_calls", None),
            extra_query=params.get("extra_query", None),
            extra_body=params.get("extra_body", None),
            extra_headers=params.get("extra_headers", None),
        )

    @staticmethod
    def convert_sp_input(
        messages: list[ChatCompletionMessageParam],
    ) -> tuple[str | None, str | list[TResponseInputItem]]:
        if isinstance(messages, str):
            return None, messages
        if messages[0].get("role", None) == "system":
            return messages[0]["content"], messages[1:]
        return None, messages

    @staticmethod
    def convert_tool(tool: ChatCompletionToolParam) -> FunctionTool:
        assert tool["type"] == "function"
        return FunctionTool(
            name=tool["function"]["name"],
            description=tool["function"].get("description", ""),
            params_json_schema=tool["function"].get("parameters", None),
            on_invoke_tool=None,
        )

    @staticmethod
    def get_message_from_image(image_url: str) -> dict:
        """Get a message dict for image input."""
        # from openai.types.responses.response_input_item_param import Message
        # from openai.types.responses.response_input_image_param import ResponseInputImageParam
        return {"role": "user", "content": [{"type": "input_image", "image_url": encode_image(image_url)}]}


class SimplifiedOpenAIChatCompletionsModel(OpenAIChatCompletionsModel):
    """extend OpenAIChatCompletionsModel to support basic api
    - enable tracing based on SimplifiedAsyncOpenAI
    """

    async def query_one(self, **kwargs) -> str:
        system_instructions, input = AgentsUtils.convert_sp_input(kwargs["messages"])
        model_settings = AgentsUtils.convert_model_settings(kwargs)
        tools = [AgentsUtils.convert_tool(tool) for tool in kwargs.get("tools", [])]
        response = await self.get_response(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.ENABLED,
            previous_response_id=None,
            prompt=None,
        )
        return ChatCompletionConverter.items_to_messages(response.to_input_items())
        # with generation_span(
        #     model=kwargs["model"],
        #     model_config=_model_settings,
        #     input=_messages,
        # ) as span_generation:
        #     result = await self.chat.completions.create(**kwargs)
        #     span_generation.span_data.output = result.choices[0].message.model_dump()
        #     return result
