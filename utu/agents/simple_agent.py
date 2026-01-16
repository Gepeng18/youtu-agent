import asyncio
import pathlib
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, Literal

from agents import (
    Agent,
    AgentOutputSchemaBase,
    Model,
    ModelSettings,
    RunConfig,
    RunHooks,
    Runner,
    StopAtTools,
    TContext,
    Tool,
    TResponseInputItem,
    trace,
)
from agents.mcp import MCPServer

from ..config import AgentConfig, ConfigLoader, ToolkitConfig
from ..context import BaseContextManager, build_context_manager
from ..db import DBService, TrajectoryModel
from ..env import _BaseEnv, get_env
from ..hooks import get_run_hooks
from ..tools import TOOLKIT_MAP, AsyncBaseToolkit
from ..utils import AgentsMCPUtils, AgentsUtils, get_logger, load_class_from_file
from .common import QueueCompleteSentinel, TaskRecorder

logger = get_logger(__name__)


# 简单代理类：封装了环境、工具、MCP服务器和上下文管理器的代理，基于openai-agents构建
# A simple agent with env, tools, mcps, and context manager, wrapped on openai-agents.
class SimpleAgent:
    """A simple agent with env, tools, mcps, and context manager, wrapped on openai-agents."""

    # 初始化简单代理实例
    def __init__(
        self,
        *,
        # 代理配置，可以是AgentConfig对象、配置文件路径字符串或None（使用默认配置）
        config: AgentConfig | str | None = None,  # use config to pass agent configs
        # 代理名称，用于标识代理实例
        name: str | None = None,
        # 代理指令，可以是字符串或可调用对象，用于指导代理行为
        instructions: str | Callable | None = None,
        # 使用的模型，可以是模型名称字符串或Model对象
        model: str | Model | None = None,
        # 模型设置项，如 temperature 等
        model_settings: ModelSettings | None = None,
        # 直接配置的工具列表，不能与toolkits同时使用
        tools: list[Tool] = None,  # config tools
        # 从工具包配置加载的工具包名称列表，不能与tools同时使用
        toolkits: list[str] | None = None,  # load tools from toolkit configs
        # 输出类型定义，支持 Pydantic 模型
        output_type: type[Any] | AgentOutputSchemaBase | None = None,
        # 工具使用行为模式，默认为运行完工具后再次运行 LLM
        tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools = "run_llm_again",
    ):
        # 确保不能同时传入tools和toolkits参数，避免配置冲突
        assert not (tools and toolkits), "You can't pass both tools and toolkits."
        # 获取代理配置
        self.config = self._get_config(config)
        # 如果提供了名称，则覆盖配置中的名称
        if name:
            self.config.agent.name = name
        # 如果提供了指令，则覆盖配置中的指令
        if instructions:
            self.config.agent.instructions = instructions
        # 如果配置了停止工具名称，则设置相应的行为模式
        if self.config.stop_at_tool_names:
            tool_use_behavior = StopAtTools(stop_at_tool_names=self.config.stop_at_tool_names)
        # 初始化模型实例
        self.model = self._get_model(self.config, model)
        # 初始化模型设置
        self.model_settings = self._get_model_settings(self.config, model_settings)
        # 初始化工具列表
        self.tools: list[Tool] = tools or []
        # 初始化工具包名称列表
        self.toolkits: list[str] = toolkits or []
        # 设置输出类型
        self.output_type: type[Any] | AgentOutputSchemaBase | None = output_type
        # 设置工具使用行为
        self.tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools = tool_use_behavior
        # 上下文管理器引用
        self.context_manager: BaseContextManager = None
        # 环境对象引用
        self.env: _BaseEnv = None
        # 工作空间目录路径
        self.workspace_dir: str = ""
        # 当前活跃的代理实例
        # 是否应该移到任务记录器中？
        self.current_agent: Agent[TContext] = None  # move to task recorder?
        # 对话输入历史项列表
        self.input_items: list[TResponseInputItem] = []
        # 运行时钩子
        self.run_hooks: RunHooks = get_run_hooks(self.config)

        # MCP 服务器列表
        self._mcp_servers: list[MCPServer] = []
        # 工具包字典，存储加载的工具包实例
        self._toolkits: dict[str, AsyncBaseToolkit] = {}
        # 用于管理 MCP 连接生命周期的异步退出栈
        self._mcps_exit_stack = AsyncExitStack()
        # 初始化状态标识
        self._initialized = False

    # 获取代理配置对象，如果传入的是字符串则从配置文件加载
    def _get_config(self, config: AgentConfig | str | None) -> AgentConfig:
        if isinstance(config, AgentConfig):
            return config
        return ConfigLoader.load_agent_config(config or "simple/base")

    # 获取模型对象，支持从配置或直接传入的模型参数创建
    def _get_model(self, config: AgentConfig, model: str | Model | None = None) -> Model:
        if isinstance(model, Model):
            return model
        model_provider_config = config.model.model_provider.model_dump()
        if isinstance(model, str):
            model_provider_config["model"] = model
        return AgentsUtils.get_agents_model(**model_provider_config)

    # 获取模型设置，如果没有传入则使用配置中的默认设置
    def _get_model_settings(self, config: AgentConfig, model_settings: ModelSettings | None = None) -> ModelSettings:
        if isinstance(model_settings, ModelSettings):
            return model_settings
        return config.model.model_settings

    # 异步上下文管理器入口，构建代理实例
    async def __aenter__(self) -> "SimpleAgent":
        await self.build()
        return self

    # 异步上下文管理器出口，清理代理资源
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    # 构建代理实例，初始化环境、工具和代理对象
    # Build the agent
    async def build(self, trace_id: str = None):
        """Build the agent"""
        if self._initialized:
            logger.info("Agent already initialized! Skipping build.")
            return
        # 初始化环境对象，使用配置和追踪ID
        self.env = await get_env(self.config, trace_id or AgentsUtils.gen_trace_id())  # Pass trace_id
        # 构建环境，准备运行时依赖
        await self.env.build()
        # 获取所有可用的工具列表
        tools = await self.get_tools(self.env)
        # 创建openai-agents的Agent实例，配置所有必要的参数
        self.current_agent = Agent(
            name=self.config.agent.name,
            instructions=self.config.agent.instructions,
            model=self.model,
            model_settings=self.model_settings,
            tools=tools,
            output_type=self.output_type,
            tool_use_behavior=self.tool_use_behavior,
            mcp_servers=self._mcp_servers,
        )
        # 构建上下文管理器，用于管理代理的上下文信息
        self.context_manager = build_context_manager(self.config)
        # 标记代理初始化完成
        self._initialized = True

    # 清理代理资源，包括MCP服务器、工具包和环境
    # Cleanup
    async def cleanup(self):
        """Cleanup"""
        logger.info("Cleaning up MCP servers...")
        await self._mcps_exit_stack.aclose()
        self._mcp_servers = []
        logger.info("Cleaning up tools...")
        self._toolkits = {}
        logger.info("Cleaning up env...")
        await self.env.cleanup()
        self._initialized = False

    # 设置工作空间目录，为需要工作空间的工具包提供环境
    def setup_workspace(self, workspace_dir: str | pathlib.Path):
        """Setup workspace for toolkits that need it"""
        assert pathlib.Path(workspace_dir).exists()
        self.workspace_dir = str(workspace_dir)
        self._setup_workspace_for_toolkits()

    # 为所有已加载的工具包设置工作空间
    def _setup_workspace_for_toolkits(self):
        for toolkit in self._toolkits.values():
            if hasattr(toolkit, "setup_workspace"):
                toolkit.setup_workspace(self.workspace_dir)

    # 获取代理可用的工具列表，支持直接配置的工具和从工具包加载的工具
    async def get_tools(self, env: _BaseEnv = None) -> list[Tool]:
        if self.tools:
            return self.tools

        # 根据配置决定工具加载方式
        if self.toolkits:
            # 从工具包名称列表加载工具
            await self._load_toolkits_config(env)
        else:
            # 从配置的工具包配置加载工具
            tools_list: list[Tool] = []
            # 添加环境提供的工具
            # 添加环境工具
            tools_list += await env.get_tools()  # add env tools
            # TODO: handle duplicate tool names
            # TODO: 处理重复的工具名称
            for _, toolkit_config in self.config.toolkits.items():
                # 加载单个工具包
                toolkit = await self._load_toolkit(toolkit_config, env)
                # 只为内置和自定义工具包获取agents可用的工具
                if toolkit_config.mode in ["customized", "builtin"]:
                    tools_list.extend(toolkit.get_tools_in_agents())
            # 记录加载的工具信息
            tool_names = [tool.name for tool in tools_list]
            logger.info(f"Loaded {len(tool_names)} tools: {tool_names}")
            self.tools = tools_list
        # 如果需要，设置工作空间
        # setup workspace if needed
        if self.workspace_dir:
            self._setup_workspace_for_toolkits()
        return self.tools

    # 从工具包配置列表加载并初始化所有工具包
    async def _load_toolkits_config(self, env: _BaseEnv = None):
        assert isinstance(self.toolkits, list) and all(isinstance(tool, str) for tool in self.toolkits)
        parsed_tools = []
        for tool_name in self.toolkits:
            config = ConfigLoader.load_toolkit_config(tool_name)
            toolkit = await self._load_toolkit(config, env)
            if config.mode in ["customized", "builtin"]:
                parsed_tools.extend(toolkit.get_tools_in_agents())
        self.tools = parsed_tools

    # 根据工具包配置的模式加载对应的工具包实现
    async def _load_toolkit(self, toolkit_config: ToolkitConfig, env: _BaseEnv = None) -> AsyncBaseToolkit | MCPServer:
        if toolkit_config.mode == "builtin":
            toolkit = await self._load_builtin_toolkit(toolkit_config, env)
        elif toolkit_config.mode == "customized":
            toolkit = await self._load_customized_toolkit(toolkit_config, env)
        elif toolkit_config.mode == "mcp":
            toolkit = await self._load_mcp_server(toolkit_config)
        else:
            raise ValueError(f"Unknown toolkit mode: {toolkit_config.mode}")
        return toolkit

    # 加载内置工具包，从TOOLKIT_MAP中获取对应的工具包类并初始化
    async def _load_builtin_toolkit(self, toolkit_config: ToolkitConfig, env: _BaseEnv = None) -> AsyncBaseToolkit:
        logger.info(f"Loading builtin toolkit `{toolkit_config.name}` with config {toolkit_config}")
        toolkit = TOOLKIT_MAP[toolkit_config.name](toolkit_config)
        toolkit.setup_env(env)
        self._toolkits[toolkit_config.name] = toolkit
        return toolkit

    # 加载自定义工具包，从指定文件动态加载工具包类并初始化
    async def _load_customized_toolkit(self, toolkit_config: ToolkitConfig, env: _BaseEnv = None) -> AsyncBaseToolkit:
        logger.info(f"Loading customized toolkit `{toolkit_config.name}` with config {toolkit_config}")
        assert toolkit_config.customized_filepath is not None and toolkit_config.customized_classname is not None
        toolkit_class = load_class_from_file(toolkit_config.customized_filepath, toolkit_config.customized_classname)
        toolkit = toolkit_class(toolkit_config)
        toolkit.setup_env(env)
        self._toolkits[toolkit_config.name] = toolkit
        return toolkit

    # 加载MCP服务器，并将其加入异步上下文管理栈进行生命周期管理
    async def _load_mcp_server(self, toolkit_config: ToolkitConfig) -> MCPServer:
        logger.info(f"Loading MCP server `{toolkit_config.name}` with params {toolkit_config.config}")
        mcp_server = AgentsMCPUtils.get_mcp_server(toolkit_config)
        server = await self._mcps_exit_stack.enter_async_context(mcp_server)
        self._mcp_servers.append(server)
        return server

    # 创建运行配置对象，包含模型、工作流名称等运行时参数
    def _get_run_config(self) -> RunConfig:
        run_config = RunConfig(
            model=self.current_agent.model,
            model_settings=self.config.model.model_settings,
            workflow_name=self.config.agent.name,
        )
        return run_config

    # 获取代理运行时的上下文信息，包含上下文管理器、环境和配置
    def _get_context(self) -> dict:
        return {
            "context_manager": self.context_manager,
            "env": self.env,
            "agent_config": self.config,
        }

    # 准备运行参数字典，整合所有必要的运行时配置
    def _prepare_run_kwargs(self, input: str | list[TResponseInputItem]) -> dict:
        return {
            "starting_agent": self.current_agent,
            "input": input,
            "context": self._get_context(),
            "max_turns": self.config.max_turns,
            "hooks": self.run_hooks,
            "run_config": self._get_run_config(),
        }

    # 封装openai-agents的Runner API，提供统一的代理运行接口
    # wrap `Runner` apis in @openai-agents
    async def run(
        self, input: str | list[TResponseInputItem], trace_id: str = None, save: bool = False, log_to_db: bool = True
    ) -> TaskRecorder:
        """Entrypoint for running the agent
        # 代理运行的入口点，支持字符串和结构化输入
        # Entrypoint for running the agent

        Args:
            trace_id: str to identify the run
            save: whether to update massage history (use `input_items`)
        """
        # 生成或使用提供的追踪ID
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        # 处理不同格式的输入，提取任务内容
        if isinstance(input, list):
            # 列表格式输入，最后一项应包含content字段
            assert isinstance(input[-1], dict) and "content" in input[-1], "invalid input format!"
            task = input[-1]["content"]
        else:
            # 字符串格式输入，直接作为任务内容
            assert isinstance(input, str), "input should be str or list of TResponseInputItem!"
            task = input
        # 创建任务记录器来跟踪运行过程
        recorder = TaskRecorder(task=task, input=input, trace_id=trace_id)

        # 如果代理未初始化，先进行构建
        if not self._initialized:
            await self.build(recorder.trace_id)
        input = recorder.input
        # 只有字符串输入时才添加历史记录
        if isinstance(input, str):  # only add history when input is str?
            input = self.input_items + [{"content": input, "role": "user"}]
        # 准备运行参数
        run_kwargs = self._prepare_run_kwargs(input)
        # 根据是否已有追踪上下文决定是否创建新的追踪
        if AgentsUtils.get_current_trace():
            run_result = await Runner.run(**run_kwargs)
        else:
            with trace(workflow_name="simple_agent", trace_id=recorder.trace_id):
                run_result = await Runner.run(**run_kwargs)
        # 保存最终输出和轨迹信息到记录器
        # save final output and trajectory
        recorder.add_run_result(run_result)
        # 如果需要保存状态，更新输入历史和当前代理
        if save:
            self.input_items = run_result.to_input_list()
            # 注意：SimpleAgent中只有一个代理
            # NOTE: acturally, there are only one agent in SimpleAgent
            self.current_agent = run_result.last_agent
        if log_to_db:
            DBService.add(TrajectoryModel.from_task_recorder(recorder))
        return recorder

    # 流式运行代理，返回支持流式事件的TaskRecorder对象
    def run_streamed(
        self, input: str | list[TResponseInputItem], trace_id: str = None, save: bool = False, log_to_db: bool = True
    ) -> TaskRecorder:
        """Entrypoint for running the agent streamly
        # 流式运行代理的入口点
        # Entrypoint for running the agent streamly

        Args:
            trace_id: str to identify the run
        """
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        if isinstance(input, list):
            assert isinstance(input[-1], dict) and "content" in input[-1], "invalid input format!"
            task = input[-1]["content"]
        else:
            assert isinstance(input, str), "input should be str or list of TResponseInputItem!"
            task = input
        recorder = TaskRecorder(task=task, input=input, trace_id=trace_id)
        recorder._run_impl_task = asyncio.create_task(self._start_streaming(recorder, save, log_to_db))
        return recorder

    async def _start_streaming(self, recorder: TaskRecorder, save: bool = False, log_to_db: bool = True):
        if not self._initialized:
            await self.build(recorder.trace_id)
        try:
            input = recorder.input
            # 只有字符串输入时才添加历史记录？
            if isinstance(input, str):  # only add history when input is str?
                input = self.input_items + [{"content": input, "role": "user"}]
            run_kwargs = self._prepare_run_kwargs(input)
            if AgentsUtils.get_current_trace():
                run_streamed_result = Runner.run_streamed(**run_kwargs)
            else:
                with trace(workflow_name="simple_agent", trace_id=recorder.trace_id):
                    run_streamed_result = Runner.run_streamed(**run_kwargs)
            async for event in run_streamed_result.stream_events():
                recorder._event_queue.put_nowait(event)
            # 保存最终输出和轨迹信息
            # save final output and trajectory
            recorder.add_run_result(run_streamed_result)
            if save:
                self.input_items = run_streamed_result.to_input_list()
                # 注意：SimpleAgent中实际上只有一个代理
                # NOTE: acturally, there are only one agent in SimpleAgent
                self.current_agent = run_streamed_result.last_agent
            if log_to_db:
                DBService.add(TrajectoryModel.from_task_recorder(recorder))
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            recorder._event_queue.put_nowait(QueueCompleteSentinel())
            recorder._is_complete = True
            raise e
        finally:
            recorder._event_queue.put_nowait(QueueCompleteSentinel())
            recorder._is_complete = True

    # 实用工具API
    # util apis
    # 简化的聊天接口，支持保存对话历史并打印新消息
    async def chat(self, input: str) -> TaskRecorder:
        # TODO: 为多轮对话设置会话级别的追踪
        # TODO: set "session-level" tracing for multi-turn chat
        recorder = await self.run(input, save=True)
        run_result = recorder.get_run_result()
        AgentsUtils.print_new_items(run_result.new_items)
        return run_result

    # 流式聊天接口，支持实时打印流式事件
    async def chat_streamed(self, input: str) -> TaskRecorder:
        recorder = self.run_streamed(input, save=True)
        await AgentsUtils.print_stream_events(recorder.stream_events())
        return recorder

    # 设置代理指令（危险操作，会重置当前代理的指令）
    def set_instructions(self, instructions: str):
        logger.warning("WARNING: reset instructions is dangerous!")
        self.current_agent.instructions = instructions

    # 清空输入历史记录，重置聊天历史
    def clear_input_items(self):
        # reset chat history
        self.input_items = []
