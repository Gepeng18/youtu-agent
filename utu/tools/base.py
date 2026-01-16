from collections.abc import Callable
from typing import TYPE_CHECKING

import mcp.types as types
from agents import FunctionTool, function_tool

from ..config import ToolkitConfig
from ..utils import ChatCompletionConverter, FileUtils, MCPConverter, get_logger
from .utils import register_tool as register_tool

if TYPE_CHECKING:
    from e2b_code_interpreter import AsyncSandbox

    from ..env import _BaseEnv

logger = get_logger(__name__)


# 异步工具包基类，提供工具注册、管理和多种格式转换的功能
class AsyncBaseToolkit:
    # 工具包的基类
    """Base class for toolkits."""

    # 初始化工具包实例
    def __init__(self, config: ToolkitConfig | dict | None = None):
        # 如果传入的不是ToolkitConfig对象，则创建新的配置
        if not isinstance(config, ToolkitConfig):
            # 将字典或 None 转换为 ToolkitConfig 对象
            config = config or {}
            config = ToolkitConfig(config=config, name=self.__class__.__name__)

        # 保存配置信息
        self.config: ToolkitConfig = config
        # 工具映射表，初始为空，用于懒加载
        self._tools_map: dict[str, Callable] = None

        # 初始化环境相关属性
        self.env: _BaseEnv = None
        # 记录环境模式
        # TODO: 弃用此属性
        self.env_mode = self.config.env_mode  # TODO: deprecate it
        # 如果是E2B环境模式，初始化E2B沙箱引用
        if self.env_mode == "e2b":
            self.e2b_sandbox: AsyncSandbox = None

    # 设置工具包运行的环境和沙箱
    # 设置环境和沙箱
    def setup_env(self, env: "_BaseEnv") -> None:
        # 设置环境和工作空间
        """Setup env and workspace."""
        # 保存环境对象的引用
        self.env = env
        # 如果是E2B环境模式，设置沙箱对象
        if self.env_mode == "e2b":  # assert is E2BEnv
            self.e2b_sandbox = env.sandbox
        # 执行工作空间的设置流程
        self.setup_workspace()

    # 设置工具运行的工作空间目录，由具体工具包子类实现
    # 设置工作空间，由具体的工具包类实现
    def setup_workspace(self, workspace_root: str = None):
        # 设置工作空间。在特定工具包内部实现。
        """Setup workspace. Implemented inside specific toolkits."""
        pass

    # 工具映射的懒加载属性
    # 工具映射的懒加载属性
    @property
    def tools_map(self) -> dict[str, Callable]:
        # 工具映射的懒加载
        # - 收集通过@register_tool注册的工具
        """Lazy loading of tools map.
        - collect tools registered by @register_tool
        """
        if self._tools_map is None:
            self._tools_map = {}
            # 遍历类的所有方法，注册带有@tool装饰器的方法
            # Iterate through all methods of the class and register @tool
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if callable(attr) and getattr(attr, "_is_tool", False):
                    self._tools_map[attr._tool_name] = attr
        return self._tools_map

    # 获取工具映射，会根据配置的激活工具进行过滤
    # 获取工具映射，会根据配置的激活工具进行过滤
    def get_tools_map_func(self) -> dict[str, Callable]:
        """Get tools map. It will filter tools by config.activated_tools if it is not None."""
        # 如果配置了激活的工具列表，则进行过滤
        if self.config.activated_tools:
            # 验证所有激活的工具都存在
            assert all(tool_name in self.tools_map for tool_name in self.config.activated_tools), (
                f"Error config activated tools: {self.config.activated_tools}! available tools: {self.tools_map.keys()}"
            )
            # 只保留激活的工具
            tools_map = {tool_name: self.tools_map[tool_name] for tool_name in self.config.activated_tools}
        else:
            # 返回所有工具
            tools_map = self.tools_map
        return tools_map

    # 获取openai-agents格式的工具列表
    # 获取openai-agents格式的工具列表
    def get_tools_in_agents(self) -> list[FunctionTool]:
        """Get tools in openai-agents format."""
        # 获取过滤后的工具映射
        tools_map = self.get_tools_map_func()
        tools = []
        # 将每个工具函数转换为FunctionTool对象
        for _, tool in tools_map.items():
            tools.append(
                function_tool(
                    tool,
                    # 关闭严格模式
                    strict_mode=False,  # turn off strict mode
                )
            )
        return tools

    # 获取OpenAI格式的工具列表
    # 获取OpenAI格式的工具列表
    def get_tools_in_openai(self) -> list[dict]:
        """Get tools in OpenAI format."""
        # 先获取agents格式的工具，然后转换为OpenAI格式
        tools = self.get_tools_in_agents()
        return [ChatCompletionConverter.tool_to_openai(tool) for tool in tools]

    # 获取MCP格式的工具列表
    # 获取MCP格式的工具列表
    def get_tools_in_mcp(self) -> list[types.Tool]:
        """Get tools in MCP format."""
        # 先获取agents格式的工具，然后转换为MCP格式
        tools = self.get_tools_in_agents()
        return [MCPConverter.function_tool_to_mcp(tool) for tool in tools]

    # 通过名称调用工具
    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool by its name."""
        tools_map = self.get_tools_map_func()
        if name not in tools_map:
            raise ValueError(f"Tool {name} not found")
        tool = tools_map[name]
        return await tool(**arguments)


TOOL_PROMPTS: dict[str, str] = FileUtils.load_prompts("tools/tools_prompts.yaml")
