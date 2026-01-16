from collections.abc import Callable
from typing import Literal

from pydantic import Field

from .base_config import ConfigBaseModel
from .model_config import ModelConfigs

# 默认指令内容
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


# 代理配置类，定义代理的基本信息
class ProfileConfig(ConfigBaseModel):
    # 配置名称
    name: str | None = "default"
    # 代理指令，可以是字符串或可调用对象
    instructions: str | Callable | None = DEFAULT_INSTRUCTIONS


# 工具包配置类
class ToolkitConfig(ConfigBaseModel):
    # 工具包配置
    """Toolkit config."""

    # 工具包模式：内置、自定义或MCP
    mode: Literal["builtin", "customized", "mcp"] = "builtin"
    """Toolkit mode."""
    # 环境模式：本地或E2B
    env_mode: Literal["local", "e2b"] = "local"
    """Environment mode for the toolkit."""
    # 工具包名称
    name: str | None = None
    """Toolkit name."""
    # 激活的工具列表，如果为None则激活所有工具
    activated_tools: list[str] | None = None
    """Activated tools, if None, all tools will be activated."""
    # 特定工具包的指定配置，使用原始字典以简化
    config: dict | None = Field(default_factory=dict)
    """Specified  configs for certain toolkit. We use raw dict for simplicity"""
    # 如果工具包中使用LLM，则为其配置
    config_llm: ModelConfigs | None = None  # | dict[str, ModelConfigs]
    """LLM config if used in toolkit."""
    # 自定义工具包的文件路径
    customized_filepath: str | None = None
    """Customized toolkit filepath."""
    # 自定义工具包的类名
    customized_classname: str | None = None
    """Customized toolkit classname."""
    # MCP传输方式
    mcp_transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    """MCP transport."""
    # MCP客户端会话读取超时秒数，设置较大值以避免超时异常
    mcp_client_session_timeout_seconds: int = 20
    """The read timeout passed to the MCP ClientSession. We set it bigger to avoid timeout expections."""


# 上下文管理器配置类
class ContextManagerConfig(ConfigBaseModel):
    # 上下文管理器名称
    name: str | None = None
    # 上下文管理器配置字典
    config: dict | None = Field(default_factory=dict)


# 环境配置类
class EnvConfig(ConfigBaseModel):
    # 环境名称
    name: str | None = None
    # 环境配置字典
    config: dict | None = Field(default_factory=dict)


# 代理总配置类，包含所有代理类型的配置
class AgentConfig(ConfigBaseModel):
    # 代理总体配置
    """Overall agent config"""

    # 代理类型：简单、管弦乐队、编排器或工作力
    type: Literal["simple", "orchestra", "orchestrator", "workforce"] = "simple"
    """Agent type"""

    # simple agent config
    # 简单代理配置
    # 简单代理配置
    # 模型配置，包含模型提供商、模型设置、模型参数
    model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Model config, with model_provider, model_settings, model_params"""
    # 代理配置信息
    # 代理基本配置信息
    agent: ProfileConfig = Field(default_factory=ProfileConfig)
    """Agent profile config"""
    # 上下文管理器配置
    # 上下文管理器配置
    context_manager: ContextManagerConfig = Field(default_factory=ContextManagerConfig)
    """Context manager config"""
    # 环境配置
    # 环境配置
    env: EnvConfig = Field(default_factory=EnvConfig)
    """Env config"""
    # 工具包配置字典
    # 工具包配置映射表
    toolkits: dict[str, ToolkitConfig] = Field(default_factory=dict)
    """Toolkits config"""
    # 简单代理的最大轮数，此参数来源于@openai-agents
    # 简单代理的最大对话轮数
    max_turns: int = 50
    """Max turns for simple agent. This param is derived from @openai-agents"""
    # 停止工具名称列表，此参数来源于@openai-agents
    # 遇到这些工具时停止执行
    stop_at_tool_names: list[str] | None = None
    """Stop at tools for simple agent. This param is derived from @openai-agents"""

    # orchestra agent config
    # 管弦乐队代理配置
    # 管弦乐队代理配置
    # 规划者模型配置
    planner_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Planner model config"""
    # 规划者配置字典
    # 规划者具体配置
    planner_config: dict = Field(default_factory=dict)
    """Planner config (dict)\n
    - `examples_path`: path to planner examples json file"""
    # 工作者配置字典
    # 工作者配置映射表
    workers: dict[str, "AgentConfig"] = Field(default_factory=dict)
    """Workers config"""
    # 工作者信息列表，包含{name, desc, strengths, weaknesses}
    # 工作者详细信息列表
    workers_info: list[dict] = Field(default_factory=list)
    """Workers info, list of {name, desc, strengths, weaknesses}\n
    - `name`: worker name
    - `desc`: worker description
    - `strengths`: worker strengths
    - `weaknesses`: worker weaknesses"""
    # 报告者模型配置
    # 报告者模型配置
    reporter_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Reporter model config"""
    # 报告者配置字典
    # 报告者具体配置
    reporter_config: dict = Field(default_factory=dict)
    """Reporter config (dict)\n
    - `template_path`: template Jinja2 file path, with `question` and `trajectory` variables"""

    # workforce agent config
    # 工作力代理配置
    # 工作力规划者模型配置
    workforce_planner_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Workforce planner model config"""
    # 工作力规划者配置字典
    # 工作力规划者具体配置
    workforce_planner_config: dict = Field(default_factory=dict)
    """Workforce planner config (dict)"""
    # 工作力分配者模型配置
    # 工作力分配者模型配置
    workforce_assigner_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Workforce assigner model config"""
    # 工作力分配者配置字典
    # 工作力分配者具体配置
    workforce_assigner_config: dict = Field(default_factory=dict)
    """Workforce assigner config (dict)"""
    # 工作力回答者模型配置
    # 工作力回答者模型配置
    workforce_answerer_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Workforce answerer model config"""
    # 工作力回答者配置字典
    # 工作力回答者具体配置
    workforce_answerer_config: dict = Field(default_factory=dict)
    """Workforce answerer config (dict)"""
    # 工作力执行者代理配置字典
    # 工作力执行者代理配置映射表
    workforce_executor_agents: dict[str, "AgentConfig"] = Field(default_factory=dict)
    """Workforce executor agents config"""
    # 工作力执行者配置字典
    # 工作力执行者具体配置
    workforce_executor_config: dict = Field(default_factory=dict)
    """Workforce executor config (dict)"""
    # 工作力执行者信息列表，包含{name, desc, strengths, weaknesses}
    # 工作力执行者详细信息列表
    workforce_executor_infos: list[dict] = Field(default_factory=list)
    """Workforce executor infos, list of {name, desc, strengths, weaknesses}"""

    # orchestrator agent config
    # 编排器代理配置
    # 编排器路由器代理配置
    orchestrator_router: "AgentConfig" = None
    """Orchestrator router agent config"""
    # 编排器配置字典
    # 编排器具体配置
    orchestrator_config: dict = Field(default_factory=dict)
    """Orchestrator config (dict)\n
    - `name`: name of the orchestrator-workers system
    - `examples_path`: path to planner examples. default utu/data/plan_examples/chain.json
    - `additional_instructions`: additional instructions for planner
    - `add_chitchat_subagent`: whether to add chitchat subagent. default True"""
    # 规划者模型配置
    # 编排器中的规划者模型配置
    orchestrator_model: ModelConfigs = Field(default_factory=ModelConfigs)
    """Planner model config"""
    # 工作者配置字典
    # 编排器中的工作者配置映射表
    orchestrator_workers: dict[str, "AgentConfig"] = Field(default_factory=dict)
    """Workers config"""
    # 工作者信息列表，包含{name, description}
    # 编排器中的工作者详细信息列表
    orchestrator_workers_info: list[dict] = Field(default_factory=list)
    """Workers info, list of {name, description}"""
