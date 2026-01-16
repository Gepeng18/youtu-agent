from typing import Literal

# from openai import NOT_GIVEN
from agents import ModelSettings
from pydantic import ConfigDict, Field

from ..utils import EnvUtils
from .base_config import ConfigBaseModel


# 模型提供商配置类，定义模型类型、名称、API 基础地址和密钥
class ModelProviderConfig(ConfigBaseModel):
    # 模型提供商配置
    """config for model provider"""

    # 模型类型，支持：chat.completions, responses, litellm
    type: Literal["chat.completions", "responses", "litellm"] = "chat.completions"
    # 模型类型，支持：chat.completions, responses
    """model type, supported types: chat.completions, responses"""
    # 模型名称，从环境变量中读取默认值
    model: str = EnvUtils.get_env("UTU_LLM_MODEL")
    # 模型名称
    """model name"""
    # 模型提供商的基础 URL 地址
    base_url: str | None = None
    # 模型提供商基础 URL
    """model provider base url"""
    # 模型提供商的 API 密钥
    api_key: str | None = None
    # 模型提供商 API 密钥
    """model provider api key"""


# 模型设置配置类，继承自 openai-agents 的 ModelSettings
class ModelSettingsConfig(ConfigBaseModel, ModelSettings):
    # openai-agents 中的 ModelSettings
    """ModelSettings in openai-agents"""

    # Pydantic 配置，允许任意类型（用于支持 ModelSettings 中的复杂类型）
    model_config = ConfigDict(arbitrary_types_allowed=True)


# 模型基础参数配置类，包含温度、采样比例和并行工具调用等参数
class ModelParamsConfig(ConfigBaseModel):
    # 在 chat.completions 和 responses 中共享的基础参数
    """Basic params shared in chat.completions and responses"""

    # 生成文本的温度参数，用于控制随机性
    temperature: float | None = None
    # 核采样参数（Top-p 采样）
    top_p: float | None = None
    # 是否允许并行工具调用
    parallel_tool_calls: bool | None = None


# 模型总配置类，整合提供商配置、设置配置和参数配置
class ModelConfigs(ConfigBaseModel):
    # 模型整体配置
    """Overall model config"""

    # 模型提供商相关的配置信息
    model_provider: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    # 模型提供商配置
    """config for model provider"""
    # 代理模型的具体设置配置
    model_settings: ModelSettingsConfig = Field(default_factory=ModelSettingsConfig)
    # 代理模型设置配置
    """config for agent's model settings"""
    # 基础模型调用的参数配置，例如在工具或判别器中使用的参数
    model_params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    # 基础模型使用配置，例如工具/判别器中的 `query_one`
    """config for basic model usage, e.g. `query_one` in tools / judger"""

    # 终止最大 token 数，用于截断逻辑
    termination_max_tokens: int | None = None
    # 模型的最大 token 数，用于截断逻辑
    """max tokens for the model, used in truncation logic"""
