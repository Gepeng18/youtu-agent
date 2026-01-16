from typing import TypeVar

from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import BaseModel

from .agent_config import AgentConfig, ToolkitConfig
from .eval_config import EvalConfig
from .model_config import ModelConfigs
from .practice_config import TrainingFreeGRPOConfig

# 配置类型变量，用于泛型约束
TConfig = TypeVar("TConfig", bound=BaseModel)


# 配置加载器类，负责加载各种配置文件
class ConfigLoader:
    # 配置加载器
    """Config loader"""

    # 默认配置路径
    config_path = "../../configs"
    # Hydra版本基线
    version_base = "1.3"

    # 加载配置文件并转换为字典格式
    @classmethod
    # 加载配置文件并转换为字典格式
    def _load_config_to_dict(cls, name: str = "default", config_path: str = None) -> dict:
        # 使用默认配置路径或指定的路径
        config_path = config_path or cls.config_path
        # 初始化Hydra配置并加载指定配置
        with initialize(config_path=config_path, version_base=cls.version_base):
            cfg = compose(config_name=name)
            OmegaConf.resolve(cfg)
        # 返回字典而不是DictConfig，避免JSON序列化错误
        # return dict instead of DictConfig -- avoid JSON serialization error
        return OmegaConf.to_container(cfg, resolve=True)

    # 测试方法：直接加载配置为指定类实例
    # @classmethod
    # def _load_config_to_cls(cls, name: str, config_type: Type[TConfig] = None) -> TConfig:
    #     # 测试中
    #     # TESTING
    #     cfg = cls._load_config_to_dict(name)
    #     return config_type(**cfg)

    # 加载模型配置
    @classmethod
    # 加载模型配置
    def load_model_config(cls, name: str = "base") -> ModelConfigs:
        # 加载模型配置
        """Load model config"""
        # 从model目录加载配置并创建ModelConfigs实例
        cfg = cls._load_config_to_dict(name, config_path="../../configs/model")
        return ModelConfigs(**cfg)

    # 加载工具包配置
    @classmethod
    # 加载工具包配置
    def load_toolkit_config(cls, name: str = "search") -> ToolkitConfig:
        # 加载工具包配置
        """Load toolkit config"""
        # 从tools目录加载配置并创建ToolkitConfig实例
        cfg = cls._load_config_to_dict(name, config_path="../../configs/tools")
        return ToolkitConfig(**cfg)

    # 加载代理配置
    @classmethod
    # 加载代理配置
    def load_agent_config(cls, name: str = "default") -> AgentConfig:
        # 加载代理配置
        """Load agent config"""
        # 如果名称不以"agents/"开头，则添加前缀
        if not name.startswith("agents/"):
            name = "agents/" + name
        # 从configs目录加载配置并创建AgentConfig实例
        cfg = cls._load_config_to_dict(name, config_path="../../configs")
        return AgentConfig(**cfg)

    # 加载评估配置
    @classmethod
    # 加载评估配置
    def load_eval_config(cls, name: str = "default") -> EvalConfig:
        # 加载评估配置
        """Load eval config"""
        # 如果名称不以"eval/"开头，则添加前缀
        if not name.startswith("eval/"):
            name = "eval/" + name
        # 从configs目录加载配置并创建EvalConfig实例
        cfg = cls._load_config_to_dict(name, config_path="../../configs")
        return EvalConfig(**cfg)

    # 加载无训练GRPO配置
    @classmethod
    # 加载无训练GRPO配置
    def load_training_free_grpo_config(cls, name: str = "default") -> TrainingFreeGRPOConfig:
        # 加载无训练GRPO配置
        """Load training-free GRPO config"""
        # 如果名称不以"practice/"开头，则添加前缀
        if not name.startswith("practice/"):
            name = "practice/" + name
        # 从configs目录加载配置并创建TrainingFreeGRPOConfig实例
        cfg = cls._load_config_to_dict(name, config_path="../../configs")
        return TrainingFreeGRPOConfig(**cfg)
