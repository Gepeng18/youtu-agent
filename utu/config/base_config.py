from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

# 表示参数的类型，用于表示方法
ReprArgs: type = Iterable[tuple[str | None, Any]]

# 需要安全处理的字段名称
SECURE_FIELDS = ("api_key", "base_url")


# 检查字段名是否需要安全处理
# 检查字段名是否需要安全处理
def if_need_secure(key: str) -> bool:
    # 检查字段名是否包含敏感字段关键词
    return any(f in key.lower() for f in SECURE_FIELDS)


# 生成安全的表示，对敏感字段进行脱敏处理
# 生成安全的表示，对敏感字段进行脱敏处理
def secure_repr(obj: ReprArgs) -> ReprArgs:
    # 遍历所有字段，对敏感字段替换为***
    for k, v in obj:
        if if_need_secure(k):
            # 敏感字段用***代替
            yield k, "***"
        else:
            # 非敏感字段保持原值
            yield k, v


# 配置基础模型类，提供安全的表示方法
class ConfigBaseModel(BaseModel):
    # 配置的基础模型，带有安全的表示方法
    """Base model for config, with secure repr"""

    # 字符串表示方法，返回与__repr__相同的内容
    # 字符串表示方法，返回与__repr__相同的内容
    def __str__(self) -> str:
        # 返回__repr__方法的输出
        return self.__repr__()

    # 对象表示方法，使用安全的字段表示
    # 对象表示方法，使用安全的字段表示
    def __repr__(self) -> str:
        # 生成安全的表示字符串，对敏感字段进行脱敏
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in secure_repr(self.__repr_args__()))})"

    # 模型序列化方法，默认排除None值
    # 模型序列化方法，默认排除None值
    def model_dump(
        self,
        *,
        # 默认排除None值，避免传递temperature=None导致SGLang错误
        exclude_none: bool = True,  # avoid passing temperature=None to avoid SGLang error
        **kwargs,
    ) -> dict[str, Any]:
        # 调用父类的model_dump方法
        return super().model_dump(exclude_none=exclude_none, **kwargs)
