import asyncio
import importlib.util
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from .path import DIR_ROOT


# 获取当前正在运行的事件循环，如果不存在则创建一个新的
def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # 尝试获取当前线程的事件循环
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 如果当前线程没有事件循环，则创建一个新的
        loop = asyncio.new_event_loop()
        # 设置为当前线程的事件循环
        asyncio.set_event_loop(loop)
    return loop


# 将 JSON Schema 转换为 Pydantic 的 BaseModel 类
def schema_to_basemodel(schema: dict, class_name: str = None) -> type[BaseModel]:
    # JSON Schema 类型到 Python 类型的映射表
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    # 根据属性 Schema 获取对应的 Python 类型
    def get_python_type(prop_schema):
        # 获取 Schema 中定义的类型
        prop_type = prop_schema.get("type")

        # 处理数组类型
        if prop_type == "array":
            # 获取数组项的类型，默认为字符串
            item_type = prop_schema.get("items", {}).get("type", "string")
            # 返回 list 泛型类型
            return list[type_map.get(item_type, str)]

        # 返回基础类型映射
        return type_map.get(prop_type, str)

    # 初始化类型注解和字段定义
    annotations = {}
    fields = {}
    # 获取 Schema 中的属性定义
    properties = schema.get("properties", {})
    # 获取必填字段集合
    required_fields = set(schema.get("required", []))

    # 遍历所有属性并构建 Pydantic 字段定义
    for field_name, field_schema in properties.items():
        # 获取该字段的 Python 类型
        annotations[field_name] = get_python_type(field_schema)
        field_kwargs = {}
        # 如果 Schema 中有描述，则添加到字段参数中
        if "description" in field_schema:
            field_kwargs["description"] = field_schema["description"]
        # 如果字段不是必填的，则设置为可选类型并赋予默认值 None
        if field_name not in required_fields:
            field_kwargs["default"] = None
            annotations[field_name] = annotations[field_name] | None
        # 如果有额外参数，则创建 Field 对象
        if field_kwargs:
            fields[field_name] = Field(**field_kwargs)
    
    # 构建类的属性字典，包含类型注解和模块名
    attrs = {
        "__annotations__": annotations,
        "__module__": __name__,
    }
    # 更新字段定义
    attrs.update(fields)

    # 确定类名，优先使用传入的类名，其次使用 Schema 中的 title，最后使用默认名
    class_name = class_name or schema.get("title", "GeneratedModel")
    # 使用 type 动态创建 BaseModel 的子类
    ModelClass = type(class_name, (BaseModel,), attrs)
    return ModelClass


# 从指定的 Python 文件中动态加载类
def load_class_from_file(filepath: str, class_name: str) -> type:
    # 从文件中加载类
    """Load class from file."""
    # 如果不是绝对路径，则相对于 DIR_ROOT 处理
    if not filepath.startswith("/"):
        filepath = str(DIR_ROOT / filepath)

    # 获取绝对路径并提取模块名
    filepath = Path(filepath).absolute()
    module_name = filepath.stem
    # 根据文件位置创建模块规范（Spec）
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    # 检查规范是否成功创建
    if spec is None:
        raise ImportError(f"Could not load spec from file '{filepath}'")

    # 根据规范创建模块对象
    module = importlib.util.module_from_spec(spec)
    # 将模块注册到 sys.modules 中
    sys.modules[module_name] = module

    # 执行模块代码，加载其中的定义
    spec.loader.exec_module(module)

    # 检查模块中是否存在指定的类名
    if hasattr(module, class_name):
        # 返回找到的类对象
        return getattr(module, class_name)
    else:
        # 如果未找到指定的类，抛出属性错误
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'")
