from collections.abc import Callable
from functools import wraps
from typing import Any

from sqlmodel import SQLModel, select

from ..utils import SQLModelUtils, get_logger

logger = get_logger(__name__)


# 数据库依赖装饰器，在执行数据库操作前检查其可用性
def require_db(safe: bool = True) -> Callable:
    # 装饰器，用于在执行数据库操作前检查数据库可用性。
    """Decorator to check database availability before executing database operations.

    参数：
    Args:
        safe: 如果为 True，失败时返回 None；如果为 False，抛出异常
        safe: If True, return None on failure; if False, raise exception
    """

    def decorator(func: Callable) -> Callable:
        # 使用 wraps 保留原始函数的元数据
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 检查数据库是否可用
            if not SQLModelUtils.check_db_available():
                # 如果是安全模式
                if safe:
                    # 记录调试信息并根据函数名返回默认值
                    logger.debug(f"Database not available, skipping {func.__name__}")
                    return None if func.__name__ != "add" else False
                else:
                    # 非安全模式抛出运行时错误
                    raise RuntimeError("Database is not available")

            try:
                # 尝试执行原始数据库操作函数
                return func(*args, **kwargs)
            except Exception as e:
                # 发生异常时，如果是安全模式则记录警告并返回默认值
                if safe:
                    logger.warning(f"Database operation {func.__name__} failed: {e}")
                    return None if func.__name__ != "add" else False
                else:
                    # 否则重新抛出异常
                    raise

        return wrapper

    return decorator


# 数据库服务类，提供基础的增删改查操作
class DBService:
    # 向数据库添加一条或多条记录
    @staticmethod
    @require_db(safe=True)
    def add(data: SQLModel | list[SQLModel]) -> bool:
        # 向数据库添加一条或多条记录。
        """Add one or more records to database.

        参数：
        Args:
            data: 要添加的 SQLModel 实例或列表
            data: SQLModel instance(s) to add

        返回：
        Returns:
            bool: 保存成功返回 True，失败返回 False
            bool: True if successfully saved, False if failed
        """
        # 将输入统一规范化为列表格式
        # Normalize input to list
        if isinstance(data, SQLModel):
            data = [data]
        elif isinstance(data, list):
            # 如果是空列表，记录警告并返回失败
            if not data:
                logger.warning("Empty data list provided")
                return False
            # 确保列表中的项是 SQLModel 实例
            assert isinstance(data[0], SQLModel), "data must be a SQLModel or list of SQLModel"
        else:
            # 输入类型错误时抛出异常
            raise ValueError("data must be a SQLModel or list of SQLModel")

        # 保存到数据库
        # Save to database
        # 创建数据库会话并执行添加操作
        with SQLModelUtils.create_session() as session:
            session.add_all(data)
            # 提交事务
            session.commit()
        # 记录保存成功的调试日志
        logger.debug(f"Successfully saved {len(data)} record(s) to database")
        return True

    # 根据可选过滤器查询记录
    @staticmethod
    @require_db(safe=True)
    def query(model_class: type[SQLModel], filters: dict = None) -> list[SQLModel]:
        # 查询记录，支持可选过滤器。
        """Query records with optional filters.

        返回：
        Returns:
            list[SQLModel]: 匹配的记录列表，如果数据库不可用则返回 None
            list[SQLModel]: List of matching records, or None if DB unavailable
        """
        # 创建数据库会话
        with SQLModelUtils.create_session() as session:
            # 构造基础查询语句
            stmt = select(model_class)
            # 如果提供了过滤器字典，则遍历并添加 where 条件
            if filters:
                for key, value in filters.items():
                    stmt = stmt.where(getattr(model_class, key) == value)
            # 执行查询并返回所有结果
            return session.exec(stmt).all()

    # 根据 ID 获取单条记录
    @staticmethod
    @require_db(safe=True)
    def get_by_id(model_class: type[SQLModel], id: int) -> SQLModel | None:
        # 根据 ID 获取单条记录。
        """Get a single record by ID.

        返回：
        Returns:
            SQLModel | None: 找到则返回记录，未找到或数据库不可用则返回 None
            SQLModel | None: The record if found, None if not found or DB unavailable
        """
        # 创建数据库会话并使用 get 方法查询
        with SQLModelUtils.create_session() as session:
            return session.get(model_class, id)

    # 更新已存在的记录
    @staticmethod
    @require_db(safe=True)
    def update(data: SQLModel) -> bool:
        # 更新一条记录。
        """Update a record.

        返回：
        Returns:
            bool: 更新成功返回 True，失败返回 False
            bool: True if successfully updated, False if failed
        """
        # 创建数据库会话
        with SQLModelUtils.create_session() as session:
            # 将对象添加到会话中（SQLModel 的 add 会处理更新）
            session.add(data)
            # 提交事务
            session.commit()
            # 刷新对象状态
            session.refresh(data)
        return True

    # 从数据库删除指定的记录
    @staticmethod
    @require_db(safe=True)
    def delete(data: SQLModel) -> bool:
        # 删除一条记录。
        """Delete a record.

        返回：
        Returns:
            bool: 删除成功返回 True，失败返回 False
            bool: True if successfully deleted, False if failed
        """
        # 创建数据库会话并执行删除操作
        with SQLModelUtils.create_session() as session:
            session.delete(data)
            # 提交事务
            session.commit()
        return True
