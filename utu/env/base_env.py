import abc
import datetime

from agents import Tool


# 环境基类，为代理提供环境接口
class _BaseEnv:
    # 代理的环境接口
    """Environment interface for agents."""

    # 获取环境的当前状态（抽象方法，由子类实现）
    @abc.abstractmethod
    def get_state(self) -> str:
        # 获取环境的当前状态
        """Get the current state of the environment."""
        raise NotImplementedError

    # 获取环境中可用的工具列表（抽象方法，由子类实现）
    @abc.abstractmethod
    async def get_tools(self) -> list[Tool]:
        # 获取环境中可用的工具
        """Get the tools available in the environment."""
        raise NotImplementedError

    # 构建环境的基础流程
    async def build(self):
        # 构建环境
        """Build the environment."""

    # 清理环境的基础流程
    async def cleanup(self):
        # 清理环境
        """Cleanup the environment."""

    # 异步上下文管理器入口，自动执行环境构建
    async def __aenter__(self):
        # 执行环境构建逻辑
        await self.build()
        return self

    # 异步上下文管理器出口，自动执行环境清理
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 执行环境清理逻辑
        await self.cleanup()


# 基础环境实现，提供默认的环境功能
class BaseEnv(_BaseEnv):
    # 获取环境状态，默认返回空字符串
    def get_state(self) -> str:
        # 返回当前环境的状态描述
        return ""

    # 获取环境中可用的工具列表，默认返回空列表
    async def get_tools(self) -> list[Tool]:
        # 返回环境中预置的工具
        return []

    # ------------------------------------------------------------------------
    # 获取当前时间的静态方法
    # 获取当前时间，返回格式化的字符串
    @staticmethod
    def get_time() -> str:
        # 返回格式为 "YYYY-MM-DD HH:MM:SS" 的当前本地时间字符串
        return datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
