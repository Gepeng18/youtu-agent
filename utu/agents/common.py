import asyncio
import traceback
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

from agents import Agent, RunResult, StreamEvent, TResponseInputItem

from ..utils import AgentsUtils, get_logger

logger = get_logger(__name__)


# 队列完成哨兵类，用于标记流式事件队列的结束
# from agents._run_impl import QueueCompleteSentinel
class QueueCompleteSentinel:
    pass


# 支持流式事件的基类，提供事件队列管理和任务取消功能
@dataclass
class DataClassWithStreamEvents:
    _is_complete: bool = False
    _stored_exception: Exception | None = field(default=None, repr=False)

    # Queues that the background run_loop writes to
    _event_queue: asyncio.Queue[StreamEvent] = field(default_factory=asyncio.Queue, repr=False)
    # Store the asyncio tasks that we're waiting on
    _run_impl_task: asyncio.Task[Any] | None = field(default=None, repr=False)

    # 取消流式运行，停止所有后台任务并标记运行完成
    # 取消流式运行，停止所有后台任务并标记运行完成
    def cancel(self) -> None:
        """Cancels the streaming run, stopping all background tasks and marking the run as
        complete."""
        # 取消所有正在运行的任务
        self._cleanup_tasks()  # Cancel all running tasks
        # 标记运行完成以停止事件流
        self._is_complete = True  # Mark the run as complete to stop event streaming

        # 可选：清空事件队列以防止处理过时事件
        # Optionally, clear the event queue to prevent processing stale events
        while not self._event_queue.empty():
            self._event_queue.get_nowait()

    # 异步生成器，提供流式事件的迭代访问
    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        while True:
            # 检查是否有错误发生
            self._check_errors()
            if self._stored_exception:
                logger.debug("Breaking due to stored exception")
                self._is_complete = True
                break

            # 如果运行完成且队列为空，则退出循环
            if self._is_complete and self._event_queue.empty():
                break

            try:
                # 从队列异步获取事件项
                item = await self._event_queue.get()
            except asyncio.CancelledError:
                logger.debug("Breaking due to asyncio.CancelledError")
                break

            # 如果是队列完成哨兵，检查错误并退出
            if isinstance(item, QueueCompleteSentinel):
                self._event_queue.task_done()
                # 检查错误，以防队列因异常而完成
                # Check for errors, in case the queue was completed due to an exception
                self._check_errors()
                break

            yield item
            self._event_queue.task_done()

        self._cleanup_tasks()

        if self._stored_exception:
            raise self._stored_exception

    # 清理后台任务
    def _cleanup_tasks(self):
        if self._run_impl_task and not self._run_impl_task.done():
            self._run_impl_task.cancel()

    # 检查任务是否出现异常
    def _check_errors(self):
        # 检查任务是否有异常
        # Check the tasks for any exceptions
        if self._run_impl_task and self._run_impl_task.done():
            run_impl_exc = self._run_impl_task.exception()
            if run_impl_exc and isinstance(run_impl_exc, Exception):
                # if isinstance(run_impl_exc, AgentsException) and run_impl_exc.run_data is None:
                #     run_impl_exc.run_data = self._create_error_details()
                logger.error(f"run_impl_exc: {run_impl_exc}")
                logger.error(
                    "".join(traceback.format_exception(type(run_impl_exc), run_impl_exc, run_impl_exc.__traceback__))
                )
                self._stored_exception = run_impl_exc

    # 转换为字典格式，排除私有属性
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


# 任务记录器，继承流式事件支持，用于记录代理运行过程和结果
@dataclass
class TaskRecorder(DataClassWithStreamEvents):
    # 任务描述
    task: str = ""
    # 追踪ID，用于跟踪运行过程
    trace_id: str = ""
    # 输入内容，可以是字符串或结构化输入项列表
    input: str | list[TResponseInputItem] = field(default_factory=dict)

    # 来自RunResultStreaming的字段
    # 最终输出结果
    # from RunResultStreaming
    final_output: str = ""
    # 最后使用的代理
    last_agent: Agent[Any] | None = None

    # 记录代理轨迹的列表
    trajectories: list = field(default_factory=list)  # record agent trajectories
    # 原始运行结果列表
    raw_run_results: list[RunResult] = field(default_factory=list)

    # 附加信息
    # additional infos
    additional_infos: dict = field(default_factory=dict)

    # 转换为输入列表格式
    def to_input_list(self) -> list[TResponseInputItem]:
        return self.get_run_result().to_input_list()

    # 添加运行结果到记录器中
    def add_run_result(self, run_result: RunResult, agent_name: str = None):
        # 添加原始运行结果
        self.raw_run_results.append(run_result)
        # 从运行结果中提取轨迹信息
        self.trajectories.append(
            AgentsUtils.get_trajectory_from_agent_result(run_result, agent_name or run_result.last_agent.name)
        )
        # sync with RunResultStreaming fields
        # 与RunResultStreaming字段同步
        self.last_agent = run_result.last_agent
        self.final_output = run_result.final_output

    # 获取最新的运行结果
    def get_run_result(self) -> RunResult:
        return self.raw_run_results[-1]

    # 设置最终输出结果
    def set_final_output(self, final_output: str):
        self.final_output = final_output

    # 设置附加信息（当前未使用）
    # set additional infos. NOT USED NOW!
    def set_attr(self, key: str, value: Any):
        self.additional_infos[key] = value

    # 获取附加信息的值
    def get_attr(self, key: str) -> Any:
        return self.additional_infos.get(key)
