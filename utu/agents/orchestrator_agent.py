"""
- [x] feat: support multi-turn chat
- [ ] feat: add reporter
    additional instructions for the last agent! (as the reporter)
- [ ] feat: replan
- [ ] add name & description for all agent types
- [x] 功能：支持多轮对话
- [ ] 功能：添加报告者
    为最后一个代理添加额外指令！（作为报告者）
- [ ] 功能：重新规划
- [ ] 为所有代理类型添加名称和描述
"""

import asyncio

from agents import trace

from ..config import AgentConfig, ConfigLoader
from ..db import DBService, TrajectoryModel
from ..utils import AgentsUtils, FileUtils, get_logger
from .common import QueueCompleteSentinel
from .orchestrator import ChainPlanner, OrchestratorStreamEvent, Recorder, Task
from .simple_agent import SimpleAgent

logger = get_logger(__name__)
PROMPTS = FileUtils.load_prompts("agents/orchestrator/chain.yaml")


# 编排器代理：使用链式规划器管理多个子代理的协作执行
class OrchestratorAgent:
    # 初始化编排器代理实例
    def __init__(self, config: AgentConfig):
        # 处理配置
        self._handle_config(config)

        # 初始化编排器和工作者
        self.orchestrator = ChainPlanner(self.config)
        self.workers = self._setup_workers()

    # 代理名称属性
    # 代理名称属性
    @property
    def name(self) -> str:
        # 从配置中获取名称，默认使用"BaseOrchestratorAgent"
        return self.config.orchestrator_config.get("name", "BaseOrchestratorAgent")

    # 处理和调整配置
    def _handle_config(self, config: AgentConfig) -> None:
        # 检查是否需要添加闲聊子代理
        add_chitchat_subagent = config.orchestrator_config.get("add_chitchat_subagent", True)
        if add_chitchat_subagent:
            # 加载并添加闲聊代理配置
            config.orchestrator_workers["ChitchatAgent"] = ConfigLoader.load_agent_config("simple/chitchat")
            config.orchestrator_workers_info.append(
                {
                    "name": "ChitchatAgent",
                    "description": "Engages in light, informal conversations and handles straightforward queries. Can optionally invoke search or Python tools for simple fact checks or quick calculations.",  # noqa: E501
                }
            )
        # 保存处理后的配置
        self.config = config

    # 设置并初始化所有工作者代理
    # 设置并初始化所有工作者代理
    def _setup_workers(self) -> dict[str, SimpleAgent]:
        # TODO: 在作为工作者时操纵代理的SP（停止序列？）
        # TODO: manipulate SP of agents when as workers
        workers = {}
        # 遍历配置中的所有工作者，为每个创建SimpleAgent实例
        for name, config in self.config.orchestrator_workers.items():
            # 目前只支持SimpleAgent作为工作者
            assert config.type == "simple", f"Only support SimpleAgent as worker in orchestra agent, get {config}"
            workers[name] = SimpleAgent(config=config)
        return workers

    # 运行编排器代理，执行完整的规划和工作流程
    async def run(self, input: str, history: Recorder = None, trace_id: str = None) -> Recorder:
        # 获取流式结果并消费所有事件
        recorder = self.run_streamed(input, history, trace_id)
        async for _ in recorder.stream_events():
            pass
        return recorder

    # 流式运行编排器代理，返回流式记录器
    def run_streamed(self, input: str, history: Recorder = None, trace_id: str = None) -> Recorder:
        # 生成或使用提供的追踪ID
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        # 创建或从历史记录创建新的记录器
        if history:
            recorder = history.new(input=input, trace_id=trace_id)
        else:
            recorder = Recorder(input=input, trace_id=trace_id)
        # 在后台启动流式处理任务
        recorder._run_impl_task = asyncio.create_task(self._start_streaming(recorder))
        return recorder

    # 启动流式处理，执行规划和任务执行流程
    async def _start_streaming(self, recorder: Recorder):
        with trace(workflow_name=self.name, trace_id=recorder.trace_id):
            try:
                # 处理输入并生成规划
                planner = await self.orchestrator.handle_input(recorder)
                # 如果有规划
                if planner:  # has a plan
                    # 循环执行所有任务
                    while True:
                        # 获取下一个要执行的任务
                        task = await self.orchestrator.get_next_task(recorder)
                        if task is None:
                            logger.error("No task available! This should not happen, please check the planner!")
                            break
                        # 执行任务
                        await self._run_task(recorder, task)
                        # 如果是最后一个任务，设置最终输出并退出
                        if task.is_last_task:
                            recorder.add_final_output(task.result)
                            break
                # log to db
                # 将轨迹记录到数据库
                DBService.add(TrajectoryModel.from_task_recorder(recorder))
            except Exception as e:
                # 发生异常时记录错误并标记完成
                logger.error(f"Error processing task: {str(e)}")
                recorder._event_queue.put_nowait(QueueCompleteSentinel())
                recorder._is_complete = True
                raise e
            finally:
                # 最终确保发送完成信号并标记任务完成
                recorder._event_queue.put_nowait(QueueCompleteSentinel())
                recorder._is_complete = True

    # 执行单个任务
    async def _run_task(self, recorder: Recorder, task: Task):
        # 获取对应的worker代理并构建
        worker = self.workers[task.agent_name]
        await worker.build()
        # build context for task
        # 为任务构建上下文信息
        task_with_context = FileUtils.get_jinja_template_str(PROMPTS["worker"]).render(
            problem=recorder.input,
            plan=recorder.get_plan_str(),
            trajectory=recorder.get_trajectory_str(),
            task=task,
        )
        # add history
        # 添加历史消息
        input = recorder.history_messages + [{"role": "user", "content": task_with_context}]
        # run the task
        # 执行任务
        recorder._event_queue.put_nowait(OrchestratorStreamEvent(name="task.start", item=task))
        result = worker.run_streamed(input)
        # 将工作结果的事件转发到记录器的队列
        async for event in result.stream_events():
            recorder._event_queue.put_nowait(event)
        # 设置任务结果
        task.result = result.final_output  # set result
        recorder._event_queue.put_nowait(OrchestratorStreamEvent(name="task.done", item=task))
        # record trajectory
        # 记录轨迹
        recorder.trajectories.append(AgentsUtils.get_trajectory_from_agent_result(result))
