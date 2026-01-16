"""
- [x] 支持规划者和报告者的流式处理
- [x] support streaming for planner & reporter
"""

import asyncio

from agents import trace

from ..config import AgentConfig, ConfigLoader
from ..utils import AgentsUtils, get_logger
from .common import QueueCompleteSentinel
from .orchestra import (
    AnalysisResult,
    CreatePlanResult,
    OrchestraTaskRecorder,
    PlannerAgent,
    ReporterAgent,
    SimpleWorkerAgent,
    Subtask,
    WorkerResult,
)

logger = get_logger(__name__)


# 管弦乐队代理：包含规划者、工作者和报告者三个角色的多代理协作系统
class OrchestraAgent:
    # 初始化管弦乐队代理
    def __init__(self, config: AgentConfig | str):
        # 初始化管弦乐队代理
        """Initialize the orchestra agent"""
        # 如果传入字符串，则从配置文件加载配置
        if isinstance(config, str):
            config = ConfigLoader.load_agent_config(config)
        self.config = config
        # init subagents
        # 初始化子代理
        self.planner_agent = PlannerAgent(config)
        self.worker_agents = self._setup_workers()
        self.reporter_agent = ReporterAgent(config)

    # 设置规划者代理
    def set_planner(self, planner: PlannerAgent):
        # 设置外部传入的规划者代理实例
        self.planner_agent = planner

    # 设置工作者代理字典
    def _setup_workers(self) -> dict[str, SimpleWorkerAgent]:
        # 初始化工作者字典
        workers = {}
        # 遍历配置中的工作者，为每个工作者创建SimpleWorkerAgent实例
        for name, config in self.config.workers.items():
            # 目前只支持SimpleAgent作为工作者
            assert config.type == "simple", f"Only support SimpleAgent as worker in orchestra agent, get {config}"
            # 创建并保存工作者代理实例
            workers[name] = SimpleWorkerAgent(config=config)
        return workers

    # 运行管弦乐队代理，执行完整的规划-工作-报告流程
    async def run(self, input: str, trace_id: str = None) -> OrchestraTaskRecorder:
        # 获取流式结果并消费所有事件
        task_recorder = self.run_streamed(input, trace_id)
        async for _ in task_recorder.stream_events():
            pass
        return task_recorder

    # 流式运行管弦乐队代理，返回流式任务记录器
    def run_streamed(self, input: str, trace_id: str = None) -> OrchestraTaskRecorder:
        # TODO: 错误追踪
        # TODO: error_tracing
        # 生成或使用提供的追踪ID
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        # 创建管弦乐队任务记录器
        task_recorder = OrchestraTaskRecorder(task=input, trace_id=trace_id)
        # Kick off the actual agent loop in the background and return the streamed result object.
        # 在后台启动实际的代理循环，并返回流式结果对象
        task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder))
        return task_recorder

    # 启动流式处理，执行规划-工作-报告的完整流程
    async def _start_streaming(self, task_recorder: OrchestraTaskRecorder):
        with trace(workflow_name="orchestra_agent", trace_id=task_recorder.trace_id):
            try:
                # 第一步：规划阶段
                await self.plan(task_recorder)

                # 【循环】第二步：依次执行每个子任务
                for task in task_recorder.plan.todo:
                    # print(f"> processing {task}")
                    # 获取对应的worker代理并构建
                    worker_agent = self.worker_agents[task.agent_name]
                    await worker_agent.build()
                    # 流式执行工作任务
                    result_streaming = worker_agent.work_streamed(task_recorder, task)
                    # 将工作结果的事件转发到任务记录器的队列
                    async for event in result_streaming.stream.stream_events():
                        task_recorder._event_queue.put_nowait(event)
                    # 设置最终输出和轨迹
                    result_streaming.output = result_streaming.stream.final_output
                    result_streaming.trajectory = AgentsUtils.get_trajectory_from_agent_result(result_streaming.stream)
                    # 添加工作结果到记录器
                    task_recorder.add_worker_result(result_streaming)
                    # print(f"< processed {task}")

                # 第三步：报告阶段，汇总所有结果
                await self.report(task_recorder)

                # 发送完成信号并标记任务完成
                task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
                task_recorder._is_complete = True
            except Exception as e:
                # 发生异常时，标记任务完成并发送完成信号
                task_recorder._is_complete = True
                task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
                raise e

    # 第一步：规划阶段，为任务创建执行计划
    """
    构建prompts，调用llm
    1. 将 planner.yaml 中的 PLANNER_SP 作为 system prompt
    2. 将 planner.yaml 中的 PLANNER_UP 作为 user prompt
    """
    async def plan(self, task_recorder: OrchestraTaskRecorder) -> CreatePlanResult:
        """Step1: Plan"""
        # 调用规划者代理创建执行计划
        plan = await self.planner_agent.create_plan(task_recorder)
        # 验证计划中的所有代理名称都在工作者代理中
        assert all(t.agent_name in self.worker_agents for t in plan.todo), (
            f"agent_name in plan.todo must be in worker_agents, get {plan.todo}"
        )
        logger.info(f"plan: {plan}")
        # 设置任务记录器的计划
        task_recorder.set_plan(plan)
        return plan

    # 第二步：工作阶段，执行单个子任务
    # 第二步：工作阶段
    async def work(self, task_recorder: OrchestraTaskRecorder, task: Subtask) -> WorkerResult:
        """Step2: Work"""
        # 获取对应的worker代理
        worker_agent = self.worker_agents[task.agent_name]
        # 执行工作任务
        result = await worker_agent.work(task_recorder, task)
        # 添加工作结果到记录器
        task_recorder.add_worker_result(result)
        return result

    # 第三步：报告阶段，汇总分析所有执行结果
    """
    构建prompts，调用llm
    将 reporter_sp.j2 作为 user prompt
    """
    async def report(self, task_recorder: OrchestraTaskRecorder) -> AnalysisResult:
        """Step3: Report"""
        # 调用报告者代理生成分析报告
        analysis_result = await self.reporter_agent.report(task_recorder)
        # 添加报告结果到记录器
        task_recorder.add_reporter_result(analysis_result)
        # 设置最终输出
        task_recorder.set_final_output(analysis_result.output)
        return analysis_result
