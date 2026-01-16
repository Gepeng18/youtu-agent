from ...config import AgentConfig
from ..simple_agent import SimpleAgent
from .common import OrchestraTaskRecorder, Subtask, WorkerResult

# 任务执行的提示模板，包含原始问题、计划、轨迹和当前任务
TEMPLATE = r"""Original Problem:
{problem}

Plan:
{plan}

Previous Trajectory:
{trajectory}

Current Task:
{task}
""".strip()


# 简单工作者代理类，负责执行管弦乐队中的单个子任务
class SimpleWorkerAgent:
    # 初始化简单工作者代理实例
    def __init__(self, config: AgentConfig):
        # 创建 SimpleAgent 实例作为底层执行核心
        self.agent = SimpleAgent(config=config)

    # 异步构建底层执行代理
    async def build(self):
        # 调用底层代理的构建方法，初始化环境和工具
        await self.agent.build()

    # 格式化子任务信息，整合上下文历史
    def _format_task(self, task_recorder: OrchestraTaskRecorder, subtask: Subtask) -> str:
        # 从记录器获取任务规划的字符串表示
        str_plan = task_recorder.get_plan_str()
        # 从记录器获取执行轨迹的字符串表示
        str_traj = task_recorder.get_trajectory_str()
        # 使用预定义模板格式化最终的用户提示词
        return TEMPLATE.format(
            problem=task_recorder.task,
            plan=str_plan,
            trajectory=str_traj,
            task=subtask.task,
        )

    # 以流式方式执行子任务
    def work_streamed(self, task_recorder: OrchestraTaskRecorder, subtask: Subtask) -> WorkerResult:
        # 待办：使用 DataClassWithStreamEvents 封装 WorkerResult
        # TODO: wrap WorkerResult with DataClassWithStreamEvents
        # 格式化增强后的任务描述（包含历史轨迹）
        aug_task = self._format_task(task_recorder, subtask)
        # 调用底层代理的流式运行接口
        run_result_streaming = self.agent.run_streamed(aug_task, trace_id=task_recorder.trace_id)
        # 初始化并返回工作者执行结果对象
        result = WorkerResult(
            task=subtask.task,
            output="",
            trajectory=[],
            stream=run_result_streaming,
        )
        return result
