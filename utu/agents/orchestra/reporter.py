import pathlib

from agents import RunResultStreaming

from ...agents.llm_agent import LLMAgent
from ...config import AgentConfig
from ...utils import FileUtils
from .common import AnalysisResult, OrchestraStreamEvent, OrchestraTaskRecorder


# 报告者代理类，负责分析任务执行结果并生成最终报告
class ReporterAgent:
    # 初始化报告者代理
    def __init__(self, config: AgentConfig):
        # 保存代理配置信息
        self.config = config
        # 初始化 LLM 代理，专门用于生成最终分析报告
        self.llm = LLMAgent(model_config=config.reporter_model, name="reporter")
        # 加载并初始化报告模板
        self.template = self._get_template()

    # 获取代理的名称属性
    @property
    def name(self) -> str:
        # 优先从配置中获取报告者名称，缺省为 "reporter"
        return self.config.reporter_config.get("name", "reporter")

    # 获取报告模板的实现方法
    def _get_template(self):
        # 尝试从配置中获取自定义模板路径
        template_path = self.config.reporter_config.get("template_path", None)
        # 如果指定了路径且文件存在，则使用该路径
        if template_path and pathlib.Path(template_path).exists():
            template_path = pathlib.Path(template_path)
        else:
            # 否则使用系统默认的报告模板路径
            template_path = "agents/orchestra/reporter_sp.j2"
        # 通过文件工具加载 Jinja2 模板对象
        return FileUtils.get_jinja_template(template_path)

    # 分析执行结果并生成汇总报告
    async def report(self, task_recorder: OrchestraTaskRecorder) -> AnalysisResult:
        # 分析子任务的结果，返回报告
        """analyze the result of a subtask, return a report"""
        # 使用记录器中的任务描述和执行轨迹渲染报告提示词
        query = self.template.render(question=task_recorder.task, trajectory=task_recorder.get_trajectory_str())
        # 将报告生成的起始事件放入事件队列
        task_recorder._event_queue.put_nowait(OrchestraStreamEvent(name="report_start"))
        # 以流式方式运行 LLM 生成报告内容
        res = self.llm.run_streamed(query)
        # 异步处理流式输出并转发事件
        await self._process_streamed(res, task_recorder)
        # 根据 LLM 的最终输出创建分析结果对象
        analysis_result = AnalysisResult(output=res.final_output)
        # 将报告生成的完成事件及其结果放入队列
        task_recorder._event_queue.put_nowait(OrchestraStreamEvent(name="report", item=analysis_result))
        return analysis_result

    # 处理流式运行结果，并将事件实时转发到任务记录器的队列
    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: OrchestraTaskRecorder):
        # 异步遍历流式响应中的所有事件
        async for event in run_result_streaming.stream_events():
            # 将每个流式事件非阻塞地放入记录器的事件队列
            task_recorder._event_queue.put_nowait(event)
