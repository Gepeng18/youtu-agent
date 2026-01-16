import json
import re

from ...config import AgentConfig, ConfigLoader
from ...utils import AgentsUtils, FileUtils, get_logger
from ..common import DataClassWithStreamEvents
from ..llm_agent import LLMAgent
from ..simple_agent import SimpleAgent
from .common import OrchestratorStreamEvent, Plan, Recorder, Task

logger = get_logger(__name__)


# 链式规划器类，负责任务分解和规划
class ChainPlanner:
    # 任务规划器，处理任务分解
    """Task planner that handles task decomposition."""

    # 初始化链式规划器
    def __init__(self, config: AgentConfig):
        # 保存代理配置
        self.config = config
        # 加载编排器链式规划的提示词
        self.prompts = FileUtils.load_prompts("agents/orchestrator/chain.yaml")

        # 如果未配置路由代理，则加载默认的编排器路由配置
        if config.orchestrator_router is None:  # set default
            # 加载默认的路由配置
            config.orchestrator_router = ConfigLoader.load_agent_config("orchestrator/router")
        # 创建路由简单代理实例
        self.router = SimpleAgent(config=config.orchestrator_router)

        # 获取规划器示例文件路径，默认为 "plan_examples/chain.json"
        examples_path = self.config.orchestrator_config.get("examples_path", "plan_examples/chain.json")
        # 加载规划器示例数据
        self.planner_examples = FileUtils.load_json_data(examples_path)
        # 获取额外的指令信息
        self.additional_instructions = self.config.orchestrator_config.get("additional_instructions", "")

    # 处理输入内容并返回生成的计划
    """
    1. 调用 SimpleAgent 的 run_streamed 方法运行路由器
    2. 路由器的输出结果如果以 <plan> 结尾，则创建计划
        2.1 chain.yaml 的 orchestrator_sp 作为系统提示词
        2.2 chain.yaml 的 orchestrator_up 作为用户提示词
    """
    async def handle_input(self, recorder: Recorder) -> None | Plan:
        # 处理输入，返回一个计划
        # handle input. return a plan
        # 使用路由代理处理输入
        async with self.router as router:
            # 整合历史消息和当前用户输入
            input = recorder.history_messages + [{"role": "user", "content": recorder.input}]
            # 以流式方式运行路由器
            res = router.run_streamed(input)
            # 处理流式输出
            await self._process_streamed(res, recorder)
            # 更新对话历史记录
            recorder.history_messages = res.to_input_list()  # update chat history
            # 添加轨迹信息
            # add trajectory
            # 获取并记录路由器的执行轨迹
            recorder.trajectories.append(AgentsUtils.get_trajectory_from_agent_result(res, "router"))
        # 检查是否需要生成计划（通过特殊的 <plan> 标记判断）
        need_plan = res.final_output.strip().endswith("<plan>")  # special token!
        if need_plan:
            # 如果需要，则创建详细的执行计划
            return await self.create_plan(recorder)

    # 根据总体任务和可用代理规划任务
    """
    构建 prompts 调用 llm
    1. chain.yaml 的 orchestrator_sp 作为系统提示词
    2. chain.yaml 的 orchestrator_up 作为用户提示词
    """
    async def create_plan(self, recorder: Recorder) -> Plan:
        # 基于总体任务和可用代理规划任务
        """Plan tasks based on the overall task and available agents."""
        # 将示例格式化为字符串。示例包含：问题、可用代理、分析、计划
        # format examples to string. example: {question, available_agents, analysis, plan}
        examples_str = []
        for example in self.planner_examples:
            # 构建 XML 风格的示例字符串
            examples_str.append(
                f"<question>{example['question']}</question>\n"
                f"<available_agents>{example['available_agents']}</available_agents>\n"
                f"<analysis>{example['analysis']}</analysis>\n"
                f"<plan>{json.dumps(example['plan'], ensure_ascii=False)}</plan>"
            )
        # 将所有示例连接成单个字符串
        examples_str = "\n".join(examples_str)
        # 渲染编排器的系统提示词
        sp = FileUtils.get_jinja_template_str(self.prompts["orchestrator_sp"]).render(planning_examples=examples_str)
        # 创建用于规划的 LLM 代理
        llm = LLMAgent(
            name="orchestrator",
            instructions=sp,
            model_config=self.config.orchestrator_model,
        )
        # 渲染编排器的用户提示词
        up = FileUtils.get_jinja_template_str(self.prompts["orchestrator_up"]).render(
            additional_instructions=self.additional_instructions,
            question=recorder.input,
            available_agents=self.config.orchestrator_workers_info,
            # background_info=await self._get_background_info(recorder),
        )
        # 如果有历史计划，则将其作为上下文输入
        if recorder.history_plan:
            input = recorder.history_plan + [{"role": "user", "content": up}]
        else:
            # 否则直接使用渲染后的用户提示词
            input = up
        # 将规划开始事件放入事件队列
        recorder._event_queue.put_nowait(OrchestratorStreamEvent(name="plan.start"))
        # 流式运行 LLM 代理进行规划
        res = llm.run_streamed(input)
        # 处理流式规划输出
        await self._process_streamed(res, recorder)
        # 解析生成的最终输出为计划对象
        plan = self._parse(res.final_output, recorder)
        # 将规划完成事件及其结果放入事件队列
        recorder._event_queue.put_nowait(OrchestratorStreamEvent(name="plan.done", item=plan))
        # 设置任务列表并记录轨迹信息
        # set tasks & record trajectories
        # 在记录器中添加生成的计划
        recorder.add_plan(plan)
        # 记录规划阶段的执行轨迹
        recorder.trajectories.append(AgentsUtils.get_trajectory_from_agent_result(res, "orchestrator"))
        return plan

    # 解析 LLM 生成的文本为计划对象
    def _parse(self, text: str, recorder: Recorder) -> Plan:
        # 使用正则表达式提取分析部分内容
        match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
        # 如果匹配成功则获取内容并去除首尾空白
        analysis = match.group(1).strip() if match else ""

        # 使用正则表达式提取计划部分的 JSON 列表字符串
        match = re.search(r"<plan>\s*\[(.*?)\]\s*</plan>", text, re.DOTALL)
        # 获取计划内容字符串
        plan_content = match.group(1).strip()
        tasks: list[Task] = []
        # 定义任务对象的正则表达式模式，提取名称和任务描述
        task_pattern = r'\{"name":\s*"([^"]+)",\s*"task":\s*"([^"]+)"\s*\}'
        # 在计划内容中查找所有匹配的任务
        task_matches = re.findall(task_pattern, plan_content, re.IGNORECASE)
        for agent_name, task_desc in task_matches:
            # 创建任务对象并添加到列表中
            tasks.append(Task(agent_name=agent_name, task=task_desc))
        # 确保解析出的任务列表不为空
        # check validity
        assert len(tasks) > 0, "No tasks parsed from plan"
        # 标记最后一个任务，用于流程控制
        tasks[-1].is_last_task = True  # FIXME: polish this
        # 返回构建好的计划对象
        return Plan(input=recorder.input, analysis=analysis, tasks=tasks)

    # 获取下一个待执行的任务，所有的任务都存储在 recorder 的 tasks 属性中
    async def get_next_task(self, recorder: Recorder) -> Task | None:
        # 获取下一个待执行的任务
        """Get the next task to be executed."""
        # 检查是否有可用任务
        if not recorder.tasks:
            raise ValueError("No tasks available. Please create a plan first.")
        # 如果当前任务 ID 超出任务列表长度，说明所有任务已执行完
        if recorder.current_task_id >= len(recorder.tasks):
            return None
        # 获取当前任务对象
        task = recorder.tasks[recorder.current_task_id]
        # 递增任务计数器
        recorder.current_task_id += 1
        return task

    # 处理流式事件并将它们放入记录器的事件队列中
    async def _process_streamed(self, res: DataClassWithStreamEvents, recorder: Recorder):
        # 异步遍历流式生成的事件
        async for event in res.stream_events():
            # 将每个事件非阻塞地放入记录器的队列中
            recorder._event_queue.put_nowait(event)

