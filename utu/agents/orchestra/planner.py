import json
import pathlib
import re

from agents import RunResultStreaming

from ...agents.llm_agent import LLMAgent
from ...config import AgentConfig
from ...utils import FileUtils
from .common import AgentInfo, CreatePlanResult, OrchestraStreamEvent, OrchestraTaskRecorder, Subtask


# 输出解析器类，用于解析LLM生成的规划输出
# 输出解析器类，用于解析LLM生成的规划输出
class OutputParser:
    # 初始化输出解析器，设置解析模式
    # 初始化输出解析器，设置解析模式
    def __init__(self):
        # 分析结果的正则表达式模式
        self.analysis_pattern = r"<analysis>(.*?)</analysis>"
        # 计划结果的正则表达式模式
        self.plan_pattern = r"<plan>\s*\[(.*?)\]\s*</plan>"
        # 注释掉的其他模式（下一步和任务完成）
        # self.next_step_pattern = r'<next_step>\s*<agent>\s*(.*?)\s*</agent>\s*<task>\s*(.*?)\s*</task>\s*</next_step>'
        # self.task_finished_pattern = r'<task_finished>\s*</task_finished>'

    # 解析输出文本，返回创建计划结果
    # 解析输出文本，返回创建计划结果
    def parse(self, output_text: str) -> CreatePlanResult:
        # 提取分析结果
        analysis = self._extract_analysis(output_text)
        # 提取计划结果
        plan = self._extract_plan(output_text)
        return CreatePlanResult(analysis=analysis, todo=plan)

    # 提取分析结果文本
    # 提取分析结果文本
    def _extract_analysis(self, text: str) -> str:
        # 使用正则表达式搜索分析标签内容
        match = re.search(self.analysis_pattern, text, re.DOTALL)
        if match:
            # 返回匹配的分析内容并去除空白字符
            return match.group(1).strip()
        return ""

    # 提取计划结果，解析为子任务列表
    # 提取计划结果，解析为子任务列表
    def _extract_plan(self, text: str) -> list[Subtask]:
        # 使用正则表达式搜索计划标签内容
        match = re.search(self.plan_pattern, text, re.DOTALL)
        if not match:
            return []
        plan_content = match.group(1).strip()
        tasks = []
        # 定义任务对象的正则表达式模式
        task_pattern = r'\{"agent_name":\s*"([^"]+)",\s*"task":\s*"([^"]+)",\s*"completed":\s*(true|false)\s*\}'
        # 查找所有匹配的任务
        task_matches = re.findall(task_pattern, plan_content, re.IGNORECASE)
        for agent_name, task_desc, completed_str in task_matches:
            # 将字符串转换为布尔值
            completed = completed_str.lower() == "true"
            # 创建子任务对象并添加到列表
            tasks.append(Subtask(agent_name=agent_name, task=task_desc, completed=completed))
        # check validity
        # 验证有效性
        assert len(tasks) > 0, "No tasks parsed from plan"
        return tasks


# 规划者代理类，负责分析任务并创建执行计划
# 规划者代理类，负责分析任务并创建执行计划
class PlannerAgent:
    # 初始化规划者代理
    # 初始化规划者代理
    def __init__(self, config: AgentConfig):
        # 保存配置
        self.config = config
        # 加载规划器提示词
        self.prompts = FileUtils.load_prompts("agents/orchestra/planner.yaml")

        # 初始化输出解析器
        self.output_parser = OutputParser()
        # 加载规划器示例
        self._load_planner_examples()
        # 加载可用代理
        self._load_available_agents()

    # 加载规划器示例数据
    # 加载规划器示例数据
    def _load_planner_examples(self) -> None:
        # 获取示例文件路径配置
        examples_path = self.config.planner_config.get("examples_path", "")
        # 如果配置的路径存在则使用，否则使用默认路径
        if examples_path and pathlib.Path(examples_path).exists():
            examples_path = pathlib.Path(examples_path)
        else:
            examples_path = pathlib.Path(__file__).parent / "data" / "planner_examples.json"
        # 读取并解析JSON示例文件
        with open(examples_path, encoding="utf-8") as f:
            self.planner_examples = json.load(f)

    # 加载可用的代理信息
    # 加载可用的代理信息
    def _load_available_agents(self) -> None:
        available_agents = []
        # 遍历配置中的工作者信息，创建AgentInfo对象
        for info in self.config.workers_info:
            available_agents.append(AgentInfo(**info))
        self.available_agents = available_agents

    # 代理名称属性
    # 代理名称属性
    @property
    def name(self) -> str:
        # 从配置中获取规划器名称，默认使用"planner"
        return self.config.planner_config.get("name", "planner")

    # 创建任务执行计划
    # 创建任务执行计划
    async def create_plan(self, task_recorder: OrchestraTaskRecorder) -> CreatePlanResult:
        # format examples to string. example: {question, available_agents, analysis, plan}
        # 将示例格式化为字符串。示例格式：{question, available_agents, analysis, plan}
        examples_str = []
        for example in self.planner_examples:
            # 构建每个示例的格式化字符串
            examples_str.append(
                f"Question: {example['question']}\n"
                f"Available Agents: {example['available_agents']}\n\n"
                f"<analysis>{example['analysis']}</analysis>\n"
                f"<plan>{json.dumps(example['plan'], ensure_ascii=False)}</plan>\n"
            )
        # 将所有示例连接成一个字符串
        examples_str = "\n".join(examples_str)
        # 渲染系统提示词（System Prompt）
        sp = FileUtils.get_jinja_template_str(self.prompts["PLANNER_SP"]).render(planning_examples=examples_str)
        # 创建LLM代理用于规划
        llm = LLMAgent(
            name="planner",
            instructions=sp,
            model_config=self.config.workforce_planner_model,
        )
        # 渲染用户提示词（User Prompt）
        up = FileUtils.get_jinja_template_str(self.prompts["PLANNER_UP"]).render(
            available_agents=self._format_available_agents(self.available_agents),
            question=task_recorder.task,
            background_info=await self._get_background_info(task_recorder),
        )
        # 发送规划开始事件
        task_recorder._event_queue.put_nowait(OrchestraStreamEvent(name="plan_start"))
        # 运行流式规划
        res = llm.run_streamed(up)
        # 处理流式输出
        await self._process_streamed(res, task_recorder)
        # 解析规划结果
        plan = self.output_parser.parse(res.final_output)
        # 发送规划完成事件
        task_recorder._event_queue.put_nowait(OrchestraStreamEvent(name="plan", item=plan))
        return plan

    # 格式化可用代理信息为字符串
    # 格式化可用代理信息为字符串
    def _format_available_agents(self, agents: list[AgentInfo]) -> str:
        agents_str = []
        for agent in agents:
            # 根据代理的优势和劣势构建描述字符串
            agents_str.append(
                f"- {agent.name}: {agent.desc}\n  Best for: {agent.strengths}\n"
                if agent.strengths
                else f"  Weaknesses: {agent.weaknesses}\n"
                if agent.weaknesses
                else ""
            )
        # 将所有代理描述连接成一个字符串
        return "\n".join(agents_str)

    # 获取查询的背景信息，默认情况下为空
    # 获取查询的背景信息，默认情况下为空
    async def _get_background_info(self, task_recorder: OrchestraTaskRecorder) -> str:
        """Get background information for the query. Leave empty by default."""
        return ""

    # 处理流式运行结果，将事件转发到任务记录器
    # 处理流式运行结果，将事件转发到任务记录器
    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: OrchestraTaskRecorder):
        # 遍历流式事件并转发到任务记录器的队列
        async for event in run_result_streaming.stream_events():
            task_recorder._event_queue.put_nowait(event)
