"""
- [x] setup tracing
- [x] purify logging
- [ ] support stream?
- [x] 设置追踪
- [x] 净化日志
- [ ] 支持流式处理？
"""

from agents import trace

from ..config import AgentConfig, ConfigLoader
from ..utils import AgentsUtils, get_logger
from .workforce import AnswererAgent, AssignerAgent, ExecutorAgent, PlannerAgent, WorkspaceTaskRecorder

logger = get_logger(__name__)


# 工作力代理：包含规划者、分配者、执行者和回答者四个角色的复杂多代理协作系统
class WorkforceAgent:
    # 代理名称
    name = "workforce_agent"

    # 初始化工作力代理
    def __init__(self, config: AgentConfig | str):
        # 初始化工作力代理
        """Initialize the workforce agent"""
        # 如果传入字符串，则从配置文件加载配置
        if isinstance(config, str):
            config = ConfigLoader.load_agent_config(config)
        self.config = config

    # 运行工作力代理，执行完整的规划-分配-执行-回答流程
    async def run(self, input: str, trace_id: str = None) -> WorkspaceTaskRecorder:
        # 生成或使用提供的追踪ID
        trace_id = trace_id or AgentsUtils.gen_trace_id()

        # 记录初始化代理的日志信息
        logger.info("Initializing agents...")
        # 初始化各个角色代理
        planner_agent = PlannerAgent(config=self.config)
        assigner_agent = AssignerAgent(config=self.config)
        answerer_agent = AnswererAgent(config=self.config)
        # 初始化执行者代理组
        executor_agent_group: dict[str, ExecutorAgent] = {}
        for name, config in self.config.workforce_executor_agents.items():
            executor_agent_group[name] = ExecutorAgent(config=config, workforce_config=self.config)

        # 创建工作空间任务记录器
        recorder = WorkspaceTaskRecorder(
            overall_task=input, executor_agent_kwargs_list=self.config.workforce_executor_infos
        )

        with trace(workflow_name=self.name, trace_id=trace_id):
            # * 1. generate plan
            # * 1. 生成计划
            # 记录生成计划的日志信息
            logger.info("Generating plan...")
            await planner_agent.plan_task(recorder)
            # 记录生成的计划详情日志
            logger.info(f"Plan: {recorder.task_plan}")

            # 讨论：是否可以合并.get_next_task和.has_uncompleted_tasks？(while True)
            # DISCUSS: merge .get_next_task and .has_uncompleted_tasks? (while True)
            # 主执行循环：处理所有未完成的任务
            while recorder.has_uncompleted_tasks:
                # * 2. assign tasks
                # * 2. 分配任务
                next_task = await assigner_agent.assign_task(recorder)
                # 记录任务分配结果日志
                logger.info(f"Assign task: {next_task.task_id} assigned to {next_task.assigned_agent}")

                # * 3. execute task
                # * 3. 执行任务
                # 记录开始执行任务的日志
                logger.info(f"Executing task: {next_task.task_id}")
                await executor_agent_group[next_task.assigned_agent].execute_task(recorder=recorder, task=next_task)
                # 记录任务执行结果日志
                logger.info(f"Task {next_task.task_id} result: {next_task.task_result}")
                # 检查任务完成情况
                await planner_agent.plan_check(recorder, next_task)
                # 记录任务检查状态日志
                logger.info(f"Task {next_task.task_id} checked: {next_task.task_status}")

                # * 4. update plan
                # * 4. 更新计划
                # 提前停止检查
                if not recorder.has_uncompleted_tasks:  # early stop
                    break
                # 决定是否更新计划
                plan_update_choice = await planner_agent.plan_update(recorder, next_task)
                # 记录规划更新选择日志
                logger.info(f"Plan update choice: {plan_update_choice}")
                if plan_update_choice == "stop":
                    # 规划者确定总体任务已完成，记录停止执行日志
                    logger.info("Planner determined overall task is complete, stopping execution")
                    break
                elif plan_update_choice == "update":
                    # 记录任务计划已更新的日志
                    logger.info(f"Task plan updated: {recorder.task_plan}")

            # 提取最终答案
            final_answer = await answerer_agent.extract_final_answer(recorder)
            # 记录提取出的最终答案日志
            logger.info(f"Extracted final answer: {final_answer}")
            recorder.set_final_output(final_answer)

            # TODO: 自我评估，将用于下一次尝试！
            # TODO: self-eval, which will be used in next attempt!
            # success = await answerer_agent.answer_check(
            #     question=agent_workspace.overall_task,
            #     model_answer=final_answer,
            #     ground_truth=task["Final answer"]
            # )
        return recorder
