"""
- [ ] error tracing
"""

import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..simple_agent import SimpleAgent
from .data import Subtask, WorkspaceTaskRecorder

logger = get_logger(__name__)

PROMPTS = FileUtils.load_prompts("agents/workforce/executor.yaml")


# 执行者代理类，负责执行由规划者分配的任务
class ExecutorAgent:
    # 执行者代理，执行由规划者分配的任务
    """Executor agent that executes tasks assigned by the planner.

    - 待办：自我反思
    - TODO: self-reflection
    """

    # 初始化执行者代理
    def __init__(self, config: AgentConfig, workforce_config: AgentConfig):
        # 保存执行代理配置
        self.config = config
        # 创建 SimpleAgent 实例作为底层的执行代理
        self.executor_agent = SimpleAgent(config=config)

        # 获取工作力执行配置
        executor_config = workforce_config.workforce_executor_config
        # 最大重试次数，默认为 1
        self.max_tries = executor_config.get("max_tries", 1)
        # 是否返回摘要结果
        self.return_summary = executor_config.get("return_summary", False)

        # 反思历史记录，用于重试时的参考
        self.reflection_history = []

    # 执行分配的任务并检查结果
    async def execute_task(
        self,
        recorder: WorkspaceTaskRecorder,
        task: Subtask,
    ) -> None:
        # 执行任务并检查结果
        """Execute the task and check the result."""
        # 设置任务状态为 "in progress"（进行中）
        task.task_status = "in progress"

        # 初始化尝试次数和结果变量
        tries = 1
        final_result = None
        executor_res = None
        # 在最大重试次数范围内进行循环尝试
        while tries <= self.max_tries:
            try:
                # 清除聊天历史，确保每次尝试都是干净的上下文
                self.executor_agent.clear_input_items()  # clear chat history!

                # * 1. 任务执行
                # * 1. Task execution
                if tries == 1:
                    # 第一次尝试时使用基础的用户提示词
                    user_prompt = PROMPTS["TASK_EXECUTE_USER_PROMPT"].format(
                        overall_task=recorder.overall_task,
                        overall_plan=recorder.formatted_task_plan,
                        task_name=task.task_name,
                        task_description=task.task_description,
                    )
                else:
                    # 后续尝试时加入反思信息
                    user_prompt = PROMPTS["TASK_EXECUTE_WITH_REFLECTION_USER_PROMPT"].format(
                        overall_task=recorder.overall_task,
                        overall_plan=recorder.formatted_task_plan,
                        task_name=task.task_name,
                        task_description=task.task_description,
                        previous_attempts=self.reflection_history[-1] if self.reflection_history else "",
                    )
                # 运行执行者代理处理任务，并保存聊天历史
                executor_res = await self.executor_agent.run(user_prompt, save=True)  # save chat history!
                # 获取最终输出结果
                final_result = executor_res.final_output

                # * 2. 任务检查
                # * 2. Task check
                # 格式化任务检查提示词
                task_check_prompt = PROMPTS["TASK_CHECK_PROMPT"].format(
                    task_name=task.task_name,
                    task_description=task.task_description,
                )
                # 运行执行者代理进行自我检查，不保存该检查过程的聊天历史
                response_content = await self.executor_agent.run(task_check_prompt)  # do not save chat history!
                # 解析检查结果，如果成功则跳出重试循环
                if self._parse_task_check_result(response_content.final_output):
                    # 记录成功日志
                    logger.info(f"Task '{task.task_name}' completed successfully.")
                    break

                # * 3. 任务反思（失败时）
                # * 3. Task reflection (when failed)
                # 格式化反思提示词
                reflection_prompt = PROMPTS["TASK_REFLECTION_PROMPT"].format(
                    task_name=task.task_name,
                    task_description=task.task_description,
                )
                # 运行执行者代理生成失败反思，不保存该过程的聊天历史
                reflection_res = await self.executor_agent.run(reflection_prompt)  # do not save chat history!
                # 将反思内容添加到历史记录中
                self.reflection_history.append(reflection_res.final_output)
                # 记录反思信息
                logger.info(f"Task '{task.task_name}' reflection: {reflection_res.final_output}")

                # 记录重试警告并递增尝试次数
                logger.warning(f"Task '{task.task_name}' not completed. Retrying... (Attempt {tries}/{self.max_tries})")
                tries += 1

            except Exception as e:
                # 记录执行过程中的异常并递增尝试次数
                logger.error(f"Error executing task `{task.task_name}` on attempt {tries}: {str(e)}")
                tries += 1
                # 如果超过最大重试次数，则设置错误结果并跳出循环
                if tries > self.max_tries:
                    final_result = f"Task execution failed: {str(e)}"
                    break

        # 如果执行结果仍为空，说明所有尝试都失败了
        if executor_res is None:
            # 记录失败日志并设置任务状态为 "failed"（失败）
            logger.error(f"Task `{task.task_name}` execution failed after {tries} attempts!")
            task.task_result = final_result
            task.task_status = "failed"
            return

        # 将执行结果添加到记录器的轨迹中
        recorder.add_run_result(executor_res.get_run_result(), "executor")  # add executor trajectory
        # 设置任务最终结果并标记为 "completed"（完成）
        task.task_result = final_result
        task.task_status = "completed"

        # 如果配置了返回摘要，则对执行结果进行汇总
        if self.return_summary:
            # 警告：重置指令很危险！不要在此处使用！
            # WARNING: reset instructions is dangerous! DONOT use here!
            # self.executor_agent.set_instructions(PROMPTS["TASK_SUMMARY_SYSTEM_PROMPT"])
            # 格式化摘要提示词
            summary_prompt = PROMPTS["TASK_SUMMARY_USER_PROMPT"].format(
                task_name=task.task_name,
                task_description=task.task_description,
            )
            # 运行执行者代理生成摘要
            summary_response = await self.executor_agent.run(summary_prompt)
            # 将摘要生成轨迹添加到记录器中
            recorder.add_run_result(summary_response.get_run_result(), "executor_summary")  # add executor trajectory
            # 更新任务结果为摘要内容
            task.task_result_detailed, task.task_result = summary_response.final_output, summary_response.final_output
            # 记录摘要成功日志
            logger.info(f"Task result summarized: {task.task_result_detailed} -> {task.task_result}")

    # 解析任务检查结果
    def _parse_task_check_result(self, response) -> bool:
        # 使用正则表达式匹配 <task_check> 标签中的内容
        task_check_result = re.search(r"<task_check>(.*?)</task_check>", response, re.DOTALL)
        # 如果匹配成功且内容为 "yes"，则返回 True，表示任务完成
        if task_check_result and task_check_result.group(1).strip().lower() == "yes":
            return True
        # 否则返回 False
        return False
