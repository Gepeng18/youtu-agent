# `youtu-agent` (utu) 源码阅读指南

## 1. 项目全景概览
`utu` 是一个基于 `openai-agents` SDK 构建的多代理（Multi-Agent）框架。它的核心目标是将复杂的任务通过不同的“编排模式”分发给多个子代理协作完成。

### 目录结构快速入口：
- `utu/agents/`: **核心地带**。包含了所有代理类型的实现（简单代理、管弦乐队模式、工作力模式等）。
- `utu/config/`: **配置中心**。决定了代理使用什么模型、哪些工具以及何种行为。
- `utu/tools/`: **能力扩展**。代理能干什么（搜网页、写文件、跑代码）都在这里定义。
- `utu/env/`: **运行环境**。代理在哪里干活（本地、Docker沙箱、E2B等）。
- `utu/utils/`: **脚手架**。日志、追踪、消息转换等工具类。

---

## 2. 核心主流程阅读路径

建议你按照以下 4 个阶段进行阅读：

### 第一阶段：单代理基石 (`SimpleAgent`)
这是所有复杂模式的“原子”单位。
1. **阅读 `utu/agents/simple_agent.py`**:
   - 关注 `__init__`: 了解如何整合 `model`, `tools`, `env` 和 `context_manager`。
   - 关注 `build()`: 重点看它是如何初始化 `agents` SDK 的 `Agent` 对象的。
   - 关注 `run()`: 了解输入是如何经过处理，最终通过 `Runner.run` 执行的。

### 第二阶段：编排模式（核心差异点）
当你理解了单个代理后，去看 `utu` 是如何让它们协作的。这里有三种主要模式：

1. **管弦乐队模式 (`OrchestraAgent`)**: *适合确定性的流水线任务*。
   - 路径：`utu/agents/orchestra_agent.py` -> `utu/agents/orchestra/`
   - 流程：`Planner` (规划) -> `Worker` (执行) -> `Reporter` (汇总报告)。

2. **链式编排模式 (`OrchestratorAgent`)**: *适合动态的多轮对话和任务切换*。
   - 路径：`utu/agents/orchestrator_agent.py` -> `utu/agents/orchestrator/chain.py`
   - 流程：`Router` (判断) -> `ChainPlanner` (按需生成链式计划) -> 顺序执行任务。

3. **工作力模式 (`WorkforceAgent`)**: *最复杂的自主循环模式*。
   - 路径：`utu/agents/workforce_agent.py` -> `utu/agents/workforce/`
   - 流程：`Planner` -> `Assigner` (分配给最合适的专家) -> `Executor` -> `Check & Replan` (检查并重新规划) -> `Answerer`。

### 第三阶段：配置与驱动 (`Config`)
了解系统是如何动态组装的。
1. **`utu/config/loader.py`**:
   - 它是整个项目的“启动器”，负责从 `configs/` 目录下的 YAML 文件加载配置。
2. **`utu/config/agent_config.py`**:
   - 所有的配置字段定义在这里，通过阅读这个文件，你可以知道一个代理到底有多少“参数”可以调优。

### 第四阶段：能力边界 (`Tools` & `Env`)
1. **`utu/tools/base.py`**:
   - 了解 `AsyncBaseToolkit` 如何封装工具，如何将函数暴露给 LLM。
2. **`utu/env/base_env.py`**:
   - 了解代理如何与外部世界（操作系统、沙箱）交互。

---

## 3. 一个典型的任务执行序列（以 Orchestra 为例）

如果你在调试代码，可以跟踪以下函数调用链路：

1. **入口**: 调用 `OrchestraAgent.run(input)`。
2. **构建**: `agent.build()` 初始化环境和子代理。
3. **规划**: 进入 `plan()`，`PlannerAgent` 调用 LLM 生成 XML 格式的步骤。
4. **循环执行**: 
   - 遍历计划中的子任务。
   - 调用 `SimpleWorkerAgent.work_streamed()`。
   - 底层最终落到 `Runner.run_streamed()`。
5. **汇总**: 调用 `ReporterAgent.report()`，将所有子任务的 `trajectory` (轨迹) 喂给 LLM 生成最终答案。
6. **落库**: `DBService.add()` 将整个执行轨迹保存到数据库。

---

## 4. 阅读建议
- **先看 `common.py`**: 在每个目录下通常都有个 `common.py`，里面定义了数据结构（如 `Task`, `Plan`, `Recorder`），理解了数据结构就理解了逻辑的一半。
- **关注 `TaskRecorder`**: 这是 `utu` 中最重要的对象，它贯穿了整个生命周期，记录了代理“想了什么”和“做了什么”。
- **对照 YAML 看代码**: 打开一个配置（如 `configs/agents/orchestra/base.yaml`），对照着 `AgentConfig` 类看，能极大地帮助你理解配置驱动的原理。
