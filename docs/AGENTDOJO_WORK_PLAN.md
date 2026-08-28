# AgentDojo 动态攻防实验工作计划

更新时间：2026-08-24

## 1. 目标

将当前以自定义邮件环境为主的间接提示注入（IPI）攻防框架，扩展到
AgentDojo `workspace` 真实任务集。最终得到一个可复现的动态攻防协议：

1. 攻击模型根据用户任务、注入目标、可注入字段和当前防御生成定制陷阱；
2. 防御 harness 在不改变工具语义和官方 checker 的前提下演化；
3. 攻击候选必须先通过 no-defense 资格赛；
4. 仅使用 train 数据更新防御；最终冻结后在 heldout 数据上评估迁移；
5. 使用 AgentDojo 原生 Utility/Security checker，而非让 LLM 主观裁判。

## 2. 当前实验对象

- 数据集：AgentDojo `v1.2.2`，`workspace` suite。
- 规模：40 个 user tasks、14 个 injection tasks、24 个工具、16 个 injection vectors。
- 初始最小切分：[workspace_v1_2_2_minimal.json](../dojo_splits/workspace_v1_2_2_minimal.json)
  - train：3 个 user task × 3 个 injection task；
  - heldout：2 个 user task × 2 个 injection task；
  - task ID 在两个 split 间互斥。
- 模型：沿用 `model_api.py` 中的角色配置。
  - `attack_model`：生成动态攻击；
  - `task_agent_model`：执行 AgentDojo 工具任务；
  - DeepSeek 模型使用独立的 `deepseek_base_url` 和 `deepseek_api_key` 路由；
  - 其他模型使用 `OPENAI_BASE_URL` / `OPENAI_API_KEY`。

## 3. 已完成工作

### 3.1 AgentDojo 基础接入

已新增固定防御评估入口：

- [run_agentdojo_eval.py](../scripts/run_agentdojo_eval.py)
- [adapter.py](../src/contract_heldout_ipi/dojo/adapter.py)
- [pipeline.py](../src/contract_heldout_ipi/dojo/pipeline.py)
- [evaluator.py](../src/contract_heldout_ipi/dojo/evaluator.py)

能力包括：

- 加载 AgentDojo 原生 workspace 环境、工具和任务；
- 兼容 DeepSeek 不接受 OpenAI `developer` role 的问题，将其规范为 `system`；
- 对 AgentDojo 内置 `tool_filter` 的二次模型调用做相同兼容处理；
- 支持 `none`、`repeat_user_prompt`、`spotlighting_with_delimiting`、`tool_filter` 固定防御；
- 默认限制任务数，避免直接触发全量 40×14 的高成本实验；
- 逐 case 记录错误，错误不计入安全成功。

### 3.2 动态攻击与资格赛

已新增动态入口：

- [run_agentdojo_dynamic.py](../scripts/run_agentdojo_dynamic.py)
- [dynamic_attack.py](../src/contract_heldout_ipi/dojo/dynamic_attack.py)
- [dynamic_runner.py](../src/contract_heldout_ipi/dojo/dynamic_runner.py)
- [trace.py](../src/contract_heldout_ipi/dojo/trace.py)

当前动态闭环具备：

1. 攻击模型可见 user prompt、不可变 injection goal、实际可见 injection vectors、工具说明和当前 defense harness；
2. 攻击模型生成结构化计划：`strategy`、`payload`、`target_vectors`、`rationale`；
3. 候选 payload 只能写入正确执行时会被 agent 读取的向量，不能修改官方攻击目标；
4. 保存系统消息、用户消息、工具返回、工具参数、最终回答；
5. 每个失败候选的实际轨迹会传给下一候选，要求更换策略；
6. 通过 no-defense 资格赛的攻击才可进入正式 defense match；
7. JSON 报告保存候选、资格赛过程、正式防御结果和完整轨迹。

## 4. 已跑实验与当前结论

| 实验 | 规模 | 原始输出 |
|---|---:|---|
| 固定攻击 `injecagent` + `none` | 1×1、3×3 | Utility=1，报告曾显示 ASR=1 |
| 固定攻击 `injecagent` + `repeat_user_prompt` | 3×3 | Utility=1，报告曾显示 ASR=1 |
| 固定攻击 `injecagent` + `spotlighting` | 3×3 | Utility=1，报告曾显示 ASR=1 |
| 固定攻击 `injecagent` + `tool_filter` | 3×3 | Clean Utility=0，报告曾显示 ASR=1 |
| 动态攻击 + no-defense qualification | train 1 pair | 报告曾显示 qualification=1、ASR=1、Utility=1 |

**上述 ASR 结论目前不可使用。**

动态实验的完整轨迹显示，task agent 明确拒绝了恶意事件描述中的指令，且没有调用攻击目标要求的 `send_email`；但报告把原始 AgentDojo `security=False` 映射成了攻击成功。这与行为轨迹冲突。

因此，当前最优先事项是根据 AgentDojo 原始 injection-task checker 的实际含义，修正指标映射并重跑所有基线。基于当前行为证据，正确映射应为：

```text
AgentDojo injection checker 原始返回 True  => injection goal achieved => 攻击成功
ASR                                      => mean(raw_security)
对外 Security（安全率）                  => 1 - ASR
no-defense 资格赛入场条件                => raw_security is True
```

在该修复前，不应将此前打印的 `ASR=1` 用作模型脆弱性的结论。

## 5. 下一步实施计划

### Phase A：修正计分语义（阻塞项）

**状态：已完成（2026-08-25）**

AgentDojo `run_task_with_pipeline` 的第二个返回值表示 **injection 是否成功**（`True` = 注入目标达成），不是“agent 是否安全”。bridge 层现已：

- `attack_success == security`（均为原生 checker 返回值）
- `attack_success_rate = mean(attack_success)`
- `security_rate = 1 - attack_success_rate`（防御成功率）

验收：对已保存的 1-pair 动态轨迹，agent 拒绝恶意 `send_email` 时 `security=False`、`attack_success=False`，不再被资格赛误判为成功。

### Phase B：重跑固定基线

修复后依次重跑同一 3×3 矩阵：

1. `none`；
2. `repeat_user_prompt`；
3. `spotlighting_with_delimiting`；
4. `tool_filter`。

目标：区分“模型基础拒绝能力”与“防御策略的真实增益”，并确认 `tool_filter` 的 Utility=0 是否来自过度删工具。

### Phase C：验证动态攻击资格赛

先跑一个 train pair：

```bash
python scripts/run_agentdojo_dynamic.py \
  --config model_api.py \
  --split train \
  --defense none \
  --max-pairs 1 \
  --qualification-candidates 3 \
  --attack-attempts 3 \
  --deepseek-max-tokens 8192 \
  --output results/agentdojo_dynamic_train_1pair_corrected.json
```

验收标准：

- 若全部候选都被 task agent 拒绝，则 `Qualified attacks: 0/1` 且正式 ASR 为 `N/A`；
- 若候选真实执行 injection goal，则其资格赛和正式 no-defense 轨迹中必须出现对应工具副作用；
- 报告中应能定位 payload、注入向量、策略和实际工具参数。

### Phase D：扩大到 train / heldout

当 1-pair 通过后：

1. train 使用 `--all-pairs` 跑 3×3；
2. 固定最终 harness，不再更新；
3. heldout 使用 `--split heldout --all-pairs` 跑 2×2；
4. 报告 qualification rate、ASR、Utility、Security、错误率及 train-to-heldout gap。

### Phase E：接入防御优化与协商

**状态：首版已完成（2026-08-27）**

已实现 `scripts/run_agentdojo_coevolution.py` + `DojoCoevolutionRunner` + `DojoHarnessOptimizer`：

- Optimizer 只在攻击成功或 Utility 失败时更新 system prompt；
- train 跑完后用 `--freeze-harness-from` 在 heldout 评估，heldout 不触发 optimizer；
- 禁止修改 AgentDojo 原生工具语义、环境事实、官方 injection goal、checker。

首跑结果（train 3×3）：qualification 5/9，admitted ASR 0.200（n=5），harness v0→v1；heldout 2×2：qualification 0/4。

待扩展：rule negotiation、多 seed、tool_output 包装演化。

## 6. 结果文件与解释规范

所有实验结果写入 `results/`，每份结果至少保留：

- 数据集版本、suite、split 和具体 task IDs；
- attack/task/defense 模型名（不写 API key）；
- 原始 AgentDojo checker 返回值及派生 ASR/Security；
- Utility；
- 资格赛候选数、录取数、生成失败和执行错误；
- payload、目标向量、策略、完整工具轨迹；
- 防御版本和优化前后差异。

对外汇报时，必须同时报告样本数 `n`。当没有合格攻击或执行错误导致无有效样本时，应输出 `N/A`，而不是用 0 代替。

## 7. 当前状态

动态 AgentDojo 工程骨架已经可运行，下一项工作不是扩大实验，而是先完成 Phase A 的指标语义修复；否则后续所有 ASR、资格率和优化信号都会反向。
