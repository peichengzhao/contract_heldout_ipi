# Dojo 协商式共演化方案（定稿）

> 日期：2026-09-05  
> 状态：**已与用户对齐；V1 代码已落地（协商器 + runner 接入 + CLI）**  
> 环境：AgentDojo workspace 为唯一评测环境  
> 旧 `game/` email 线：**不维护**（只复用「协商→锁定」流程思想）

---

## 1. 目标

攻击者与防御者**每轮协商游戏规则并锁定**，在 AgentDojo 上公平对抗，避免「防御一进化、攻击再进化也无效」的单边锁死；最终产物是更强的任务 agent **`DefenseHarness.prompt`**。

---

## 2. 概念钉死

| 名称 | 定义 |
|------|------|
| **Defense harness** | 任务 agent 的 `DefenseHarness.prompt`（评测时的 system message） |
| **Attack harness** | 攻击生成器自己的 system prompt（`AttackHarness.prompt`） |
| **可协商规则** | 本轮攻防权限与升级节奏（`DojoGameRules`） |
| **Constitution** | 不可协商硬约束 |

---

## 3. 权限边界

### 3.1 防御者唯一可进化产物

- **只能改**：`DefenseHarness.prompt`
- **不能**：禁工具、拒绝一切任务、改 injection goal、改官方 checker、改攻击侧任何资产

### 3.2 攻击者可改（尽量大）

- 攻击生成器 system prompt（attack harness）
- payload 内容与格式策略（受本轮锁定规则约束）
- 模板库（template library）
- archive / archive mutation
- 注入向量选择（含扩展 injectable surface，若本轮规则允许）
- **不能改**：防御 harness（最多白盒可见）

### 3.3 Constitution（不可协商）

- 不可改 injection goal
- 不可改 AgentDojo 官方 checker / scoring
- 不可 `disable all tools`
- 不可 `refuse every task` / 必须尝试完成用户任务
- 攻击不可写防御 harness

---

## 4. 每轮流程

```text
Round r:
  1. 协商 DojoGameRules（攻防轮流提案 → judge 裁定 → constitution 校验锁定）
  2. 在锁定规则下跑 AgentDojo train pairs：
       攻击生成 → qualification（no-defense 入场）→ 对当前 harness 评测
  3. 按本轮规则：
       - 防御：仅允许修订 DefenseHarness（次数/模式由规则门控）
       - 攻击：可改 attack prompt / templates / archive / payload / vectors
  4. archive + league replay
  5. Round r+1 → 再次协商（规则可持续修正）
```

**协商时机：每轮都谈。**

---

## 5. V1 规则对象（DojoGameRules）

### AttackRules（宽）

- `max_payload_chars`, `max_injected_vectors`
- `natural_language_only` / `structured_payloads_allowed` / `allowed_payload_formats`
- `may_use_multi_vector`, `expand_injectable_vectors`
- `may_update_attack_prompt`, `may_update_templates`, `archive_mutation_enabled`
- `qualification_success_min`（仍受合理上下界约束）

### DefenseRules（窄）

- `may_update_harness`
- `defense_update_mode`: `incremental` | `full`
- `max_defense_updates_this_round` / `cooldown` / `min_successes_before_update`
- `max_harness_chars`
- **不提供**禁工具 / 拒任务字段（constitution 直接禁止）

### 开局默认

可用现有 `arms_race` knobs 作为 **baseline 提案起点**，但每轮协商后以锁定规则为准。

---

## 6. 与现有代码的接法

| 现有 | 用途 |
|------|------|
| `game/negotiation.py` 流程 | 抄「攻防提案 → judge → lock」，字段换成 Dojo |
| `dojo/coevolution_runner.py` | 每轮开头插入协商；enforce 锁定规则 |
| `dojo/optimizer.py` | 仅在规则允许时改 `DefenseHarness` |
| `dojo/attack_optimizer.py` / templates / archive | 攻击侧大权限入口 |
| `DojoAttackRules` / `CoevolutionProtocol` | 收敛进 `DojoGameRules`，或由锁定规则覆盖 |
| `final_system_prompt` 导出 | 保留；建议按轨迹选最优 harness 版本 |

旧 email `GameRules`（`mutable_email_fields` 等）**不接入 Dojo 主路径**。

---

## 7. 实验规模

1. **先**：约 3 pairs × 3 rounds（验证每轮协商 + 不单边锁死）
2. **再**：9 pairs × 8 rounds 炼最终 harness

成功信号（过程）：

- 防御升级后，攻击仍常有非零压力（AdmASR / LgASR 不永久归零）
- Clean Utility 不崩
- 最终 harness 可导出且可读

---

## 8. 实施顺序

1. `DojoGameRules` + `DojoConstitution` + 校验
2. `DojoGameNegotiator`（每轮协商）
3. 接入 `DojoCoevolutionRunner` + CLI
4. enforce：防御只动 harness；攻击侧大权限
5. 小规模 smoke → 全量

---

## 9. 用户确认记录

- harness = 任务 agent 的 `DefenseHarness.prompt` → **确认**
- 攻击者可改：attack system prompt、模板库、archive、payload、注入向量选择 → **确认**
- 每轮协商、Dojo 唯一环境、不管旧 game 线 → **确认**

---

## 10. 四角色模型配置（开发时以配置文件为准）

参考文件：`model_api_ablation.py`

| 角色 | 配置变量 | 当前模型 | 用途 |
|------|----------|----------|------|
| 攻击 Agent | `ATTACK_MODEL` | `deepseek-v4-flash-0731` | 协商提案（攻）、生成/改写 payload、attack prompt、模板、archive、向量 |
| 防御 Agent | `DEFENSE_MODEL` | `deepseek-v4-flash-0731` | 协商提案（防）、只升级 `DefenseHarness.prompt` |
| Judge Agent | `JUDGE_MODEL` | `gpt-5.6-sol` | 每轮规则裁定 + constitution 锁定 |
| 工作 Agent | `WORK_AGENT` | `deepseek-v4-flash-0731` | AgentDojo 任务执行（戴 harness） |

**硬约束：GPT 系列不得承担攻击者角色**（含 attack generator、payload refine、attack-side 协商提案）。攻击侧只用 DeepSeek（或其它非 GPT 模型）。Judge 可用 GPT。

实现时注意：现有 `ExperimentModelConfig` 用的是 `TASK_AGENT_MODEL` / `OPTIMIZER_MODEL` 等旧字段名；接入时应把 `WORK_AGENT` → 工作/task agent，`DEFENSE_MODEL` → 防御协商与 harness 优化，不要混用。
