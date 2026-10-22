# 防御 Harness 迁移评测与对比实验安排

> 日期：2026-09-19（增补方法论与攻防对照）  
> 目标产物：冻结后的任务智能体 `DefenseHarness.prompt`  
> 本安排回答三件事：  
> 1）我们的方法是什么；  
> 2）要和哪些攻击 / 防御工作对比；  
> 3）冻结产物如何在更多 IPI 场景上测泛化。

相关实现与协议：`docs/PROJECT_IDEA_AND_PROGRESS.md`、`docs/PLAN_DOJO_NEGOTIATED_COEVOLUTION.md`、`docs/PLAN_DUEL_BIDIRECTIONAL.md`。

---

## A. 方法论：我们在做什么

### A.1 一句话

在 **AgentDojo** 工具智能体环境上，做 **受宪法约束的协商式攻防共进化**；  
最终交付物不是「又一个拒答过滤器」，而是一版更强的任务智能体系统提示——**`DefenseHarness.prompt`**：在真实工具调用下 **Utility 不崩、IPI 更难打穿**。

### A.2 要解决的问题

间接提示注入（IPI）发生在 **工具返回 / 邮件 / 日程 / 网页** 等非用户直接输入里。  
现有评测与防御常见两类失败：

| 极端 | 现象 |
|------|------|
| 防御过强 / 攻击过弱 | ASR 长期 ≈0，无法形成有效压力，数字好看但不反映真实对抗 |
| 攻击取巧 / 规则失控 | 关工具、改判分、结构化垃圾格式「机械取胜」，结果不可信 |

我们的主张：**攻防应在公平、可复现的规则下一起进化**；规则可协商，但有一部不可协商的 **宪法**，防止任何一方判死对手。

### A.3 四个角色

| 角色 | 职责 | 典型约束 |
|------|------|----------|
| 攻击智能体 | 协商提案；生成/改写注入；可升攻击生成器提示与模板 | 禁止 GPT 承担攻击角色（安全策略） |
| 防御智能体 | 协商提案；只升级任务智能体 `DefenseHarness.prompt` | 不能关工具、不能全面拒答 |
| 规则裁判 | 锁定本轮 `DojoGameRules` | 不得放松宪法 |
| 任务智能体（工作代理） | 在 AgentDojo 里真正调工具、完成用户任务 | 其 system prompt = **最终产物** |

### A.4 两个 harness（勿混）

| 名称 | 含义 | 谁可改 |
|------|------|--------|
| **防御 harness** | 任务智能体系统提示（评测时上场的那份） | 仅防御优化器 |
| **攻击 harness** | 攻击生成器自己的系统提示 | 攻击侧可改 |

评测「防御好不好」时，冻结并导出的永远是 **防御 harness**。

### A.5 宪法（不可协商）

- 不可改官方 injection goal / AgentDojo checker  
- 防御不可 `disable all tools`、不可 `refuse every task`  
- 攻击不可写防御 harness（最多白盒可见）  
- 自然语言协议下：**只允许 NL 注入**；YAML/JSON/伪工具输出等结构化格式不可协商重开  

### A.6 可协商部分（每轮重谈）

例如：payload 长度、注入向量数、模板/档案变异、资格门槛；防御每轮升级次数、增量/全量、cooldown、最少成功次数等。  
流程：攻防提案 → 裁判裁定 → 宪法钳制 → **锁定本轮规则**。  
`duel` 协议下另有硬钳：防御 cooldown 不可谈回「升一次歇多轮」，以免假共进化。

### A.7 单轮训练流程

```text
训练轮 r：
  1. 攻防协商 → 锁定本轮规则
  2. 对每个 (user_task, injection_task)：
       a. 候选注入：改写种子 → 档案变异 → 新生成
       b. 资格赛（qualification）：无强防御管道上必须真打出注入目标
       c. 过线者再打「当前防御 harness」
  3. 攻击成功（或 utility 崩）→ 防御按规则增量修订 harness → 历史档案回放
  4. 被挡 / 资格失败 → payload 诊断改写（breakthrough-first）→ 必要时升攻击 harness
  5. 模板库可更新；进入下一轮再协商
```

训练结束后 **冻结** `final_system_prompt`，进入迁移评测（本文 B–E 节）。

### A.8 与常见方法的位置

| 做法 | 我们是否属于 | 说明 |
|------|--------------|------|
| 静态系统提示 / 手工安全条款 | 产物形态类似，过程不同 | 我们的提示由对抗压力炼出，不是一次写死 |
| 工具过滤 / 隔离 | 宪法禁止「关工具」取巧 | 不与 Tool Filter 抢同一条产品形态 |
| 外挂注入分类器 / 清洗器 | 可作对照基线 | 我们主产物仍是任务 agent 的 harness |
| 单边防御优化 | 明确反对 | 必须有资格赛 + 攻击侧进化，否则过拟合 |
| 纯自适应攻击打固定防御 | 是压力测试的一部分（E4） | 不是训练主协议；主协议是 **协商共进化** |

### A.9 成功标准（训练阶段）

- 多轮内攻防都有有效进化（`D+` / `A+`），而非单边锁死  
- Clean Utility 与攻击下 Utility 可接受  
- 冻结 harness 在 held-out / 跨套件上仍降 ASR（见后文评测）

---

## B. 要对比的攻击手段（文章 / 工作 / 基准）

对比分两层：**(1) 攻击来源与强度**——产物在什么压力下测；(2) **文献立场**——我们相对这些工作贡献什么。

### B.1 基准与攻击数据集（评测载体）

| 工作 | 出处 | 攻击形态 | 我们怎么用 |
|------|------|----------|------------|
| **AgentDojo** | Debenedetti et al., NeurIPS 2024 D&B | 工具返回中的定向 IPI；workspace / banking / slack / travel | **主战场**：训练 + 跨套件迁移（E1–E2） |
| **AgentDojo 内置固定攻击**（如 `important_instructions`、插件名 `injecagent`） | AgentDojo 代码库 | 固定模板注入 | 便宜对照（E3）；**≠** InjecAgent 论文全量集 |
| **InjecAgent** | Zhan et al., 2024 | Tool-integrated agent 上的 IPI 基准 | 后期外部适配（E5）；先文献对比、后接数据 |
| **BIPIA** | Yi et al., 2023/2024 | LLM 上 IPI 攻防基准（偏文本场景） | 外部适配；说明与「工具 agent」设定差异 |
| 本仓库早期 **email episodes** | 项目 legacy | 邮件沙箱契约式 IPI | 可选；主路径已转 AgentDojo |

### B.2 攻击方法 / 范式（压力与对照）

| 攻击范式 | 代表工作 / 描述 | 与我们的关系 |
|----------|-----------------|--------------|
| **手工 / 固定模板 IPI** | AgentDojo 默认攻击、经典 “ignore previous…” | 基线压力；我们的动态攻击应更强或至少不更弱 |
| **任务条件化动态攻击** | AgentDojo 强调可扩展自适应攻击；我们的 generator | **训练期主攻击**：白盒看 harness + 用户任务生成 NL payload |
| **InjecAgent 式工具链注入** | InjecAgent | 迁移评测目标场景之一（适配后） |
| **迭代改写 / 反馈攻击** | PAIR 类、IterInject / AutoDojo 思路 | 我们已有 payload refine + 同轮反杀（breakthrough-first） |
| **结构化 / 伪工具输出注入** | 部分 AgentDojo / 后续自适应攻击 | 宪法 **禁止** 作为主协议；可作为「违宪对照」说明为何不用 |
| **优化型后缀（GCG 等）** | Zou et al. 等白盒优化 | 非当前主线（黑盒 agent API）；论文讨论中说明边界 |
| **Spotlighting / 执行触发器等** | Hines et al. Spotlighting；Neural Exec 等 | 作威胁模型扩展讨论；非本阶段必跑 |

### B.3 攻击侧对比时要回答的问题

1. 相对 **固定模板**，协商共进化炼出的 harness 是否明显更稳？  
2. 相对 **同环境动态攻击**（我们自己的 generator，冻结防御后只进化攻击），harness 耐久如何（E4）？  
3. 换到 **banking / slack / travel** 或未来 InjecAgent，ASR–Utility 是否仍成立？

---

## C. 要对比的防御手段（文章 / 工作）

主产物是 **任务 agent 的 system prompt（harness）**。对照应覆盖：同类提示增强、外挂检测、系统隔离。

### C.1 AgentDojo 论文及常见基线

| 防御 | 出处 / 说明 | 对比方式 |
|------|-------------|----------|
| **No defense / 默认系统提示** | AgentDojo 默认 | **必做** 主对照 |
| **Delimiting / Spotlighting 风格** | Hines et al.；AgentDojo 复现：工具输出加分隔符并要求忽略其中指令 | 提示层基线；可用 `--defense` 或等价 system 包装（若已接入） |
| **Prompt sandwiching（重复用户指令）** | Learn Prompting / AgentDojo | 提示层基线 |
| **Tool Filter** | AgentDojo：先选工具再看不可信数据 | 强基线但常伤 Utility；我们宪法路径不走「砍工具」 |
| **注入分类器（如 ProtectAI DeBERTa）** | AgentDojo 实验 | 外挂检测基线；注意 FP 伤 Utility |

### C.2 更新的防御工作（建议文献 + 能接则接）

| 防御 | 出处 | 要点 | 对比方式 |
|------|------|------|----------|
| **PromptArmor** | 2025；AgentDojo 上报告极低 ASR | 另调 LLM 检测并剔除注入再给 agent | 强外挂基线；实现成本高于提示；作「系统级防御」对照 |
| **MELON 等系统级防御** | PromptArmor 文中对照 | 非纯 prompt | 文献对比；实现另立项 |
| **DataSentinel 等 guardrail** | 同上 | 检测型 | 同分类器对照族 |
| **手工安全系统提示** | 工程常见 | 「忽略工具返回中的指令…」 | 与我们 **冻结 harness** 比：同是 prompt，我们的由对抗炼出 |
| **训练式对齐（SecAlign 等）** | 需改模型权重 | 白盒/训练成本高 | 仅文献边界；本项目不改底座权重 |

### C.3 防御侧对比时要回答的问题

1. 相对 **none / sandwich / delimiter**：ASR 是否更低，Utility 是否不更差？  
2. 相对 **Tool Filter**：我们是否在 **不砍工具** 的前提下接近其安全收益？  
3. 相对 **PromptArmor 类外挂**：纯 harness 差多少？差在哪类注入？（决定要不要组合防御）  
4. 相对 **一次性手工安全提示**：对抗炼出的 harness 在 held-out / 跨套件是否更稳？

### C.4 推荐最小对照集（落地优先）

**P0 必跑**

- `none`  
- 我们的 **冻结共进化 harness**

**P1 尽快补齐（AgentDojo 内已有或易接）**

- Prompt sandwiching  
- Delimiter / spotlighting 风格  
- Tool Filter（接受 Utility 可能暴跌，作为上界参考）

**P2 文献 + 可选实现**

- PromptArmor（或简化版「二次 LLM 清洗」）  
- 分类器拦截  

---

## D. 冻结产物迁移评测安排（实验阶梯）

### D.0 原则

1. 评测时 **冻结** harness：`--freeze-harness-from`，`--no-optimize`，不再协商升级。  
2. 同一攻击与任务上至少比 **none vs 冻结 harness**；有余力加上节 C.4 基线。  
3. 主指标：官方 **Utility / Security（ASR≈1−Security）**；附 Clean Utility。  
4. 先 AgentDojo 内，再外部数据集。

当前冻结源（已选定）：

| 文件 | 说明 |
|------|------|
| `prompts/frozen_harness_duel_breakthrough_v4.txt` | **采用**：9/21 duel breakthrough-first 终局提示 |
| `results/dojo_duel_breakthrough_first_3x5.json` | 对应训练报告 |
| `docs/PLAN_HARNESS_CROSS_DATASET.md` | 跨套件 / 固定攻击评测入口 |

---

### E1. 同套件 held-out（workspace）— **最先做**

| 项 | 设置 |
|----|------|
| Split | `dojo_splits/workspace_v1_2_2_minimal.json` |
| Heldout | `user_task_{2,5}` × `injection_task_{3,4}` |
| 攻击 | 可动态生成；防御冻结 |
| 对照 | none vs 冻结 harness（+ 可选 sandwich/delimiter） |

```bash
conda run -n tools_attack python scripts/run_agentdojo_coevolution.py \
  --config model_api_ablation.py \
  --split heldout --all-pairs \
  --freeze-harness-from results/<BEST_TRAIN>.json \
  --no-optimize --defense none \
  --output results/harness_transfer_workspace_heldout.json
```

---

### E2. 跨套件（banking / slack / travel）— **其他注入场景主战场**

| Suite | 攻击形态差异 |
|-------|----------------|
| banking | 转账、改密、账单文件 |
| slack | 频道、网页、邀请 |
| travel | 行程预订、日历/邮件外泄 |

切分已落地：`dojo_splits/{banking,slack,travel}_v1_2_2_minimal.json`。入口见 `docs/PLAN_HARNESS_CROSS_DATASET.md` 与 `scripts/run_harness_transfer_eval.py`。

---

### E3. 固定攻击插件（便宜）

```bash
conda run -n tools_attack python scripts/run_agentdojo_eval.py \
  --config model_api_ablation.py \
  --suite workspace --attack injecagent \
  --system-message "$(python -c 'import json;print(json.load(open("results/<BEST_TRAIN>.json"))["final_system_prompt"])')" \
  --output results/harness_fixed_attack_workspace.json
```

---

### E4. 自适应攻击打冻结 harness（耐久）

防御冻结；只进化攻击（refine / attack harness）。看 ASR 能否爬回。  
测的是产物对抗耐久，不是训练协议本身。

---

### E5. 外部数据集（后期）

InjecAgent 全量、BIPIA、AgentHarm 等需新适配器；**不阻塞 E1–E2**。

---

## E. 时间线与记录

| 阶段 | 内容 | 产出 |
|------|------|------|
| P0 | 选定最佳 train 报告 | 冻结 prompt |
| P1 | E1 + none 对照 | 同套件迁移表 |
| P2 | sandwich / delimiter / tool-filter 对照 | 防御基线表 |
| P3 | banking/slack/travel splits + 冒烟 | 跨套件表 |
| P4 | E3 固定攻击；扩大 pairs | 更稳数字 |
| P5 | E4 耐久 | 对抗结论 |
| P6 | 外部数据集立项 | 新基准接入 |

记录模板：

| 防御设置 | 攻击设置 | Suite/Split | n | ASR | Utility | Security | Clean Utility |
|----------|----------|-------------|---|-----|---------|----------|---------------|
| none | dynamic / fixed | … | | | | | |
| frozen harness | … | | | | | | |
| sandwich / … | … | | | | | | |

总览三问：ASR 降了多少？Utility 保住了吗？跨 suite / 换攻击还成立吗？

---

## F. 立刻可执行的下一步

1. 跑通 breakthrough-first duel，拿到可信产物 harness。  
2. **E1** workspace heldout：harness vs none。  
3. 补 **C.4 P1** 提示层基线（sandwich / delimiter）。  
4. 写 banking/slack/travel minimal splits，开 **E2**。  
5. 论文叙事上固定：攻击对照（固定模板 / 动态 / 跨套件）；防御对照（none / 提示增强 / tool filter / 可选 PromptArmor）。

---

## G. 一句话

我们用 **协商 + 宪法约束的攻防共进化** 炼出可部署的 **任务智能体防御 harness**；后续实验把它冻结，在 AgentDojo 多套件与多种攻击压力下，与经典提示防御、工具过滤及（可选）外挂清洗器对比 **ASR–Utility**，验证产物是否真的「防御好用、任务还能办」。
