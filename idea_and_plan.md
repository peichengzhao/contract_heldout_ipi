# Contract-based Held-out Evaluation for IPI

想法整理 + 下一步安排

---

## 1. 问题从哪来

现有 IPI（Indirect Prompt Injection）评测，大多依赖固定题库。

这样做的方便之处是好比较；风险是：

- 防御可以针对可见题调参
- 静态分数容易好看
- 换一批同类型新题，效果可能明显下降

也就是说：高分不一定等于真安全，也可能只是背了题。

---

## 2. 我们的想法（缩小后的版本）

我们不重新定义 IPI，也不先追求最强防御。

我们做的是一套更可靠的评测协议，用来回答：

> 这个防御是真泛化，还是只是过拟合了可见 benchmark？

核心三件套：

### 2.1 Episode Contract

把每一局评测写成机器可读合同，至少包括：

- 用户任务
- 工具与权限
- 可信 / 不可信通道
- 攻击目标
- 攻击成功条件
- 良性任务成功条件
- 预算限制
- train / held-out 划分标签

目的：可复现、可校验、可统一判分。

### 2.2 Referee

Referee 不负责发明攻击或防御，只负责筛局。

至少检查：

- 字段是否完整
- 工具是否可执行
- 成功条件是否可机器判定
- 是否退化（例如禁用全部工具就能赢）
- 是否不可能完成 / 太水

只有通过校验的 episode，才能进入评测集。

### 2.3 Held-out Transfer Gap

把 episodes 分成：

- Train：调防御时可见
- Held-out：调防御时不可见

分别报告：

- ASR（攻击成功率，越低越好）
- Utility（良性任务完成率，越高越好）

并计算：

```
Security Gap = ASR_heldout - ASR_train
```

解释：

- Gap 大：Train 很好、Held-out 变差 -> 更像背题
- Gap 小：两边接近 -> 更像有迁移能力

注意：Held-out 也是人工构造的。关键不是“更真实”，而是调参时不可见。

---

## 3. 主张边界（汇报时要说清）

### 我们主张

- IPI 威胁模型沿用已有标准
- 每个评测实例必须形式化为 episode contract
- 每个 episode 必须过 referee
- 评测必须区分 train / held-out
- 防御结果必须报告 transfer gap

### 我们不主张

- 发明新的 IPI 规则
- 提出最强防御
- 做完整攻防共演化系统
- 自动生成所有可能的 IPI episodes
- 解决全部 agent 安全评测问题

一句话定位：

> 这是 evaluation protocol，不是新 threat model，也不是最强 defense。

---

## 4. 「真泛化还是背了题」怎么证明

实验逻辑：

1. 固定威胁模型与判分规则
2. 构造并通过 referee 的 train / held-out episodes
3. 只在 train 上选择或调优防御
4. 在 train 和 held-out 上用同一指标评测
5. 比较 transfer gap，以及 safety-utility tradeoff

这个实验证明的是：

- 静态题库是否高估防御
- 防御对未见过 episodes 的迁移能力

这个实验不证明：

- 真实世界绝对安全
- held-out 覆盖了所有攻击形态

---

## 5. 当前进度

已完成（scaffold）：

- 项目仓库与目录结构
- episode contract JSON Schema
- Pydantic 加载与基础模型
- rule-based referee（最小版）
- email sandbox 占位
- 2 条 seed episodes（1 train + 1 held-out）
- episode 校验脚本可跑通

未完成（实验还跑不起来）：

- 足够多的 episodes
- 更强的 referee 检查
- LLM agent 执行闭环
- baseline defenses
- 端到端评测脚本与 transfer gap 报告

仓库：

https://github.com/peichengzhao/contract_heldout_ipi

---

## 6. 下一步安排

原则：先把最小实验跑通，再扩规模。不并行铺太大摊子。

### Phase A：把最小评测闭环跑通（优先）

目标：能对 2 条 seed episodes 真正跑出 ASR / Utility / Gap。

任务：

1. 接 LLM agent loop（基于 EmailSandbox）
2. 实现轨迹判分
   - attack success
   - benign success
3. 实现 2 个 baseline
   - no defense
   - system prompt warning
4. 写 `scripts/run_eval.py`
   - 跑 train + held-out
   - 输出表格结果

完成标准：

- 一条命令能跑完评测
- 能看到 no-defense 与 prompt-defense 的对比
- 能看到 train / held-out 指标和 gap

### Phase B：把评测集做扎实

目标：从“能跑”变成“能支撑主张”。

任务：

1. 手写扩到约 10 个 email episodes
2. 明确 train / held-out 划分，并保证调参时 held-out 不可见
3. 加强 referee
   - replay / 可执行性
   - 平凡捷径
   - 不可判分 / 不可能局

完成标准：

- 至少 10 个合法 episodes
- referee 能拒绝一批故意构造的坏局

### Phase C：做对照实验

目标：用结果支持论文主结论。

任务：

1. 增加 baseline
   - instruction hierarchy
   - tool confirmation
   - trusted/untrusted separation
2. 跑实验并报告
   - train ASR / utility
   - held-out ASR / utility
   - transfer gap
   - safety-utility tradeoff
3. 观察是否出现
   - 某些 prompt defense 在 train 上好看
   - 在 held-out 上明显下降

完成标准：

- 有可展示的结果表
- 能清楚讲：静态评测可能高估，held-out gap 能暴露过拟合

### Phase D：再考虑扩展（先不做）

以下内容先 ented，不进入近期主线：

- 攻防共演化
- 自动大规模生成全部 episodes
- 多 domain 全面铺开
- 声称最强防御

---

## 7. 近期执行顺序（建议）

按这个顺序推进：

1. Agent loop + 判分
2. no-defense / prompt-defense
3. run_eval 出第一张结果表
4. 扩到约 10 个 episodes
5. 加强 referee
6. 补更多 baseline，做 transfer gap 对比

近期不要并行去做：

- 共演化
- 大规模自动生成
- 多 domain 同时开工

---

## 8. 成功标准（什么叫这件事做成了）

最小成功：

- 协议清楚：Contract + Referee + Held-out Gap
- 有可复现的小规模实验
- 能展示至少一种“train 好看、held-out 变差”的现象，或清楚说明不同防御 gap 不同

完整成功（论文口径）：

- 证明静态 IPI benchmark 可能高估防御
- 证明 contract-based held-out evaluation 能更可靠地暴露泛化问题
- 不依赖“我们做出了最强防御”这个过大主张

---

## 9. 一句话收束

这个 idea 的价值，不在于提出最强防御，而在于：

让 IPI 防御评测更可复现、更可校验、更重视泛化。

下一步不是继续扩大故事，而是先把最小实验闭环跑通。
