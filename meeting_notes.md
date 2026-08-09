# Contract-based Held-out IPI Evaluation

汇报用简版说明

---

## 1. 一句话

现有 IPI 评测太像固定试卷，防御容易对答案。

我们要的是一套能检查「换题还会不会」的评测方法。

---

## 2. 核心思路（三步）

### Step 1: Contract

把每一局评测写成机器可读合同：

- 用户任务
- 工具
- 不可信内容
- 攻击怎样算成功
- 正常任务怎样算完成

### Step 2: Referee

自动筛掉坏局：

- 跑不了
- 判不了分
- 太水（例如关掉所有工具就赢了）

### Step 3: Held-out Gap

防御在可见题上好，不等于真安全。

必须看没见过的题掉了多少。

流程：

```
Episode
  -> Contract
  -> Referee 校验
  -> Train / Held-out 划分
  -> 测防御
  -> 比 Transfer Gap
```

---

## 3. 我们主张什么 / 不主张什么

我们做：

- 把评测协议做扎实
- 测防御会不会过拟合题库
- 报告 train 到 held-out 的迁移差距

我们不做：

- 不发明新的 IPI 威胁模型
- 不声称做出最强防御
- 不做完整攻防共演化系统

---

## 4. 「真泛化还是背了题」怎么证明

### 做法

1. 准备两套题（都过 Referee，威胁模型相同）
   - Train：调防御时可以看到
   - Held-out：调的时候完全不能看

2. 防御只在 Train 上选择 / 调参

3. 用同一套指标分别在 Train 和 Held-out 上测
   - ASR：攻击成功率，越低越好
   - Utility：良性任务完成率，越高越好

4. 看差距

```
Security Gap = ASR_heldout - ASR_train
```

- Gap 大：Train 很安全，Held-out 明显变差 -> 更像背了题
- Gap 小：两边差不多 -> 更像真有点泛化

### 能说明什么

能说明：

- 防御是否过拟合可见题库
- 静态 benchmark 会不会高估防御
- 不同防御谁更抗换题

不能说明：

- 真实世界 100% 安全
- Held-out 覆盖了所有攻击
- 某个防御绝对最强

说明：

Held-out 也是人写的。关键不是更真实，而是调参时不可见。

测的是从可见分布迁到未见分布的能力。

---

## 5. 30 秒口述版

现在很多 IPI benchmark 是静态题库。防御在这些题上调好了，分数会很好看，但换一批同类型新题可能就塌。

所以我们不先卷防御，而先把评测规则定清楚：每局有机器可读合同、有裁判校验、并且必须报告在未见过题上的表现差距。

目标不是证明谁最强，而是证明：这个防御是真泛化，还是只是背了题。

---

## 6. 当前进度

- Contract schema + seed episodes：已有（2 条）
- Rule-based referee：最小版已有
- Email sandbox：占位
- Agent loop / 防御 / 完整跑分：待做

仓库：

https://github.com/peichengzhao/contract_heldout_ipi
