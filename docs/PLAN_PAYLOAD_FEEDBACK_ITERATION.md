# Payload / Template 反馈迭代改动说明

> 2026-09-07  
> 对照：IterInject、AutoDojo、AgentVigil（EMNLP’25）、PAIR

## 问题

3×3 协商实验显示：改 **attack system prompt** 后新攻击变差（YAML/`event_metadata`），而 R0 自然语言老攻击仍有 LgASR。顶会自适应 IPI 几乎都在迭代 **payload 文本**，不是生成器 system prompt。

## 对照顶会 → 我们的改动

| 工作 | 核心机制 | 落地 |
|------|----------|------|
| **IterInject** | 结构化诊断标签 + 按历史改 payload | `attack_diagnosis.py` + `payload_optimizer.py` |
| **AutoDojo** | feedback → diagnosis → 生成下一候选注入 | defense_blocked 时 refine → `payload_seeds` 下一轮优先试用 |
| **AgentVigil** | 种子突变 / 搜索 | `archive_mutator` 优先 `paraphrase_nl` / `strategy_reskin`，结构化 reformat 降级 |
| **PAIR** | 改攻击内容本身，保留历史 | prior_attempts + diagnosis 喂给 payload optimizer |

## 代码

- `dojo/attack_diagnosis.py` — success/partial/detected/ignored/error  
- `dojo/payload_optimizer.py` — 只改 `vector_payloads`  
- `dojo/archive_mutator.py` — NL paraphrase 优先；NL-only 时跳过 YAML/tool_output  
- `dojo/coevolution_runner.py` — defense_blocked：**先 payload refine**；成功则 **跳过** attack-prompt 升级  
- `dojo/dynamic_runner.py` — `payload_seeds` 优先于 archive mutation / generate  
- CLI 默认挂上 `DojoPayloadOptimizer`（attack client）

## 建议下一跑

```bash
conda run -n tools_attack python scripts/run_agentdojo_coevolution.py \
  --config model_api_ablation.py \
  --negotiate-each-round \
  --max-pairs 3 --train-rounds 3 \
  --qualification-candidates 3 \
  --output results/dojo_negotiated_payload_iter_3x3.json
```

看：`payload_refines` 数量、种子是否入场、AdmASR 在 R1+ 是否不再掉光、attack harness 版本是否少涨。
