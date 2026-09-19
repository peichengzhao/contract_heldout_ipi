# Duel：多轮双向共进化协议

> 状态：已落地（`duel` protocol + `DUEL_DOJO_CONSTITUTION`）  
> 目的：避免「防御只升一轮、攻击 harness 不动」的假共进化

## 相对 arms_race 的关键改动

| 项 | arms_race | duel |
|----|-----------|------|
| 防御 cooldown | 3（协商还可再加） | **0**（宪法硬钳） |
| 全局防御升级上限 | 2 | **无上限** |
| 升级所需攻击成功数 | 2 | **1** |
| 攻击 prompt 优化触发 | 仅 defense_blocked | defense_blocked **或** qual_failure |
| 每轮强制升攻击 harness | 否 | **是**（若本轮尚未升） |
| payload 出 seed 就跳过攻击 harness | 旧逻辑会跳过 | **仅同轮打穿才跳过** |
| 协商宪法 | `dojo-negotiated-v1.1` | `dojo-negotiated-duel-v1` |

## 推荐命令

协商是主设定（`duel` 会自动挂上 `DUEL_DOJO_CONSTITUTION`，cooldown 等门控无法谈回单边锁死）：

```bash
conda run -n tools_attack python scripts/run_agentdojo_coevolution.py \
  --config model_api_ablation.py \
  --coevolution-protocol duel \
  --negotiate-each-round \
  --max-pairs 3 \
  --train-rounds 5 \
  --qualification-candidates 6 \
  --payload-refine-candidates 4 \
  --same-round-attack-iters 3 \
  --defense none \
  --output results/dojo_duel_bidirectional_3x5.json
```

若轨迹异常、需要排查是 runner 还是协商问题时，可临时去掉 `--negotiate-each-round` 做对照；正式实验始终带协商。
## Breakthrough-first（2026-09-19）

针对「攻击优化很多但打不穿」补丁：

1. 攻击 / payload / NL 生成提示改为要求 **工具轨迹成功**，禁止「已完成历史记录」式空转  
2. 同轮反杀防御回放：`duel` 默认 3 次  
3. ban 阈值抬高（duel=5）；同轮失败只记 fingerprint，不封整类 strategy  
4. 生成时跳过 `banned_strategies`，被 ban 后强制轮换 strategy  
5. 防御 incremental 补丁要求 **只挡本次机制**，避免一次写死「忽略一切嵌入指令」
