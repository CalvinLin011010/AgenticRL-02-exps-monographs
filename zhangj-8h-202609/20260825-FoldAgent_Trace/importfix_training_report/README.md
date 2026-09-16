# Qwen3-8B 强化学习训练曲线统计

## 数据来源

主实验对应：`AgenticRL-Lab-long/logs/browsecomp_qwen3_8b_trace_importfix_20260818_0205/train/metrics.jsonl`，本次绘图严格截取 step 1--300；后续 step 不纳入。

同时扫描到两个相关目录：

- `browsecomp_qwen3_8b_agentgrpo_trace_ctx40k_full_20260817_174200`：存在目录与训练配置，但没有 `train/metrics.jsonl`，因此没有可绘制的 step 级曲线。
- `browsecomp_qwen3_8b_agentgrpo_trace_em_branch8k_20260823_retry3_uuidfix`：有 22 个 step，配置与主实验不同，仅单独展示，不并入主实验统计。

## 输出文件

- `training_metrics_all_runs.csv`：所有可读取 run 的 step 级宽表。
- `training_metrics_summary.json`：每个 run、每个指标的记录数、均值、标准差、最小值、最大值、首值和末值。
- `training_metrics_summary.md`：面向阅读的统计摘要。
- `*_actor_loss.svg`：PG loss、TRACE loss 和派生的 total loss。
- `*_optimization_stability.svg`：梯度、PPO KL、rollout KL、entropy、概率差异和 clip 相关曲线。
- `*_critic_signals.svg`：critic score/reward、advantage 和 return 相关曲线。
- `*_reward_train.svg`：训练 batch 的 reward 均值、reward_score、标准差、最小值、最大值、轨迹数和 UID 数。
- `*_reward_critic.svg`：critic score/reward 的均值、最小值和最大值。
- `*_reward_rl_signals.svg`：advantage 和 return 的均值、最小值和最大值。
- `*_reward_validation.svg`：固定验证集的 reward、reward_score、avg_score、标准差、最小值和最大值；仅在验证 step 有数据。
- `*_trace.svg`：TRACE credit 及 actor 侧 TRACE credit 统计曲线。
- `*_behavior.svg`：response 长度、turn 数、aborted、overlong 和 clipping 曲线。
- `*_efficiency.svg`：step/gen/adv/update_actor 耗时、throughput 和 MFU 曲线。

## 运行方式

在仓库根目录执行：

```text
python 00-docs/实验/wanghb-calvin/FoldAgent-1/2026-08-25-Qwen3-8B-训练曲线统计/plot_training_curves.py
```

脚本只使用 Python 标准库，直接生成 CSV、JSON、Markdown 和 SVG，不依赖 matplotlib 或 pandas。

## 统计口径

- `reward/avg_score` 与 `reward/reward_score` 是训练 batch 的在线奖励聚合，不是固定验证集准确率。
- `actor/pg_loss`、`actor/trace_loss` 是 actor 更新中的 loss 指标；`actor/total_loss` 在主实验中由 `pg_loss + trace_loss` 派生得到。
- `actor/grad_norm`、`actor/ppo_kl`、rollout KL、entropy 和 clip fraction 是训练稳定性辅助指标，不是 loss。
- `critic/score/*`、`critic/rewards/*`、`critic/advantages/*`、`critic/returns/*` 保留为 critic 与 advantage 相关分量。
- reward 曲线按训练 batch、critic 聚合、RL 信号和固定验证集分图；所有曲线严格使用 step 1--300。
- 主实验原始记录为 319 条；本次统计固定使用 step 1--300，共 300 条。
- retry3 与 importfix 是不同实验，不在本次图表中混用。
