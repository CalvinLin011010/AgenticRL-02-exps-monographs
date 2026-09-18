# Qwen3-8B 三组 Rollout / Trajectory / Tool / Token 统计报告

- 日期：2026-09-10
- 三组设置：Trace+FoldGRPO、no-Trace、Trace-only
- 公平训练区间：step 21-100
- Validation 对比：step 100，150 个 BrowseComp 问题
- 分析脚本：[analyze_three_way.py](../../../../01-exps/zhangj-8h-202609/2026-09-10-qwen3-8b-three-way-rollout-analysis/analyze_three_way.py)

> 用户原文中的“on Trace”按上下文解释为“no-Trace”。如其本意是另一组 on-policy Trace 运行，需要另行指定 run path。

## 1. 设置和数据口径

| 设置 | Actor advantage | process reward | 运行 |
|---|---|---|---|
| Trace+FoldGRPO | A_FoldGRPO + 0.2 A_TRACE | flat | browsecomp_qwen3_8b_trace_long_fix_20260904_2248 |
| no-Trace | A_FoldGRPO | flat | browsecomp_qwen3_8b_ablation_no_trace_20260906_0200 |
| Trace-only | 0 A_FoldGRPO + 0.2 A_TRACE | null | browsecomp_qwen3_8b_trace_only_20260909_v2 |

三组的主要 rollout 配置一致：Qwen3-8B、train batch 2、rollout n=2、max_traj=5、prompt 8192、response 32768、model context 40960、max turn 100、max session 10、branch max turn 12、search/open_page observation 上限 8192。Trace-only 的 objective 和 process reward 处理不同，这是实验变量，不是完全同构的 reward 记录。

历史 Trace 保留的逐 step 指标和训练 rollout 从 step 21 开始，因此训练统计统一限制为 21-100，不对缺失 step 1-20 插值。Validation 统一用 step 100。

### 分母定义

- raw rollout：训练 rollout JSONL 中实际保存的 trajectory 行数；动态 session/retry/max_traj 使它不等于 batch_size × n 的简单固定值。
- retained rollout：每 step 的 num_unique_gen_uids × avg_trajs_per_gen_uid，表示进入后续统计/更新的估计数量。
- question success：val/avg_score，在 150 个问题聚合后计算。
- trajectory success：保存的 trajectory 行中 score=1 的比例，等于 val/raw_environment_reward。
- aborted ratio：使用 step metrics 的 response/aborted_ratio；逐 trajectory JSON 没有稳定的 aborted 标志，不用空 output 猜测。
- at limit：达到 32768 response token 上限；overflow：越界长度。二者分开统计。
- 重复 search：同一 transcript 中规范化后的 search function body 完全重复，是保守的 exact-repeat 定义。
- fold/context management：模型没有 fold tool；报告分别列 branch、return、branch+return，以及 harness summary/forced return。

## 2. 训练 Rollout：Step 21-100

![Training rollout dynamics](../figures/training_rollout_dynamics.png)

| 设置 | raw rollout | retained | retained/step | step avg score | trajectory score | aborted ratio | 平均 turns | 平均 response tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trace+FoldGRPO | 792 | 645 | 8.06 | 0.2625 | 0.2866 | 18.88% | 27.26 | 14,639 |
| no-Trace | 844 | 707 | 8.84 | 0.2969 | 0.2915 | 15.79% | 31.25 | 14,734 |
| Trace-only | 876 | 749 | 9.36 | 0.2813 | 0.2660 | 14.91% | 33.60 | 16,326 |

Trace-only 保存和保留的 rollout 最多，aborted ratio 最低，但轨迹更长。它的 response tokens 比 Trace+FoldGRPO 高约 11.5%，而 generated tokens 均值反而略低（1388 vs 1452）；因此增长主要来自更长的工具 observation/折叠后上下文，而不是模型每次生成更多 token。

### 训练轨迹工具行为

| 设置 | tools/traj | search | open_page | branch | return | assistant turns | exact duplicate search | branch+return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trace+FoldGRPO | 5.72 | 1.93 | 1.01 | 1.40 | 0.60 | 5.72 | 19.95% | 40.40% |
| no-Trace | 6.12 | 1.76 | 0.96 | 1.89 | 0.69 | 6.10 | 17.77% | 44.31% |
| Trace-only | 5.98 | 1.73 | 0.94 | 1.84 | 0.71 | 6.24 | 17.58% | 45.89% |

Trace+FoldGRPO 更偏向 search，后两组更偏向 branch。Trace-only 的总工具量略低于 no-Trace，但 assistant turns 更高，说明其单 turn 的工具密度略低。

## 3. Step-100 Validation：Rollout 数量和成功率

![Step-100 trajectory comparison](../figures/step100_trajectory_comparison.png)

| 设置 | 问题数 | trajectory 数 | rollout/问题 | question success | trajectory success | easy traj | medium traj | hard traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trace+FoldGRPO | 150 | 293 | 1.95 | 0.3533 | 0.3754 | 0.5957 | 0.2178 | 0.3265 |
| no-Trace | 150 | 315 | 2.10 | 0.3267 | 0.2508 | 0.4773 | 0.1491 | 0.1770 |
| Trace-only | 150 | 321 | 2.14 | 0.4133 | 0.3925 | 0.6436 | 0.2742 | 0.2813 |

Trace-only 的 question success 和 trajectory success 都最高；Trace+FoldGRPO 在 hard trajectory 上高于 Trace-only。三个 split 的 trajectory 数不相等，是动态 session/重试导致，不能把 trajectory-level split rate 当作固定 50 题的独立评测率。

## 4. Step-100 Validation：工具与轨迹结构

![Tool and context events](../figures/tool_and_context_events.png)

| 设置 | tools/traj | search | open_page | branch | return | assistant turns | exact duplicate search | branch+return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trace+FoldGRPO | 5.59 | 1.80 | 0.98 | 1.60 | 0.67 | 5.61 | 23.89% | 47.78% |
| no-Trace | 6.61 | 1.82 | 1.16 | 2.25 | 0.85 | 6.19 | 18.73% | 52.38% |
| Trace-only | 6.08 | 1.63 | 1.20 | 1.84 | 0.79 | 6.25 | 18.07% | 51.40% |

Trace-only 比 no-Trace 少 8.1% 总工具调用，主要少 branch；但 open_page 略多，表现为更少分支、更直接读取证据。Trace+FoldGRPO 总工具最少，但 exact duplicate search 比例最高。

### 成功与失败轨迹条件统计

| 设置/结果 | N | tools | search | open | branch | turns | response tokens | duplicate search |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trace+FoldGRPO / correct | 110 | 5.70 | 1.81 | 0.98 | 1.65 | 5.50 | 14,583 | 20.00% |
| Trace+FoldGRPO / wrong | 183 | 5.53 | 1.79 | 0.98 | 1.57 | 5.68 | 15,056 | 26.23% |
| no-Trace / correct | 79 | 4.58 | 1.70 | 1.04 | 0.77 | 3.96 | 12,129 | 12.66% |
| no-Trace / wrong | 236 | 7.29 | 1.86 | 1.20 | 2.75 | 6.93 | 15,434 | 20.76% |
| Trace-only / correct | 126 | 5.77 | 1.58 | 1.37 | 1.57 | 5.79 | 15,797 | 18.25% |
| Trace-only / wrong | 195 | 6.28 | 1.66 | 1.09 | 2.01 | 6.55 | 17,046 | 17.95% |

no-Trace 的失败轨迹相对成功轨迹多 59% 工具调用和约 3,305 response tokens，主要来自 branch 膨胀；这是三组里最明显的失败时无效扩展。Trace-only 的失败轨迹也更长，但膨胀较温和。Trace+FoldGRPO 的正确/错误工具数量接近，错误更常发生 exact-repeat search。

## 5. Tool Observation Token

使用 Qwen3-8B tokenizer，对扁平 transcript 中 search/open_page 后的完整 observation 编码。多个同 turn 调用共享一条 user observation 时只归属一次；内联 observation 使用 </function> 到下一个 think/function 的区间，和已有 context-management 审计规则一致。

| 设置 | search observation 数 | 平均 tokens | 最大 tokens | open observation 数 | 平均 tokens | 最大 tokens |
|---|---:|---:|---:|---:|---:|---:|
| Trace+FoldGRPO | 471 | 6,329 | 25,343 | 267 | 2,399 | 20,601 |
| no-Trace | 504 | 5,943 | 25,913 | 339 | 2,097 | 8,347 |
| Trace-only | 462 | 7,305 | 27,120 | 385 | 2,718 | 8,363 |

Trace-only 的 search observation 平均长度比 Trace+FoldGRPO 高 15.4%、比 no-Trace 高 22.9%；open_page observation 也分别高 13.3% 和 29.6%。这解释了它在 generated tokens 接近时拥有更长 response context。最大 search observation 远超配置中的 8192，是因为这里统计的是保存 transcript 中完整 user observation（可包含组合输出和附加 guidance），不是后端单次搜索结果的硬截断字段。

## 6. Token 上限、超长与 Overflow

| 设置 | train response at-limit | train overlong_masked | 受影响 step | val response at-limit | val overlong rate | 真正 response overflow |
|---|---:|---:|---:|---:|---:|---:|
| Trace+FoldGRPO | 8/792 (1.01%) | 0 | 0 | 3/293 (1.02%) | 0 | 0 |
| no-Trace | 11/844 (1.30%) | 0 | 0 | 7/315 (2.22%) | 0 | 0 |
| Trace-only | 18/876 (2.05%) | 18 | 4 | 5/321 (1.56%) | 0.67% | 0 |

Validation response token P95 为 30,956 / 30,738 / 31,153；三组都靠近 32,768 上限。Trace-only 的上下文负载最高，训练中也是唯一触发 overlong masking 的设置。三组 generated_at_limit 和 generated overflow 都为 0，response overflow 也为 0，因此不能把 at-limit 样本描述为已经越界。

## 7. 独立 150 题评测

本报告对应的 `trace_long_fix_20260904_2248` 未保留同口径独立 local-Qwen 评测。当前与历史结果把所有已知判分口径并列如下；`—` 表示该实验没有运行该 scorer，不能跨列补值。

| 实验族 | 模型/阶段 | checkpoint | local normalized exact | local-Qwen hybrid accepted | relaxed diagnostic match | historical strict/TRACE normalized EM | DeepSeek accepted（模型见注） | 历史旧 scorer |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 当前 202609 | no-Trace | step 100 | 21/150 = 14.00% | 24/150 = 16.00% | 43/150 = 28.67% | **20/150 = 13.33%** | **25/150 = 16.67%**（3 err） | — |
| 当前 202609 | Trace-only | step 100 | 17/150 = 11.33% | 22/150 = 14.67% | 44/150 = 29.33% | **14/150 = 9.33%** | **22/150 = 14.67%**（3 err） | — |
| 历史 202608 | 未训练 Qwen3-8B | base | — | — | — | 8/150 = 5.33% | 12/150 = 8.00% | 未运行 |
| 历史 202608 | FoldAgent+TRACE（实际 estimator 为 AgentGRPO） | step 100 | — | — | — | 16/150 = 10.67% | 23/150 = 15.33% | 54/150 = 36.00% |
| 历史 202608 | FoldAgent+TRACE（实际 estimator 为 AgentGRPO） | step 300 | — | — | — | 14/150 = 9.33% | 27/150 = 18.00% | 59/150 = 39.33% |

当前两组的多项分数来自同一批 150 题答案：historical strict 按历史脚本在 CPU 上重算，Easy/Medium/Hard 分别为 no-Trace 16/3/1、Trace-only 13/1/0；另用同一个 `deepseek-flash`、temperature 0、`correct && confidence≥0.8` 统一复评，分别接受 25/150 和 22/150。每组仍有 3 条 JSON 解析错误，按历史 contract fail-closed 计 0，因此 DeepSeek 数值带错误数披露。当前 model id 与历史 `deepseek-v4-flash` 不同，只作初步对照。见 [统一 DeepSeek 复评汇总](../../../../01-exps/zhangj-8h-202609/2026-09-10-qwen3-8b-three-way-rollout-analysis/data/deepseek_flash_fixed_judge/summary.json)。

历史 `browsecomp_qwen3_8b_trace_importfix_20260818_0205` 不是本报告的 Trace+FoldGRPO run，且名称虽包含 FoldAgent，保存配置的 advantage estimator 实际是 `agentgrpo`，不是 `foldgrpo`。它与当前 run 的训练配置、checkpoint 和 grader 均不同，只能按列作初步旁证；旧 scorer 已确认明显高估。详见 [zhangj-calvin 的实验谱系说明](../../../../00-docs/experiments/zhangj-calvin/foldagent-1/2026-08-30-Qwen3-4B-Thinking训练设计错误复盘与全量训练推理方案/README.md) 与 [完整 150 题评测报告](../../../../00-docs/experiments/wanghb-calvin/foldagent-1/2026-08-24-Qwen3-8B-FoldAgent-TRACE科研汇报-修订版/README.md)。

## 8. 历史 FoldGRPO / TRACE 结果的可比性核对

本次在 lr/00-docs/experiments/wanghb-calvin 与 lr/00-docs/experiments/zhangj-calvin 中复核了历史记录。结论是：**没有找到与当前 Qwen3-8B、202609 三组运行完全同口径且明确属于 FoldGRPO+Trace 的历史独立评测结果**。历史材料中有两类容易混淆的结果：

| 历史材料 | 实际设置 | 可核对结果 | 能否作为当前 FoldGRPO+Trace 结果 |
|---|---|---:|---|
| wanghb-calvin，2026-08-13 Qwen3-4B-Thinking-2507 FoldGRPO 完整总结 | Qwen3-4B-Thinking-2507 + FoldGRPO + search_branch | 最终离线 strict/本地评测 22/150 = 14.67%；训练内 validation 最终约 5/150 | 否，backbone、run、时间和评测产物不同；只能作历史 FoldGRPO 背景 |
| wanghb-calvin，2026-08-24 Qwen3-8B FoldAgent-TRACE 修订报告 | Qwen3-8B + **AgentGRPO** + FoldAgent search_branch + frozen-reference online TRACE | step 100 historical strict 16/150 = 10.67%，DeepSeek 23/150 = 15.33%；step 300 strict 14/150 = 9.33%，DeepSeek 27/150 = 18.00% | 否，文档明确不是纯 FoldGRPO；且使用历史 deepseek-v4-flash |
| zhangj-calvin，2026-08-06 Qwen3-8B TP2 BrowseComp 全量分析 | Qwen3-8B 推理/本地 judge 基线，非当前 FoldGRPO+Trace 训练 run | 本地 judge 35/150 = 23.33%；其中确定性 em_score 仅 25/150 = 16.67% | 否，非当前训练设置，且不是严格统一 grader |

4B 历史结果的原始记录为 22/150，但不能拿来填当前表格的 FoldGRPO+Trace 行。历史 8B AgentGRPO+TRACE 的 strict/DeepSeek 数字也只能作为旁证，不能与当前 no-Trace、Trace-only 的独立评测直接做同实验比较。当前三组中 FoldGRPO+Trace 缺失同口径的一题一条独立评测输出，仍应记为 —，不能从历史数字或 validation 的 53/150 反推。

此外，历史 8B 报告明确指出旧 scorer 假阳性严重：step 300 的旧 scorer 为 59/150 = 39.3%，但 strict/TRACE normalized EM 只有 14/150 = 9.3%；因此历史表中的 scorer 也必须按列解释，不能把旧 score 当语义准确率。详细来源：

- [4B FoldGRPO 完整实验总结](../../../../00-docs/experiments/wanghb-calvin/foldagent-1/2026-08-13-Qwen3-4B-Thinking-2507-FoldGRPO完整实验总结/README.md)
- [8B FoldAgent-TRACE 修订报告](../../../../00-docs/experiments/wanghb-calvin/foldagent-1/2026-08-24-Qwen3-8B-FoldAgent-TRACE科研汇报-修订版/README.md)
- [8B 历史 scorer/奖励链复盘](../../../../00-docs/experiments/wanghb-calvin/foldagent-1/2026-08-22-Qwen3-8B实验结果、评分修复与性能下降完整复盘/README.md)
- [8B TP2 历史评测分析](../../../../00-docs/experiments/zhangj-calvin/foldagent-1/2026-08-06-Qwen3-8B-TP2-BrowseComp全量实验结果分析/README.md)

## 9. 与公开 FoldAgent Trace 诊断的关系

参考页面：[Qwen3-8B FoldAgent TRACE 实验报告](https://calvinlin011010.github.io/Agentic_RL/20260825-FoldAgent_Trace/)，源码入口：[CalvinLin011010/Agentic_RL](https://github.com/CalvinLin011010/Agentic_RL)。公开中文增强诊断报告针对 29 个显式 main+branch 重建 case，报告：

- 成功 9/29；
- 重复 search case 比例 44.8%；
- 主上下文 fold case 比例 65.5%；
- 成功后仍有冗余候选 3/9；
- 失败但词面 coverage 达 100% 为 4/20。

本报告借用了“逐轨迹工具事件、重复检索、context fold、token evidence 长度、成功/失败条件切片”的分析框架，但不把公开数字当基线。原因是公开报告只有 29 个诊断 case，并采用更宽的重复检索/主 fold 定义；本地统计覆盖 293-321 条 validation trajectory，exact duplicate search 更保守。公开页面可从服务器访问；GitHub 源码在本次核验时经代理返回 503，因此未复制其代码或中间数据。

## 10. 综合判断

1. Trace-only 确实改变了行为：保留 rollout 更多、aborted ratio 最低、validation success 最高，但轨迹和 observation 最长，并出现额外 overlong masking。
2. no-Trace 的主要问题是失败轨迹膨胀：错误样本工具调用和 branch 明显多于正确样本。
3. Trace+FoldGRPO validation 较稳健且总工具最少，但 exact duplicate search 最高；hard trajectory success 最高。
4. Trace-only 在本轮 validation 上最好，但独立 local-Qwen 评测略低于 no-Trace，说明 grader 路径差异或单 seed 方差仍然很大。
5. 三组 rollout 数量并不完全相同，Trace-only 的更高 retained count 是潜在 compute/sample confound。下一轮应固定实际 retained trajectories 或报告按有效 trajectory 归一化的训练预算。
6. 建议至少做 3 个 matched seeds，并新增 semantic duplicate query、每题累计 tool budget、fold 前后 evidence retention、success-per-1k-context-token 和 overlong-mask 后的样本去向统计。

## 11. 产物

- [逐 step 训练指标](../data/training_step_metrics.csv)
- [逐 trajectory 明细](../../../../01-exps/zhangj-8h-202609/2026-09-10-qwen3-8b-three-way-rollout-analysis/data/trajectory_details.csv)
- [训练 trajectory 汇总](../data/training_trajectory_summary.csv)
- [Validation trajectory 汇总](../data/validation_trajectory_summary.csv)
- [Step-100 validation 汇总](../data/validation_step100_summary.csv)
- [独立评测汇总](../data/independent_evaluation_summary.csv)
- [来源与定义 manifest](../data/source_manifest.json)
- [运行日志](../../../../01-exps/zhangj-8h-202609/2026-09-10-qwen3-8b-three-way-rollout-analysis/run.log)
