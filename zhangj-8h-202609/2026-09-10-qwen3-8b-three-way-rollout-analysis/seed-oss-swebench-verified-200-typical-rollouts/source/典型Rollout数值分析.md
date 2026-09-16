# 典型 Rollout 数值分析

## 1. 选择方法

对每条 case 统一提取 session time、total token、action count、generation request、max prompt token 和 summary count。对成功组（41 条）与有正式报告的未解决组（150 条）分别计算各特征中位数与 MAD，逐维绝对偏差除以 MAD 后求和；距离最小者定义为组内数值中位代表。MAD 为 0 时以 1 代替。其余样本按预先声明的边界规则选择，不能解释为总体“平均 case”。

## 2. 选中样本

| 类别 | instance | 结果 | session s | token | request | summary | F2P | P2P |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 成功组 medoid | pytest-dev__pytest-5262 | FULL | 391.8 | 23,063 | 16 | 0 | 1/1 | 108/108 |
| 未解决组 medoid | django__django-14351 | NO | 302.9 | 26,273 | 16 | 0 | 0/1 | 58/58 |
| 高效成功边界 | django__django-15467 | FULL | 285.7 | 12,751 | 18 | 0 | 1/1 | 62/62 |
| 折叠后成功 | sphinx-doc__sphinx-9673 | FULL | 353.0 | 54,314 | 17 | 1 | 1/1 | 24/24 |
| partial near miss | django__django-14315 | PARTIAL | 411.4 | 22,768 | 18 | 0 | 7/11 | 0/0 |
| 回归 near miss | astropy__astropy-14508 | NO | 426.5 | 24,825 | 16 | 0 | 1/1 | 173/174 |
| 高成本失败异常值 | sphinx-doc__sphinx-8056 | NO | 1,252.6 | 54,061 | 16 | 1 | 0/1 | 40/40 |
| 基础设施超时 | scikit-learn__scikit-learn-14087 | 无报告 | 325.4 | 34,477 | 19 | 1 | — | — |

## 3. 数值解释

成功 medoid 与失败 medoid 都使用 16 次请求，token 分别为 23,063 与 26,273；单纯增加轨迹长度没有保证修复成功。高效成功样本只用 12,751 tokens，而成本异常失败达到 54,061 tokens、1,252.6 秒和 78,588 response chars，仍未通过唯一目标测试。

折叠成功样本证明 summary 链路可在 54,314-token telemetry 下保留足够信息，但全量中未使用 summary 的 128 题通过 28 题，使用 summary 的 72 题通过 13 题；这是选择偏差明显的描述统计，不能推断 summary 导致成功或失败。

错误至少分三类：普通目标未命中（失败 medoid）、目标部分修复（PARTIAL）、目标修好但引入回归（regression-only）。基础设施超时没有官方测试报告，应从模型补丁错误中单列。

## 4. 全量背景

全量 resolved 41/200；191 条完整报告中，41 条目标和回归测试均通过，9 条目标通过但回归失败，130 条目标失败但回归通过，11 条两者均失败。正确 case 的 token 中位数 24,296，未解决 case 为 27,537；action 中位数均为 16。

## 5. 可追溯性与限制

逐题数据见 `../data/case_metrics.csv`，精确选择记录见 `../data/typical_cases.json`，生成逻辑见 `../analyze_seed_oss_swebench.py`。原始 events、eval.log 和 2.9 MB 结果 JSON 留在 runtime，不复制进本目录。本分析是单次运行的描述性审计，不估计跨 seed 方差，也不把 token、折叠或 finish 状态解释为因果因素。
