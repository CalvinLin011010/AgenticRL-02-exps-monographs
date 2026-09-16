# Seed-OSS SWE-bench Verified 200 题典型 Rollout 数值分析

本文件夹按用户指定放在既有三路 rollout 分析目录下，保存 Seed-OSS-36B-Instruct 的独立 SWE-bench case 分析；它不属于 Qwen3 三路对比样本。

## 入口

- [完整分析](./source/典型Rollout数值分析.md)
- [全量逐题指标](./data/case_metrics.csv)
- [聚合统计](./data/aggregate_stats.json)
- [典型 case 详情](./data/typical_cases.json)
- [来源 manifest](./data/source_manifest.json)
- [可复现脚本](./analyze_seed_oss_swebench.py)

## 口径

主结果为 41/200（20.50%，Wilson 95% CI 15.49%–26.63%）；191 题有完整官方报告。典型样本不是主观挑选：成功组和有效未解决组各使用六个数值特征经 median/MAD 归一化后选取距中位数最近者，并另选高效成功、折叠成功、partial near miss、回归 near miss、成本异常值和基础设施超时作为边界样本。

复现：

```bash
python3 analyze_seed_oss_swebench.py
```
