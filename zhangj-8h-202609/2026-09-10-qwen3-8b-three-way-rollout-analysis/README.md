# 2026-09-10 Qwen3-8B 三路 Rollout 阶段分析

本目录是可直接部署的静态实验报告包，对比 Trace+FoldGRPO、no-Trace 与 Trace-only。

## 页面入口

- [增强分析首页](./index.html)
- [8 案例完整轨迹、coverage 与 token 诊断](./diagnostics.html#cases)
- [诊断页生成脚本](./build_diagnostics.py)
- [完整案例轨迹 JSON](./data/detailed_rollout_cases.json)
- [validation score / finish / gold 语义审计](./data/validation_semantic_audit.json)
- [8 个典型案例的 strict EM / DeepSeek 审计](./data/typical_cases_external_audit.json)
- [DeepSeek 独立评测汇总](./data/deepseek_flash_fixed_judge_summary.json)
- [新增派生统计](./data/derived_statistics.json)
- [逐 step 训练指标](./data/training_step_metrics.csv)
- [Step-100 validation 汇总](./data/validation_step100_summary.csv)
- [独立评测汇总](./data/independent_evaluation_summary.csv)
- [来源与定义 manifest](./data/source_manifest.json)
- [Seed-OSS SWE-bench Verified 200 题典型 Rollout 数值分析](./seed-oss-swebench-verified-200-typical-rollouts/README.md)

## 统计口径

- 训练统计统一使用 step 21-100，共 80 个在线 actor 状态。
- validation 使用完成 step 100 更新后的 checkpoint，每组 150 题。
- 独立评测使用导出的 `hf_global_step_100`，grader 路径与训练 validation 不同。
- validation 差值的 p 值为未做多重比较校正的双侧 pooled two-proportion z-test，仅用于描述不确定性。
- 95% 区间使用 Wilson score interval。当前只有单 seed，不能把区间或 p 值解释为跨 seed 的训练方差。

## 可追溯材料

`data/` 和 `figures/` 保存页面所用结构化数据与原图，`source/` 保存原始完整报告及计算审计。页面不依赖外部 CDN，可直接打开或部署到静态站点。
