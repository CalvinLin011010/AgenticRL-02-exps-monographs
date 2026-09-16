import csv
import html
import json
import statistics
from pathlib import Path


BASE = Path(__file__).resolve().parent
PHASES = [(1, 50), (51, 100), (101, 150), (151, 200), (201, 250), (251, 300)]
PHASE_NAMES = ["阶段 1（1--50）", "阶段 2（51--100）", "阶段 3（101--150）", "阶段 4（151--200）", "阶段 5（201--250）", "阶段 6（251--300）"]
REWARD_GROUPS = {
    "训练 Batch Reward": ["reward/avg_score", "reward/reward_score", "reward/std_score", "reward/min_score", "reward/max_score", "reward/avg_trajs_per_gen_uid", "reward/num_unique_gen_uids"],
    "Critic Reward 与 Score": ["critic/score/mean", "critic/score/min", "critic/score/max", "critic/rewards/mean", "critic/rewards/min", "critic/rewards/max"],
    "Advantage 与 Return": ["critic/advantages/mean", "critic/advantages/min", "critic/advantages/max", "critic/returns/mean", "critic/returns/min", "critic/returns/max"],
    "验证集 Reward": ["val/reward", "val/reward_score", "val/avg_score", "val/std_score", "val/min_score", "val/max_score"],
}
LABELS = {
    "actor/pg_loss": "PG loss",
    "actor/trace_loss": "TRACE loss",
    "actor/total_loss": "Total loss",
    "actor/grad_norm": "Gradient norm",
    "actor/ppo_kl": "PPO KL",
    "actor/pg_clipfrac": "PPO upper clip fraction",
    "actor/pg_clipfrac_lower": "PPO lower clip fraction",
    "actor/entropy": "Actor entropy",
    "rollout_corr/kl": "Rollout KL",
    "rollout_corr/k3_kl": "K3 KL",
    "training/rollout_probs_diff_mean": "Rollout probability difference",
    "critic/advantages/mean": "Advantage mean",
    "critic/returns/mean": "Return mean",
    "critic/score/mean": "Critic score mean",
    "critic/rewards/mean": "Critic reward mean",
    "reward/avg_score": "Average score",
    "reward/reward_score": "Reward score",
    "reward/std_score": "Score std",
    "reward/min_score": "Minimum score",
    "reward/max_score": "Maximum score",
    "reward/avg_trajs_per_gen_uid": "Average trajectories per gen UID",
    "reward/num_unique_gen_uids": "Unique generation UIDs",
    "val/reward": "Validation reward",
    "val/reward_score": "Validation reward score",
    "val/avg_score": "Validation average score",
    "val/std_score": "Validation score std",
    "val/min_score": "Validation minimum score",
    "val/max_score": "Validation maximum score",
}


def read_rows():
    with (BASE / "training_metrics_all_runs.csv").open(encoding="utf-8") as handle:
        return [{k: (float(v) if k != "run" and v else v) for k, v in row.items()} for row in csv.DictReader(handle)]


def fmt(value):
    if value is None:
        return "--"
    if abs(value) >= 1000:
        return f"{value:,.3f}"
    return f"{value:.6f}"


def stat_table(rows, metrics):
    out = ["<table><thead><tr><th>指标</th><th>均值</th><th>标准差</th><th>最小值</th><th>最大值</th><th>首值</th><th>末值</th></tr></thead><tbody>"]
    for key in metrics:
        values = [r[key] for r in rows if isinstance(r.get(key), float)]
        if not values:
            continue
        out.append(f"<tr><td><code>{html.escape(key)}</code><br><span>{html.escape(LABELS.get(key, key))}</span></td><td>{fmt(statistics.fmean(values))}</td><td>{fmt(statistics.stdev(values) if len(values) > 1 else 0)}</td><td>{fmt(min(values))}</td><td>{fmt(max(values))}</td><td>{fmt(values[0])}</td><td>{fmt(values[-1])}</td></tr>")
    return "".join(out) + "</tbody></table>"


def phase_table(rows, metrics):
    out = ["<table><thead><tr><th>阶段</th>" + "".join(f"<th>{html.escape(LABELS.get(m, m))}</th>" for m in metrics) + "</tr></thead><tbody>"]
    for (start, end), name in zip(PHASES, PHASE_NAMES):
        phase_rows = [r for r in rows if start <= r["step"] <= end]
        cells = []
        for metric in metrics:
            values = [r[metric] for r in phase_rows if isinstance(r.get(metric), float)]
            cells.append(f"<td>{fmt(statistics.fmean(values)) if values else '--'}</td>")
        out.append(f"<tr><td><strong>{name}</strong></td>{''.join(cells)}</tr>")
    return "".join(out) + "</tbody></table>"


def phase_mean(rows, metric, start, end):
    values = [r[metric] for r in rows if start <= r["step"] <= end and isinstance(r.get(metric), float)]
    return statistics.fmean(values) if values else None


def phase_change(rows, metric):
    first = phase_mean(rows, metric, 1, 50)
    last = phase_mean(rows, metric, 251, 300)
    if first is None or last is None:
        return "没有足够的连续训练记录进行阶段首尾比较。"
    delta = last - first
    if abs(delta) < max(0.001, abs(first) * 0.05):
        return f"阶段首尾均值接近（{fmt(first)} → {fmt(last)}），整体以横向波动为主。"
    direction = "上升" if delta > 0 else "下降"
    return f"阶段首尾均值由 {fmt(first)} 变为 {fmt(last)}，总体呈{direction}，但这不代表每个 step 单调变化。"


def chart_analysis(rows, key):
    if key == "loss":
        return "".join(["<p><strong>直观观察：</strong>PG loss 在 300 步内上下摆动，局部峰值对应较强的 outcome policy-gradient 更新；TRACE loss 的振幅明显更小，通常作为 PG 更新上的细粒度修正；total loss 处于两者叠加后的水平。", f"{phase_change(rows, 'actor/pg_loss')} ", "不能把曲线最后变低简单等同于训练收敛，因为 RL loss 受每个 batch 的 advantage 和 rollout 难度影响。 </p>"])
    if key == "stability":
        return "<p><strong>直观观察：</strong>梯度范数反映更新力度，PPO KL 反映新旧策略的采样分布变化，clip fraction 反映有多少 token 被 PPO 保护机制限制。若三者同时持续抬升才更像更新过激；本图整体表现为 KL 和 clipping 较低、梯度范数有 batch-level 起伏，未出现持续性爆炸。</p>"
    if key == "critic":
        return "<p><strong>直观观察：</strong>score/reward 曲线描述每个 batch 中轨迹结果的聚合水平；advantage/return 曲线描述这些结果相对于组内基线的学习信号。advantage 与 return 围绕相对中心波动是 AgentGRPO 的预期现象，不应按监督学习 loss 的单调下降来判断。</p>"
    if key == "train_reward":
        return "<p><strong>直观观察：</strong>训练 reward 的均值曲线在中后期出现抬升，但后段又有所回落；std、min、max 展示同一 batch 内轨迹质量的离散程度和边界。max 偶尔升高只能说明 batch 中出现过高分轨迹，不能说明平均策略已经变好。阶段首尾 reward 均值的变化应结合固定验证曲线共同判断。</p>"
    if key == "critic_reward":
        return "<p><strong>直观观察：</strong>critic score 与 critic reward 的均值曲线用于检查奖励进入 RL 更新前后是否发生变化；min/max 用于观察 batch 内是否存在极端轨迹。本 run 中两组均值基本一致，说明该统计口径下额外 reward 处理没有造成明显偏移。</p>"
    if key == "rl_signals":
        return "<p><strong>直观观察：</strong>advantage 的正负决定策略倾向：正值对应相对更值得增加概率的轨迹，负值对应相对需要抑制的轨迹；return 是该实现中用于形成 outcome 信号的返回量。均值接近中心、min/max 对称波动，说明每个 batch 的组内相对关系在变化，而不是 value loss 发散。</p>"
    if key == "validation":
        return "<p><strong>直观观察：</strong>验证曲线只有固定 checkpoint 才有点，不能把空白区间当成性能没有变化，也不应对其进行视觉上的连续趋势解读。step 300 的验证 avg score 高于若干中期点，但应与 strict/TRACE EM 单独对照，不能用训练 reward 代替严格评测。</p>"
    if key == "trace":
        return "<p><strong>直观观察：</strong>TRACE credit 的 nonzero 数量表示有多少有效 action token 得到非零局部信用，abs mean 表示信用强度；credit 的正负分布决定哪些 action 被鼓励或抑制。持续非零说明 TRACE 路径在工作，但 credit 强度本身不是答案准确率。</p>"
    if key == "behavior":
        return "<p><strong>直观观察：</strong>行为曲线反映轨迹长度、turn 数、终止和中止情况。若 response length 或 turn 数长期升高，可能表示搜索更深，也可能表示效率变差；需要和 reward、overlong rate、验证 EM 一起判断。本 run 的长轨迹特征不能单独证明主动 fold/return 已经学会。</p>"
    return "<p><strong>直观观察：</strong>效率曲线用于观察每个训练 step 的 token、轨迹和有效样本开销。它更适合识别计算负担和数据规模变化，不是性能指标；异常尖峰应与 batch size、response length 和训练吞吐一起核对。</p>"


def chart(title, filename, analysis):
    path = BASE / filename
    if not path.exists():
        return ""
    return f'<section class="chart"><h3>{html.escape(title)}</h3><div class="chart-frame">{path.read_text(encoding="utf-8")}</div><p class="caption">图中横轴为训练 step 1--300；曲线、图例和坐标轴均来自重新排版后的 SVG。</p><div class="chart-analysis">{analysis}</div></section>'


def build():
    rows = read_rows()
    summary = json.loads((BASE / "training_metrics_summary.json").read_text(encoding="utf-8"))
    data = summary["runs"]["importfix_step1_300"]
    charts = [
        ("Actor loss", "importfix_step1_300_actor_loss.svg", "loss"),
        ("Optimization stability", "importfix_step1_300_optimization_stability.svg", "stability"),
        ("Critic signals", "importfix_step1_300_critic_signals.svg", "critic"),
        ("Training reward", "importfix_step1_300_reward_train.svg", "train_reward"),
        ("Critic reward and score", "importfix_step1_300_reward_critic.svg", "critic_reward"),
        ("Advantage and return", "importfix_step1_300_reward_rl_signals.svg", "rl_signals"),
        ("Validation reward", "importfix_step1_300_reward_validation.svg", "validation"),
        ("TRACE credit", "importfix_step1_300_trace.svg", "trace"),
        ("Behavior", "importfix_step1_300_behavior.svg", "behavior"),
        ("Efficiency", "importfix_step1_300_efficiency.svg", "efficiency"),
    ]
    stage_metrics = ["reward/avg_score", "reward/reward_score", "critic/score/mean", "critic/advantages/mean", "critic/returns/mean"]
    html_doc = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Qwen3-8B importfix 训练曲线完整图像描述</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#526174; --line:#d7dee8; --blue:#1d4ed8; --panel:#f7f9fc; --accent:#0f766e; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#eef2f7; color:var(--ink); font-family:Arial,"Microsoft YaHei",sans-serif; font-size:20px; line-height:1.75; }}
main {{ max-width:1900px; margin:0 auto; padding:42px 56px 90px; background:white; }} h1 {{ font-size:42px; line-height:1.25; margin:0 0 18px; }} h2 {{ font-size:32px; border-bottom:3px solid var(--blue); padding-bottom:8px; margin-top:54px; }} h3 {{ font-size:26px; margin:18px 0 12px; }} h4 {{ font-size:22px; margin:18px 0 8px; }} p {{ margin:10px 0; }} .lead {{ font-size:23px; color:#26364d; }} .meta {{ background:#f1f5f9; border-left:6px solid var(--blue); padding:18px 24px; margin:24px 0; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin:22px 0; }} .card {{ border:1px solid var(--line); background:var(--panel); padding:18px 22px; }} .card strong {{ display:block; font-size:25px; color:var(--blue); }} .small {{ color:var(--muted); font-size:17px; }} table {{ width:100%; border-collapse:collapse; margin:18px 0 30px; font-size:17px; }} th,td {{ border:1px solid var(--line); padding:10px 12px; text-align:right; vertical-align:top; }} th {{ background:#e8eef8; font-size:18px; }} th:first-child,td:first-child {{ text-align:left; min-width:230px; }} td span {{ color:var(--muted); font-size:15px; }} code {{ font-family:Consolas,"Courier New",monospace; font-size:16px; color:#9f1239; }} .chart {{ margin:30px 0 42px; page-break-inside:avoid; }} .chart-frame {{ width:100%; overflow-x:auto; border:1px solid var(--line); background:white; padding:12px; }} .chart-frame svg {{ display:block; width:100%; min-width:1200px; height:auto; }} .caption {{ color:var(--muted); font-size:17px; }} .chart-analysis {{ background:#f8fafc; border:1px solid var(--line); border-left:6px solid var(--accent); padding:10px 20px; margin-top:12px; }} .note {{ background:#fff7ed; border-left:6px solid #ea580c; padding:16px 22px; margin:18px 0; }} .good {{ background:#ecfdf5; border-left:6px solid var(--accent); padding:16px 22px; margin:18px 0; }} ul {{ padding-left:32px; }} li {{ margin:5px 0; }} .toc a {{ color:var(--blue); text-decoration:none; }} .formula {{ background:#f8fafc; border:1px solid var(--line); padding:14px 20px; font-family:Consolas,"Courier New",monospace; font-size:19px; overflow:auto; }} .nav {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:24px; }} .nav a,.nav button {{ border:1px solid #9fb3cc; background:#f8fafc; color:#173b71; padding:8px 16px; border-radius:6px; font:inherit; font-size:18px; text-decoration:none; cursor:pointer; }} .nav a:hover,.nav button:hover {{ background:#e8eef8; }} .top-button {{ position:fixed; right:24px; bottom:24px; display:none; border:1px solid #315a91; border-radius:50%; width:58px; height:58px; background:#1d4ed8; color:white; font-size:26px; cursor:pointer; box-shadow:0 3px 12px #0003; }} .top-button.visible {{ display:block; }} @media(max-width:800px) {{ main {{ padding:24px 18px 60px; }} body {{ font-size:17px; }} h1 {{ font-size:32px; }} h2 {{ font-size:26px; }} table {{ font-size:14px; }} .top-button {{ right:16px; bottom:16px; }} }}
</style></head><body id="top"><main>
<nav class="nav"><a href="../index.html">报告首页</a><button type="button" onclick="history.back()">返回上一页</button></nav>
<h1>Qwen3-8B FoldAgent/TRACE importfix 训练曲线完整图像描述与阶段梳理</h1>
<p class="lead">对象：<strong>browsecomp_qwen3_8b_trace_importfix_20260818_0205</strong>。本文只分析 2026-08-18 启动、截至训练 step 300 的 importfix run，不混入 retry3 或其他 Qwen3-8B 实验。</p>
<div class="meta"><strong>数据范围：</strong>step 1--300，共 {data['records']} 条训练记录。原始日志还保存了 step 301--319，但本报告严格按需求截取前 300 步。<br><strong>数据源：</strong><code>AgenticRL-Lab-long/logs/browsecomp_qwen3_8b_trace_importfix_20260818_0205/train/metrics.jsonl</code><br><strong>训练配置：</strong>AgentGRPO、FoldAgent/search_branch、在线 frozen-reference TRACE、<code>TRACE_ALPHA_TURN=0.2</code>、<code>PROCESS_REWARD=[flat]</code>。</div>
<nav class="toc"><h2>目录</h2><ul><li><a href="#overview">1. 总体概览</a></li><li><a href="#semantics">2. 指标与代码语义</a></li><li><a href="#loss">3. Loss 与优化过程</a></li><li><a href="#reward">4. Reward 全部曲线</a></li><li><a href="#phases">5. 六阶段波动梳理</a></li><li><a href="#charts">6. 全部图像说明</a></li><li><a href="#conclusion">7. 结论与边界</a></li></ul></nav>
<h2 id="overview">1. 总体概览</h2>
<div class="grid"><div class="card"><strong>{fmt(data['metrics']['reward/avg_score']['mean'])}</strong>训练 batch avg score 均值<div class="small">范围 {fmt(data['metrics']['reward/avg_score']['min'])}--{fmt(data['metrics']['reward/avg_score']['max'])}</div></div><div class="card"><strong>{fmt(data['metrics']['actor/pg_loss']['mean'])}</strong>PG loss 均值<div class="small">范围 {fmt(data['metrics']['actor/pg_loss']['min'])}--{fmt(data['metrics']['actor/pg_loss']['max'])}</div></div><div class="card"><strong>{fmt(data['metrics']['actor/trace_loss']['mean'])}</strong>TRACE loss 均值<div class="small">绝对量约为 PG loss 的十分之一</div></div><div class="card"><strong>{fmt(data['metrics']['actor/grad_norm']['mean'])}</strong>梯度范数均值<div class="small">最大值 {fmt(data['metrics']['actor/grad_norm']['max'])}</div></div><div class="card"><strong>{fmt(data['metrics']['trace/credit_nonzero']['mean'])}</strong>TRACE nonzero 均值<div class="small">在线 credit 持续非零</div></div><div class="card"><strong>{fmt(data['metrics']['val/avg_score']['last'])}</strong>step 300 验证 avg score<div class="small">验证 reward 为 {fmt(data['metrics']['val/reward']['last'])}</div></div></div>
<div class="note"><strong>核心判断：</strong>训练数值上没有从 PG/TRACE loss、梯度范数、PPO KL 或 clip fraction 单独显示出典型发散；但训练 batch reward 在六个阶段间只是小幅波动，没有呈现稳定单调增长。step 300 的固定验证值高于部分中期 checkpoint，但不能仅凭这条曲线声称稳定 RL 能力提升。</div>
<h2 id="semantics">2. 指标与代码语义</h2>
<p>本报告的整理流程是：<strong>读取原始 metrics.jsonl → 固定 importfix 并截取 step 1--300 → 按指标前缀分组 → 根据训练代码确认张量来源 → 计算全程与六阶段统计 → 绘制 SVG → 在本 HTML 中嵌入图像并解释。</strong>因此，图中的每条曲线都能回溯到原始字段；派生的 <code>actor/total_loss</code> 明确标记为 <code>pg_loss + trace_loss</code>。</p>
<div class="grid"><div class="card"><strong>Reward 层</strong><span>从 reward manager 的 score 到 batch reward 聚合；回答“轨迹结果怎样”。</span></div><div class="card"><strong>RL 信号层</strong><span>由 group-relative score 形成 advantage/return；回答“哪些轨迹相对更值得强化”。</span></div><div class="card"><strong>TRACE 层</strong><span>由 frozen reference 的 prefix likelihood 生成局部 credit；回答“哪些 action span 带来进展”。</span></div><div class="card"><strong>Actor 优化层</strong><span>PG loss 与 TRACE loss 合成策略更新；回答“模型怎样被反向传播更新”。</span></div><div class="card"><strong>诊断层</strong><span>梯度、KL、clip fraction、长度与吞吐；回答“更新是否稳定、代价是否异常”。</span></div><div class="card"><strong>验证层</strong><span>固定 checkpoint 的 val 指标和严格 EM；回答“训练变化是否转化为可比较性能”。</span></div></div>
<h3>2.1 Reward、score、critic reward 的关系</h3><p><code>reward/avg_score</code>、<code>reward/reward_score</code> 是训练 batch 的在线 reward 聚合；<code>critic/score/*</code> 是 token-level score 求和后的统计；<code>critic/rewards/*</code> 是进入 RL 更新的 token-level reward 聚合。在本 run 中两组均值基本一致，说明当前统计口径下没有明显额外偏移。</p>
<h3>2.2 AgentGRPO 的 advantage 与 return</h3><p>代码按 UID 进行 group-relative outcome 计算，并将组内相对优势广播到有效 action token。由于当前没有 process reward mask，advantage 主要是组内相对 outcome 信号，不是普通监督学习 loss。</p><div class="formula">sequence advantage = (score_i - group_mean) / (group_std + epsilon)</div><p>本实现中的 <code>returns</code> 与 outcome advantage 使用同一类返回张量，因此不能把 return 曲线理解为独立 value network 的回归损失。</p>
<h3>2.3 TRACE credit 与 TRACE loss</h3><p>TRACE 在基础 advantage 之后注入：frozen reference 评估 prefix 的平均 log-probability，经过 potential、相邻 decision 差分和有限 horizon 加权，最后使用 <code>alpha_turn=0.2</code> 加入 advantage。</p><div class="formula">final advantage = base outcome advantage + 0.2 × token-level TRACE credit</div><p><code>actor/trace_loss</code> 是 TRACE likelihood objective；正 credit 鼓励对应 action，负 credit 抑制对应 action。它不是最终答案准确率。</p>
<h3>2.4 PG、TRACE 与 total loss</h3><p><code>actor/pg_loss</code> 是 clipped PPO/GRPO policy loss，<code>actor/trace_loss</code> 是 TRACE 辅助 loss。本报告用以下关系展示合成趋势：</p><div class="formula">total loss = PG loss + TRACE loss</div><p>total loss 是展示两个 actor 优化分量合成后的派生量，不把它误认为另一个独立 reward。</p>
<h2 id="loss">3. Loss 与优化过程</h2>
<h3>3.1 Actor loss 统计</h3>{stat_table(rows, ['actor/pg_loss','actor/trace_loss','actor/total_loss'])}<p><code>PG loss</code> 的均值为 {fmt(data['metrics']['actor/pg_loss']['mean'])}，标准差为 {fmt(data['metrics']['actor/pg_loss']['std'])}；<code>TRACE loss</code> 均值为 {fmt(data['metrics']['actor/trace_loss']['mean'])}；派生的 <code>total loss</code> 均值为 {fmt(data['metrics']['actor/total_loss']['mean'])}。PG loss 波动幅度明显大于 TRACE loss，说明这 300 步中 actor 更新主要受 outcome policy gradient 驱动。</p>
<h3>3.2 优化稳定性统计</h3>{stat_table(rows, ['actor/grad_norm','actor/ppo_kl','actor/pg_clipfrac','actor/pg_clipfrac_lower','actor/entropy','rollout_corr/kl','rollout_corr/k3_kl','training/rollout_probs_diff_mean'])}<p><code>actor/ppo_kl</code> 是 sampled log-ratio 近似，不要求每个 step 都非负；接近零表示新旧策略在采样 token 上平均变化较小。<code>pg_clipfrac</code> 均值约为 {data['metrics']['actor/pg_clipfrac']['mean']*100:.4f}%，下界 clip 几乎为零，说明 PPO 裁剪并未大规模限制更新。</p>
<h2 id="reward">4. Reward 全部曲线</h2>
<p>本节按 reward 语义分成四类。每一类既给出全程统计，也给出六阶段均值。训练 reward 和验证 reward 必须分开阅读：前者来自训练 batch 与随机 rollout，后者只在 step 50、100、150、200、250、300 等 checkpoint 记录。</p>
<h3>4.0 统计量应该怎样读</h3><p><strong>均值（mean）</strong>回答“这一段训练中，典型 batch 的水平是多少”，用于看总体中心趋势；<strong>标准差（std）</strong>回答“不同 step 或同一类 batch 之间波动有多大”，std 大说明曲线不稳定或 batch 难度差异大；<strong>最小值（min）</strong>回答“最差观察点有多差”，适合发现失败 batch、负向 advantage 或训练退化；<strong>最大值（max）</strong>回答“最好观察点能达到什么程度”，适合发现偶发高分轨迹，但不能代表通常能力；<strong>首值和末值</strong>只用于比较训练起点和终点，不能证明中间过程单调。</p><div class="note"><strong>是否冗余：</strong>如果目标只是汇报平均趋势，min/max 和 std 可以显得冗余；但对于 RL，它们不是重复信息，因为 reward 常由离散的成功、失败和格式结果组成，极值能揭示稀有成功，std 能揭示 batch 构成变化。本文保留它们作为诊断信息，并把真正的趋势判断放在阶段均值、验证 checkpoint 和曲线形状上，而不是单看某一个极值。</div>
<h3>4.1 训练 Batch Reward</h3>{stat_table(rows, REWARD_GROUPS['训练 Batch Reward'])}<p><code>reward/avg_score</code> 全程均值为 {fmt(data['metrics']['reward/avg_score']['mean'])}，首值为 {fmt(data['metrics']['reward/avg_score']['first'])}，末值为 {fmt(data['metrics']['reward/avg_score']['last'])}。但它在阶段 5 达到最高阶段均值后，阶段 6 回落，显示为波动而非稳定上升。</p>
<h3>4.2 Critic Reward 与 Score</h3>{stat_table(rows, REWARD_GROUPS['Critic Reward 与 Score'])}<p>在本 run 中 <code>critic/score/mean</code> 与 <code>critic/rewards/mean</code> 的统计值一致，均值为 {fmt(data['metrics']['critic/score/mean']['mean'])}。min 曲线全程为 0，max 曲线反映每个 batch 是否出现高分轨迹，不能直接当作平均性能。</p>
<h3>4.3 Advantage 与 Return</h3>{stat_table(rows, REWARD_GROUPS['Advantage 与 Return'])}<p>advantage 和 return 围绕相对中心波动。均值分别为 {fmt(data['metrics']['critic/advantages/mean']['mean'])} 和 {fmt(data['metrics']['critic/returns/mean']['mean'])}；它们在阶段 3、4 和阶段 6 的均值变化，主要反映不同训练 batch 的组内相对 reward 组成变化，不能按普通 loss 的“下降”标准评价。</p>
<h3>4.4 固定验证集 Reward</h3>{stat_table(rows, REWARD_GROUPS['验证集 Reward'])}<p>有效验证点为 step 50、100、150、200、250、300。<code>val/avg_score</code> 依次为 0.3067、0.2867、0.2733、0.2800、0.2333、0.3467；step 300 回升，但仍需结合 strict/TRACE EM 与外部复评解读。汇报记录的 step 300 strict/TRACE normalized EM 为 14/150=9.3%，不能用训练 reward 曲线替代。</p>
<h2 id="phases">5. 六阶段波动梳理</h2>
<p>下表为每 50 个 step 的阶段均值。阶段均值用于观察趋势，不代表同一固定验证集上的性能。</p>{phase_table(rows, stage_metrics)}
<h3>阶段 1：step 1--50</h3><p>训练 avg score 均值约 0.2138，reward score 约 0.2067，critic advantage 均值接近 0。此阶段 PG/TRACE 更新初始波动较大，属于 rollout 分布和策略更新共同调整的早期阶段。</p>
<h3>阶段 2：step 51--100</h3><p>avg score 升至约 0.2313，reward score 约 0.2238，advantage/return 均值同步上移。step 100 的固定验证 avg score 为 0.2867，低于 step 50 的 0.3067，说明训练 batch reward 的小幅抬升没有同步转化为验证集改善。</p>
<h3>阶段 3：step 101--150</h3><p>这是训练 batch reward 较高的阶段，avg score 约 0.2550，reward score 约 0.2421。advantage 均值约 0.0531，但验证 avg score 在 step 150 降至 0.2733；因此不能把训练 reward 的上升直接解释为泛化能力上升。</p>
<h3>阶段 4：step 151--200</h3><p>avg score 回落到约 0.2475，reward score 回落到约 0.2241。advantage 和 return 仍为正均值，但训练 reward 与验证 reward 并不严格同步，step 200 验证 avg score 约 0.2800。</p>
<h3>阶段 5：step 201--250</h3><p>这是训练 batch reward 的最高阶段，avg score 约 0.2575，reward score 约 0.2501；但固定验证 avg score 降至 step 250 的 0.2333，形成明显的“训练 batch reward 较高、验证表现较低”现象，是 reward 与真实性能脱钩的重要证据。</p>
<h3>阶段 6：step 251--300</h3><p>训练 avg score 回落到约 0.2413，reward score 约 0.2335；advantage/return 均值仍为正。step 300 验证 avg score 回升到 0.3467，验证 reward 为 0.3159，但主汇报指出 strict/TRACE EM 仅为 14/150=9.3%，旧 scorer 为 59/150=39.3%，因此必须使用严格评测口径解释该回升。</p>
<h2 id="charts">6. 全部图像说明</h2>
<p>以下图像均已嵌入本 HTML。每张图的绘图区、坐标轴、刻度和图例已分区排版，字体和线宽已放大。</p>
{''.join(chart(title, filename, chart_analysis(rows, analysis_key)) for title, filename, analysis_key in charts)}
<h2 id="conclusion">7. 结论与边界</h2>
<div class="good"><strong>可以确认：</strong><ul><li>importfix 的 PG、TRACE 和 total loss 在 300 步内持续有有限值。</li><li>TRACE credit 持续非零，说明在线 frozen-reference TRACE 确实参与训练。</li><li>PPO KL 接近零、clip fraction 很低，未显示典型的策略更新失控。</li><li>训练 reward 在不同阶段上下波动，阶段 5 较高、阶段 6 回落；验证 reward 在 step 300 回升。</li></ul></div>
<div class="note"><strong>不能确认：</strong><ul><li>不能把 <code>reward/avg_score</code> 或 <code>reward/reward_score</code> 当作 BrowseComp-Plus 固定测试准确率。</li><li>不能把 TRACE loss 越低解释为答案能力越强；TRACE 是 prefix likelihood progress 的辅助 credit。</li><li>不能仅凭 loss 曲线证明模型学会了主动 fold、主动 return 或高效上下文管理。</li><li>step 300 的验证回升仍需与 strict/TRACE normalized EM 和 DeepSeek 旁路复评分开报告。</li></ul></div>
<p><strong>最终汇报表述：</strong>importfix 的 300-step 训练过程没有显示常规数值发散，但在线 reward、critic 相对信号和固定验证表现之间存在明显波动与脱钩。PG 是主要 actor 更新来源，TRACE 提供较小但持续的辅助修正；step 300 的严格评测结果只能作为初步同口径对照，不能单独证明稳定能力提升或主动上下文管理已经学会。</p>
<p class="small">报告生成自 <code>training_metrics_all_runs.csv</code>、<code>training_metrics_summary.json</code>、importfix SVG 图和 2026-08-24 汇报文档。生成脚本见 <code>generate_report.py</code>。</p>
</main><button class="top-button" id="topButton" type="button" aria-label="回到顶端" title="回到顶端">↑</button><script>const b=document.getElementById("topButton");window.addEventListener("scroll",()=>b.classList.toggle("visible",window.scrollY>500));b.addEventListener("click",()=>window.scrollTo({{top:0,behavior:"smooth"}}));</script></body></html>'''
    (BASE / "importfix_训练曲线完整图像描述.html").write_text(html_doc, encoding="utf-8")


if __name__ == "__main__":
    build()
