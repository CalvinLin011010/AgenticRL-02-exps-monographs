#!/usr/bin/env python3
"""Reproduce aggregate and typical-case statistics for the Seed-OSS SWE-bench run."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_RUN_DIR = Path("/gemini/space/TablePipeline-lr/runtime/experiments/swebench_seed_oss_36b_inference_20260914/verified_stratified_200/sample200")
DEFAULT_INSTANCES = Path("/gemini/space/TablePipeline-lr/lr/00-docs/experiments/shared/settings/datasets/2026-09-14-swebench-verified-stratified-200/instances.csv")

def mean(xs):
    xs=list(xs); return statistics.fmean(xs) if xs else None

def median(xs):
    xs=list(xs); return statistics.median(xs) if xs else None

def percentile(xs,q):
    xs=sorted(xs)
    if not xs: return None
    pos=(len(xs)-1)*q; lo,hi=math.floor(pos),math.ceil(pos)
    return xs[lo] if lo==hi else xs[lo]*(hi-pos)+xs[hi]*(pos-lo)

def wilson(k,n,z=1.959963984540054):
    if not n: return None
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

def numeric_summary(rows,key):
    xs=[float(r[key]) for r in rows if r.get(key) is not None]
    return {"n":len(xs),"mean":mean(xs),"median":median(xs),"p25":percentile(xs,.25),"p75":percentile(xs,.75),"min":min(xs) if xs else None,"max":max(xs) if xs else None}

def grouped(rows,key):
    groups=defaultdict(list)
    for row in rows: groups[str(row.get(key) or "<missing>")].append(row)
    out=[]
    for value,items in groups.items():
        k=sum(bool(x["resolved"]) for x in items)
        out.append({key:value,"n":len(items),"resolved":k,"resolve_rate":k/len(items),"wilson_95":wilson(k,len(items)),"median_total_token":median(x["total_token"] for x in items if x.get("total_token") is not None),"median_session_time_s":median(x["session_time_s"] for x in items if x.get("session_time_s") is not None)})
    return sorted(out,key=lambda x:(-x["n"],x[key]))

def load_events(path):
    cases=defaultdict(list); types=Counter(); current=None
    with path.open(encoding="utf-8") as f:
        for line in f:
            event=json.loads(line); types[event["event"]]+=1
            if event["event"]=="case_start": current=event["uid"]
            if current is not None: cases[current].append(event)
    return cases,types

def event_metrics(events):
    req=[e for e in events if e["event"]=="generation_request"]
    resp=[e for e in events if e["event"]=="generation_response"]
    prompts=[e.get("prompt_tokens",0) for e in req]; times=[e["time"] for e in events if "time" in e]
    return {"generation_requests":len(req),"generation_responses":len(resp),"prompt_tokens_mean":mean(prompts),"prompt_tokens_max":max(prompts) if prompts else None,"prompt_tokens_sum":sum(prompts),"response_chars_sum":sum(e.get("response_chars",0) for e in resp),"summary_due_events":sum(e["event"]=="summary_due" for e in events),"summary_created_events":sum(e["event"]=="summary_created" for e in events),"event_span_s":max(times)-min(times) if len(times)>1 else 0}

def robust_medoid(candidates):
    features=["session_time_s","total_token","action_count","generation_requests","prompt_tokens_max","summary_count"]
    centers={k:median(r[k] for r in candidates if r.get(k) is not None) for k in features}
    scales={k:median(abs(r[k]-centers[k]) for r in candidates if r.get(k) is not None) or 1 for k in features}
    def distance(row):
        return sum(abs(row[k]-centers[k])/scales[k] for k in features if row.get(k) is not None)
    return sorted(candidates,key=lambda r:(distance(r),r["position"]))

def select_typical(rows):
    selected=[]; used=set()
    def take(label,candidates,reason):
        for row in candidates:
            if row["instance_id"] not in used:
                selected.append({"category":label,"selection_reason":reason,**row}); used.add(row["instance_id"]); return
    ok=[r for r in rows if r["resolved"]]
    valid=[r for r in rows if not r["resolved"] and r["resolution_status"] in {"RESOLVED_NO","RESOLVED_PARTIAL"}]
    infra=[r for r in rows if r["resolution_status"]=="NO_REPORT"]
    take("resolved-medoid",robust_medoid(ok),"在成功组内按时长、token、动作、请求、上下文峰值和折叠数做 MAD 归一化后，距组中位数最近")
    take("unresolved-medoid",robust_medoid(valid),"在有正式报告的未解决组内按同一规则选取中位代表")
    take("successful-efficient",sorted(ok,key=lambda r:(r["total_token"],r["session_time_s"])),"成功样本中 total_token 最低，代表高效修复边界")
    take("successful-context-folded",sorted([r for r in ok if r["summary_count"]>0],key=lambda r:(r["summary_count"],r["total_token"]),reverse=True),"成功且触发上下文折叠，优先选择 token 消耗较高者")
    take("partial-near-miss",sorted([r for r in valid if r["resolution_status"]=="RESOLVED_PARTIAL"],key=lambda r:r["f2p_success_count"]/max(1,r["f2p_total"]),reverse=True),"官方 PARTIAL 中目标测试通过比例最高")
    take("regression-only-near-miss",sorted([r for r in valid if r["f2p_failure_count"]==0 and r["p2p_failure_count"]>0],key=lambda r:r["p2p_failure_count"]),"目标测试全过但出现回归，按回归数最少优先")
    take("costly-failure-outlier",sorted(valid,key=lambda r:(r["session_time_s"],r["response_chars_sum"]),reverse=True),"有正式报告的未解决样本中 session_time 最高，明确作为成本异常值而非中位代表")
    take("infrastructure-timeout",sorted(infra,key=lambda r:(r["session_time_s"],r["event_span_s"]),reverse=True),"无完整官方报告样本中活跃时长最高，用于区分基础设施失败")
    return selected

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,default=DEFAULT_RUN_DIR); p.add_argument("--instances",type=Path,default=DEFAULT_INSTANCES); p.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"data"); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    result_path=a.run_dir/"swe_results_20260915_133213.json"
    with result_path.open(encoding="utf-8") as f: raw=json.load(f)
    with a.instances.open(encoding="utf-8",newline="") as f: meta={r["instance_id"]:r for r in csv.DictReader(f)}
    events,event_types=load_events(a.run_dir/"events.jsonl"); rows=[]
    for result in raw["results"]:
        uid=result["instance_id"]; env=result.get("env_stats") or {}; report=result.get("report") or {}; f2p=report.get("fail_to_pass") or {}; p2p=report.get("pass_to_pass") or {}; m=meta.get(uid,{})
        rows.append({"position":int(m["position"]) if m.get("position") else None,"instance_id":uid,"repo":result.get("repo") or m.get("repo"),"difficulty":m.get("difficulty","<missing>"),"version":m.get("version"),"resolved":bool(result.get("resolved")),"score":result.get("score"),"resolution_status":report.get("resolution_status","NO_REPORT"),"num_turns":result.get("num_turns"),"action_count":env.get("action"),"main_turn":env.get("main_turn"),"traj_num":env.get("traj_num"),"session_time_s":env.get("session_time"),"env_init_time_s":env.get("env_init_time"),"main_len":env.get("main_len"),"total_token":env.get("total_token"),"summary_count":env.get("summary_count",0),"finish_accepted":env.get("finish_accepted",0),"stop_reason":env.get("stop_reason","<missing>"),"f2p_success_count":f2p.get("success_count",0),"f2p_failure_count":f2p.get("failure_count",0),"f2p_total":f2p.get("total",0),"p2p_success_count":p2p.get("success_count",0),"p2p_failure_count":p2p.get("failure_count",0),"p2p_total":p2p.get("total",0),**event_metrics(events.get(uid,[]))})
    with (a.output_dir/"case_metrics.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    ok=[r for r in rows if r["resolved"]]; bad=[r for r in rows if not r["resolved"]]; official=[r for r in rows if r["resolution_status"] in {"RESOLVED_FULL","RESOLVED_NO","RESOLVED_PARTIAL","PARTIAL"}]
    keys=["total_token","main_len","num_turns","action_count","session_time_s","env_init_time_s","summary_count","generation_requests","prompt_tokens_max","prompt_tokens_mean","response_chars_sum"]
    taxonomy={
        "target_and_regression_pass":sum(r["f2p_failure_count"]==0 and r["p2p_failure_count"]==0 for r in official),
        "target_pass_regression_fail":sum(r["f2p_failure_count"]==0 and r["p2p_failure_count"]>0 for r in official),
        "target_fail_regression_pass":sum(r["f2p_failure_count"]>0 and r["p2p_failure_count"]==0 for r in official),
        "target_and_regression_fail":sum(r["f2p_failure_count"]>0 and r["p2p_failure_count"]>0 for r in official),
        "no_official_report":len(rows)-len(official),
    }
    summary={
        "source":{"results":str(result_path),"events":str(a.run_dir/"events.jsonl"),"instances":str(a.instances)},
        "overall":{"total":len(rows),"resolved":len(ok),"resolve_rate":len(ok)/len(rows),"wilson_95":wilson(len(ok),len(rows)),"valid_official_reports":len(official),"valid_report_resolve_rate":len(ok)/len(official),"valid_report_wilson_95":wilson(len(ok),len(official)),"no_or_nonstandard_report":len(rows)-len(official),"total_telemetry_tokens":sum(r["total_token"] for r in rows),"summed_session_hours":sum(r["session_time_s"] for r in rows)/3600,"resolves_per_million_telemetry_tokens":len(ok)/(sum(r["total_token"] for r in rows)/1_000_000),"event_type_counts":dict(sorted(event_types.items()))},
        "by_difficulty":grouped(rows,"difficulty"),"by_repo":grouped(rows,"repo"),"by_resolution_status":grouped(rows,"resolution_status"),"by_stop_reason":grouped(rows,"stop_reason"),"by_summary_used":grouped([{**r,"summary_used":"yes" if r["summary_count"] else "no"} for r in rows],"summary_used"),
        "resolved_vs_unresolved":{"resolved":{k:numeric_summary(ok,k) for k in keys},"unresolved":{k:numeric_summary(bad,k) for k in keys}},
        "case_outcome_taxonomy":taxonomy,
        "test_outcomes":{"f2p_success":sum(r["f2p_success_count"] for r in rows),"f2p_failure":sum(r["f2p_failure_count"] for r in rows),"p2p_success":sum(r["p2p_success_count"] for r in rows),"p2p_failure":sum(r["p2p_failure_count"] for r in rows),"unresolved_with_partial_f2p_success":sum((not r["resolved"]) and r["f2p_success_count"]>0 for r in rows),"cases_with_p2p_regression":sum(r["p2p_failure_count"]>0 for r in rows)}
    }
    with (a.output_dir/"aggregate_stats.json").open("w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2); f.write("\n")
    typical=select_typical(rows); details={r["instance_id"]:r for r in raw["results"]}
    for case in typical:
        report=details[case["instance_id"]].get("report") or {}; case["f2p_success_tests"]=(report.get("fail_to_pass") or {}).get("success",[]); case["f2p_failure_tests"]=(report.get("fail_to_pass") or {}).get("failure",[]); case["p2p_failure_tests"]=(report.get("pass_to_pass") or {}).get("failure",[]); case["test_command"]=report.get("test_command")
    with (a.output_dir/"typical_cases.json").open("w",encoding="utf-8") as f: json.dump(typical,f,ensure_ascii=False,indent=2); f.write("\n")
    print(json.dumps({"rows":len(rows),"resolved":len(ok),"typical_cases":[x["instance_id"] for x in typical]},ensure_ascii=False))
if __name__=="__main__": main()
