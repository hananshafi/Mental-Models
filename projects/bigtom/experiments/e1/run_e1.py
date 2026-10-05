#!/usr/bin/env python3
import sys, os, json, argparse, time
sys.path.insert(0, "projects/bigtom/scripts")
from official_eval_common import load_policy_bundle, build_benchmark_prompt, normalize_text

BASE="Qwen/Qwen2.5-7B-Instruct"
STAGE1="projects/bigtom/checkpoints/stage1_qwen_5k/best_ckpt"
POLICY="projects/bigtom/checkpoints/stage4_qwen_5k/step_300"
EDIR="projects/bigtom/runs/e1"

ap=argparse.ArgumentParser()
ap.add_argument("--mode", choices=["base","grpo"], required=True)
ap.add_argument("--max_new_tokens", type=int, default=24)
ap.add_argument("--pairs_only", action="store_true")
ap.add_argument("--pairs_file", default="e1_pairs.jsonl")
ap.add_argument("--tag", default="")
args=ap.parse_args()

bundle = load_policy_bundle(
    mode=args.mode, base_model=BASE,
    stage1_ckpt=(None if args.mode=="base" else STAGE1),
    policy_ckpt=(None if args.mode=="base" else POLICY),
    z_dim=128,
)
print(f"[loaded] mode={args.mode}", flush=True)

def run_file(infile, outfile):
    rows=[json.loads(l) for l in open(infile) if l.strip()]
    preds=[]; t0=time.time()
    for i,r in enumerate(rows,1):
        prompt=build_benchmark_prompt(r["story"], r["question"], dataset_name="ToMi")
        out=bundle.generate(prompt, story=r["story"], question=r["question"], max_new_tokens=args.max_new_tokens)
        pn=normalize_text(out); gn=normalize_text(r["answer"])
        r=dict(r)
        r["prediction"]=out
        r["correct"]=1.0 if (gn==pn or gn in pn) else 0.0
        # which candidate location did the model name (for flip/specificity direction)
        if "loc1" in r:
            l1=normalize_text(r["loc1"]); l2=normalize_text(r["loc2"])
            r["pred_loc"] = ("loc1" if l1 in pn else "loc2" if l2 in pn else "other")
        preds.append(r)
        if i%60==0: print(f"  {infile.split('/')[-1]} {i}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    with open(outfile,"w") as f:
        for r in preds: f.write(json.dumps(r)+"\n")
    acc=sum(p["correct"] for p in preds)/len(preds)
    print(f"[done] {outfile}  acc={acc:.3f}  n={len(preds)}", flush=True)

tag = args.tag or ""
if not args.pairs_only:
    run_file(f"{EDIR}/e1_sanity.jsonl", f"{EDIR}/preds_sanity_{args.mode}.jsonl")
run_file(f"{EDIR}/{args.pairs_file}",  f"{EDIR}/preds_pairs{tag}_{args.mode}.jsonl")
print("ALL DONE", flush=True)
