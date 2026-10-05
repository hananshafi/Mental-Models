#!/usr/bin/env python3
import json, collections
EDIR="projects/bigtom/runs/e1"

def load(f):
    try: return [json.loads(l) for l in open(f) if l.strip()]
    except FileNotFoundError: return None

def sanity_stats(mode):
    rows=load(f"{EDIR}/preds_sanity_{mode}.jsonl")
    if rows is None: return None
    by=collections.defaultdict(lambda:[0,0])
    for r in rows:
        fb = r.get("qtype","").endswith("_tom") and "no_tom" not in r.get("qtype","")
        k="first-order FALSE-belief" if fb else "first-order TRUE-belief"
        by[k][0]+=r["correct"]; by[k][1]+=1
    tot=sum(r["correct"] for r in rows)/len(rows)
    return tot, {k:(v[0]/v[1]*100,v[1]) for k,v in by.items()}

def pair_stats(mode):
    rows=load(f"{EDIR}/preds_pairs_{mode}.jsonl")
    if rows is None: return None
    byid=collections.defaultdict(dict)
    for r in rows: byid[r["pair_id"]][r["condition"]]=r
    # per-condition accuracy
    cond_acc=collections.defaultdict(lambda:[0,0])
    for r in rows: cond_acc[r["condition"]][0]+=r["correct"]; cond_acc[r["condition"]][1]+=1
    n=0; both_correct=0; dir_flip=0; spec_ok=0; spec_den=0; fb_world_bias=0
    for pid,d in byid.items():
        if not {"fb","tb"} <= set(d): continue
        n+=1
        fb,tb=d["fb"],d["tb"]
        both_correct += 1 if (fb["correct"]==1 and tb["correct"]==1) else 0
        dir_flip += 1 if (fb.get("pred_loc")=="loc1" and tb.get("pred_loc")=="loc2") else 0
        # FB world-state shortcut: model answers loc2 (reality) instead of belief loc1
        if fb.get("pred_loc")=="loc2": fb_world_bias+=1
        # specificity: prediction unchanged under belief-irrelevant perturbation
        for base_c,pert_c in [("fb","fb_pert"),("tb","tb_pert")]:
            if pert_c in d:
                spec_den+=1
                if d[base_c].get("pred_loc")==d[pert_c].get("pred_loc"): spec_ok+=1
    return dict(
        n=n,
        cond_acc={c:(v[0]/v[1]*100,v[1]) for c,v in cond_acc.items()},
        belief_flip_acc=both_correct/n*100,
        directional_flip=dir_flip/n*100,
        fb_world_bias=fb_world_bias/n*100,
        specificity=spec_ok/spec_den*100 if spec_den else 0,
    )

print("="*66)
print("E1 — ToMi belief minimal-pairs: sensitivity + specificity")
print("="*66)
for mode,label in [("base","Base (Qwen2.5-7B)"),("grpo","Ours (mental-model GRPO)")]:
    s=sanity_stats(mode); p=pair_stats(mode)
    print(f"\n### {label}")
    if s:
        tot,by=s
        print(f"  [sanity] real ToMi first-order acc: {tot*100:.1f}%  "
              + " | ".join(f"{k}: {v[0]:.1f}%" for k,v in by.items()))
    if p:
        ca=p["cond_acc"]
        print(f"  [pairs] per-condition acc:  "
              + "  ".join(f"{c}={ca[c][0]:.1f}%" for c in ["fb","tb","fb_pert","tb_pert"] if c in ca))
        print(f"  Belief-flip accuracy (both FB&TB correct) : {p['belief_flip_acc']:.1f}%")
        print(f"  Directional flip (loc1 on FB, loc2 on TB) : {p['directional_flip']:.1f}%")
        print(f"  Specificity (pred unchanged under pert)   : {p['specificity']:.1f}%")
        print(f"  FB world-state shortcut (answers reality) : {p['fb_world_bias']:.1f}%")
