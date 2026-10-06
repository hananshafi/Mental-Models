#!/usr/bin/env python3
"""
Stage-1-only, NON-CIRCULAR test of mental-state reconstruction:
does the frozen reward model correctly track belief across matched aware/not-aware branches?

For each held-out BigToM (scenario, task, init_belief), the reward must prefer the branch-consistent
answer (pos) over the opposite-branch answer (neg) in BOTH the aware and not-aware conditions.
"Paired belief-flip" = both conditions correct for the same scenario -> the reward flipped its
preference exactly with the belief. This probes belief-tracking (recon's target), not decoding.
"""
import sys, json, argparse, collections, torch
sys.path.insert(0, "projects/bigtom/scripts")
from stage2_policy_sft import load_stage1_encoder
from stage1_train_mental_reward import BigToMRecursiveDataset, collate
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
import random

ap=argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--data", default="projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl")
ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
args=ap.parse_args()

device=torch.device("cuda:0")
tok=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=load_stage1_encoder(args.base_model, __import__("pathlib").Path(args.ckpt), 128, device)
model.eval()

ds=BigToMRecursiveDataset(args.data, tok, max_ctx_len=768)
# same scenario-level val split as training (seed 123, first 10%)
sids=sorted({s["sid"] for s in ds.samples})
rng=random.Random(123); rng.shuffle(sids)
val_sids=set(sids[:max(1,int(len(sids)*0.1))])
va_idx=[i for i,s in enumerate(ds.samples) if s["sid"] in val_sids]
loader=DataLoader(Subset(ds,va_idx), batch_size=8, shuffle=False,
                  collate_fn=lambda b: collate(b, pad_id=tok.pad_token_id), num_workers=2)

# we also need per-example sid/condition/task -> read them from ds.samples in order
meta=[ds.samples[i] for i in va_idx]

recs=[]
mi=0
with torch.no_grad():
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        out=model.forward_all(
            ctx_ids=batch["ctx_ids"], ctx_mask=batch["ctx_mask"],
            pos_ids=batch["pos_ids"], pos_mask=batch["pos_mask"],
            neg_ids=batch["neg_ids"], neg_mask=batch["neg_mask"],
            m1_ids=batch["m1_ids"], m1_mask=batch["m1_mask"],
            m2_ids=batch["m2_ids"], m2_mask=batch["m2_mask"],
            first_pos_token=batch["first_pos"], belief_label=batch["belief_label"],
            stop_grad_z1=False)
        d=(out["pos_r"]-out["neg_r"]).squeeze(-1).float().cpu()
        for j in range(d.numel()):
            s=meta[mi]; mi+=1
            recs.append((s["sid"], s.get("task"), s.get("condition"),
                         1 if d[j]>0 else 0, float(d[j])))

# overall pairwise
correct=sum(r[3] for r in recs); n=len(recs)
margin=sum(r[4] for r in recs)/n
# paired belief-flip: group by (sid, task) -> require BOTH conditions (aware & not_aware) correct
grp=collections.defaultdict(dict)
for sid,task,cond,ok,mg in recs:
    grp[(sid,task)][cond]=ok
pair_ok=pair_n=0
for k,d in grp.items():
    if "aware" in d and "not_aware" in d:
        pair_n+=1; pair_ok+= 1 if (d["aware"]==1 and d["not_aware"]==1) else 0
res={"tag":args.tag,"ckpt":args.ckpt,"pairwise_acc":correct/n*100,
     "mean_margin":margin,"paired_belief_flip":pair_ok/pair_n*100,
     "n":n,"n_pairs":pair_n}
print("REWARD_FLIP_RESULT", json.dumps(res))
json.dump(res, open(f"projects/bigtom/runs/posterior/rflip_{args.tag}.json","w"), indent=2)
