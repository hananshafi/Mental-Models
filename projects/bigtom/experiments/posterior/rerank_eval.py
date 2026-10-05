#!/usr/bin/env python3
"""
Reward-reranking single metric: for each hard (mixed-pool) BigToM question, the reward model scores
all N candidates and picks its top-1. Metric = fraction of top-1 picks that are correct.
As auxiliary components are dropped from the reward model, this single number moves.
"""
import sys, json, argparse, torch, torch.nn.functional as F
sys.path.insert(0, "projects/bigtom/scripts")
from stage3_policy_sft import load_stage1_encoder
from stage1_train_mental_reward import build_encoder_context
from transformers import AutoTokenizer
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--cands", default="projects/bigtom/runs/posterior/rerank_candidates.jsonl")
ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
args=ap.parse_args()

dev=torch.device("cuda:0")
tok=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=load_stage1_encoder(args.base_model, Path(args.ckpt), 128, dev); model.eval()

def last_hidden(ids, mask):
    out=model.transformer(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    h=out.last_hidden_state; idx=mask.sum(1)-1
    return h.gather(1, idx.view(-1,1,1).expand(-1,1,h.size(-1))).squeeze(1)

@torch.no_grad()
def reward(ctx_text, resp_text):
    c=tok(ctx_text, return_tensors="pt", truncation=True, max_length=768).to(dev)
    mu1,mu2=model.encode_z1_z2_deterministic(c["input_ids"], c["attention_mask"])
    r=tok(resp_text, return_tensors="pt", truncation=True, max_length=128).to(dev)
    rh=last_hidden(r["input_ids"], r["attention_mask"]).to(mu1.dtype)
    x=torch.cat([mu1, mu2, rh], dim=1).to(model.joint_outcome_head[0].weight.dtype)
    return float(model.joint_outcome_head(x).squeeze())

rows=[json.loads(l) for l in open(args.cands)]
top1_correct=0; n=0; rand_exp=0.0
for q in rows:
    ctx=q["context"]  # already includes story + question
    scored=[]
    for c in q["candidates"]:
        if c["label"]<0: continue           # skip ambiguous candidates
        scored.append((reward(ctx, c["text"]), c["label"]))
    if not scored or all(l==scored[0][1] for _,l in scored): continue
    scored.sort(key=lambda t:-t[0])
    top1_correct += scored[0][1]; n+=1
    labs=[l for _,l in scored]; rand_exp += sum(labs)/len(labs)
res={"tag":args.tag,"rerank_top1_acc":top1_correct/n*100,"random_baseline":rand_exp/n*100,"n":n}
print("RERANK_RESULT", json.dumps(res))
json.dump(res, open(f"projects/bigtom/runs/posterior/rerank_{args.tag}.json","w"), indent=2)
