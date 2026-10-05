#!/usr/bin/env python3
"""
Generate a best-of-N candidate pool for reward-reranking, focused on the HARD stratum
(false-belief-change belief questions) where the base model is near chance -> the pool is
~50/50 correct/wrong, so the reward model's ranking quality is stressed and discriminative.
Each candidate is labelled correct/wrong by which belief the answer expresses (pos vs neg key).
"""
import sys, json, random, re, torch
sys.path.insert(0, "projects/bigtom/scripts")
from stage1_train_mental_reward import BigToMRecursiveDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(0)
BASE="Qwen/Qwen2.5-7B-Instruct"
DATA="projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl"
OUT="projects/bigtom/runs/posterior/rerank_candidates.jsonl"
N=6; N_Q=150; SCAN=600   # scan up to SCAN questions, keep N_Q with MIXED (discriminative) pools

tok=AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"":"cuda:0"}).eval()

ds=BigToMRecursiveDataset(DATA, tok, max_ctx_len=768)
# HARD stratum: true-belief (aware) belief questions -> base is near chance on the init-belief cases,
# so many pools come out mixed. We scan and keep only questions with a MIXED pool (base uncertain).
hard=[s for s in ds.samples if s["task"] in ("forward_belief","backward_belief")
      and s["condition"]=="aware"]
random.shuffle(hard); hard=hard[:SCAN]

def key_token(pos, neg):
    """the distinguishing content word between the pos and neg answer sentences."""
    pw=set(re.findall(r"[a-z_]+", pos.lower())); nw=set(re.findall(r"[a-z_]+", neg.lower()))
    pk=[w for w in pw-nw if len(w)>2]; nk=[w for w in nw-pw if len(w)>2]
    return (pk[0] if pk else None), (nk[0] if nk else None)

out=[]
with torch.no_grad():
    for i,s in enumerate(hard):
        pk,nk=key_token(s["pos_answer"], s["neg_answer"])
        if not pk or not nk: continue
        prompt=f"{s['context']}\nAnswer:"
        ids=tok(prompt, return_tensors="pt", truncation=True, max_length=900).input_ids.to("cuda:0")
        gen=model.generate(ids, max_new_tokens=24, do_sample=True, temperature=0.9, top_p=0.95,
                           num_return_sequences=N, pad_token_id=tok.pad_token_id)
        cands=[]
        for g in gen:
            txt=tok.decode(g[ids.size(1):], skip_special_tokens=True).strip().lower()
            if pk in txt and nk not in txt: lab=1
            elif nk in txt and pk not in txt: lab=0
            else: lab=-1   # ambiguous/other
            cands.append({"text":txt[:120],"label":lab})
        n_ok=sum(c["label"]==1 for c in cands); n_bad=sum(c["label"]==0 for c in cands)
        if n_ok>=1 and n_bad>=1:   # keep only MIXED (discriminative) pools
            out.append({"context":s["context"], "pos_answer":s["pos_answer"], "neg_answer":s["neg_answer"],
                        "pk":pk, "nk":nk, "candidates":cands, "n_ok":n_ok, "n_bad":n_bad})
        if (i+1)%50==0: print(f"  scanned {i+1}  kept {len(out)}/{N_Q} mixed", flush=True)
        if len(out)>=N_Q: break

with open(OUT,"w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
# pool difficulty summary
tot_ok=sum(r["n_ok"] for r in out); tot=sum(len(r["candidates"]) for r in out)
print(f"wrote {len(out)} questions; candidate correct-rate={tot_ok/tot:.2f} (want ~0.5 for a discriminative pool)")
