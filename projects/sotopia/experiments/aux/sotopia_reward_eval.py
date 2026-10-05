#!/usr/bin/env python3
"""
SOTOPIA reward-model eval (7-dim graded reward). Well-defined, non-binary metric:
  - hard-negative preference accuracy: does the reward prefer the good response over the
    fluent-but-socially-poor hard negative? (scalarized + per-dim)
  - 7-dim reward regression: correlation between predicted and gold reward vectors.
Usage: point --ckpt at a checkpoint dir (lora_adapter/ + *.pth heads).
"""
import sys, os, json, argparse, torch
sys.path.insert(0, "projects/sotopia")
import stage1_train_coupled_mental_reward_v3 as S3
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader, random_split

ap=argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--data", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
args=ap.parse_args()

dev=torch.device("cuda:0")
tok=AutoTokenizer.from_pretrained(args.model_name)
if tok.pad_token is None: tok.pad_token=tok.eos_token
base=AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map={"":dev})
base=PeftModel.from_pretrained(base, os.path.join(args.ckpt,"lora_adapter"))
model=S3.RecursiveToMModel(base, reward_dim=7, z_dim=128).to(dev)
# load heads
for name in ["z1_mu","z1_logvar","z2_mu","z2_logvar","joint_outcome_head","z1_only_reward_head",
             "z_combined_reward_head","z_to_hidden","mental1_decoder","mental2_decoder",
             "expl_cross_attn","expl_reward_head"]:
    p=os.path.join(args.ckpt,f"{name}.pth")
    if os.path.exists(p):
        getattr(model,name).load_state_dict(torch.load(p,map_location=dev));
for pmt in model.parameters(): pmt.requires_grad=False
model.eval()
# cast head modules to bf16 to match the (bf16) transformer hidden states
for n,pmt in model.named_parameters():
    if not n.startswith("base_model."): pmt.data=pmt.data.to(torch.bfloat16)

ds=S3.RecursiveToMDataset(args.data, tok, max_ctx_len=1024)
n_val=max(16,int(len(ds)*0.1))
g=torch.Generator().manual_seed(42)
_, val = random_split(ds, [len(ds)-n_val, n_val], generator=g)
loader=DataLoader(val, batch_size=4, shuffle=False,
                  collate_fn=lambda b: S3.collate_fn(b, tok))

import numpy as np
DIMS=S3.REWARD_DIMS if hasattr(S3,"REWARD_DIMS") else list(range(7))
scal_correct=0; n_neg=0
perdim_correct=np.zeros(7); preds=[]; golds=[]
with torch.no_grad():
    for batch in loader:
        batch={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}
        posr=model.predict_reward(batch["ctx_input_ids"],batch["ctx_attention_mask"],
                                  batch["pos_input_ids"],batch["pos_attention_mask"]).float()
        negr=model.predict_reward(batch["ctx_input_ids"],batch["ctx_attention_mask"],
                                  batch["neg_input_ids"],batch["neg_attention_mask"]).float()
        hn=batch["has_negative"].float().cpu().numpy()
        ps=posr.sum(-1).cpu().numpy(); ns=negr.sum(-1).cpu().numpy()
        for i in range(len(hn)):
            if hn[i]>0:
                n_neg+=1
                scal_correct += 1 if ps[i]>ns[i] else 0
                perdim_correct += (posr[i].cpu().numpy()>negr[i].cpu().numpy()).astype(float)
        preds.append(posr.cpu().numpy()); golds.append(batch["reward_vec"].float().cpu().numpy())
preds=np.concatenate(preds); golds=np.concatenate(golds)
# per-dim Pearson corr (pred pos reward vs gold)
corrs=[float(np.corrcoef(preds[:,d],golds[:,d])[0,1]) if preds[:,d].std()>1e-6 else float("nan") for d in range(7)]
res={"tag":args.tag,
     "hardneg_pref_acc_scalar": scal_correct/max(1,n_neg)*100,
     "hardneg_pref_acc_perdim": (perdim_correct/max(1,n_neg)*100).round(1).tolist(),
     "reward_regression_corr_mean": float(np.nanmean(corrs)),
     "reward_regression_corr_perdim": [round(c,3) for c in corrs],
     "n_neg": n_neg, "n_val": len(preds)}
print("SOTOPIA_REWARD_RESULT", json.dumps(res))
os.makedirs("projects/sotopia/runs/aux", exist_ok=True)
json.dump(res, open(f"projects/sotopia/runs/aux/sreval_{args.tag}.json","w"), indent=2)
