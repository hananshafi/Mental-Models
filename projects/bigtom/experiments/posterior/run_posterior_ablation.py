#!/usr/bin/env python3
"""
Posterior-family ablation for the Stage-1 recursive mental+reward model (BigToM).
Keeps EVERYTHING fixed (data, z_dim, reward coupling, budget, seed) and swaps only
the posterior family / structure:

  --posterior diag          diagonal Gaussian (ours)
  --posterior lowrank       low-rank + diagonal covariance  Sigma = diag(sigma^2) + U U^T  (captures within-level correlations)
  --posterior deterministic z = mu, no sampling, no KL      (ablates the variational bottleneck)
  --recursive 0             parallel z1,z2 (z2 does NOT condition on z1) -> isolates recursion vs covariance

Eval: held-out (scenario-level) pairwise reward accuracy, margin, and belief-probe accuracy.
"""
import os, sys, json, time, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

SCRIPTS="projects/bigtom/scripts"
sys.path.insert(0, SCRIPTS)
import stage1_train_mental_reward as S
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Subset

# ---------------- posterior-family model ----------------
class AblationModel(S.RecursiveToMModel):
    def __init__(self, base, z_dim=128, posterior="diag", recursive=1, cov_rank=8):
        super().__init__(base, z_dim=z_dim)
        self.posterior=posterior; self.recursive=int(recursive); self.cov_rank=cov_rank
        self._U1=None; self._U2=None
        if posterior=="lowrank":
            self.z1_U=nn.Linear(self.hidden_size, z_dim*cov_rank)
            self.z2_U=nn.Linear(self.hidden_size+z_dim, z_dim*cov_rank)
            for m in (self.z1_U, self.z2_U):
                nn.init.xavier_uniform_(m.weight, gain=0.01); nn.init.zeros_(m.bias)

    def _sample_family(self, mu_proj, logvar_proj, U_proj, h, which):
        h=h.to(mu_proj.weight.dtype); mu=mu_proj(h)
        if self.posterior=="deterministic":
            setattr(self, f"_U{which}", None)
            return mu, mu, torch.zeros_like(mu)
        logvar=logvar_proj(h).clamp(min=-8.0, max=6.0)
        if self.posterior=="lowrank":
            U=U_proj(h).view(h.size(0), self.z_dim, self.cov_rank).float()
            setattr(self, f"_U{which}", U)
            if self.training:
                std=(0.5*logvar).exp()
                eps_d=torch.randn_like(std)
                eps_r=torch.randn(h.size(0), self.cov_rank, device=h.device, dtype=U.dtype)
                z=mu + std*eps_d + torch.einsum('bdr,br->bd', U, eps_r).to(mu.dtype)
            else:
                z=mu
            return z, mu, logvar
        # diagonal
        setattr(self, f"_U{which}", None)
        if self.training:
            std=(0.5*logvar).exp(); z=mu+std*torch.randn_like(std)
        else:
            z=mu
        return z, mu, logvar

    def encode_z1_z2(self, ctx_ids, ctx_mask, stop_grad_z1=False):
        ctx_last,_=self._encode(ctx_ids, ctx_mask)
        z1,mu1,logvar1=self._sample_family(self.z1_mu, self.z1_logvar, getattr(self,"z1_U",None), ctx_last, 1)
        z1_for_z2=z1.detach() if stop_grad_z1 else z1
        if self.recursive:
            z2_in=torch.cat([ctx_last, z1_for_z2], dim=1)
        else:
            z2_in=torch.cat([ctx_last, torch.zeros_like(z1_for_z2)], dim=1)  # parallel: z2 ⟂ z1
        z2,mu2,logvar2=self._sample_family(self.z2_mu, self.z2_logvar, getattr(self,"z2_U",None), z2_in, 2)
        return ctx_last, z1, mu1, logvar1, z2, mu2, logvar2

# ---------------- KL ----------------
def kl_diag(mu, logvar):
    return -0.5*torch.mean(torch.sum(1+logvar-mu.pow(2)-logvar.exp(), dim=1))

def kl_lowrank(mu, logvar, U):
    sigma2=logvar.exp(); d=mu.size(1); r=U.size(2)
    trace=sigma2.sum(1)+(U**2).sum((1,2))
    quad=(mu**2).sum(1)
    Ut=torch.einsum('bdi,bd,bdj->bij', U, 1.0/sigma2, U)   # [B,r,r]
    M=torch.eye(r, device=U.device, dtype=U.dtype).unsqueeze(0)+Ut
    logdet=torch.log(sigma2).sum(1)+torch.logdet(M)
    return (0.5*(trace+quad-d-logdet)).mean()

def compute_loss(model, batch, device, args, step):
    batch={k:v.to(device) for k,v in batch.items()}
    stop=step<args.z1_stop_grad_steps
    out=model.forward_all(
        ctx_ids=batch["ctx_ids"], ctx_mask=batch["ctx_mask"],
        pos_ids=batch["pos_ids"], pos_mask=batch["pos_mask"],
        neg_ids=batch["neg_ids"], neg_mask=batch["neg_mask"],
        m1_ids=batch["m1_ids"], m1_mask=batch["m1_mask"],
        m2_ids=batch["m2_ids"], m2_mask=batch["m2_mask"],
        first_pos_token=batch["first_pos"], belief_label=batch["belief_label"],
        stop_grad_z1=stop)
    pref=F.softplus(out["neg_r"]-out["pos_r"]).mean()
    reward_reg=(F.smooth_l1_loss(out["pos_r"], torch.ones_like(out["pos_r"]))
                +F.smooth_l1_loss(out["neg_r"], torch.zeros_like(out["neg_r"])))
    branch=(1.0-batch["belief_label"].float()).unsqueeze(-1)
    zreg=F.smooth_l1_loss(out["z1_only_r"], branch)+F.smooth_l1_loss(out["zc_r"], branch)
    belief_ce=F.cross_entropy(out["belief_logits"], batch["belief_label"])
    with torch.amp.autocast(device_type="cuda", enabled=False):
        if model.posterior=="deterministic":
            kl1=kl2=torch.zeros((), device=device)
        elif model.posterior=="lowrank":
            kl1=kl_lowrank(out["mu1"].float(), out["logvar1"].float(), model._U1)
            kl2=kl_lowrank(out["mu2"].float(), out["logvar2"].float(), model._U2)
        else:
            kl1=kl_diag(out["mu1"].float(), out["logvar1"].float())
            kl2=kl_diag(out["mu2"].float(), out["logvar2"].float())
    if args.kl_anneal_steps>0:
        a1=min(1.0, step/args.kl_anneal_steps)
        z2s=args.kl_anneal_steps+args.z2_kl_delay_steps
        a2=min(1.0, max(0.0, step-z2s)/args.kl_anneal_steps)
    else: a1=a2=1.0
    total=(args.pref_weight*pref+reward_reg+args.z_only_weight*zreg+args.belief_weight*belief_ce
           +args.kl_weight*a1*kl1+args.kl_weight*a2*kl2
           +args.m1_weight*out["m1_loss"]+args.m2_weight*out["m2_loss"]
           +args.future_weight*out["future_loss"])
    return total, {"total":total.item(),"pref":pref.item(),"kl1":float(kl1),"kl2":float(kl2),
                   "belief_acc":(out["belief_logits"].argmax(-1)==batch["belief_label"]).float().mean().item()}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct=0; n=0; bacc=0; bn=0
    margins=[]; belief_nll=0.0; m1_sum=0.0; m2_sum=0.0; nb=0
    conf_bins=[[0.0,0] for _ in range(10)]  # (sum_correct, count) per confidence bin for ECE
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
        d=(out["pos_r"]-out["neg_r"]).squeeze(-1)
        correct+=(d>0).sum().item(); margins.append(d.float().cpu()); n+=d.numel()
        bl=batch["belief_label"]
        probs=torch.softmax(out["belief_logits"].float(), dim=-1)
        pred=probs.argmax(-1); conf=probs.max(-1).values
        bacc+=(pred==bl).sum().item(); bn+=bl.numel()
        belief_nll+=F.nll_loss(torch.log(probs+1e-9), bl, reduction="sum").item()
        for c,ok in zip(conf.cpu().tolist(), (pred==bl).cpu().tolist()):
            b=min(9,int(c*10)); conf_bins[b][0]+=1.0 if ok else 0.0; conf_bins[b][1]+=1
        # second-order (z2) and first-order (z1) decode NLL on val
        m1_sum+=out["m1_loss"].item()*bl.numel(); m2_sum+=out["m2_loss"].item()*bl.numel(); nb+=bl.numel()
    margins=torch.cat(margins)
    # ECE
    ece=0.0
    for i,(sc,cnt) in enumerate(conf_bins):
        if cnt==0: continue
        acc_bin=sc/cnt; conf_bin=(i+0.5)/10.0; ece+=(cnt/n)*abs(acc_bin-conf_bin)
    return {"pairwise_acc":correct/n*100, "mean_margin":float(margins.mean()),
            "margin_std":float(margins.std()), "belief_probe_acc":bacc/bn*100,
            "belief_nll":belief_nll/bn, "belief_ece":ece*100,
            "first_order_decode_nll":m1_sum/nb, "second_order_decode_nll":m2_sum/nb, "n":n}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--posterior", choices=["diag","lowrank","deterministic"], default="diag")
    ap.add_argument("--recursive", type=int, default=1)
    ap.add_argument("--cov_rank", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="projects/bigtom/runs/posterior")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--kl_weight", type=float, default=0.1)
    ap.add_argument("--kl_anneal_steps", type=int, default=500)
    ap.add_argument("--z2_kl_delay_steps", type=int, default=500)
    ap.add_argument("--z1_stop_grad_steps", type=int, default=300)
    ap.add_argument("--m1_weight", type=float, default=0.5)
    ap.add_argument("--m2_weight", type=float, default=0.3)
    ap.add_argument("--future_weight", type=float, default=0.3)
    ap.add_argument("--z_only_weight", type=float, default=0.3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_ctx_len", type=int, default=768)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--pref_weight", type=float, default=1.0)     # hard-negative preference term
    ap.add_argument("--belief_weight", type=float, default=1.0)   # belief classifier CE
    ap.add_argument("--save_ckpt", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda")
    tok=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                              trust_remote_code=True, device_map={"":device})
    base.config.pad_token_id=tok.pad_token_id; base.gradient_checkpointing_enable()
    base=get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
        lora_alpha=args.lora_alpha, target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, bias="none"))
    model=AblationModel(base, z_dim=args.z_dim, posterior=args.posterior,
                        recursive=args.recursive, cov_rank=args.cov_rank).to(device)
    for nme,p in model.named_parameters():
        if not nme.startswith("base_model.") and not nme.startswith("transformer."):
            p.data=p.data.float()

    ds=S.BigToMRecursiveDataset(args.data, tok, max_ctx_len=args.max_ctx_len)
    # scenario-level split
    sids=sorted({s["sid"] for s in ds.samples})
    rng=random.Random(123); rng.shuffle(sids)
    n_val=max(1, int(len(sids)*args.val_frac)); val_sids=set(sids[:n_val])
    tr_idx=[i for i,s in enumerate(ds.samples) if s["sid"] not in val_sids]
    va_idx=[i for i,s in enumerate(ds.samples) if s["sid"] in val_sids]
    coll=lambda b: S.collate(b, pad_id=tok.pad_token_id)
    tl=DataLoader(Subset(ds,tr_idx), batch_size=args.batch_size, shuffle=True, collate_fn=coll, num_workers=2, drop_last=True)
    vl=DataLoader(Subset(ds,va_idx), batch_size=args.batch_size, shuffle=False, collate_fn=coll, num_workers=2)
    print(f"[{args.tag}] train_samples={len(tr_idx)} val_samples={len(va_idx)} "
          f"posterior={args.posterior} recursive={args.recursive}", flush=True)

    total_steps=max(1,(len(tl)//args.grad_accum)*args.epochs)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sch=get_linear_schedule_with_warmup(opt, int(0.05*total_steps), total_steps)

    gstep=0; t0=time.time(); opt.zero_grad(set_to_none=True)
    for ep in range(args.epochs):
        model.train()
        for i,batch in enumerate(tl):
            loss,m=compute_loss(model, batch, device, args, gstep)
            (loss/args.grad_accum).backward()
            if (i+1)%args.grad_accum: continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); gstep+=1
            if gstep==1 or gstep%25==0:
                print(f"[{args.tag}] ep{ep} step {gstep}/{total_steps} {(time.time()-t0)/60:.1f}m "
                      f"total={m['total']:.3f} pref={m['pref']:.3f} kl1={m['kl1']:.3f} kl2={m['kl2']:.3f} bacc={m['belief_acc']:.2f}", flush=True)
    res=evaluate(model, vl, device)
    res.update({"posterior":args.posterior,"recursive":args.recursive,"tag":args.tag,
                "epochs":args.epochs,"total_steps":total_steps,"train_min":(time.time()-t0)/60})
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}/result_{args.tag}.json","w") as f: json.dump(res,f,indent=2)
    print(f"[{args.tag}] RESULT {json.dumps(res)}", flush=True)

    if args.save_ckpt:
        # Save in the format load_stage1_encoder expects: <ckpt>/lora + <ckpt>/heads.pt
        # For the non-recursive (parallel) variant, the stock loader still feeds z1 into z2's
        # input; zero the z1 columns of the z2 heads so the recursive deploy path reproduces the
        # parallel computation (z2 depends only on context) exactly.
        if not model.recursive:
            with torch.no_grad():
                model.z2_mu.weight[:, model.hidden_size:].zero_()
                model.z2_logvar.weight[:, model.hidden_size:].zero_()
        ck=Path(args.save_ckpt); ck.mkdir(parents=True, exist_ok=True)
        model.base_model.save_pretrained(str(ck/"lora"))
        heads={k:v.detach().cpu() for k,v in model.state_dict().items()
               if not k.startswith("base_model.") and not k.startswith("transformer.")}
        torch.save({"state_dict":heads,"args":vars(args)}, ck/"heads.pt")
        print(f"[{args.tag}] saved stage1 ckpt -> {ck}", flush=True)

if __name__=="__main__":
    main()
