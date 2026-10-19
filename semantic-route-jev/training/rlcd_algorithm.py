import torch

def sigma_at_step(step,total,start=.4,end=.1):
    if total < 1 or step < 0 or step >= total:
        raise ValueError("Invalid optimizer step")
    fraction=step/max(1,total-1)
    return start+(end-start)*fraction

def advantages(rewards,mode):
    if rewards.ndim!=2 or rewards.shape[0]<2:
        raise ValueError("Expected G>=2 by batch rewards")
    r=rewards.detach()
    if mode=="rloo":
        return r-(r.sum(0,keepdim=True)-r)/(r.shape[0]-1)
    if mode=="legacy":
        a=r-r.mean(0,keepdim=True)
        return a/(a.std()+1e-6)
    raise ValueError(mode)

def logit_gradient_stats(logits,parts):
    gradients={name:torch.autograd.grad(loss,logits,retain_graph=True)[0].detach().float() for name,loss in parts.items()}
    out={name+"_logit_grad_norm":float(g.norm()) for name,g in gradients.items()}
    a,b=gradients["rl"].flatten(),(gradients["ce"]+gradients["kl"]).flatten()
    out["rl_vs_ce_kl_cosine"]=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
    return out
