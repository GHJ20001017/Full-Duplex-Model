import json,hashlib
from pathlib import Path
import numpy as np
import torch

def load_reference(path,data_path,n):
    c=json.loads(Path(path).read_text())
    assert c['data_sha256']==hashlib.sha256(Path(data_path).read_bytes()).hexdigest(), 'Reference/data mismatch'
    assert c['labels']==['continue','yield','wait']
    assert len(c['probabilities'])==n
    probs=np.array([x if x is not None else [1/3]*3 for x in c['probabilities']],dtype=np.float32)
    masks=np.array([x is not None for x in c['probabilities']],dtype=np.float32)
    assert probs.shape==(n,3) and np.isfinite(probs).all() and (probs>=0).all()
    assert np.allclose(probs.sum(1),1,atol=1e-5)
    return probs,masks

def preservation_kl(logits,reference,mask):
    # Batch-mean masked KL: augmented rows have exactly zero preservation gradient.
    q=reference.detach().float()
    kl=(q*(q.clamp_min(1e-12).log()-logits.float().log_softmax(-1))).sum(-1)
    return (kl*mask.detach().float()).mean()
