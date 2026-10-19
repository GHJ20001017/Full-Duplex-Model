import unittest,tempfile,json,hashlib
from pathlib import Path
import torch
from preservation import preservation_kl,load_reference
class Tests(unittest.TestCase):
 def test_identity(self):
  q=torch.tensor([[.1,.7,.2]]);z=q.log().requires_grad_();l=preservation_kl(z,q,torch.ones(1));l.backward();self.assertLess(abs(l.item()),1e-6);self.assertLess(z.grad.abs().max().item(),1e-6)
 def test_direction_and_mask(self):
  z=torch.zeros(2,3,requires_grad=True);q=torch.tensor([[.05,.05,.9],[.9,.05,.05]])
  l=preservation_kl(z,q,torch.tensor([1.,0.]));l.backward();self.assertLess(z.grad[0,2].item(),0);self.assertTrue(torch.equal(z.grad[1],torch.zeros(3)))
  self.assertTrue(torch.allclose(z.grad[0],(torch.ones(3)/3-q[0])/2,atol=1e-6))
 def test_all_masked(self):
  z=torch.randn(3,3,requires_grad=True);l=preservation_kl(z,torch.ones(3,3)/3,torch.zeros(3));l.backward();self.assertEqual(l.item(),0);self.assertEqual(z.grad.abs().sum().item(),0)
 def test_zero_probability(self):
  z=torch.zeros(1,3,requires_grad=True);l=preservation_kl(z,torch.tensor([[1.,0.,0.]]),torch.ones(1));l.backward();self.assertTrue(torch.isfinite(l));self.assertTrue(torch.isfinite(z.grad).all())
 def test_cache_integrity(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t)/'data';d.write_text('example');r=Path(t)/'ref';c={'data_sha256':hashlib.sha256(d.read_bytes()).hexdigest(),'labels':['continue','yield','wait'],'probabilities':[[.1,.2,.7],None]};r.write_text(json.dumps(c));p,m=load_reference(r,d,2);self.assertEqual(m.tolist(),[1.,0.]);d.write_text('changed')
   with self.assertRaises(AssertionError):load_reference(r,d,2)
if __name__=='__main__':unittest.main()
