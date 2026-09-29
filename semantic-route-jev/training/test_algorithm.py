import unittest,torch
from rlcd_algorithm import sigma_at_step,advantages,logit_gradient_stats
class Tests(unittest.TestCase):
 def test_schedule_endpoints(self):
  self.assertAlmostEqual(sigma_at_step(0,912),.4)
  self.assertAlmostEqual(sigma_at_step(911,912),.1)
  ss=[sigma_at_step(i,912) for i in range(912)]
  self.assertTrue(all(a>=b for a,b in zip(ss,ss[1:])))
 def test_accumulation(self):
  steps=[i//16 for i in range(14577)]
  self.assertEqual(steps[-1],911)
  self.assertEqual(len(set(sigma_at_step(i,912) for i in steps)),912)
 def test_one_update(self):
  self.assertEqual(sigma_at_step(0,1),.4)
  with self.assertRaises(ValueError):sigma_at_step(1,1)
 def test_rloo(self):
  r=torch.tensor([[1.,10.],[2.,20.],[3.,30.],[4.,40.]],requires_grad=True)
  a=advantages(r,'rloo')
  expected=torch.stack([r[i]-torch.cat([r[:i],r[i+1:]]).mean(0) for i in range(4)])
  self.assertTrue(torch.allclose(a,expected))
  self.assertFalse(a.requires_grad)
  self.assertTrue(torch.allclose(a,(r-r.mean(0))*4/3))
 def test_no_cross_sample_scaling(self):
  r=torch.tensor([[1.,4.],[2.,8.],[3.,9.],[4.,12.]])
  a=advantages(r,'rloo');r[:,1]*=1000
  self.assertTrue(torch.equal(a[:,0],advantages(r,'rloo')[:,0]))
 def test_legacy_parity(self):
  torch.manual_seed(4);r=torch.randn(4,2);a=r-r.mean(0,keepdim=True)
  self.assertTrue(torch.equal(advantages(r,'legacy'),a/(a.std()+1e-6)))
 def test_constant_rewards(self):
  for m in ['rloo','legacy']:self.assertEqual(advantages(torch.ones(4,2),m).abs().sum().item(),0)
 def test_diagnostics_preserve_gradient(self):
  x=torch.tensor([[.2,.3,.4]],requires_grad=True);parts={'rl':x.square().sum(),'ce':x.sin().sum(),'kl':x.exp().sum()}
  expected=torch.autograd.grad(sum(parts.values()),x,retain_graph=True)[0]
  d=logit_gradient_stats(x,parts);self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in d.values()))
  sum(parts.values()).backward();self.assertTrue(torch.equal(x.grad,expected))
if __name__=='__main__':unittest.main()
