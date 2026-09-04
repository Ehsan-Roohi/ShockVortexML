import unittest
import numpy as np
import torch
from jcp2026.models import make_model
from jcp2026.evaluate import hybrid_masks
from train_stage5_joint import masked_bce_dice_loss, gradients_or_zeros, project_conflicting_gradients


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_forward_loss_and_backward_all_architectures(self):
        for arch,diagnostics in [("joint_unet",True),("joint_unet",False),("segformer_b0",True)]:
            with self.subTest(architecture=arch,diagnostics=diagnostics):
                model=make_model({"architecture":arch,"diagnostics":diagnostics})
                x=torch.randn(2,7 if diagnostics else 4,128,128)
                y=model(x)
                self.assertEqual(tuple(y.shape),(2,6,128,128))
                losses=[masked_bce_dice_loss(y[:,h],torch.zeros_like(y[:,h]),torch.ones_like(y[:,h]),20)[0] for h in range(6)]
                shared=model.shared_encoder_parameters()
                if arch=="joint_unet":
                    gs=gradients_or_zeros(losses[0],shared,retain_graph=True)
                    gv=gradients_or_zeros(losses[1],shared,retain_graph=True)
                    a,b,cos,_=project_conflicting_gradients(gs,gv)
                    self.assertTrue(np.isfinite(cos))
                    self.assertEqual(len(a),len(shared))
                sum(losses).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_hybrid_has_no_physics_only_creation_or_overlap_veto(self):
        shape=(32,32)
        loaded={"observation_mask":np.ones(shape,bool),"geometry":np.zeros(shape,bool),"targets":np.ones((6,*shape),np.uint8),"inputs":np.ones((7,*shape),np.float32),"wall_band":np.zeros(shape,bool)}
        policy={"shock_dilation_cells":6,"shock_seed_floor":.75,"vortex_seed_floor":.75,"rotation_support_fraction":.5,"maximum_wall_fraction":.25}
        probabilities=np.zeros((6,*shape),np.float32)
        high=np.zeros((2,*shape),bool)
        self.assertFalse(hybrid_masks(probabilities,high,loaded,policy,9).any())
        high[:,10:14,10:14]=True
        result=hybrid_masks(probabilities,high,loaded,policy,9)
        self.assertTrue(np.all(result[:,10:14,10:14]))


if __name__=="__main__": unittest.main()
