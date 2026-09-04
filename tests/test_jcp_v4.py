import unittest
import numpy as np
from jcp2026.diagnostics import gradient_diagnostics, build_inputs
from jcp2026.common import validate_groups
from ml.flow_aligned import FreestreamReference


class DiagnosticContractTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-1, 1, 25)
        self.y = np.linspace(-1, 1, 23)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def test_rotation_with_compression(self):
        d = gradient_diagnostics(-2*self.X-3*self.Y, 3*self.X-2*self.Y, self.x, self.y)
        np.testing.assert_allclose(d["q_deviatoric"], 9, atol=1e-11)
        np.testing.assert_allclose(d["q_legacy"], 5, atol=1e-11)
        np.testing.assert_allclose(d["lambda_ci"], 3, atol=1e-11)

    def test_shear_is_not_rotation(self):
        d = gradient_diagnostics(4*self.Y, np.zeros_like(self.X), self.x, self.y)
        np.testing.assert_allclose(d["q_deviatoric"], 0, atol=1e-11)

    def test_pure_isotropic_compression(self):
        d = gradient_diagnostics(-2*self.X, -2*self.Y, self.x, self.y)
        np.testing.assert_allclose(d["q_deviatoric"], 0, atol=1e-11)
        np.testing.assert_allclose(d["q_legacy"], -4, atol=1e-11)

    def test_reversed_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            gradient_diagnostics(self.X, self.Y, self.x[::-1], self.y)

    def test_nonuniform_linear_gradient(self):
        x = np.linspace(0.1, 2, 25)**2
        X, Y = np.meshgrid(x, self.y)
        d = gradient_diagnostics(-3*Y, 3*X, x, self.y)
        np.testing.assert_allclose(d["q_deviatoric"], 9, atol=1e-10)

    def test_group_leakage_rejected(self):
        with self.assertRaises(ValueError):
            validate_groups([dict(dataset_id="a",leakage_group_id="same",split="train"),dict(dataset_id="b",leakage_group_id="same",split="test")])

    def test_no_stored_q_shortcut(self):
        f = dict(rho=np.ones_like(self.X), pressure=np.ones_like(self.X), u=3-3*self.Y, v=3*self.X, x=self.x, y=self.y, q_criterion=np.full_like(self.X, -999))
        c,d=build_inputs(f, FreestreamReference(1,1,3,0))
        self.assertTrue(np.all(c[5]>0))


if __name__ == "__main__":
    unittest.main()
