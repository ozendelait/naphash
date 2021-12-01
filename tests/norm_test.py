import unittest
import numpy as np

class MainTest(unittest.TestCase):
    def test_norm(self):
        from naphash_py import naphash as nhpy, rot_inv_type
        dct_dim = 32
        init_weights = np.ones(324,np.float32)
        h_crop0 = nhpy(dct_dim=dct_dim, rot_inv_mode=rot_inv_type.none, apply_center_crop=False, is_rgb=False)
        h_crop0.get_norm(ret_coeffs=init_weights)
        #check NPHASH weights
        self.assertEqual(int(init_weights[0]+0.5), 421)
        self.assertEqual(int(init_weights[2]+0.5), 317)
        init_weights.fill(1)
        h_crop0 = nhpy(dct_dim=dct_dim, rot_inv_mode=rot_inv_type.full, apply_center_crop=False, is_rgb=False)
        h_crop0.get_norm(ret_coeffs = init_weights)
        #check NAPHASH weights
        self.assertEqual(int(init_weights[0]+0.5), 359)
        self.assertEqual(int(init_weights[2]+0.5), 348)
        #check custom weights (these are nonsensical weight values used only during test case)
        for i in range(324):
          init_weights[i] = (i*7+5)
        h_crop0.set_norm(coeffs=init_weights, do_normalization=False)
        init_weights.fill(1)
        h_crop0.get_norm(ret_coeffs=init_weights)
        read_back_ok = all([init_weights[i] == (i*7+5) for i in range(324)])
        self.assertEqual(read_back_ok, True)
        #check custom weights normalized
        h_crop0.set_norm(coeffs=init_weights, do_normalization=True)
        h_crop0.get_norm(ret_coeffs=init_weights)
        self.assertEqual(int(init_weights[0]+0.5), 435)
        self.assertEqual(int(init_weights[2]+0.5), 1152)

if __name__ == '__main__':
    unittest.main()