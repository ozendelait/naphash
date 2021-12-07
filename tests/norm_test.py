import unittest
import numpy as np

class MainTest(unittest.TestCase):
    #test custom weight norms
    def test_norm(self):
        from naphash_py import naphash_obj as nhpy, rot_inv_type
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

    #test hamming distance calculation
    def test_hamming(self):
        from naphash_py import naphash_rgb, nphash_rgb, hamming_dist
        import numpy as np
        #create simple test image (non-symmetric in x and y)
        dct_dim=32
        grad_im1 = np.uint8(np.mgrid[0:dct_dim*9,0:dct_dim*16][0,])
        grad_im1[:,dct_dim*3] = 0
        grad_im1[dct_dim*4,:] = 0
        #create all 8 orientation versions (== combinations of flipping and 90deg. rotations)
        #NAPHash is invariant to these orientation changes; NPHash is not
        cmp_ims = [grad_im1, 
                   np.ascontiguousarray(np.rot90(grad_im1)),
                   np.ascontiguousarray(np.rot90(grad_im1, k=-1)),
                   grad_im1[::-1],
                   np.ascontiguousarray(grad_im1[:,::-1]),
                   np.ascontiguousarray(grad_im1[::-1,::-1]),
                   np.ascontiguousarray(np.rot90(grad_im1))[::-1],
                   np.ascontiguousarray(np.rot90(grad_im1, k=-1))[::-1]
                   ]
        
        cmp_hamming = [[0, 48, 34, 24, 17, 28, 45, 32],
                       [0, 0, 0, 0, 0, 0, 0, 0]]
        
        for j,napvers in enumerate([nphash_rgb, naphash_rgb]):
          hashes = [napvers(i) for i in cmp_ims]
          hdists = [hamming_dist(hashes[0],h) for h in hashes]
          self.assertEqual(hdists, cmp_hamming[j])
    
if __name__ == '__main__':
    unittest.main()