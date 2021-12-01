import unittest
import numpy as np

class MainTest(unittest.TestCase):
    def test_hash(self):
        from naphash_py import naphash as nhpy, rot_inv_type
        dct_dim = 32
        #creating test images 
        ones_im0 = np.ones((dct_dim,dct_dim),np.uint8)
        ones_im1 = np.ones((dct_dim*9,dct_dim*16),np.uint8)
        grad_im0 = np.uint8(np.mgrid[0:dct_dim,0:dct_dim][0,])
        grad_im1 = np.uint8(np.mgrid[0:dct_dim*9,0:dct_dim*16][0,])
        test_ims = [np.uint8(ones_im0*7),
                    np.uint8(ones_im1*8+grad_im1*2),
                    np.uint8(grad_im0*7+grad_im0.T*3),
                    np.float32(grad_im0)*13+grad_im0.T*11+grad_im0*7,
                    np.float32(grad_im1)*11]
        test_ims[1][:,dct_dim*3] = 0
        test_ims[1][dct_dim*4,:] = 0
        test_ims_c3 = []
        for i,t in enumerate(test_ims):
          t_c3 = np.tile(t[:,:,None],[1,1,3])
          t_c3[:,:,0] += 3
          t_c3[:,:,1]*= 5
          test_ims_c3.append(np.uint8(t_c3))
        test_ims += test_ims_c3
        nphash0 = nhpy(dct_dim=dct_dim, rot_inv_mode=rot_inv_type.none, apply_center_crop=False, is_rgb=False)
        naphash0 = nhpy(dct_dim=dct_dim, rot_inv_mode=rot_inv_type.full, apply_center_crop=False, is_rgb=False)
        self.assertEqual(nphash0.get_bitlen(), 320)
        self.assertEqual(naphash0.get_bitlen(), 168)
        
        calc_u32 = np.uint32(np.array([0,256,256*256,256*256*256]))
        #comparision values can be generated by calling program without
        cmp_hash_start =  [0, 4294966016, 2331779072, 4291821312, 3221224192, 0, 4294966016, 3397216512, 3362306048, 3221224192, 0, 134292480, 1663845120, 138421248, 4269056, 0, 138485760, 4054614784, 3558644480, 138486784]
        for j,napvers in enumerate([nphash0, naphash0]):
          for i,t in enumerate(test_ims):
            tmp_dct = np.ones((dct_dim,dct_dim),np.float32)
            napvers.get_dct(img=t, ret_dct=tmp_dct)
            if i == 0:
            #the first  image has no gradient -> should have only DC-content set, remainder to zero
              ret_dct = tmp_dct.reshape(dct_dim*dct_dim)
              self.assertEqual(np.all(ret_dct[1:] == 0), True)
              self.assertEqual(int(ret_dct[0]) == 0, False)
            ret_hash0 = np.ones(324,np.uint8)
            ret_hash1 = np.ones(324,np.uint8)
            # calculating from dct should also work for 1D flattened data
            napvers.get_hash_dct(dct_inp = tmp_dct, ret_hash = ret_hash0)
            napvers.get_hash_dct(dct_inp = tmp_dct.reshape(dct_dim*dct_dim), ret_hash = ret_hash1)
            self.assertEqual(np.all(ret_hash1 == ret_hash0), True)
            ret_hash1.fill(1)
            #get_hash should result in the same as when calc. DCT first
            napvers.get_hash(img = t, ret_hash = ret_hash1)
            self.assertEqual(np.all(ret_hash1 == ret_hash0), True)
            res_u32 = abs(np.sum(calc_u32*np.uint32(ret_hash1[:4])))
            #cmp_hash_start.append(res_u32) #uncomment to generate comparison values
            #compare generated hash vs. comparison value
            self.assertEqual(res_u32, cmp_hash_start[j*10+i])
            
        #print("cmp_hash_start = ",[f for f in cmp_hash_start]) #uncomment to generate comparison values


if __name__ == '__main__':
    unittest.main()