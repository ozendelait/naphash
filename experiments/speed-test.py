from cifar10_trainer import files_in_subdirs
from build.naphash_cpp import naphash as nhcpp, rot_inv_type

import cv2                                    # cv2.img_hash installed via pip install opencv-contrib-python
import time, sys
import numpy as np

from PIL import Image as pilim
from imagehash import phash as pip_phash # installed via pip install imagehash


versions = ['NPHash','NAPHash','Phash8 (ocv)', 'Phash8 (raw)', 'Phash12 (raw)'] 

def ocv_phash(i):
    return cv2.img_hash.pHash(i)

def check_speed(v):
    cifar10_train = files_in_subdirs('./cifar10/train')
    cifar10_val = files_in_subdirs('./cifar10/test')
    cifar10_paths = cifar10_val+cifar10_train
    print("Loaded: ",len(cifar10_train), len(cifar10_val))
    all_imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in cifar10_paths]
    if v > 2: # pip_phash operates on PIL images
        all_imgs = [pilim.fromarray(i) for i in all_imgs]
    
    itter_times = 20
    if v < 2:
        naphobj = nhcpp(dct_dim=32, rot_inv_mode=(rot_inv_type.full if v == 1 else rot_inv_type.none), apply_center_crop=False, is_rgb=False)
        hash_bitlen = naphobj.get_bitlen()
        tmp_hash = np.zeros(32*32, np.float32)
        hash_ret = np.zeros((len(all_imgs),hash_bitlen//8), np.uint8)
    freq_size = 12 if v > 3 else 8
    best_time = []
    for i in range(itter_times):
        if v >= 2:
            hash_ret = []
        t0 = time.time()
        if v < 2:
            for i in range(len(all_imgs)):
                naphobj.get_hash_fast(all_imgs[i],tmp_hash,hash_ret[i])
        elif v == 2:
            for i in range(len(all_imgs)):
                hash_ret.append(ocv_phash(all_imgs[i]))
        else:
            for i in range(len(all_imgs)):
                hash_ret.append(pip_phash(all_imgs[i],freq_size))
        t1 = time.time()
        best_time.append(t1-t0)
    best_time = min(best_time)
    print(versions[v]," Time: ",best_time," per img: ", best_time/len(all_imgs), "imgs/sec",len(all_imgs)/best_time)
    print("Example results: ", hash_ret[0],hash_ret[1],hash_ret[-2],hash_ret[-1])
    return {versions[v]:best_time}

def main(argv):
    check_speed(int(argv[0]))

if __name__ == "__main__":
    main(sys.argv[1:])