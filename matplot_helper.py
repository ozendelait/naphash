import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from tqdm.notebook import tqdm as tqdm_nb
import numpy as np

from async_dct_loader import load_img_tar
import cv2

bit_counts = np.array([int(bin(x).count("1")) for x in range(256)]).astype(np.uint8)
def hamming_dist(a,b,axis=None):
    return np.sum(bit_counts[np.bitwise_xor(a,b)],axis=axis)
def hex2hash(s):
    return np.frombuffer(bytes.fromhex(s), dtype=np.uint8)
def hash2hex(h):
    return h.tobytes().hex()
def calc_inter_dist(hashes, eye_val=1024):
    all_d = np.zeros((hashes.shape[0],hashes.shape[0]),dtype=np.int32)
    for i in range(hashes.shape[0]):
        all_d[i,i+1:] = hamming_dist(hashes[i],hashes[(i+1):], axis=1)
    return all_d+all_d.T+np.eye(hashes.shape[0],dtype=np.int32)*eye_val

def load_cntr(path0, dim=128):
    i0 = load_img_tar(path0)
    fx = dim/i0.shape[0]
    x0, y0 = (int(i0.shape[1]*fx+0.5)-dim)//2, 0
    if i0.shape[0] > i0.shape[1]:
        fx = dim/i0.shape[1]
        x0, y0 = 0, (int(i0.shape[0]*fx+0.5)-dim)//2
    return cv2.resize(i0,(0,0),fx=fx,fy=fx,interpolation=cv2.INTER_AREA)[y0:(y0+dim),x0:(x0+dim)]

def vis_pairs(pairs0,paths, dim=128):
    loaded_pairs = []
    spacer=np.ones((dim,dim//8,3),np.uint8)*255
    for (p0,p1) in pairs0:
        i0 = load_cntr(paths[p0], dim=dim)
        i1 = load_cntr(paths[p1], dim=dim)
        loaded_pairs+=[i0,i1,spacer]
    return loaded_pairs

def cls_fig():
    plt.close() 
    plt.subplots(figsize=(16, 9))
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

def get_hash_dists(h, name0, paths, distname='nap0'):
    name0 = name0+'_'
    dists0 = calc_inter_dist(h,1024)
    dists0 = np.float32(dists0)
    dists0[dists0 > 1023] = np.nan
    m_mean = np.int32(np.nanmean(dists0,axis= 1)+0.5)
    m_max = np.int32(np.nanmax(dists0,axis= 1)+0.5)
    m_min = np.int32(np.nanmin(dists0,axis= 1)+0.5)
    v0,c0 = np.unique(m_mean, return_counts=True)
    v1,c1 = np.unique(m_max, return_counts=True)
    v2,c2 = np.unique(m_min, return_counts=True)

    plt.subplots(figsize=(16, 9))
    for store_one in [True,False]:
        for (vals, counts, name1) in [(v0,c0,'mean'),(v1,c1,'max'),(v2,c2,'min')]:
            if store_one: cls_fig()
            if len(vals) < 6:
                if min(vals) > 2:
                    vals = [(min(vals)-2),(min(vals)-1)]+vals.tolist()+[(max(vals)+1),(max(vals)+2)]
                    counts = [0,0]+c0.tolist()+[0,0]
                else:
                    vals = vals.tolist()+[(max(vals)+i) for i in range(1,5)]
                    counts = c0.tolist()+[0,0,0,0]
            plt.bar(vals, counts, label=name0+name1+' dists')
            plt.legend()
            if store_one: plt.savefig('hist_'+distname+'_'+name0+name1+'.png')  
        if not store_one: plt.savefig('hist_'+distname+'_'+name0+'all.png')
        cls_fig()
    #show nearest hits
    np.nan_to_num(dists0, nan=1024, copy=False)
    dists0 = np.int32(dists0)
    pairs0, dists0 = find_closest(m_min, dists0)
    json.dump({'pairs':pairs0,'dists':dists0,'name0':name0}, open('hist_'+distname+'_nearest_'+name0+'.json','wt'))
    nearest_matches = np.hstack(vis_pairs(pairs0,paths))
    cv2.imwrite('hist_'+distname+'_nearest_'+name0+'.jpg',nearest_matches)
    
def find_closest(m_min, c0, max_num=8):
    solved_pairs = []
    ret_pairs, ret_dists = [], []
    best_ord = np.argsort(m_min)
    for p0 in best_ord:
        if p0 in solved_pairs:
            continue
        p1 = np.argmin(c0[p0])
        if p1 in solved_pairs:
            continue
        ret_pairs.append((int(p0),int(p1)))
        solved_pairs.append(p0)
        solved_pairs.append(p1)
        ret_dists.append(int(c0[p0,p1]))
        if len(ret_pairs) >= max_num:
            break
    return ret_pairs, ret_dists