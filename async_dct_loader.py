from tqdm import tqdm as tqdm_con
from tqdm.notebook import tqdm as tqdm_nb
import cv2
import numpy as np
import asyncio
from PIL import Image as pilim
#original phash by calling pip install imagehash
from imagehash import phash as pip_phash_call
import math

#global objects allow parallel executions
nhcpp_objs_g = None

# allows loading of images either from disk or from inside a tar archive; 
# path for tar archives: <path/to/tarfile.tar>[/<internal_path>]:offset_bytes:num_bytes
def load_img_tar(fn):
    pos_tar = fn.find('.tar')
    if pos_tar <= 0: return cv2.imread(fn)
    pos_sl = fn.rfind('/',0,pos_tar)+1
    tarname, tarroot, fname = fn[pos_sl:pos_tar]+'.tar', fn[:pos_sl], fn[pos_sl:].split(':')
    if len(tarroot)==0: tarroot="."
    im = None
    with open(tarroot+'/'+tarname, 'rb') as file_obj:
        file_obj.seek(int(fname[1]))
        try:
            bytes_inmem = file_obj.read(int(fname[2].strip()))
            im = cv2.imdecode(np.frombuffer(bytes_inmem,dtype="uint8"), cv2.IMREAD_COLOR) 
        except Exception as e:
            print(e)
            return None
    return im

def pip_phash(i, hash_size=8):
    return np.packbits(np.uint8(pip_phash_call(pilim.fromarray(i), hash_size).hash.flatten()))

def nap_resize(i,trg_wh):
    return np.array(pilim.fromarray(i).resize(trg_wh, pilim.LANCZOS))

def get_hashes(i, nhcpp_obj, nhcpp_obj_rotinv, hashlen=8):
    assert(hashlen>= 1 and hashlen <= 40)
    hash_bitlen = nhcpp_obj.get_bitlen()
    hash_ret = np.zeros(hash_bitlen//8, np.uint8)
    nhcpp_obj.get_hash(i,hash_ret)
    hash_bitlen = nhcpp_obj_rotinv.get_bitlen()
    hash_ret_rotinv = np.zeros(hash_bitlen//8, np.uint8)
    nhcpp_obj_rotinv.get_hash(i,hash_ret_rotinv)
    phashmin = math.ceil(math.sqrt(8*hashlen))
    hash_ph8 = pip_phash(i,phashmin)
    return hash_ret[:hashlen], hash_ret_rotinv[:hashlen], hash_ph8[:hashlen]

async def get_dct_parallel(fn, obj_idx, dct_dim, min_img_dims, ret_hash, pil_sz=-1, phsz=0):
    global nhcpp_objs_g
    try:
        i = load_img_tar(fn)
        if i is None or min(i.shape[:2]) < min_img_dims: #skip invalid or tiny images
            return None
        if pil_sz > 0:
            i = nap_resize(i,(pil_sz,pil_sz))
        if pil_sz == 0:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return None
    if ret_hash == 2:
        return i
    if ret_hash > 0 and phsz > 0:
        return pip_phash(i,phsz)
    dct_crop0 = np.zeros((dct_dim,dct_dim), np.float32)
    nhcpp_objs_g[obj_idx].get_dct(i,dct_crop0)
    if ret_hash == 1:
        hash_bitlen = nhcpp_objs_g[obj_idx].get_bitlen()
        hash_ret = np.zeros(hash_bitlen//8, np.uint8)
        nhcpp_objs_g[obj_idx].get_hash_dct(dct_crop0,hash_ret)
        return hash_ret
    return dct_crop0

async def get_hash_dct_parallel(dct, obj_idx):
    global nhcpp_objs_g
    hash_bitlen = nhcpp_objs_g[obj_idx].get_bitlen()
    hash_ret = np.zeros(hash_bitlen//8, np.uint8)
    nhcpp_objs_g[obj_idx].get_hash_dct(dct,hash_ret)
    return hash_ret


#call result = await async_load_dct_paths(..) from jupyter nb
#other wise use result = asyncio.run(async_load_dct_paths(...))
# ret_val-> 0:dct, 1:hash, 2:img
async def async_load_dct_paths(nhcpp_objs, all_paths, num_threads=8, dct_dim=32, min_img_dims=128, tqdm_vers = tqdm_nb, ret_hash=0, pil_sz=-1, phsz=0):
    global nhcpp_objs_g
    assert len(nhcpp_objs) == num_threads, "Number of objects in nhcpp_objs must be the same as num_threads"
    num_threads = min(len(all_paths),num_threads)
    nhcpp_objs_g = nhcpp_objs
    num_passes = (len(all_paths)+num_threads-1)//num_threads
    results = []
    for i in tqdm_vers(range(num_passes)):
        i0, tasks = i*num_threads, []
        for t in range(num_threads):
            p = min(i0+t,len(all_paths)-1)
            tasks.append(asyncio.create_task(get_dct_parallel(all_paths[p],t, dct_dim, min_img_dims, ret_hash, pil_sz, phsz)))
        for t in range(num_threads):
            results.append(await tasks[t])
    #curiously, joblib did not work and produced a runtime comparable to a single core
    #result = joblib.Parallel(n_jobs=num_threads,require='sharedmem')(joblib.delayed(get_dct_parallel)(str(all_paths[i]), int(i%num_threads), dct_dim, min_img_dims) for i in tqdm(range(len(all_paths))))
    return results[:len(all_paths)]

#call result = await async_load_dct_paths(..) from jupyter nb
#other wise use result = asyncio.run(async_load_dct_paths(...))
async def async_hash_dcts(nhcpp_objs, all_dcts, num_threads=8, tqdm_vers = tqdm_nb):
    global nhcpp_objs_g
    assert len(nhcpp_objs) == num_threads, "Number of objects in nhcpp_objs must be the same as num_threads"
    num_threads = min(len(all_dcts),num_threads)
    nhcpp_objs_g = nhcpp_objs
    num_passes = (len(all_dcts)+num_threads-1)//num_threads
    results = []
    for i in tqdm_vers(range(num_passes)):
        i0, tasks = i*num_threads, []
        for t in range(num_threads):
            p = min(i0+t,len(all_dcts)-1)
            tasks.append(asyncio.create_task(get_hash_dct_parallel(all_dcts[p],t)))
        for t in range(num_threads):
            results.append(await tasks[t])
    #curiously, joblib did not work and produced a runtime comparable to a single core
    #result = joblib.Parallel(n_jobs=num_threads,require='sharedmem')(joblib.delayed(get_dct_parallel)(str(all_paths[i]), int(i%num_threads), dct_dim, min_img_dims) for i in tqdm(range(len(all_paths))))
    return results[:len(all_dcts)]