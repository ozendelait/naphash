from tqdm import tqdm as tqdm_con
from tqdm.notebook import tqdm as tqdm_nb
import cv2
import numpy as np
import asyncio

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

async def get_dct_parallel(fn, obj_idx, dct_dim, min_img_dims, ret_hash):
    global nhcpp_objs_g
    try:
        i = load_img_tar(fn)
        if i is None or min(i.shape[:2]) < min_img_dims: #skip invalid or tiny images
            return None
    except Exception as e:
        return None
    dct_crop0 = np.zeros((dct_dim,dct_dim), np.float32)
    nhcpp_objs_g[obj_idx].get_dct(i,dct_crop0)
    if ret_hash:
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
async def async_load_dct_paths(nhcpp_objs, all_paths, num_threads=8, dct_dim=32, min_img_dims=128, tqdm_vers = tqdm_nb, ret_hash=False):
    global nhcpp_objs_g
    assert len(nhcpp_objs) == num_threads, "Number of objects in nhcpp_objs must be the same as num_threads"
    num_threads = min(len(all_paths),num_threads)
    nhcpp_objs_g = nhcpp_objs
    num_passes = (len(all_paths)+num_threads-1)//num_threads
    results = []
    for i in tqdm_vers(range(num_passes)):
        i0, tasks = i*num_threads, []
        for t in range(num_threads):
            p = min(i0+t,len(all_paths))
            tasks.append(asyncio.create_task(get_dct_parallel(all_paths[p],t, dct_dim, min_img_dims, ret_hash)))
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
            p = min(i0+t,len(all_dcts))
            tasks.append(asyncio.create_task(get_hash_dct_parallel(all_dcts[p],t)))
        for t in range(num_threads):
            results.append(await tasks[t])
    #curiously, joblib did not work and produced a runtime comparable to a single core
    #result = joblib.Parallel(n_jobs=num_threads,require='sharedmem')(joblib.delayed(get_dct_parallel)(str(all_paths[i]), int(i%num_threads), dct_dim, min_img_dims) for i in tqdm(range(len(all_paths))))
    return results[:len(all_dcts)]