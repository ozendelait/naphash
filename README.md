# NAPHash - A fast orientation-invariant perceptual image hash

## Install:
`python -m pip install git+https://github.com/ozendelait/naphash`

In conda environments, install pybind11 first: `conda install pybind11` .
You can put ` OpenCV_DIR=<path_to_ocv_root> ` in front if you need to link to a specific OpenCV version.
Note: C++ OpenCV libs are required to build the project (e.g.  `conda install -c conda-forge "opencv==4.5.3"`)

## Usage:
For faster calculations, create a naphash_obj and use its functions.

```python
from naphash_py import naphash_obj as npobj
from skimage import io
test_img = io.imread('https://upload.wikimedia.org/wikipedia/commons/b/b6/SIPI_Jelly_Beans_4.1.07.tiff')
calcnap_rgb = npobj(dct_dim = 32, rot_inv_mode = rot_inv_type.full, apply_center_crop = False, is_rgb = True)
calcnp_bgr = npobj(dct_dim = 32, rot_inv_mode = rot_inv_type.none, apply_center_crop = False, is_rgb = False)
calcnap_rgb.get_hash(test_img)
> array([ 69,  21,  53,  77, 108,  13,  35, 212,  21,  85, 186, 135,   5,
   212,  17,  31, 181, 116, 189, 127, 125], dtype=uint8)
calcnp_bgr.get_hash(test_img)
> array([118, 137, 183,  48, 123, 219, 176, 168,  51, 163,  93, 248,  91,
   230, 127, 117,  16, 110, 136, 138, 172, 177, 255, 249, 213, 221,
   23, 102,  87, 114,  38, 143, 155, 170, 171,  50,  93, 225, 145, 85], dtype=uint8)
```      
There are four convenience functions which directly calculate hashes from images: nphash_bgr, naphash_bgr, nphash_rgb, naphash_rgb

Check if your color images are in RGB (skimage, PIL) or BGR (OpenCV) and use the correct function to calculate the standard 40-byte NPHash or 21-byte NAPHash (=orientation-invariant).
Example:
```python
from naphash_py import naphash_bgr
naphash_rgb(test_img)
> array([ 69,  21,  53,  77, 108,  13,  35, 212,  21,  85, 186, 135,   5,
       212,  17,  31, 181, 116, 189, 127, 125], dtype=uint8)
```
## Citation:
If you use NAPHash, please cite our associated paper:

    @inproceedings{Zendel_2021_ICECET,
    author = {Zendel, Oliver and Zinner, Christian},
    title = {NAPHash: Efficient Image Hash to Reduce Dataset Redundancy}
    booktitle = {Proceedings of the International Conference on Electrical, Computer and Energy Technologies (ICECET)},
    year = {2021},
    month = {December},
    address = {Cape Town, South Africa}
    }

NAPHash copyright by AIT - Austrian Institute of Technology Gmbh

## Acknowledgement: 
This research has received funding from Mobility of the Future; a research, technology, and innovation funding program of the Austrian Ministry of Climate Action.
