/******************************************************************************
*
* FILENAME:     naphash_py.cpp
*
* PURPOSE:      Python bindings for C++ implementation of NAPHash; 
*               see https://github.com/ozendelait/naphash for more information
*               
* AUTHOR:       Oliver Zendel, AIT Austrian Institute of Technology GmbH
*
*  Copyright (C) 2021 AIT Austrian Institute of Technology GmbH
*  All rights reserved.
******************************************************************************/ 

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <naphash.hpp>
#include <iostream>
#include <optional>

namespace py = pybind11;

class pynaphash
{   
    naphash nobj;
    int check_dct_dim;

public:
    pynaphash(const int _dct_dim=32, 
              const naphash::rot_inv_type _rot_inv_mode=naphash::rot_inv_full, 
              const bool _apply_center_crop=false, 
              const bool _c3_is_rgb=true): 
        nobj(_dct_dim, _rot_inv_mode, _apply_center_crop, _c3_is_rgb), check_dct_dim(_dct_dim) {}
    
    // wrap C++ function with NumPy array IO
    py::object get_dct(py::array img,
                        py::array ret_dct) {
          // check input dimensions
          if ( img.ndim() < 2 || img.ndim() > 3 )
            throw std::runtime_error("Input should be 2-D/3-D NumPy array");
          if ( ret_dct.ndim() != 2 )
            throw std::runtime_error("Target should be 2-D NumPy array");
          if(img.itemsize() != 4 && img.itemsize() != 1)
            throw std::runtime_error("Input data should be unsigned char or 32bit float");

          auto buf = img.request();
          auto buf2 = ret_dct.request();
          if (ret_dct.shape()[0] !=  check_dct_dim || 
             ret_dct.shape()[1] !=  check_dct_dim ||
            ret_dct.itemsize() != 4) throw std::runtime_error("ret_dct dimensions invalid! Needs dct_dim x dct_dim float32 target buffer.");

          int h = (int)img.shape()[0], w = (int)img.shape()[1], c = ((int)img.ndim() == 2)?1:(int)img.shape()[2];
          unsigned char* ptr = (unsigned char*) buf.ptr;
          unsigned char* ptr_trg = (unsigned char*) buf2.ptr;
          int stepsz = (int)img.strides()[0];
          if(img.strides()[1] != img.itemsize()*c)
              throw std::runtime_error("Non-standard channel stride not supported. Use np.ascontiguousarray for img!");
          // call pure C++ function
          if(img.itemsize() == 1)
              nobj.get_dct_u8(ptr, w, h, c, stepsz, ptr_trg);
          else
              nobj.get_dct_f32((float*)ptr, w, h, c, stepsz, ptr_trg);
          return py::cast<py::none>(Py_None);
    }
        
    py::object get_hash_dct(py::array dct_inp,
                        py::array ret_hash) {
          // check input dimensions
          if (dct_inp.ndim() > 2 || dct_inp.itemsize() != 4)
            throw std::runtime_error("Input should be 1-D or 2-D NumPy float array");
          if (ret_hash.ndim() != 1 || ret_hash.itemsize() != 1)
            throw std::runtime_error("Target should be 1-D NumPy uint8 array");

          auto buf = dct_inp.request();
          auto buf2 = ret_hash.request();
          if(dct_inp.ndim() == 2 && (dct_inp.shape()[0] !=  check_dct_dim 
                               || dct_inp.shape()[1] !=  check_dct_dim 
                               || dct_inp.strides()[0] != check_dct_dim*sizeof(float)))
             throw std::runtime_error("Input 2D dct dimensions/step width invalid!");
          else if(dct_inp.ndim() == 1 && dct_inp.shape()[0] !=  check_dct_dim*check_dct_dim)
             throw std::runtime_error("Input 1D dct size invalid!");
          int num_trg_bytes = nobj.get_bitlen()/8;
          if(ret_hash.shape()[0] < num_trg_bytes){
              char errorstr[64]={0};sprintf(errorstr,"Output needs minimum size of %i",  num_trg_bytes);
              throw std::runtime_error(std::string(errorstr));
          }
        
          float* ptr = (float*) buf.ptr;
          unsigned char* ptr_trg = (unsigned char*) buf2.ptr;
          nobj.get_hash_dct(ptr, ptr_trg);
          return py::cast<py::none>(Py_None);
    }
    
    // wrap C++ function with NumPy array IO
    py::array get_hash(py::array img,
                       std::optional<py::array> ret_hash) {
      float dct_tmp_[32*32];
      // return newly allocated data pointer in case of None, is handled by python garbage collector
      py::array _ret_hash = ret_hash.has_value()?ret_hash.value():py::array_t<unsigned char>((const int)((get_bitlen()+7)/8));
      if(_ret_hash.is_none() || _ret_hash.ndim() == 0) // sometimes None gets cast to a 0 dimensional py::array
         _ret_hash = py::array_t<unsigned char>((const int)((get_bitlen()+7)/8));
      float *pDct_tmp_ = &dct_tmp_[0];
      bool bFreeDyn = (check_dct_dim > 32);
      if (bFreeDyn)
        pDct_tmp_ = new float[check_dct_dim*check_dct_dim];
          py::array dct_tmp = py::array_t<float>(std::vector<ptrdiff_t>{check_dct_dim,check_dct_dim}, &dct_tmp_[0]);
          get_dct(img, dct_tmp);
          get_hash_dct(dct_tmp, _ret_hash);
      if (bFreeDyn)
        delete[] pDct_tmp_;
      return _ret_hash;
    }
    
    // wrap C++ function with NumPy array IO
    int get_hash_fast(py::array_t<unsigned char> img,
                      py::array_t<float> dct_tmp_f32,
                      py::array_t<unsigned char> ret_hash) {  
          unsigned char *inpc = (unsigned char *)&(img.unchecked<2>()(0,0)), *outpc = (unsigned char *)&(ret_hash.unchecked<1>()(0));
          unsigned char * tmpc = (unsigned char *)&(dct_tmp_f32.unchecked<1>()(0));
          nobj.get_dct_u8(inpc, 32, 32, 1, 32, tmpc);
          nobj.get_hash_dct((float*)tmpc, outpc);
          return 0;
    }
             
    int get_bitlen() {
        return nobj.get_bitlen();
    }
             
    py::object get_norm(py::array ret_coeffs) {
          if ( ret_coeffs.ndim() != 1 ||  ret_coeffs.itemsize() != 4 || ret_coeffs.shape()[0] != nap_norm_len)
            throw std::runtime_error("Target should be 1-D NumPy float array with a size of 324");
          auto buf = ret_coeffs.request();
          nobj.get_nap_norm((float*)buf.ptr);
          return py::cast<py::none>(Py_None);
    }
    
    py::object set_norm(py::array coeffs, const bool do_normalization=true) {
          if ( coeffs.ndim() != 1 ||  coeffs.itemsize() != 4 )
            throw std::runtime_error("Input should be 1-D NumPy float array.");
          auto buf = coeffs.request();
          nobj.set_nap_norm((float*)buf.ptr, (int)coeffs.shape()[0], do_normalization);
          return py::cast<py::none>(Py_None);
    }
    
    static int hamming_dist(py::array_t<unsigned char> h0,
                            py::array_t<unsigned char> h1,
                            int                 num_bytes) {
        unsigned char *h0c = (unsigned char *)&(h0.unchecked<1>()(0)), *h1c = (unsigned char *)&(h1.unchecked<1>()(0));
        if(num_bytes <= 0) {
            if (h0.ndim() != 1 || h1.ndim() != 1)
                throw std::runtime_error("Inputs should be 1-D NumPy arrays!");
            auto check_h0 = h0.request(), check_h1 = h1.request();
            num_bytes = (int)std::min(h0.shape()[0], h1.shape()[0]);
        }
        return naphash::hamming_dist(h0c,h1c,num_bytes);
    }
};


// slower convenience functions which create/delete pynaphash objects for each call
static py::array naphash_rgb_func(py::array img) {
  pynaphash nap_obj = pynaphash(32, naphash::rot_inv_full);
  return nap_obj.get_hash(img, std::nullopt);
}

static py::array nphash_rgb_func(py::array img) {
  pynaphash nap_obj = pynaphash(32, naphash::rot_inv_none);
  return nap_obj.get_hash(img, std::nullopt);
}

static py::array naphash_bgr_func(py::array img) {
  pynaphash nap_obj = pynaphash(32, naphash::rot_inv_full,false,false);
  return nap_obj.get_hash(img, std::nullopt);
}

static py::array nphash_bgr_func(py::array img) {
  pynaphash nap_obj = pynaphash(32, naphash::rot_inv_none,false,false);
  return nap_obj.get_hash(img, std::nullopt);
}


PYBIND11_MODULE(naphash_py, m) {
     m.doc() = "NAPHash; a dct-based image hash";

    #ifdef PYBIND11_VERSION_INFO
    #define STRINGIFY(x) #x
    #define TOSTRINGCONV(x) STRINGIFY(x)
    m.attr("__version__") = TOSTRINGCONV(PYBIND11_VERSION_INFO);
    #endif

    py::enum_<naphash::rot_inv_type>(m, "rot_inv_type")
        .value("none", naphash::rot_inv_none)
        .value("swap", naphash::rot_inv_swap)
        .value("full", naphash::rot_inv_full)
        .export_values();
    
    py::class_<pynaphash>(m, "naphash_obj")
        .def(py::init<const int, const naphash::rot_inv_type, const bool, const bool>(), py::arg("dct_dim") = 32, py::arg("rot_inv_mode") = naphash::rot_inv_full,  py::arg("apply_center_crop") = false, py::arg("is_rgb") = true)
        .def("get_hash",  &pynaphash::get_hash, "Calculate naphash based on input image", py::arg("img"), py::arg("ret_hash") = py::none())
        .def("get_dct",  &pynaphash::get_dct, "Calculate dct based on input image", py::arg("img"), py::arg("ret_dct"))
        .def("get_hash_dct",  &pynaphash::get_hash_dct, "Calculate naphash based on dct input", py::arg("dct_inp"), py::arg("ret_hash"))
        .def("get_hash_fast",  &pynaphash::get_hash_fast, "Calculate naphash based on 32x32x1_u8 image", py::arg("img"), py::arg("dct_tmp_f32"), py::arg("ret_hash"))
        .def("get_bitlen",  &pynaphash::get_bitlen, "Returns number of usable bits of resulting naphashes")
        .def("set_norm",  &pynaphash::set_norm, "Set custom naphash norm coeff weights", py::arg("coeffs"), py::arg("do_normalization") = true)
        .def("get_norm",  &pynaphash::get_norm, "Get naphash norm coeff weights", py::arg("ret_coeffs"));
        
   m.def("naphash_bgr", &naphash_bgr_func, "Return standard NPHash for one bgr image (slow conviniece function; consider using naphash_obj).", py::arg("img"));
   m.def("nphash_bgr", &nphash_bgr_func, "Return standard NAPHash for one bgr image (slow conviniece function; consider using naphash_obj).", py::arg("img"));
   
   m.def("naphash_rgb", &naphash_rgb_func, "Return standard NPHash for one rgb image (slow conviniece function; consider using naphash_obj).", py::arg("img"));
   m.def("nphash_rgb", &nphash_rgb_func, "Return standard NAPHash for one rgb image (slow conviniece function; consider using naphash_obj).", py::arg("img"));
   m.def("hamming_dist", &pynaphash::hamming_dist, "Hamming distance between two np.uint8 arrays (of the same length, specify length using num_bytes to speed up function).", py::arg("h0"), py::arg("h1"), py::arg("num_bytes")= 0);
}