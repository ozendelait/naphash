#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <naphash.hpp>
#include <iostream>

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
    py::object get_dct(py::array inp,
                        py::array trg) {
          // check input dimensions
          if ( inp.ndim() < 2 || inp.ndim() > 3 )
            throw std::runtime_error("Input should be 2-D/3-D NumPy array");
          if ( trg.ndim() != 2 )
            throw std::runtime_error("Target should be 2-D NumPy array");
          if(inp.itemsize() != 4 && inp.itemsize() != 1)
            throw std::runtime_error("Input data should be unsigned char or 32bit float");

          auto buf = inp.request();
          auto buf2 = trg.request();
          if (trg.shape()[0] !=  check_dct_dim || 
             trg.shape()[1] !=  check_dct_dim ||
            trg.itemsize() != 4) throw std::runtime_error("trg dimensions invalid!");


          int h = inp.shape()[0], w = inp.shape()[1], c = (inp.ndim() == 2)?1:inp.shape()[2];
          unsigned char* ptr = (unsigned char*) buf.ptr;
          unsigned char* ptr_trg = (unsigned char*) buf2.ptr;
          int stepsz = inp.strides()[0];
          if(inp.strides()[1] != inp.itemsize()*c)
              throw std::runtime_error("Non-standard channel stride not supported. Use np.ascontiguousarray for inp!");
          // call pure C++ function
          if(inp.itemsize() == 1)
              nobj.get_dct_u8(ptr, w, h, c, stepsz, ptr_trg);
          else
              nobj.get_dct_f32((float*)ptr, w, h, c, stepsz, ptr_trg);
          return py::cast<py::none>(Py_None);
    }
        
    py::object get_hash_dct(py::array inp,
                        py::array trg) {
          // check input dimensions
          if (inp.ndim() > 2 || inp.itemsize() != 4)
            throw std::runtime_error("Input should be 1-D or 2-D NumPy float array");
          if (trg.ndim() != 1 || trg.itemsize() != 1)
            throw std::runtime_error("Target should be 1-D NumPy uint8 array");

          auto buf = inp.request();
          auto buf2 = trg.request();
          if(inp.ndim() == 2 && (inp.shape()[0] !=  check_dct_dim 
                               || inp.shape()[1] !=  check_dct_dim 
                               || inp.strides()[0] != check_dct_dim*sizeof(float)))
             throw std::runtime_error("Input 2D dct dimensions/step width invalid!");
          else if(inp.ndim() == 1 && inp.shape()[0] !=  check_dct_dim*check_dct_dim)
             throw std::runtime_error("Input 1D dct size invalid!");
          int num_trg_bytes = nobj.get_bitlen()/8;
          if(trg.shape()[0] < num_trg_bytes){
              char errorstr[64]={0};sprintf(errorstr,"Output needs minimum size of %i",  num_trg_bytes);
              throw std::runtime_error(std::string(errorstr));
          }
        
          float* ptr = (float*) buf.ptr;
          unsigned char* ptr_trg = (unsigned char*) buf2.ptr;
          nobj.get_hash_dct(ptr, ptr_trg);
          return py::cast<py::none>(Py_None);
    }
    
    // wrap C++ function with NumPy array IO
    py::array get_hash(py::array inp,
                        py::array trg) {
          float dct_tmp_[check_dct_dim][check_dct_dim];
          py::array dct_tmp = py::array_t<float>(std::vector<ptrdiff_t>{check_dct_dim,check_dct_dim}, &dct_tmp_[0][0]);
          get_dct(inp, dct_tmp);
          get_hash_dct(dct_tmp, trg);
          return py::cast<py::none>(Py_None);
    }
    
    // wrap C++ function with NumPy array IO
    int get_hash_fast(py::array_t<unsigned char> inp,
                      py::array_t<float> dct_tmp_f32,
                      py::array_t<unsigned char> trg) {  
    //int get_hash_fast(py::array inp,
    //                  py::array dct_tmp_f32,
    //                  py::array trg) {      
          //auto buf = inp.request();
          //auto buf2 = dct_tmp_f32.request();
          //auto buf3 = trg.request();
          //nobj.get_dct_u8((unsigned char*)buf.ptr, 32, 32, 1, 32, (unsigned char*)buf2.ptr);
          //nobj.get_hash_dct((float*)buf2.ptr, (unsigned char*)buf3.ptr);
          unsigned char *inpc = (unsigned char *)&(inp.unchecked<2>()(0,0)), *outpc = (unsigned char *)&(trg.unchecked<1>()(0));
          unsigned char * tmpc = (unsigned char *)&(dct_tmp_f32.unchecked<1>()(0));
          nobj.get_dct_u8(inpc, 32, 32, 1, 32, tmpc);
          nobj.get_hash_dct((float*)tmpc, outpc);
          return 0;
    }
             
    int get_bitlen() {
        return nobj.get_bitlen();
    }
             
    py::object get_norm(py::array trg) {
          if ( trg.ndim() != 1 ||  trg.itemsize() != 4 || trg.shape()[0] != nap_norm_len)
            throw std::runtime_error("Target should be 1-D NumPy float array with a size of 324");
          auto buf = trg.request();
          nobj.get_nap_norm((float*)buf.ptr);
          return py::cast<py::none>(Py_None);
    }
    
    py::object set_norm(py::array inp, const bool do_normalization=true) {
          if ( inp.ndim() != 1 ||  inp.itemsize() != 4 )
            throw std::runtime_error("Input should be 1-D NumPy float array.");
          auto buf = inp.request();
          nobj.set_nap_norm((float*)buf.ptr, inp.shape()[0], do_normalization);
          return py::cast<py::none>(Py_None);
    }
};

PYBIND11_MODULE(naphash_cpp, m) {
     m.doc() = "naphash; a dct-based image hash";
    // bindings to Pet class
    py::enum_<naphash::rot_inv_type>(m, "rot_inv_type")
        .value("none", naphash::rot_inv_none)
        .value("swap", naphash::rot_inv_swap)
        .value("full", naphash::rot_inv_full)
        .export_values();
    
    py::class_<pynaphash>(m, "naphash")
        .def(py::init<const int, const naphash::rot_inv_type, const bool, const bool>(), py::arg("dct_dim") = 32, py::arg("rot_inv_mode") = naphash::rot_inv_full,  py::arg("apply_center_crop") = false, py::arg("is_rgb") = true)
        .def("get_hash",  &pynaphash::get_hash, "Calculate naphash based on input image")
        .def("get_dct",  &pynaphash::get_dct, "Calculate dct based on input image")
        .def("get_hash_dct",  &pynaphash::get_hash_dct, "Calculate naphash based on dct input")
        .def("get_hash_fast",  &pynaphash::get_hash_fast, "Calculate naphash based on 32x32x1_u8 image")
        .def("get_bitlen",  &pynaphash::get_bitlen, "Returns number of usable bits of resulting naphashes")
        .def("set_norm",  &pynaphash::set_norm, "Set custom naphash norm coeff weights")
        .def("get_norm",  &pynaphash::get_norm, "Get naphash norm coeff weights");
}