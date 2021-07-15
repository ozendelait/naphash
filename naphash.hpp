#include <napnorm.hpp>

class naphash
{
public:
    enum rot_inv_type
    {
        rot_inv_none = 0,
        rot_inv_swap = 1,
        rot_inv_full = 2
    };

    naphash(int _dct_dim=32, rot_inv_type _rot_inv_mode=rot_inv_none, bool _apply_center_crop=false, bool _c3_is_rgb=true);
    void get_dct_f32(float* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg); //ptr_trg must have a size of _dct_dim*_dct_dim*sizeof(float)
    void get_dct_u8(unsigned char* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg); //ptr_trg must have a size of _dct_dim*_dct_dim*sizeof(float)
    void get_hash_f32(float* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg); //ptr_trg must have a size of _dct_dim*_dct_dim*sizeof(float)
    void get_hash_u8(unsigned char* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg); //ptr_trg must have a size of _dct_dim*_dct_dim*sizeof(float)
    void get_hash_dct(float* dct, unsigned char* ptr_trg); //dct must have size dct_dim*dct_dim*sizeof(float), output should have enough bytes to fit get_bitlen() 
    
    int get_bitlen(); //returns number of bits of resulting naphash
    
    void set_nap_norm(const float *ptr, int num_coeffs);// input ptr must have size num_coeffs*sizeof(float)                                   
    void get_nap_norm(float *ptr_trg);//ptr_trg must have size nap_norm_len*sizeof(float)    
    
private:
    int  dct_dim; //dct dimension (square matrix) 
    bool apply_center_crop; //use square center crops instead of squishing for non-square input
    bool c3_is_rgb; //input with three components is in rgb (PIL) rather than bgr (opencv)
    rot_inv_type rot_inv_mode; //set nap-hash rotation invariance mode
    float nap_norm_w[nap_norm_len]; //currently active norm weights, allows user-supplied norm
    int nap_norm_idx[nap_norm_len]; //norm indices accessed from square dct matrix
    int nap_norm_idx_tr0[nap_norm_len]; //norm indices from square dct matrix having a transpose
    int nap_norm_idx_tr1[nap_norm_len]; //transposed norm indices from square dct matrix having a transpose
    int nap_norm_idx_nodiag[nap_norm_len]; //norm indices from square dct matrix without diagonal idx
    int nap_norm_len_tr; //number of transposable norm indices
};