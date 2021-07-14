#include "naphash.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

//internal method to convert, crop and resize input image to single channel cv::Mat
static cv::Mat _naphash_getMat(uchar* ptr, int w, int h, int c, int imtype, int stepsz, bool is_rgb, bool apply_center_crop, int trg_sz) {
    if(c!=1 && c!=3)
        throw std::runtime_error("Input should have 1 or 3 channels (mono or rgb)");
    cv::Mat inp(h, w, imtype, ptr, stepsz);
    if(c == 3) //convert to 1 channel
        cv::cvtColor(inp, inp, is_rgb?cv::COLOR_RGB2GRAY:cv::COLOR_BGR2GRAY);
    if(apply_center_crop && (w!=h)) {
        int fix_offset = (w%2 != h%2);
        int c0 = (w - h + 1 - fix_offset)/2, m0 = std::min(w,h);
        if(c0 > 0){
            int x1 = c0+m0+fix_offset;
            if(x1 < w)
                inp = cv::Mat(inp, cv::Rect(c0,0, x1-c0, m0));
        } else {
            int y0 = -c0-fix_offset;
            if(y0 > 0)
                inp = cv::Mat(inp, cv::Rect(0,y0,m0,m0-y0));
        }
    }
    //PIL's LANCZOS works best for downsampling; INTER_AREA is second place (INTER_LANCZOS is not suitable)
    if(w != trg_sz || h!= trg_sz)
        cv::resize(inp, inp, cv::Size(trg_sz, trg_sz), 0, 0, cv::INTER_AREA);
    return inp;
}

static void _naphash_cpyNorm(const float* pSrc, float* pTrg, int num_coeffs, const bool equalize_coeffs) {
    memcpy(pTrg, pSrc, std::min(nap_norm_len,num_coeffs)*sizeof(float));
    float fill_value = pTrg[std::max(num_coeffs-1,0)]; //fill remainder with last supplied value
    while(num_coeffs < nap_norm_len)
        pTrg[num_coeffs++] = fill_value;
    //use mean of transposed indices for both coefficients -> both matrix and transposed matrix are the same
    if(equalize_coeffs) {
        //sloppy, writes each transposable index twice (mean of identical values on second pass does not change both values).
        for(int i=0; i<nap_norm_len;++i) {
            int i_tr = naphash_idx_tr[i];
            float res = (pTrg[i]+pTrg[i_tr])*0.5;
            pTrg[i] = pTrg[i_tr] = res;
        }
    }
}

naphash::naphash(int _dct_dim, rot_inv_type _rot_inv_mode, bool _apply_center_crop, bool _c3_is_rgb): 
        dct_dim(_dct_dim), rot_inv_mode(_rot_inv_mode), apply_center_crop(_apply_center_crop), c3_is_rgb(_c3_is_rgb) {
     const float * pDefaultNorm = _apply_center_crop?naphash_norm_crop:((_rot_inv_mode == rot_inv_full)?naphash_norm_rotinv:naphash_norm);
    _naphash_cpyNorm(pDefaultNorm, this->nap_norm_w, nap_norm_len, (_rot_inv_mode != rot_inv_none));
}

void naphash::get_dct_f32(float* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg)
{
    int imtype = (c==3)?CV_32FC3:CV_32FC1;
    cv::Mat inp = _naphash_getMat((uchar*) ptr, w, h, c, imtype, stepsz, this->c3_is_rgb, this->apply_center_crop, this->dct_dim);
    cv::Mat dct(this->dct_dim, this->dct_dim, CV_32FC1, ptr_trg, this->dct_dim*sizeof(float));
    cv::dct(inp,dct);
    if(this->rot_inv_mode != rot_inv_none)
        dct = cv::abs(dct);
}

void naphash::get_dct_u8(unsigned char* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg)
{
    int imtype = (c==3)?CV_8UC3:CV_8UC1;
    cv::Mat inp = _naphash_getMat(ptr, w, h, c, imtype, stepsz, this->c3_is_rgb, this->apply_center_crop, this->dct_dim);
    inp.convertTo(inp, CV_32FC1);
    cv::Mat dct(this->dct_dim, this->dct_dim, CV_32FC1, ptr_trg, this->dct_dim*sizeof(float));
    cv::dct(inp,dct);
    if(this->rot_inv_mode != rot_inv_none)
        dct = cv::abs(dct);
}
                                                                  
void naphash::get_hash_dct(float* dct, unsigned char* ptr_trg)
{
    memcpy(ptr_trg, dct, this->dct_dim*this->dct_dim*sizeof(float));
}

void naphash::get_hash_f32(float* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg)
{
    cv::Mat dct(this->dct_dim,this->dct_dim,CV_32F);
    get_dct_f32(ptr, w, h, c, stepsz, dct.data);
    get_hash_dct((float*)dct.data, ptr_trg);
}

void naphash::get_hash_u8(unsigned char* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg)
{
    cv::Mat dct(this->dct_dim,this->dct_dim,CV_32F);
    get_dct_u8(ptr, w, h, c, stepsz, dct.data);
    get_hash_dct((float*)dct.data, ptr_trg);
}
                                                                  

void naphash::set_nap_norm(const float *ptr, int num_coeffs)
{
    _naphash_cpyNorm(ptr, this->nap_norm_w, num_coeffs, false);
}
                                                                  
void naphash::get_nap_norm(float *ptr_trg)
{
    memcpy(ptr_trg, (unsigned char*)this->nap_norm_w, nap_norm_len*sizeof(float));
}