/******************************************************************************
*
* FILENAME:     naphash_cpp.cpp
*
* PURPOSE:      C++ implementation of NAPHash; 
*               see https://github.com/ozendelait/naphash for more information
*               
* AUTHOR:       Oliver Zendel, AIT Austrian Institute of Technology GmbH
*
*  Copyright (C) 2021 AIT Austrian Institute of Technology GmbH
*  All rights reserved.
******************************************************************************/ 

#include "naphash.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <numeric> //needed for std::accumulate
#include <algorithm>

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
            if((x1 < w)&& c0 > 0)
                inp = cv::Mat(inp, cv::Rect(c0, 0, m0, m0));
        } else {
            int y0 = -c0-fix_offset;
            if(y0 > 0)
                inp = cv::Mat(inp, cv::Rect(0, y0, m0, m0));
        }
    }
    //PIL's LANCZOS works best for downsampling; INTER_AREA is second place (INTER_LANCZOS is not suitable)
    if(w != trg_sz || h!= trg_sz)
        cv::resize(inp, inp, cv::Size(trg_sz, trg_sz), 0, 0, cv::INTER_AREA);
    return inp;
}

//internal method to copy and normalize weights. 
//Normalized weights are positive int16 values 
//(done to be compatible with future SSE optimizations); 
//The smallest value is normalized to 256 (== 1.0)
//equalize_coeffs: Invariance towards transposition of input image requires 
//identical coefficients for weights affecting 
//DCT coefficients with transposed indices

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
            float res = (pTrg[i]+pTrg[i_tr])*0.5f;
            pTrg[i] = pTrg[i_tr] = res;
        }
    }
}

//Main constructor dct_dim
DLLEXPORT naphash::naphash(int _dct_dim, rot_inv_type _rot_inv_mode, bool _apply_center_crop, bool _c3_is_rgb): 
        dct_dim(_dct_dim), rot_inv_mode(_rot_inv_mode), 
        apply_center_crop(_apply_center_crop), c3_is_rgb(_c3_is_rgb), nap_norm_len_tr(0) {
     const float * pDefaultNorm = _apply_center_crop?naphash_norm_crop:((_rot_inv_mode == rot_inv_full)?naphash_norm_rotinv:naphash_norm);
    _naphash_cpyNorm(pDefaultNorm, this->nap_norm_w, nap_norm_len, (_rot_inv_mode != rot_inv_none));
    for(int i=0; i<nap_norm_len; ++i){ //setup access indices based on zig-zag pattern (increasing entropy)
        nap_norm_idx[i] = naphash_idx0[i]*_dct_dim+naphash_idx1[i];
        if(naphash_idx0[i] > naphash_idx1[i])
            continue;
        nap_norm_idx_tr0[nap_norm_len_tr] = i;
        nap_norm_idx_tr1[nap_norm_len_tr++] = (naphash_idx0[i] == naphash_idx1[i])?i:i+1; //transposed idx is next (or same for diagonals)
    }
            
}

int naphash::hamming_dist(unsigned char* ptr0, unsigned char* ptr1, int num_bytes){
  if(!ptr0 || !ptr1 || (num_bytes<=0))
    return -1;
  cv::Mat p0(1, num_bytes, CV_8UC1, ptr0, num_bytes), p1(1, num_bytes, CV_8UC1, ptr1, num_bytes);
  return (int)cv::norm(p0,p1,cv::NORM_HAMMING);
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
    
    char dbgtxt[1024]={0};
    cv::dct(inp,dct);
    if(this->rot_inv_mode != rot_inv_none)
        dct = cv::abs(dct);
}

int naphash::get_bitlen()
{
    return (this->rot_inv_mode == rot_inv_full)?this->nap_norm_len_tr:((this->rot_inv_mode == rot_inv_swap)?naphash_nondiag_pack_len:naphash_pack_len);
    
}

void naphash::get_hash_dct(float* dct, unsigned char* ptr_trg)
{
    const bool concatenate_transposed = (this->rot_inv_mode == rot_inv_full);
    const int num_add = concatenate_transposed?32:64; //calculate mean using this many normed entries
    const float thr_f = concatenate_transposed?(float)(230.0/(256.0*num_add)):(float)(206.0/(256.0*num_add)); //norm mean by this factor
    const int num_transp = concatenate_transposed?nap_norm_len_tr:0;
    
    float normed_coeffs[nap_norm_len], normed_coeffs_tr[nap_norm_len];
    for(int i=0; i<nap_norm_len; ++i)
        normed_coeffs[i]=this->nap_norm_w[i]*dct[nap_norm_idx[i]];
    for(int i = 0; i < num_transp; ++i)
        normed_coeffs_tr[i] = normed_coeffs[nap_norm_idx_tr0[i]]+normed_coeffs[nap_norm_idx_tr1[i]];
    const float *pAcc =  concatenate_transposed?normed_coeffs_tr:normed_coeffs;
    // regardless of result length, we use the same number of initial coefficients for mean (64 or 32)
    const float thr_val = thr_f * (float)std::accumulate(pAcc, pAcc+num_add,0.0f);
    char dbgtxt[1024]={0};
    
    const unsigned int *idx_pack = (this->rot_inv_mode == rot_inv_swap)?naphash_pack_nondiag_idx:naphash_pack_idx;
    const unsigned int len_pack = this->get_bitlen();
    const unsigned char set_bit_lut[8]={0x80,0x40,0x20,0x10,0x8,0x4,0x2,0x1};
        
    //pack result into ptr_trg bits
    for(int i=0; i<(int)(len_pack/8); ++i)
        ptr_trg[i] = 0;
    for(int i=0; i<(int)len_pack; ++i){
        if(pAcc[idx_pack[i]] > thr_val)
            ptr_trg[i/8] |= set_bit_lut[i&0x7];
    }
    /*sprintf(dbgtxt,"DEBUG: %f; \n%f %f %f %f %f %f %f %f\n %f %f %f %f %f %f %f %f", thr_val, 
            pAcc[idx_pack[0]], pAcc[idx_pack[1]], pAcc[idx_pack[2]], pAcc[idx_pack[3]],
            pAcc[idx_pack[4]], pAcc[idx_pack[5]], pAcc[idx_pack[6]], pAcc[idx_pack[7]],
            pAcc[idx_pack[8]], pAcc[idx_pack[9]], pAcc[idx_pack[10]], pAcc[idx_pack[11]],
            pAcc[idx_pack[12]], pAcc[idx_pack[13]], pAcc[idx_pack[14]], pAcc[idx_pack[15]]);
    throw std::runtime_error(std::string(dbgtxt));*/
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
                                                                  

void naphash::set_nap_norm(const float *ptr, int num_coeffs, bool do_normalization)
{
    float nap_norm_normalize[nap_norm_len];
    const float *ptr_use = do_normalization ? nap_norm_normalize : ptr;
    num_coeffs = std::min(nap_norm_len,num_coeffs);
    if(do_normalization) {
        const float norm256 = 256.0f/(*std::min_element(ptr,ptr+num_coeffs));
        const float max_s16 = 32767.0f;
        for(int i = 0; i < num_coeffs; ++i)
            nap_norm_normalize[i] = (float)((int)(std::min(ptr[i]*norm256,max_s16)+0.5));
    }
    _naphash_cpyNorm(ptr_use, this->nap_norm_w, num_coeffs, (do_normalization && (rot_inv_mode != rot_inv_none)));
}
                                                                  
void naphash::get_nap_norm(float *ptr_trg)
{
    memcpy(ptr_trg, (unsigned char*)this->nap_norm_w, nap_norm_len*sizeof(float));
}