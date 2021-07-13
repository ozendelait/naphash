#include "naphash.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

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
    if(w != trg_sz || h!= trg_sz)
        cv::resize(inp, inp, cv::Size(trg_sz, trg_sz), 0, 0, cv::INTER_AREA);
    return inp;
}

void naphash::get_hash_f32(float* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg)
{
    int imtype = (c==3)?CV_32FC3:CV_32FC1;
    cv::Mat inp = _naphash_getMat((uchar*) ptr, w, h, c, imtype, stepsz, this->c3_is_rgb, this->apply_center_crop, this->dct_dim);
    memcpy(ptr_trg, inp.data, this->dct_dim*this->dct_dim*sizeof(float));
}

void naphash::get_hash_u8(unsigned char* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg)
{
    int imtype = (c==3)?CV_8UC3:CV_8UC1;
    cv::Mat inp = _naphash_getMat(ptr, w, h, c, imtype, stepsz, this->c3_is_rgb, this->apply_center_crop, this->dct_dim);
    inp.convertTo(inp, CV_32FC1);
    memcpy(ptr_trg, inp.data, this->dct_dim*this->dct_dim*sizeof(float));
}