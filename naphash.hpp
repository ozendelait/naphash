class naphash
{
    int  dct_dim; //dct dimension (square matrix) 
    bool apply_center_crop; //use square center crops instead of squishing for non-square input
    bool c3_is_rgb; //input with three components is in rgb (PIL) rather than bgr (opencv)
public:
    naphash(int _dct_dim=32, bool _apply_center_crop=false, bool _c3_is_rgb=true): 
        dct_dim(_dct_dim), apply_center_crop(_apply_center_crop), c3_is_rgb(_c3_is_rgb) {};
    void get_hash_f32(float* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg);
    void get_hash_u8(unsigned char* ptr, int w, int h, int c, int stepsz, unsigned char* ptr_trg);
};