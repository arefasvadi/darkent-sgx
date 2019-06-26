#ifndef COL2IM_H
#define COL2IM_H

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
#include "BlockEngine.hpp"
#include <array>
void col2im_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & data_col, int data_col_offset,
        int channels, int height, int width,
        int ksize, int stride, int pad, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &data_im,int data_im_offset);
#endif

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef GPU
void col2im_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);
#endif
#endif
