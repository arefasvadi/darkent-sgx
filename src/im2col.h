#ifndef IM2COL_H
#define IM2COL_H
//#pragma once
#ifdef USE_SGX
#include "timingdefs.h"
#endif
void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

void im2col_cpu1D(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);
#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
#include "BlockEngine.hpp"
void im2col_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & data_im, int data_im_offset,
        int channels, int height, int width,
        int ksize, int stride, int pad, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & data_col, int data_col_offset);
#endif

#endif
