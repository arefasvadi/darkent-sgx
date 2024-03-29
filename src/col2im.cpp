#include <stdio.h>
#include <math.h>
#include "col2im.h"


inline void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    // if (row < 0 || col < 0 ||
    //     row >= height || col >= width) return;
    if (((unsigned) row) < ((unsigned) height) && ((unsigned) col) < ((unsigned) width)) {
        im[col + width*(row + height*channel)] += val;
    }
}

void col2im_add_pixel1D(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    //row -= pad;
    col -= pad;

    // if (row < 0 || col < 0 ||
    //     row >= height || col >= width) return;
    if (((unsigned) row) < ((unsigned) height) && ((unsigned) col) < ((unsigned) width)) {
        im[col + width*(row + height*channel)] += val;
    }

}

//This one might be too, can't remember.
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im) 
{
    #ifdef USE_SGX
    SET_START_TIMING(SGX_TIMING_CONV_COL2IM);
    #endif
    int c,h,w;
    const int height_col = (height + 2*pad - ksize) / stride + 1;
    const int width_col = (width + 2*pad - ksize) / stride + 1;

    const int channels_col = channels * ksize * ksize;
    // #pragma omp parallel for
    int col_major_ind=0;
    // #pragma omp parallel for collapse(3)
    // #pragma parallel for num_threads(6)
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            int im_row = h_offset + h * stride;
            for (int w = 0; w < width_col; ++w) {
                int im_col = w_offset + w * stride;
                // https://software.intel.com/content/www/us/en/develop/articles/caffe-optimized-for-intel-architecture-applying-modern-code-techniques.html
                // int col_index = (c * height_col + h) * width_col + w;
                // double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, data_col[col_major_ind]);
                ++col_major_ind;
            }
        }
    }
    #ifdef USE_SGX
    SET_FINISH_TIMING(SGX_TIMING_CONV_COL2IM);
    #endif
}

void col2im_cpu1D(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im) 
{
    int c,h,w;
    int height_col = 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * 1 * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        //int h_offset = 0;
        int c_im = c / ksize;
        int im_row = 0;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                //int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel1D(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
void col2im_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & data_col, int data_col_offset,
        int channels, int height, int width,
        int ksize, int stride, int pad, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &data_im,int data_im_offset) {
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    //BLOCK_ENGINE_INIT_FOR_LOOP(data_col, data_col_valid_range, data_col_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(data_col, data_col_valid_range, data_col_block_val_ptr,data_col_current_index,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(data_im, data_im_valid_range, data_im_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(data_im, data_im_valid_range, data_im_block_val_ptr,data_im_current_index,true, float)
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                //double val = data_col[col_index];
                
                //col2im_add_pixel(data_im, height, width, channels,
                        //im_row, im_col, c_im, pad, val);
                int row = im_row - pad;
                int col = im_col - pad;

                if (!(row < 0 || col < 0 ||
                    row >= height || col >= width)) {
                        int col_index = (c * height_col + h) * width_col + w;
                        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(data_col, data_col_valid_range, data_col_block_val_ptr, false, data_col_current_index, col_index+data_col_offset)
                        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(data_col, data_col_valid_range, data_col_block_val_ptr, false, data_col_current_index, col_index+data_col_offset)
                        double val = *(data_col_block_val_ptr+data_col_current_index-data_col_valid_range.block_requested_ind);
                        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(data_im, data_im_valid_range, data_im_block_val_ptr, true, data_im_current_index,col + width*(row + height*c_im) + data_im_offset)
                        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(data_im, data_im_valid_range, data_im_block_val_ptr, true, data_im_current_index,col + width*(row + height*c_im) + data_im_offset)
                        *(data_im_block_val_ptr+data_im_current_index-data_im_valid_range.block_requested_ind) += val;
                        //im[col + width*(row + height*channel)] += val;
                }
            }
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(data_col, data_col_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(data_im, data_im_valid_range)
}
#endif

