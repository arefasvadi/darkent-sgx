#include "im2col.h"
#include <stdio.h>

inline float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    // if (row < 0 || col < 0 ||
    //     row >= height || col >= width) return 0;
    if (((unsigned) row) < ((unsigned) height) && ((unsigned) col) < ((unsigned) width)) {
        return im[col + width*(row + height*channel)];
    }
    return 0;
}

float im2col_get_pixel1D(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    //row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    #ifdef USE_SGX
    SET_START_TIMING(SGX_TIMING_CONV_IM2COL);
    #endif
    int c,h,w;
    const int height_col = (height + 2*pad - ksize) / stride + 1;
    const int width_col = (width + 2*pad - ksize) / stride + 1;

    const int channels_col = channels * ksize * ksize;
    // #pragma omp parallel for
    // int col_major_ind=0;
    // #pragma omp parallel for collapse(2)
    // #pragma parallel for num_threads(6)
    int w_offset,h_offset, c_im,im_col,im_row,row,col,row_offset,srow_offset;
    // schedule(dynamic,ksize)
    // #pragma parallel for num_threads(6) schedule(static,ksize) private(c,h,w, w_offset, h_offset, c_im,im_row,im_row,im_col,row,col,row_offset,srow_offset) shared(ksize,stride,height_col,width_col,channels_col,data_im,height, width, channels,pad,data_im,data_col)
    for (c = 0; c < channels_col; ++c) {
        // int w_offset = c % ksize;
        // int h_offset = (c / ksize) % ksize;
        // int c_im = c / ksize / ksize;
        w_offset = (c % ksize) - pad;
        // h_offset = (c / ksize) % ksize;
        h_offset = ((c / ksize) % ksize) - pad;
        c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            // int im_row = h_offset + h * stride;
            // im_row = h_offset + h * stride;
            row = h_offset + h * stride;
            row_offset = (c * height_col + h) * width_col;
            srow_offset = width*(row + height*c_im);
            for (w = 0; w < width_col; ++w) {
                // int im_col = w_offset + w * stride;
                // im_col = w_offset + w * stride;
                // https://software.intel.com/content/www/us/en/develop/articles/caffe-optimized-for-intel-architecture-applying-modern-code-techniques.html
                // int col_index = (c * height_col + h) * width_col + w;
                col = w_offset + w * stride;
                if (((unsigned) row) < ((unsigned) height) && ((unsigned) col) < ((unsigned) width)) {
                    data_col[row_offset + w] = data_im[srow_offset+col];
                }
                else {
                    data_col[row_offset + w] = 0;
                }
                // data_col[(c * height_col + h) * width_col + w] = im2col_get_pixel(data_im, height, width, channels,
                //         im_row, im_col,c_im , pad);
                // data_col[col_major_ind] = im2col_get_pixel(data_im, height, width, channels,
                //         im_row, im_col, c_im, pad);
                // ++col_major_ind;
            }
        }
    }
    #ifdef USE_SGX
    SET_FINISH_TIMING(SGX_TIMING_CONV_IM2COL);
    #endif
}

void im2col_cpu1D(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    #ifdef USE_SGX
    
    #endif
    int c,h,w;
    int height_col = 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int im_row = 0;
    int channels_col = channels * 1 * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        //int h_offset = 0;
        int c_im = c / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                //int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel1D(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
void im2col_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & data_im, int data_im_offset,
        int channels, int height, int width,
        int ksize, int stride, int pad, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & data_col, int data_col_offset) {
        int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    //BLOCK_ENGINE_INIT_FOR_LOOP(data_col, data_col_valid_range, data_col_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(data_col, data_col_valid_range, data_col_block_val_ptr,data_col_index_var,true, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(data_im, data_im_valid_range, data_im_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(data_im, data_im_valid_range, data_im_block_val_ptr,data_im_index_var,false, float)
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                int row = im_row - pad;
                int col = im_col - pad;
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(data_col, data_col_valid_range, data_col_block_val_ptr, true, data_col_index_var, col_index+data_col_offset)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(data_col, data_col_valid_range, data_col_block_val_ptr, true, data_col_index_var, col_index+data_col_offset)
                if (row < 0 || col < 0 ||
                        row >= height || col >= width) {
                            // return 0;
                            *(data_col_block_val_ptr+data_col_index_var-data_col_valid_range.block_requested_ind) = 0.0;
                }
                else {
                    // data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        // im_row, im_col, c_im, pad);
                    //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(data_im, data_im_valid_range, data_im_block_val_ptr, false, data_im_index_var,col + width*(row + height*c_im)+data_im_offset)
                    BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(data_im, data_im_valid_range, data_im_block_val_ptr, false, data_im_index_var,col + width*(row + height*c_im)+data_im_offset)
                     *(data_col_block_val_ptr+data_col_index_var-data_col_valid_range.block_requested_ind) = *(data_im_block_val_ptr+data_im_index_var-data_im_valid_range.block_requested_ind);   
                }
                
            }
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(data_col, data_col_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(data_im, data_im_valid_range)
}
#endif
