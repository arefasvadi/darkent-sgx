#include "avgpoolx1D_layer.h"
#include "cuda.h"
#include <stdio.h>

#ifndef USE_SGX_LAYERWISE
avgpoolx1D_layer make_avgpoolx1D_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    avgpoolx1D_layer l = {};
    l.type = AVGPOOLX1D;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.outputs * batch;
    l.output =  (float*)calloc(output_size, sizeof(float));
    if (global_training) l.delta =   (float*)calloc(output_size, sizeof(float));
    l.forward = forward_avgpoolx1D_layer;
    l.backward = backward_avgpoolx1D_layer;
    
    #ifdef GPU
    error("avgpoolx1D has not implemented for gpu yet!");
    // l.forward_gpu = forward_avgpool_layer_gpu;
    // l.backward_gpu = backward_avgpool_layer_gpu;
    // l.output_gpu  = cuda_make_array(l.output, output_size);
    // l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "avgx1D          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, 1, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}
#endif

/* void resize_avgpoolx_layer(avgpoolx_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    error("avgpoolx not yet implemented for gpu")
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif

} */

#ifndef USE_SGX_LAYERWISE
void forward_avgpoolx1D_layer(const avgpoolx1D_layer l, network net)
{
    /* int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    } */
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = 0;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float sum = 0.0;
                    for(n = 0; n < 1; ++n){
                        for(m = 0; m < l.size; ++m){
                            //int cur_h = h_offset + i*l.stride + n;
                            int cur_h = 0;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            if (valid) {
                                sum += net.input[index];
                            }
                        }
                    }
                    l.output[out_index] = sum / (l.size);
                }
            }
        }
    }
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_avgpoolx1D_layer(const avgpoolx1D_layer l, network net)
{
    /* int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    } */

    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = 0;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float val = l.delta[out_index]/(1*l.size);
                    for(n = 0; n < 1; ++n){
                        for(m = 0; m < l.size; ++m){
                            //int cur_h = h_offset + i*l.stride + n;
                            int cur_h = 0;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            if (valid) {
                                net.delta[index] += val;
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif
#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
avgpoolx1D_layer make_avgpoolx1D_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    avgpoolx1D_layer l = {};
    l.type = AVGPOOLX1D;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.outputs * batch;
    //l.output =  (float*)calloc(output_size, sizeof(float));
    l.output =  sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(output_size);
    //l.delta =   (float*)calloc(output_size, sizeof(float));
    l.delta =   sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(output_size);
    l.forward = forward_avgpoolx1D_layer;
    l.backward = backward_avgpoolx1D_layer;
    fprintf(stderr, "avgx1D          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, 1, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void forward_avgpoolx1D_layer(const avgpoolx1D_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = 0;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float sum = 0.0;
                    for(n = 0; n < 1; ++n){
                        for(m = 0; m < l.size; ++m){
                            //int cur_h = h_offset + i*l.stride + n;
                            int cur_h = 0;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            if (valid) {
                                sum += net_input[index];
                            }
                        }
                    }
                    l_output[out_index] = sum / (l.size);
                }
            }
        }
    }
    l.output->setItemsInRange(0, l.output->getBufferSize(), l_output);
}

void backward_avgpoolx1D_layer(const avgpoolx1D_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = 0;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float val = l_delta[out_index]/(1*l.size);
                    for(n = 0; n < 1; ++n){
                        for(m = 0; m < l.size; ++m){
                            //int cur_h = h_offset + i*l.stride + n;
                            int cur_h = 0;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            if (valid) {
                                net_delta[index] += val;
                            }
                        }
                    }
                }
            }
        }
    }
    net.delta->setItemsInRange(0, net.delta->getBufferSize(), net_delta);
}
#endif