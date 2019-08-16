#include "maxpool1D_layer.h"
#include "cuda.h"
#include <stdio.h>

/* image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
} */

/* image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
} */
#ifndef USE_SGX_LAYERWISE
maxpool1D_layer make_maxpool1D_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool1D_layer l = {};
    l.type = MAXPOOL1D;
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
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = (int*)calloc(output_size, sizeof(int));
    l.output =  (float*)calloc(output_size, sizeof(float));
    l.delta =   (float*)calloc(output_size, sizeof(float));
    l.forward = forward_maxpool1D_layer;
    l.backward = backward_maxpool1D_layer;
    #ifdef GPU
    error("GPU for maxpool 1D not yet implemented!")
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max1D          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, 1, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}
#endif

/* void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
} */

#ifndef USE_SGX_LAYERWISE
void forward_maxpool1D_layer(const maxpool1D_layer l, network net)
{
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
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < 1; ++n){
                        for(m = 0; m < l.size; ++m){
                            //int cur_h = h_offset + i*l.stride + n;
                            int cur_h = 0;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_maxpool1D_layer(const maxpool1D_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}
#endif
#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
maxpool1D_layer make_maxpool1D_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool1D_layer l = {};
    l.type = MAXPOOL1D;
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
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    //l.indexes = (int*)calloc(output_size, sizeof(int));
    l.indexes = sgx::trusted::SpecialBuffer<int>::GetNewSpecialBuffer(output_size);
    //l.output =  (float*)calloc(output_size, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(output_size);
    //l.delta =   (float*)calloc(output_size, sizeof(float));
    l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(output_size);
    l.forward = forward_maxpool1D_layer;
    l.backward = backward_maxpool1D_layer;
    fprintf(stderr, "max1D          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, 1, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void forward_maxpool1D_layer(const maxpool1D_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = 0;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto l_indexes = l.indexes->getItemsInRange(0, l.indexes->getBufferSize());

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < 1; ++n){
                        for(m = 0; m < l.size; ++m){
                            //int cur_h = h_offset + i*l.stride + n;
                            int cur_h = 0;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net_input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l_output[out_index] = max;
                    l_indexes[out_index] = max_i;
                }
            }
        }
    }
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    l.indexes->setItemsInRange(0, l.indexes->getBufferSize(),l_indexes);
}

void backward_maxpool1D_layer(const maxpool1D_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    auto l_indexes = l.indexes->getItemsInRange(0, l.indexes->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l_indexes[i];
        net_delta[index] += l_delta[i];
    }
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
}
#endif