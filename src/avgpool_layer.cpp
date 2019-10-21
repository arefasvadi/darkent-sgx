#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>

#ifndef USE_SGX_LAYERWISE
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    #ifndef USE_SGX
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    #endif
    avgpool_layer l = {};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  (float*)calloc(output_size, sizeof(float));
    l.delta =   (float*)calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_avgpool_layer_gpu;
    l.backward_gpu = backward_avgpool_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}
#endif

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

#ifndef USE_SGX_LAYERWISE
void forward_avgpool_layer(avgpool_layer& l, network& net)
{
    int b,i,k;

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
    }
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_avgpool_layer(avgpool_layer& l, network& net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    
    //fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    //l.output =  (float*)calloc(output_size, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(output_size);
    //l.delta =   (float*)calloc(output_size, sizeof(float));
    if (global_training) l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(output_size);
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    return l;
}

void forward_avgpool_layer(avgpool_layer& l, network& net)
{
    int b,i,k;
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l_output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l_output[out_index] += net_input[in_index];
            }
            l_output[out_index] /= l.h*l.w;
        }
    }
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
}

void backward_avgpool_layer(avgpool_layer& l, network& net)
{
    int b,i,k;
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net_delta[in_index] += l_delta[out_index] / (l.h*l.w);
            }
        }
    }
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)

avgpool_layer_blocked make_avgpool_layer_blocked(int batch, int w, int h, int c) {
    avgpool_layer_blocked l = {};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    // l.output =  (float*)calloc(output_size, sizeof(float));
    l.output =  sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({output_size});
    // l.delta =   (float*)calloc(output_size, sizeof(float));
    l.delta =   sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({output_size});
    l.forward_blocked = forward_avgpool_layer_blocked;
    l.backward_blocked = backward_avgpool_layer_blocked;
    
    return l;
}
void forward_avgpool_layer_blocked(const avgpool_layer_blocked l, network_blocked net) {
    int b,i,k;

    BLOCK_ENGINE_INIT_FOR_LOOP(l.output, output_valid_range, output_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(net.input, input_valid_range, input_block_val_ptr, float)
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.output, output_valid_range, output_block_val_ptr, true, output_index_var, out_index)
            //l.output[out_index] = 0;
            *(output_block_val_ptr+output_index_var-output_valid_range.block_requested_ind) = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.input, input_valid_range, input_block_val_ptr, false, input_index_var, in_index)
                // l.output[out_index] += net.input[in_index];
                *(output_block_val_ptr+output_index_var-output_valid_range.block_requested_ind) += *(input_block_val_ptr+input_index_var-input_valid_range.block_requested_ind);
            }
            // l.output[out_index] /= l.h*l.w;
             *(output_block_val_ptr+output_index_var-output_valid_range.block_requested_ind) /= l.h*l.w;
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.output, output_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(net.input, input_valid_range)
}
void backward_avgpool_layer_blocked(const avgpool_layer_blocked l, network_blocked net) {
    int b,i,k;
    BLOCK_ENGINE_INIT_FOR_LOOP(l.delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(net.delta, net_delta_valid_range, net_delta_block_val_ptr, float)
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, out_index)
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.delta, net_delta_valid_range, net_delta_block_val_ptr, true, net_delta_index_var, in_index)
                // net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
                *(net_delta_block_val_ptr+net_delta_index_var-net_delta_valid_range.block_requested_ind) += *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) / (l.h*l.w);
            }
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(net.delta, net_delta_valid_range)
}
#endif