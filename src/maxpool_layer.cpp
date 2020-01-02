#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

#ifndef USE_SGX_LAYERWISE
image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}
#endif

#ifndef USE_SGX_LAYERWISE
image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}
#endif

#ifndef USE_SGX_LAYERWISE
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = (int*)calloc(output_size, sizeof(int));
    l.output =  (float*)calloc(output_size, sizeof(float));
    l.delta =   (float*)calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #if defined(SGX_VERIFIES) && defined(GPU)
    l.forward_gpu_sgx_verifies = forward_maxpool_gpu_sgx_verifies_fbv; 
    l.backward_gpu_sgx_verifies = backward_maxpool_gpu_sgx_verifies_fbv;
    // l.create_snapshot_for_sgx = create_maxpool_snapshot_for_sgx_fbv;
    #endif 
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}
#endif

#ifndef USE_SGX_LAYERWISE
void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = (int*)realloc(l->indexes, output_size * sizeof(int));
    l->output = (float*)realloc(l->output, output_size * sizeof(float));
    l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}
#endif

#ifndef USE_SGX_LAYERWISE
void forward_maxpool_layer(maxpool_layer& l, network& net)
{
    // TODO: Should be data oblivious
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

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
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
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
void backward_maxpool_layer(maxpool_layer& l, network& net)
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

#if defined(SGX_VERIFIES) && defined(GPU)
void
forward_maxpool_gpu_sgx_verifies_fbv(maxpool_layer l, network net) {
  forward_maxpool_layer_gpu(l, net);
}
void
backward_maxpool_gpu_sgx_verifies_fbv(maxpool_layer l, network net) {
  backward_maxpool_layer_gpu(l, net);
}

// void
// create_convolutional_snapshot_for_sgx_fbv(struct layer &  l,
//                                           struct network &net,
//                                           uint8_t **      out,
//                                           uint8_t **      sha256_out) {
//   if (gpu_index >= 0) {
//     pull_convolutional_layer(l);
//   }
//   int total_bytes = (l.nbiases + l.nweights) * sizeof(float);
//   if (l.batch_normalize) {
//     total_bytes += (3 * l.nbiases) * sizeof(float);
//   }
//   size_t buff_ind = 0;
//   *out            = new uint8_t[total_bytes];
//   *sha256_out     = new uint8_t[SHA256_DIGEST_LENGTH];

//   std::memcpy((*out + buff_ind), l.biases, l.nbiases * sizeof(float));
//   buff_ind += l.nbiases * sizeof(float);
//   std::memcpy((*out + buff_ind), l.weights, l.nweights * sizeof(float));
//   buff_ind += l.nweights * sizeof(float);

//   if (l.batch_normalize) {
//     std::memcpy((*out + buff_ind), l.scales, l.nbiases * sizeof(float));
//     buff_ind += l.nbiases * sizeof(float);
//     std::memcpy((*out + buff_ind), l.rolling_mean, l.nbiases * sizeof(float));
//     buff_ind += l.nbiases * sizeof(float);
//     std::memcpy(
//         (*out + buff_ind), l.rolling_variance, l.nbiases * sizeof(float));
//     buff_ind += l.nbiases * sizeof(float);
//   }
//   if (buff_ind != total_bytes) {
//     LOG_ERROR("size mismatch\n")
//     abort();
//   }
//   gen_sha256(*out, total_bytes, *sha256_out);
// }
#endif

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
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
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    //fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void forward_maxpool_layer(maxpool_layer& l, network& net)
{
    // TODO: Should be data oblivious
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

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
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
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
                    if (max_i == -1) {
                        auto aaa = 0;
                        LOG_ERROR("in image %d for channel %d there was negative index!\n",b,k)
                        abort();
                    }
                    l_indexes[out_index] = max_i;
                }
            }
        }
    }
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    l.indexes->setItemsInRange(0, l.indexes->getBufferSize(),l_indexes);
}

void backward_maxpool_layer(maxpool_layer& l, network& net)
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
        /* if (index == -1) {
            //auto aaa = 0;
            LOG_ERROR("index -1 seen at i %d out of %d\n",i,h*w*c*l.batch)
            abort();
        } */
        net_delta[index] += l_delta[i];
    }
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
maxpool_layer_blocked make_maxpool_layer_blocked(int batch, int h, int w, int c, int size, int stride, int padding) {
    maxpool_layer_blocked l = {};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    // l.indexes = (int*)calloc(output_size, sizeof(int));
    l.indexes = sgx::trusted::BlockedBuffer<int, 1>::MakeBlockedBuffer({output_size});
    // l.output =  (float*)calloc(output_size, sizeof(float));
    l.output =  sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({output_size});
    // l.delta =   (float*)calloc(output_size, sizeof(float));
    l.delta =   sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({output_size});
    l.forward_blocked = forward_maxpool_layer_blocked;
    l.backward_blocked = backward_maxpool_layer_blocked;

    return l;
}
void forward_maxpool_layer_blocked(const maxpool_layer_blocked l, network_blocked net) {
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    //BLOCK_ENGINE_INIT_FOR_LOOP(net.input, net_input_valid_range, net_input_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(net.input, net_input_valid_range, net_input_block_val_ptr,net_input_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(l.indexes, indexes_valid_range, indexes_block_val_ptr, int)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(l.indexes, indexes_valid_range, indexes_block_val_ptr,indexes_index_var,true, int)
    //BLOCK_ENGINE_INIT_FOR_LOOP(l.output, output_valid_range, output_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(l.output, output_valid_range, output_block_val_ptr,output_index_var,true, float)

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            // float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            float val = -FLT_MAX;
                            if (valid != 0) {
                                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.input, net_input_valid_range, net_input_block_val_ptr, false, net_input_index_var, index)
                                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(net.input, net_input_valid_range, net_input_block_val_ptr, false, net_input_index_var, index)
                                // val =  net.input[index]
                                val = *(net_input_block_val_ptr+net_input_index_var-net_input_valid_range.block_requested_ind);
                                if (val > max) {
                                    max_i = index;
                                    max   = val;
                                }
                            }
                        }
                    }
                    //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.output, output_valid_range, output_block_val_ptr, true, output_index_var, out_index)
                    BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(l.output, output_valid_range, output_block_val_ptr, true, output_index_var, out_index)
                    //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.indexes, indexes_valid_range, indexes_block_val_ptr, true, indexes_index_var, out_index)
                    BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(l.indexes, indexes_valid_range, indexes_block_val_ptr, true, indexes_index_var, out_index)
                    // l.output[out_index] = max;
                    *(output_block_val_ptr+output_index_var-output_valid_range.block_requested_ind) = max;
                    // l.indexes[out_index] = max_i;
                    *(indexes_block_val_ptr+indexes_index_var-indexes_valid_range.block_requested_ind) = max_i;
                }
            }
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(net.input, net_input_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(l.indexes, indexes_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(l.output, output_valid_range)
}
void backward_maxpool_layer_blocked(const maxpool_layer_blocked l, network_blocked net) {
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    //BLOCK_ENGINE_INIT_FOR_LOOP(l.indexes, indexes_valid_range, indexes_block_val_ptr, int)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(l.indexes, indexes_valid_range, indexes_block_val_ptr,indexes_index_var,false, int)
    //BLOCK_ENGINE_INIT_FOR_LOOP(l.delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(l.delta, delta_valid_range, delta_block_val_ptr,delta_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(net.delta, net_delta_valid_range, net_delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(net.delta, net_delta_valid_range, net_delta_block_val_ptr,net_delta_index_var,true, float)
    for(i = 0; i < h*w*c*l.batch; ++i){
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.indexes, indexes_valid_range, indexes_block_val_ptr, false, indexes_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(l.indexes, indexes_valid_range, indexes_block_val_ptr, false, indexes_index_var, i)
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(l.delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, i)
        // int index = l.indexes[i];
        int index = *(indexes_block_val_ptr+indexes_index_var-indexes_valid_range.block_requested_ind);
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.delta, net_delta_valid_range, net_delta_block_val_ptr, true, net_delta_index_var, index)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(net.delta, net_delta_valid_range, net_delta_block_val_ptr, true, net_delta_index_var, index)
        // net.delta[index] += l.delta[i];
        *(net_delta_block_val_ptr+net_delta_index_var-net_delta_valid_range.block_requested_ind) += *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.indexes, indexes_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(l.delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(net.delta, net_delta_valid_range)

}
#endif
