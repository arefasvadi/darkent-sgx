#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef USE_SGX_LAYERWISE
dropout_layer make_dropout_layer(int batch, int inputs, float probability,PRNG& net_layer_rng_deriver)
{
    dropout_layer l = {};
    l.layer_rng = std::make_shared<PRNG>(generate_random_seed_from(net_layer_rng_deriver));
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = (float*)calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    
    #if defined(SGX_VERIFIES) && defined(GPU)
    l.forward_gpu_sgx_verifies = forward_dropout_gpu_sgx_verifies_fbv; 
    l.backward_gpu_sgx_verifies = backward_dropout_gpu_sgx_verifies_fbv;
    // l.update_gpu_sgx_verifies = update_dropout_gpu_sgx_verifies_fbv;
    // l.create_snapshot_for_sgx = create_dropout_snapshot_for_sgx_fbv;
    #endif 
    
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 
#endif

#if defined(SGX_VERIFIES) && defined(GPU)
void
forward_dropout_gpu_sgx_verifies_fbv(dropout_layer l, network net) {
  forward_dropout_layer_gpu(l, net);
}
void
backward_dropout_gpu_sgx_verifies_fbv(dropout_layer l, network net) {
  backward_dropout_layer_gpu(l, net);
}
// void
// update_dropout_gpu_sgx_verifies_fbv(dropout_layer l, update_args a) {
//   update_dropout_layer_gpu(l, a);
// }

// void
// create_dropout_snapshot_for_sgx_fbv(struct layer &  l,
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

#ifndef USE_SGX_LAYERWISE
void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = (float*)realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}
#endif

#ifndef USE_SGX_LAYERWISE
void forward_dropout_layer(dropout_layer& l, network& net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        // float r = rand_uniform(0, 1);
        float r = rand_uniform(*(l.layer_rng),0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_dropout_layer(dropout_layer& l, network& net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}
#endif

#if defined(USE_SGX) && defined(USE_SGX_LAYERWISE)
dropout_layer make_dropout_layer(int batch, int inputs, float probability,PRNG& net_layer_rng_deriver)
{
    dropout_layer l = {};
    l.layer_rng = std::make_shared<PRNG>(generate_random_seed_from(net_layer_rng_deriver));
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    //l.rand = (float*)calloc(inputs*batch, sizeof(float));
    l.rand = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*batch);
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    //fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
}

void forward_dropout_layer(dropout_layer& l, network& net)
{
    int i;
    if (!net.train) return;
    auto l_rand = l.rand->getItemsInRange(0, l.rand->getBufferSize());
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    for(i = 0; i < l.batch * l.inputs; ++i){
        // float r = rand_uniform(0, 1);
        float r = rand_uniform(*(l.layer_rng),0, 1);
        l_rand[i] = r;
        if(r < l.probability) net_input[i] = 0;
        else net_input[i] *= l.scale;
    }
    // if (net.index == 12) {
    //     print_array(&l_rand[0],l.batch * l.inputs,0,"SGX dropout rand vals");
    // }
    l.rand->setItemsInRange(0, l.rand->getBufferSize(),l_rand);
    net.input->setItemsInRange(0, net.input->getBufferSize(),net_input);
}

void backward_dropout_layer(dropout_layer& l, network& net)
{
    int i;
    if(!net.delta) return;
    auto l_rand = l.rand->getItemsInRange(0, l.rand->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l_rand[i];
        if(r < l.probability) net_delta[i] = 0;
        else net_delta[i] *= l.scale;
    }
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
}
#endif

#if defined(USE_SGX) && defined(USE_SGX_BLOCKING)

dropout_layer_blocked make_dropout_layer_blocked(int batch, int inputs, float probability) {
    dropout_layer_blocked l = {};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    // l.rand = (float*)calloc(inputs*batch, sizeof(float));
    l.rand = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({inputs*batch});
    l.scale = 1./(1.-probability);
    l.forward_blocked = forward_dropout_layer_blocked;
    l.backward_blocked = backward_dropout_layer_blocked;
    
    return l;
}

void forward_dropout_layer_blocked(dropout_layer_blocked l, network_blocked net) {
    int i;
    if (!net.train) return;
    BLOCK_ENGINE_INIT_FOR_LOOP(l.rand, rand_valid_range, rand_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(net.input, input_valid_range, input_block_val_ptr, float)
    for(i = 0; i < l.batch * l.inputs; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.rand, rand_valid_range, rand_block_val_ptr, true, rand_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.input, input_valid_range, input_block_val_ptr, true, input_index_var, i)
        float r = rand_uniform(0, 1);
        // l.rand[i] = r;
        *(rand_block_val_ptr+rand_index_var-rand_valid_range.block_requested_ind) = r;
        if(r < l.probability) {
            // net.input[i] = 0;
            *(input_block_val_ptr+input_index_var-input_valid_range.block_requested_ind) = 0;
        }
        else {
            // net.input[i] *= l.scale;
            *(input_block_val_ptr+input_index_var-input_valid_range.block_requested_ind) *= l.scale;
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.rand, rand_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(net.input, input_valid_range)
}
void backward_dropout_layer_blocked(dropout_layer_blocked l, network_blocked net) {
    //LOG_DEBUG("dropout backward")
    int i;
    if(!net.delta) {
        //LOG_DEBUG("dropout backward early return")
        return;
    }
    BLOCK_ENGINE_INIT_FOR_LOOP(l.rand, rand_valid_range, rand_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(net.delta, delta_valid_range, delta_block_val_ptr, float)
    for(i = 0; i < l.batch * l.inputs; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.rand, rand_valid_range, rand_block_val_ptr, false, rand_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.delta, delta_valid_range, delta_block_val_ptr, true, delta_index_var, i)
        // float r = l.rand[i];
        float r =  *(rand_block_val_ptr+rand_index_var-rand_valid_range.block_requested_ind);
        if(r < l.probability) {
            // net.delta[i] = 0;
            *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) = 0;
        }
        else {
            // net.delta[i] *= l.scale;
             *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) *= l.scale;
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.rand, rand_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(net.delta, delta_valid_range)
}
#endif
