#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer dropout_layer;

dropout_layer make_dropout_layer(int batch, int inputs, float probability,PRNG& net_layer_rng_deriver);

void forward_dropout_layer(dropout_layer& l, network& net);
void backward_dropout_layer(dropout_layer& l, network& net);
void resize_dropout_layer(dropout_layer *l, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer l, network net);
void backward_dropout_layer_gpu(dropout_layer l, network net);

#endif

#if defined(USE_SGX) && defined(USE_SGX_LAYERWISE)
    #include "sgxlwfit/sgxlwfit.h"
#endif

#if defined(SGX_VERIFIES) && defined(GPU)
    void forward_dropout_gpu_sgx_verifies_fbv     (dropout_layer l, network net);
    void backward_dropout_gpu_sgx_verifies_fbv    (dropout_layer l, network net);
    // void update_dropout_gpu_sgx_verifies_fbv      (dropout_layer l, update_args);
    // void create_dropout_snapshot_for_sgx_fbv      (struct layer&, struct network&, uint8_t** out, uint8_t**sha256_out);
#endif

#if defined(USE_SGX) && defined(USE_SGX_BLOCKING)
typedef layer_blocked dropout_layer_blocked;

dropout_layer_blocked make_dropout_layer_blocked(int batch, int inputs, float probability);

void forward_dropout_layer_blocked(dropout_layer_blocked l, network_blocked net);
void backward_dropout_layer_blocked(dropout_layer_blocked l, network_blocked net);
#endif
#endif
