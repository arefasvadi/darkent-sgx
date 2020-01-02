#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam,PRNG& net_layer_rng_deriver);

void forward_connected_layer(layer& l, network& net);
void backward_connected_layer(layer& l, network& net);
void update_connected_layer(layer& l, update_args a);

#if defined(SGX_VERIFIES) && defined(GPU)
    void forward_connected_gpu_sgx_verifies_fbv     (struct layer, struct network);
    void backward_connected_gpu_sgx_verifies_fbv    (struct layer, struct network);
    void update_connected_gpu_sgx_verifies_fbv      (struct layer, update_args);
    void create_connected_snapshot_for_sgx_fbv      (struct layer&, struct network&, uint8_t** out, uint8_t**sha256_out);
#endif

#ifdef GPU
void forward_connected_layer_gpu(layer l, network net);
void backward_connected_layer_gpu(layer l, network net);
void update_connected_layer_gpu(layer l, update_args a);
void push_connected_layer(layer l);
void pull_connected_layer(layer l);
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
layer_blocked make_connected_layer_blocked(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void forward_connected_layer_blocked(layer_blocked l, network_blocked net);
void backward_connected_layer_blocked(layer_blocked l, network_blocked net);
void update_connected_layer_blocked(layer_blocked l, update_args a);

#endif

#endif

