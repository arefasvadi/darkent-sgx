#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;

void softmax_array(float *input, int n, float temp, float *output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(softmax_layer& l, network& net);
void backward_softmax_layer(softmax_layer& l, network& net);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network net);
void backward_softmax_layer_gpu(const softmax_layer l, network net);
#endif

#if defined(SGX_VERIFIES) && defined(GPU)
    void forward_softmax_gpu_sgx_verifies_fbv     (softmax_layer l, network net);
    void backward_softmax_gpu_sgx_verifies_fbv    (softmax_layer l, network net);
    // void update_softmax_gpu_sgx_verifies_fbv      (softmax_layer l, update_args);
    // void create_softmax_snapshot_for_sgx_fbv      (struct layer&, struct network&, uint8_t** out, uint8_t**sha256_out);
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
typedef layer_blocked softmax_layer_blocked;
softmax_layer_blocked make_softmax_layer_blocked(int batch, int inputs, int groups);
void forward_softmax_layer_blocked(const softmax_layer_blocked l, network_blocked net);
void backward_softmax_layer_blocked(const softmax_layer_blocked l, network_blocked net);
#endif

#endif
