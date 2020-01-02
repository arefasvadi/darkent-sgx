#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional_layer;

#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif
#endif

#if defined(SGX_VERIFIES) && defined(GPU)
    void forward_convolutional_gpu_sgx_verifies_fbv     (convolutional_layer l, network net);
    void backward_convolutional_gpu_sgx_verifies_fbv    (convolutional_layer l, network net);
    void update_convolutional_gpu_sgx_verifies_fbv      (convolutional_layer l, update_args);
    void create_convolutional_snapshot_for_sgx_fbv      (struct layer&, struct network&, uint8_t** out, uint8_t**sha256_out);
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam,PRNG& net_layer_rng_deriver);
void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void forward_convolutional_layer(convolutional_layer& layer, network& net);
void update_convolutional_layer(convolutional_layer& layer, update_args a);
image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_weights);
void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(convolutional_layer *l);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void backward_convolutional_layer(convolutional_layer& layer, network& net);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional_image(const convolutional_layer& layer);
image get_convolutional_delta(const convolutional_layer& layer);
image get_convolutional_weight(const convolutional_layer& layer, int i);

int convolutional_out_height(const convolutional_layer& layer);
int convolutional_out_width(const convolutional_layer& layer);

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
typedef layer_blocked convolutional_layer_blocked;
convolutional_layer_blocked make_convolutional_layer_blocked(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
int convolutional_out_height_blocked(convolutional_layer_blocked layer);
int convolutional_out_width_blocked(convolutional_layer_blocked layer);
void forward_convolutional_layer_blocked(const convolutional_layer_blocked layer, network_blocked net);
void update_convolutional_layer_blocked(convolutional_layer_blocked layer, update_args a);
void backward_convolutional_layer_blocked(convolutional_layer_blocked layer, network_blocked net);
void add_bias_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output, const  std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &biases, int batch, int n, int size);
void backward_bias_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & bias_updates, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, int batch, int n, int size);
#endif

#endif

