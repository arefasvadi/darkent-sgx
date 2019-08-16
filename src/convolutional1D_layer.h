#ifndef CONVOLUTIONAL1D_LAYER_H
#define CONVOLUTIONAL1D_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional1D_layer;

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

convolutional1D_layer make_convolutional1D_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void resize_convolutional1D_layer(convolutional1D_layer *layer, int w, int h);
void forward_convolutional1D_layer(const convolutional1D_layer layer, network net);
void update_convolutional1D_layer(convolutional1D_layer layer, update_args a);
image *visualize_convolutional1D_layer(convolutional1D_layer layer, char *window, image *prev_weights);
void binarize_weights(float *weights, int n, int size, float *binary);
//void swap_binary(convolutional1D_layer *l);
//void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void backward_convolutional1D_layer(convolutional1D_layer layer, network net);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional1D_image(convolutional1D_layer layer);
image get_convolutional1D_delta(convolutional1D_layer layer);
image get_convolutional1D_weight(convolutional1D_layer layer, int i);

int convolutional1D_out_height(convolutional1D_layer layer);
int convolutional1D_out_width(convolutional1D_layer layer);

#endif