#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_batchnorm_layer(int batch, int w, int h, int c);
void forward_batchnorm_layer(layer& l, network& net);
void backward_batchnorm_layer(layer& l, network& net);

#ifdef GPU
void forward_batchnorm_layer_gpu(layer l, network net);
void backward_batchnorm_layer_gpu(layer l, network net);
void pull_batchnorm_layer(layer l);
void push_batchnorm_layer(layer l);
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
layer_blocked make_batchnorm_layer_blocked(int batch, int w, int h, int c);
void forward_batchnorm_layer_blocked(layer_blocked l, network_blocked net);
void backward_batchnorm_layer_blocked(layer_blocked l, network_blocked net);
#endif

#endif
