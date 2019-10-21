#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

void forward_activation_layer(layer& l, network& net);
void backward_activation_layer(layer& l, network& net);

#ifdef GPU
void forward_activation_layer_gpu(layer l, network net);
void backward_activation_layer_gpu(layer l, network net);
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
layer_blocked make_activation_layer_blocked(int batch, int inputs, ACTIVATION activation);

void forward_activation_layer_blocked(layer_blocked l, network_blocked net);
void backward_activation_layer_blocked(layer_blocked l, network_blocked net);
#endif

#endif

