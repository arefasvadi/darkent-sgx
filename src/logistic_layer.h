#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H
#include "layer.h"
#include "network.h"

layer make_logistic_layer(int batch, int inputs);
void forward_logistic_layer(const layer l, network net);
void backward_logistic_layer(const layer l, network net);

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
layer_blocked make_logistic_layer_blocked(int batch, int inputs);
void forward_logistic_layer_blocked(const layer_blocked l, network_blocked net);
void backward_logistic_layer_blocked(const layer_blocked l, network_blocked net);

#endif

#ifdef GPU
void forward_logistic_layer_gpu(const layer l, network net);
void backward_logistic_layer_gpu(const layer l, network net);
#endif

#endif
