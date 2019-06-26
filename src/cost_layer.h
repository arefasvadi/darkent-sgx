#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer cost_layer;


COST_TYPE get_cost_type(char *s);
char *get_cost_string(COST_TYPE a);
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE type, float scale);
void forward_cost_layer(const cost_layer l, network net);
void backward_cost_layer(const cost_layer l, network net);
void resize_cost_layer(cost_layer *l, int inputs);

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
typedef layer_blocked cost_layer_blocked;
cost_layer_blocked make_cost_layer_blocked(int batch, int inputs, COST_TYPE type, float scale);
void forward_cost_layer_blocked(const cost_layer_blocked l, network_blocked net);
void backward_cost_layer_blocked(const cost_layer_blocked l, network_blocked net);
#endif

#ifdef GPU
void forward_cost_layer_gpu(cost_layer l, network net);
void backward_cost_layer_gpu(const cost_layer l, network net);
#endif

#endif
