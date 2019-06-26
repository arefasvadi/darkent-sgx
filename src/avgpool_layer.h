#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer avgpool_layer;

image get_avgpool_image(avgpool_layer l);
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void resize_avgpool_layer(avgpool_layer *l, int w, int h);
void forward_avgpool_layer(const avgpool_layer l, network net);
void backward_avgpool_layer(const avgpool_layer l, network net);

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
typedef layer_blocked avgpool_layer_blocked;
avgpool_layer_blocked make_avgpool_layer_blocked(int batch, int w, int h, int c);
void forward_avgpool_layer_blocked(const avgpool_layer_blocked l, network_blocked net);
void backward_avgpool_layer_blocked(const avgpool_layer_blocked l, network_blocked net);
#endif

#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network net);
void backward_avgpool_layer_gpu(avgpool_layer l, network net);
#endif

#endif

