#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer maxpool_layer;

image get_maxpool_image(const maxpool_layer& l);
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(maxpool_layer& l, network& net);
void backward_maxpool_layer(maxpool_layer& l, network& net);

#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer l, network net);
void backward_maxpool_layer_gpu(maxpool_layer l, network net);
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
typedef layer_blocked maxpool_layer_blocked;

maxpool_layer_blocked make_maxpool_layer_blocked(int batch, int h, int w, int c, int size, int stride, int padding);
void forward_maxpool_layer_blocked(const maxpool_layer_blocked l, network_blocked net);
void backward_maxpool_layer_blocked(const maxpool_layer_blocked l, network_blocked net);
#endif

#endif

