#ifndef AVGPOOLX_LAYER_H
#define AVGPOOLX_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer avgpoolx_layer;

image get_avgpoolx_image(avgpoolx_layer l);
avgpoolx_layer make_avgpoolx_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_avgpoolx_layer(avgpoolx_layer *l, int w, int h);
void forward_avgpoolx_layer(avgpoolx_layer& l, network& net);
void backward_avgpoolx_layer(avgpoolx_layer& l, network& net);

#ifdef GPU
// void forward_avgpool_layer_gpu(avgpool_layer l, network net);
// void backward_avgpool_layer_gpu(avgpool_layer l, network net);
#endif

#endif