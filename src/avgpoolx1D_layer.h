#ifndef AVGPOOLX1D_LAYER_H
#define AVGPOOLX1D_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer avgpoolx1D_layer;

image get_avgpoolx_image(avgpoolx1D_layer l);
avgpoolx1D_layer make_avgpoolx1D_layer(int batch, int h, int w, int c, int size, int stride, int padding);
//void resize_avgpoolx1D_layer(avgpoolx1D_layer *l, int w, int h);
void forward_avgpoolx1D_layer(avgpoolx1D_layer& l, network &net);
void backward_avgpoolx1D_layer(avgpoolx1D_layer& l, network &net);

#ifdef GPU
// void forward_avgpool_layer_gpu(avgpool_layer l, network net);
// void backward_avgpool_layer_gpu(avgpool_layer l, network net);
#endif

#endif