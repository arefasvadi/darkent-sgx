#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

typedef layer crop_layer;

image get_crop_image(crop_layer l);
crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void forward_crop_layer(crop_layer &l, network &net);
void resize_crop_layer(layer *l, int w, int h);

#ifdef GPU
void forward_crop_layer_gpu(crop_layer l, network net);
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
typedef layer_blocked crop_layer_blocked;
crop_layer_blocked make_crop_layer_blocked(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void forward_crop_layer_blocked(const crop_layer_blocked l, network_blocked net);
#endif

#endif

