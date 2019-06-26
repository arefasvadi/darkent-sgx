#include "crop_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_crop_image(crop_layer l) {
  int h = l.out_h;
  int w = l.out_w;
  int c = l.out_c;
  return float_to_image(w, h, c, l.output);
}

void backward_crop_layer(const crop_layer l, network net) {}
void backward_crop_layer_gpu(const crop_layer l, network net) {}

crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height,
                           int crop_width, int flip, float angle,
                           float saturation, float exposure) {
  fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h, w,
          crop_height, crop_width, c);
  crop_layer l = {};
  l.type = CROP;
  l.batch = batch;
  l.h = h;
  l.w = w;
  l.c = c;
  l.scale = (float)crop_height / h;
  l.flip = flip;
  l.angle = angle;
  l.saturation = saturation;
  l.exposure = exposure;
  l.out_w = crop_width;
  l.out_h = crop_height;
  l.out_c = c;
  l.inputs = l.w * l.h * l.c;
  l.outputs = l.out_w * l.out_h * l.out_c;
  l.output = (float *)calloc(l.outputs * batch, sizeof(float));
  l.forward = forward_crop_layer;
  l.backward = backward_crop_layer;

#ifdef GPU
  l.forward_gpu = forward_crop_layer_gpu;
  l.backward_gpu = backward_crop_layer_gpu;
  l.output_gpu = cuda_make_array(l.output, l.outputs * batch);
  l.rand_gpu = cuda_make_array(0, l.batch * 8);
#endif
  return l;
}

void resize_crop_layer(layer *l, int w, int h) {
  l->w = w;
  l->h = h;

  l->out_w = l->scale * w;
  l->out_h = l->scale * h;

  l->inputs = l->w * l->h * l->c;
  l->outputs = l->out_h * l->out_w * l->out_c;

  l->output =
      (float *)realloc(l->output, l->batch * l->outputs * sizeof(float));
#ifdef GPU
  cuda_free(l->output_gpu);
  l->output_gpu = cuda_make_array(l->output, l->outputs * l->batch);
#endif
}

void forward_crop_layer(const crop_layer l, network net) {
  int i, j, c, b, row, col;
  int index;
  int count = 0;
  int flip = (l.flip && rand() % 2);
  int dh = rand() % (l.h - l.out_h + 1);
  int dw = rand() % (l.w - l.out_w + 1);
  float scale = 2;
  float trans = -1;
  if (l.noadjust) {
    scale = 1;
    trans = 0;
  }
  if (!net.train) {
    flip = 0;
    dh = (l.h - l.out_h) / 2;
    dw = (l.w - l.out_w) / 2;
  }
  for (b = 0; b < l.batch; ++b) {
    for (c = 0; c < l.c; ++c) {
      for (i = 0; i < l.out_h; ++i) {
        for (j = 0; j < l.out_w; ++j) {
          if (flip) {
            col = l.w - dw - j - 1;
          } else {
            col = j + dw;
          }
          row = i + dh;
          index = col + l.w * (row + l.h * (c + l.c * b));
          l.output[count++] = net.input[index] * scale + trans;
        }
      }
    }
  }
}

#if defined(USE_SGX) && defined(USE_SGX_BLOCKING)
void backward_crop_layer_blocked(const crop_layer_blocked l,
                                 network_blocked net) {}

crop_layer_blocked make_crop_layer_blocked(int batch, int h, int w, int c,
                                           int crop_height, int crop_width,
                                           int flip, float angle,
                                           float saturation, float exposure) {
  crop_layer_blocked l = {};
  l.type = CROP;
  l.batch = batch;
  l.h = h;
  l.w = w;
  l.c = c;
  l.scale = (float)crop_height / h;
  l.flip = flip;
  l.angle = angle;
  l.saturation = saturation;
  l.exposure = exposure;
  l.out_w = crop_width;
  l.out_h = crop_height;
  l.out_c = c;
  l.inputs = l.w * l.h * l.c;
  l.outputs = l.out_w * l.out_h * l.out_c;
  // l.output = (float*)calloc(l.outputs*batch, sizeof(float));
  l.output = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer(
      {l.outputs * batch});
  l.forward_blocked = forward_crop_layer_blocked;
  l.backward_blocked = backward_crop_layer_blocked;

  return l;
}
void forward_crop_layer_blocked(const crop_layer_blocked l,
                                network_blocked net) {
  int i, j, c, b, row, col;
  int index;
  int count = 0;
  int flip = (l.flip && rand() % 2);
  int dh = rand() % (l.h - l.out_h + 1);
  int dw = rand() % (l.w - l.out_w + 1);
  float scale = 2;
  float trans = -1;
  if (l.noadjust) {
    scale = 1;
    trans = 0;
  }
  if (!net.train) {
    flip = 0;
    dh = (l.h - l.out_h) / 2;
    dw = (l.w - l.out_w) / 2;
  }
  BLOCK_ENGINE_INIT_FOR_LOOP(net.input, net_input_valid_range,
                             net_input_block_val_ptr, float)
  BLOCK_ENGINE_INIT_FOR_LOOP(l.output, l_output_valid_range,
                             l_output_block_val_ptr, float)
  for (b = 0; b < l.batch; ++b) {
    for (c = 0; c < l.c; ++c) {
      for (i = 0; i < l.out_h; ++i) {
        for (j = 0; j < l.out_w; ++j) {
          if (flip) {
            col = l.w - dw - j - 1;
          } else {
            col = j + dw;
          }
          row = i + dh;
          index = col + l.w * (row + l.h * (c + l.c * b));

          BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.input, net_input_valid_range,
                                              net_input_block_val_ptr, false,
                                              net_input_index_var, index)
          BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.output, l_output_valid_range,
                                              l_output_block_val_ptr, true,
                                              l_output_index_var, count)
          // l.output[count++] = net.input[index]*scale + trans;
          *(l_output_block_val_ptr + l_output_index_var -
            l_output_valid_range.block_requested_ind) =
              (*(net_input_block_val_ptr+net_input_index_var-net_input_valid_range.block_requested_ind)) * scale + trans;
          count++;
        }
      }
    }
  }
  BLOCK_ENGINE_LAST_UNLOCK(net.input, net_input_valid_range)
  BLOCK_ENGINE_LAST_UNLOCK(l.output, l_output_valid_range)
}
#endif