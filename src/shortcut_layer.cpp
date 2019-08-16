#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>

#ifndef USE_SGX_LAYERWISE
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l = {};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta =  (float*)calloc(l.outputs*batch, sizeof(float));
    l.output = (float*)calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
    #ifdef GPU
    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}
#endif

#ifndef USE_SGX_LAYERWISE
void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta =  (float*)realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = (float*)realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}
#endif

#ifndef USE_SGX_LAYERWISE
void forward_shortcut_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}
#endif

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l = {};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    //l.delta =  (float*)calloc(l.outputs*batch, sizeof(float));
    l.delta =  sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.outputs*batch);
    //l.output = (float*)calloc(l.outputs*batch, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.outputs*batch);

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
    return l;
}

void forward_shortcut_layer(const layer l, network net)
{
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto index_output = net.layers[l.index].output->getItemsInRange(0, net.layers[l.index].output->getBufferSize());
    copy_cpu(l.outputs*l.batch, &net_input[0], 1, &l_output[0], 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, &index_output[0], l.out_w, l.out_h, l.out_c, l.alpha, l.beta, &l_output[0]);
    activate_array(&l_output[0], l.outputs*l.batch, l.activation);
    
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
}

void backward_shortcut_layer(const layer l, network net)
{
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto index_delta = net.layers[l.index].delta->getItemsInRange(0, net.layers[l.index].delta->getBufferSize());
    gradient_array(&l_output[0], l.outputs*l.batch, l.activation, &l_delta[0]);
    axpy_cpu(l.outputs*l.batch, l.alpha, &l_delta[0], 1, &net_delta[0], 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, &l_delta[0], l.w, l.h, l.c, 1, l.beta, &index_delta[0]);
    
    l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
    net.layers[l.index].delta->setItemsInRange(0, net.layers[l.index].delta->getBufferSize(),index_delta);
}
#endif