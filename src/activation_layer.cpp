#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = (float*)calloc(batch*inputs, sizeof(float));
    l.delta = (float*)calloc(batch*inputs, sizeof(float));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
layer_blocked make_activation_layer_blocked(int batch, int inputs, ACTIVATION activation) {
    layer_blocked l = {};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    // l.output = (float*)calloc(batch*inputs, sizeof(float));
    l.output = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({batch*inputs});
    // l.delta = (float*)calloc(batch*inputs, sizeof(float));
    l.delta = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({batch*inputs});

    l.forward_blocked = forward_activation_layer_blocked;
    l.backward_blocked = backward_activation_layer_blocked;

    l.activation = activation;
    // fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer_blocked(layer_blocked l, network_blocked net) {
    // copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu_blocked(l.outputs*l.batch, net.input, 1, l.output, 1);
    // activate_array(l.output, l.outputs*l.batch, l.activation);
    activate_array_blocked(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_blocked(layer_blocked l, network_blocked net){
    // gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    gradient_array_blocked(l.output, l.outputs*l.batch, l.activation, l.delta);
    // copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
    copy_cpu_blocked(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#endif
#ifdef GPU

void forward_activation_layer_gpu(layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
