#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_SGX_LAYERWISE
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = (float*)calloc(batch*inputs, sizeof(float));
    if (global_training) l.delta = (float*)calloc(batch*inputs, sizeof(float));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %s %d inputs\n", get_activation_string(activation),inputs);
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

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    //l.output = (float*)calloc(batch*inputs, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(batch*inputs);
    //l.delta = (float*)calloc(batch*inputs, sizeof(float));
    l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(batch*inputs);

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.activation = activation;
    //fprintf(stderr, "Activation Layer: %s %d inputs\n", get_activation_string(activation),inputs);
    return l;
}

void forward_activation_layer(layer l, network net)
{
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    //float* net_input_ptr = &net_input[0];
    copy_cpu(l.outputs*l.batch, &net_input[0], 1, &l_output[0], 1);
    activate_array(&l_output[0], l.outputs*l.batch, l.activation);
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);

}

void backward_activation_layer(layer l, network net)
{
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    gradient_array(&l_output[0], l.outputs*l.batch, l.activation, &l_delta[0]);
    copy_cpu(l.outputs*l.batch, &l_delta[0], 1, &net_delta[0], 1);
    l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
}
#endif

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
    // ffprintf(stderr, "Activation Layer: %s %d inputs\n", get_activation_string(activation),inputs);
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