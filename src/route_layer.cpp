#include "route_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

#ifndef USE_SGX_LAYERWISE
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    fprintf(stderr,"route ");
    route_layer l = {};
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    l.delta =  (float*)calloc(outputs*batch, sizeof(float));
    l.output = (float*)calloc(outputs*batch, sizeof(float));;

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
    #ifdef GPU
    l.forward_gpu = forward_route_layer_gpu;
    l.backward_gpu = backward_route_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    return l;
}
#endif

#ifndef USE_SGX_LAYERWISE
void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
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
void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#endif

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_gpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}

void backward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#endif
#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    fprintf(stderr,"route ");
    route_layer l = {};
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    //l.delta =  (float*)calloc(outputs*batch, sizeof(float));
    l.delta =  sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs*batch);
    //l.output = (float*)calloc(outputs*batch, sizeof(float));;
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs*batch);

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
    return l;
}

void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        //float *input = net.layers[index].output;
        auto input = net.layers[index].output->getItemsInRange(0,net.layers[index].output->getBufferSize());
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_cpu(input_size, &input[0] + j*input_size, 1, &l_output[0] + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
}

void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        //float *delta = net.layers[index].delta;
        auto delta = net.layers[index].delta->getItemsInRange(0, net.layers[index].delta->getBufferSize());
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_cpu(input_size, 1, &l_delta[0] + offset + j*l.outputs, 1, &delta[0] + j*input_size, 1);
        }
        offset += input_size;
        net.layers[index].delta->setItemsInRange(0, net.layers[index].delta->getBufferSize(),delta);
    }
}
#endif