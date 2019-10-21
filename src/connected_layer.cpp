#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef USE_SGX_LAYERWISE
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.nweights = outputs*inputs;

    l.output = (float*)calloc(batch*outputs, sizeof(float));
    if (global_training) l.delta = (float*)calloc(batch*outputs, sizeof(float));

    if (global_training) l.weight_updates = (float*)calloc(inputs*outputs, sizeof(float));
    if (global_training) l.bias_updates = (float*)calloc(outputs, sizeof(float));

    l.weights = (float*)calloc(outputs*inputs, sizeof(float));
    l.biases = (float*)calloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }
    if (global_training) {
        if(adam){
            l.m = (float*)calloc(l.inputs*l.outputs, sizeof(float));
            l.v = (float*)calloc(l.inputs*l.outputs, sizeof(float));
            l.bias_m = (float*)calloc(l.outputs, sizeof(float));
            l.scale_m = (float*)calloc(l.outputs, sizeof(float));
            l.bias_v = (float*)calloc(l.outputs, sizeof(float));
            l.scale_v = (float*)calloc(l.outputs, sizeof(float));
        }
    }
    
    if(batch_normalize){
        l.scales = (float*)calloc(outputs, sizeof(float));
        if (global_training) l.scale_updates = (float*)calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        if (global_training) {
            l.mean = (float*)calloc(outputs, sizeof(float));
            l.mean_delta = (float*)calloc(outputs, sizeof(float));
            l.variance = (float*)calloc(outputs, sizeof(float));
            l.variance_delta = (float*)calloc(outputs, sizeof(float));
        }
        

        l.rolling_mean = (float*)calloc(outputs, sizeof(float));
        l.rolling_variance = (float*)calloc(outputs, sizeof(float));

        if (global_training) l.x = (float*)calloc(batch*outputs, sizeof(float));
        if (global_training) l.x_norm = (float*)calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}
#endif

#ifndef USE_SGX_LAYERWISE
void update_connected_layer(layer& l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}
#endif

#ifndef USE_SGX_LAYERWISE
void forward_connected_layer(layer& l, network& net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_connected_layer(layer& l, network& net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
#endif
#ifndef USE_SGX_LAYERWISE
void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j){
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}
#endif


void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if(l.batch_normalize){
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }

        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}

void forward_connected_layer_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_gpu(layer l, network net)
{
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i,j;
    layer l = {};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.nweights = outputs*inputs;

    //l.output = (float*)calloc(batch*outputs, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(batch*outputs);
    //l.delta = (float*)calloc(batch*outputs, sizeof(float));
    if (global_training) l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(batch*outputs);

    //l.weight_updates = (float*)calloc(inputs*outputs, sizeof(float));
    if (global_training) l.weight_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*outputs);
    //l.bias_updates = (float*)calloc(outputs, sizeof(float));
    if (global_training) l.bias_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);

    //l.weights = (float*)calloc(outputs*inputs, sizeof(float));
    l.weights = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs*inputs);
    //l.biases = (float*)calloc(outputs, sizeof(float));
    l.biases = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    
    for(i = 0; i < outputs; ++i){
        auto l_weights = l.weights->getItemsInRange(i*inputs,(i+1)*inputs);
        for (j=0;j<inputs;++j) {
            l_weights[j] = scale*rand_uniform(-1, 1);
        }
        l.weights->setItemsInRange(i*inputs,(i+1)*inputs,l_weights);
    }
    
    // already initialized to zero
    {
        auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
        for(i = 0; i < outputs; ++i){
            //l.biases[i] = 0;
            l_biases[i] = 0;
        }
        l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
    }
    if (global_training) {
        if(adam){
            //l.m = (float*)calloc(l.inputs*l.outputs, sizeof(float));
            l.m = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.inputs*l.outputs);
            //l.v = (float*)calloc(l.inputs*l.outputs, sizeof(float));
            l.v = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.inputs*l.outputs);
            //l.bias_m = (float*)calloc(l.outputs, sizeof(float));
            l.bias_m = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.outputs);
            //l.scale_m = (float*)calloc(l.outputs, sizeof(float));
            l.scale_m = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.outputs);
            //l.bias_v = (float*)calloc(l.outputs, sizeof(float));
            l.bias_v = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.outputs);
            //l.scale_v = (float*)calloc(l.outputs, sizeof(float));
            l.scale_v = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.outputs);
        }
    }
    if(batch_normalize){
        //l.scales = (float*)calloc(outputs, sizeof(float));
        l.scales = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);
        //l.scale_updates = (float*)calloc(outputs, sizeof(float));
        if (global_training) l.scale_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);
        
        if (global_training) {
            auto scs = l.scales->getItemsInRange(0, l.scales->getBufferSize());
            for(i = 0; i < outputs; ++i){
                scs[i] = 1;
            }
            l.scales->setItemsInRange(0, l.scales->getBufferSize(), scs);
        }

        if (global_training) {
            //l.mean = (float*)calloc(outputs, sizeof(float));
            l.mean = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);
            //l.mean_delta = (float*)calloc(outputs, sizeof(float));
            l.mean_delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);
            //l.variance = (float*)calloc(outputs, sizeof(float));
            l.variance = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);
            //l.variance_delta = (float*)calloc(outputs, sizeof(float));
            l.variance_delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);
            //l.x = (float*)calloc(batch*outputs, sizeof(float));
            l.x = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(batch*outputs);
            //l.x_norm = (float*)calloc(batch*outputs, sizeof(float));
            l.x_norm = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(batch*outputs);
        }
        //l.rolling_mean = (float*)calloc(outputs, sizeof(float));
        l.rolling_mean = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);
        //l.rolling_variance = (float*)calloc(outputs, sizeof(float));
        l.rolling_variance = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(outputs);
        
    }
    l.activation = activation;

    uint64_t consumed_space_wo_weights_bytes = (l.inputs + l.outputs + l.nbiases)*sizeof(float);
    l.enclave_layered_batch = (SGX_LAYERWISE_MAX_LAYER_SIZE - (consumed_space_wo_weights_bytes));
    if (l.enclave_layered_batch <=0) {
        LOG_ERROR("remaining space is negative!!!!\n");
        abort();
    }
    l.enclave_layered_batch = 
        (l.enclave_layered_batch / (sizeof(float)*l.nweights / l.outputs));

    if (l.enclave_layered_batch <= 0) {
        LOG_ERROR("remaining space is not enough for even a single batch!!!!\n");
        abort();
    }
    if (l.enclave_layered_batch > l.outputs) {
        l.enclave_layered_batch = l.outputs;
    }
    //fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void update_connected_layer(layer& l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    int enclave_update_batch = l.enclave_layered_batch / 2;
    int q = l.outputs / enclave_update_batch;
    int r = l.outputs % enclave_update_batch;
    {
        auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
        auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        axpy_cpu(l.outputs, learning_rate/batch, &l_bias_updates[0], 1, &l_biases[0], 1);
        scal_cpu(l.outputs, momentum, &l_bias_updates[0], 1);
        l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
        l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(),l_bias_updates);
    }

    if(l.batch_normalize){
        auto l_scale_updates = l.scale_updates->getItemsInRange(0, l.scale_updates->getBufferSize());
        auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
        axpy_cpu(l.outputs, learning_rate/batch, &l_scale_updates[0], 1, &l_scales[0], 1);
        scal_cpu(l.outputs, momentum, &l_scale_updates[0], 1);
        l.scale_updates->setItemsInRange(0, l.scale_updates->getBufferSize(),l_scale_updates);
        l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
    }

    {
        for (int i=0;i<q;++i) {
            auto l_weights = l.weights->getItemsInRange(i*enclave_update_batch*l.inputs, (i+1)*enclave_update_batch*l.inputs);
            auto l_weight_updates = l.weight_updates->getItemsInRange(i*enclave_update_batch*l.inputs, (i+1)*enclave_update_batch*l.inputs);
            axpy_cpu(l.inputs, -decay*batch, &l_weights[0], 1, &l_weight_updates[0], 1);
            axpy_cpu(l.inputs, learning_rate/batch, &l_weight_updates[0], 1, &l_weights[0], 1);
            scal_cpu(l.inputs, momentum, &l_weight_updates[0], 1);
            l.weights->setItemsInRange(i*enclave_update_batch*l.inputs, (i+1)*enclave_update_batch*l.inputs,l_weights);
            l.weight_updates->setItemsInRange(i*enclave_update_batch*l.inputs, (i+1)*enclave_update_batch*l.inputs,l_weight_updates);
        }
        if (r > 0) {
            auto l_weights = l.weights->getItemsInRange(q*enclave_update_batch*l.inputs, q*enclave_update_batch*l.inputs + r*l.inputs);
            auto l_weight_updates = l.weight_updates->getItemsInRange(q*enclave_update_batch*l.inputs, q*enclave_update_batch*l.inputs+r*l.inputs);
            axpy_cpu(l.inputs, -decay*batch, &l_weights[0], 1, &l_weight_updates[0], 1);
            axpy_cpu(l.inputs, learning_rate/batch, &l_weight_updates[0], 1, &l_weights[0], 1);
            scal_cpu(l.inputs, momentum, &l_weight_updates[0], 1);
            l.weights->setItemsInRange(q*enclave_update_batch*l.inputs, q*enclave_update_batch*l.inputs + r*l.inputs,l_weights);
            l.weight_updates->setItemsInRange(q*enclave_update_batch*l.inputs, q*enclave_update_batch*l.inputs+r*l.inputs,l_weight_updates);
        }
        //axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
        //axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
        //scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
    }
}

void forward_connected_layer(layer& l, network& net)
{
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    fill_cpu(l.outputs*l.batch, 0, &l_output[0], 1);
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    
    int q = l.outputs / l.enclave_layered_batch;
    int r = l.outputs % l.enclave_layered_batch;

    {
        std::vector<float> temp_out(l.batch*l.enclave_layered_batch,0.0);
        float *a = &net_input[0];
        for (int i=0;i<q;++i) {
            //std::vector<float> temp_out;
            for (int j=0;j<l.batch;++j) {
                //temp_out.push_back(l_output[j*l.outputs + i]);
                for (int p=0;p<l.enclave_layered_batch;++p) {
                    temp_out[j*l.enclave_layered_batch+p] = 
                        l_output[j*l.outputs + i*l.enclave_layered_batch + p];
                }
            }
            float *c = &temp_out[0];
            auto l_weights = l.weights->getItemsInRange(i*l.enclave_layered_batch*l.inputs,(i+1)*l.enclave_layered_batch*l.inputs); 
            float *b = &l_weights[0];        
            gemm(0,1,m,l.enclave_layered_batch,k,1,a,k,b,k,1,c,l.enclave_layered_batch);
            for (int j=0;j<l.batch;++j) {
                //l_output[j*l.outputs + i] = temp_out[j];
                for (int p=0;p<l.enclave_layered_batch;++p) {
                    l_output[j*l.outputs + i*l.enclave_layered_batch + p] = 
                        temp_out[j*l.enclave_layered_batch+p] ;
                }
            }
            //l.weights->setItemsInRange(i*l.inputs,(i+1)*l.inputs,l_weights);
        }
        if (r > 0) {
            for (int j=0;j<l.batch;++j) {
                //temp_out.push_back(l_output[j*l.outputs + i]);
                for (int p=0;p<r;++p) {
                    temp_out[j*r+p] = 
                        l_output[j*l.outputs + q*l.enclave_layered_batch + p];
                }
            }
            float *c = &temp_out[0];
            auto l_weights = l.weights->getItemsInRange(q*l.enclave_layered_batch*l.inputs,q*l.enclave_layered_batch*l.inputs+r*l.inputs); 
            float *b = &l_weights[0];        
            gemm(0,1,m,r,k,1,a,k,b,k,1,c,r);
            for (int j=0;j<l.batch;++j) {
                for (int p=0;p<r;++p) {
                    l_output[j*l.outputs + q*l.enclave_layered_batch + p] = 
                        temp_out[j*r+p];
                }
            }
        }
        //gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    }
    
    if(l.batch_normalize){
        l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
        forward_batchnorm_layer(l, net);
        l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    } else {
        auto l_biases = l.biases->getItemsInRange(0,l.biases->getBufferSize()); 
        add_bias(&l_output[0], &l_biases[0], l.batch, l.outputs, 1);
    }
    activate_array(&l_output[0], l.outputs*l.batch, l.activation);
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
}

void backward_connected_layer(layer& l, network& net)
{
    
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    {
        auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
        gradient_array(&l_output[0], l.outputs*l.batch, l.activation, &l_delta[0]);
    }

    if(l.batch_normalize){
        l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
        backward_batchnorm_layer(l, net);
        l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    } else {
        auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        backward_bias(&l_bias_updates[0], &l_delta[0], l.batch, l.outputs, 1);
        l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(), l_bias_updates);
    }

    int q = l.outputs / l.enclave_layered_batch;
    int r = l.outputs % l.enclave_layered_batch;

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    float *b = &net_input[0];
    {
        std::vector<float> temp_delta(l.batch*l.enclave_layered_batch,0.0);
        for (int i=0;i<q;++i) {
            for (int j=0;j<l.batch;++j) {
                for (int p=0;p<l.enclave_layered_batch;++p) {
                    temp_delta[j*l.enclave_layered_batch+p] = 
                        l_delta[j*l.outputs + i*l.enclave_layered_batch + p];
                }
            }
            float *a = &temp_delta[0];
            auto l_weight_updates = l.weight_updates->getItemsInRange(i*l.enclave_layered_batch*n, (i+1)*l.enclave_layered_batch*n);
            float *c = &l_weight_updates[0];
            gemm(1,0,l.enclave_layered_batch,n,k,1,a,l.enclave_layered_batch,b,n,1,c,n);
            l.weight_updates->setItemsInRange(i*l.enclave_layered_batch*n, (i+1)*l.enclave_layered_batch*n,l_weight_updates);
        }
        if (r > 0) {
            for (int j=0;j<l.batch;++j) {
                for (int p=0;p<r;++p) {
                    temp_delta[j*r+p] = 
                        l_delta[j*l.outputs + q*l.enclave_layered_batch + p];
                }
            }
            float *a = &temp_delta[0];
            auto l_weight_updates = l.weight_updates->getItemsInRange(q*l.enclave_layered_batch*n, q*l.enclave_layered_batch*n + r*n);
            float *c = &l_weight_updates[0];
            gemm(1,0,r,n,k,1,a,r,b,n,1,c,n);
            l.weight_updates->setItemsInRange(q*l.enclave_layered_batch*n, q*l.enclave_layered_batch*n+r*n,l_weight_updates);
        }
        //gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);
    }
    

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    if(net.delta) {
        auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
        std::vector<float> temp_delta(l.batch*l.enclave_layered_batch,0.0);
        for (int i=0;i<q;++i) {
            //std::vector<float> temp_delta;
            for (int j=0;j<l.batch;++j) {
                for (int p=0;p<l.enclave_layered_batch;++p) {
                    temp_delta[j*l.enclave_layered_batch+p] = 
                        l_delta[j*l.outputs + i*l.enclave_layered_batch + p];
                }  
            }
            float *a = &temp_delta[0];
            auto l_weights = l.weights->getItemsInRange(i*l.enclave_layered_batch*(l.inputs), (i+1)*l.enclave_layered_batch*l.inputs);
            b = &l_weights[0];
            float* c = &net_delta[0];
            gemm(0,0,m,n,l.enclave_layered_batch,1,a,l.enclave_layered_batch,b,n,1,c,n);
        }
        if (r > 0) {
            for (int j=0;j<l.batch;++j) {
                for (int p=0;p<r;++p) {
                    temp_delta[j*r+p] = 
                        l_delta[j*l.outputs + q*l.enclave_layered_batch + p];
                }  
            }
            float *a = &temp_delta[0];
            auto l_weights = l.weights->getItemsInRange(q*l.enclave_layered_batch*(l.inputs), q*l.enclave_layered_batch*l.inputs+r*l.inputs);
            b = &l_weights[0];
            float* c = &net_delta[0];
            gemm(0,0,m,n,r,1,a,r,b,n,1,c,n);
        }
        //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
    }
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
layer_blocked make_connected_layer_blocked(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam) {
    int i;
    layer_blocked l = {};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.nweights = outputs*inputs;

    //l.output = (float*)calloc(batch*outputs, sizeof(float));
    l.output = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({batch*outputs});
    //l.delta = (float*)calloc(batch*outputs, sizeof(float));
    l.delta = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({batch*outputs});

    //l.weight_updates = (float*)calloc(inputs*outputs, sizeof(float));
    l.weight_updates = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({inputs*outputs});
    //l.bias_updates = (float*)calloc(outputs, sizeof(float));
    l.bias_updates = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});

    //l.weights = (float*)calloc(outputs*inputs, sizeof(float));
    l.weights = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs*inputs});
    //l.biases = (float*)calloc(outputs, sizeof(float));
    l.biases = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});

    l.forward_blocked = forward_connected_layer_blocked;
    l.backward_blocked = backward_connected_layer_blocked;
    l.update_blocked = update_connected_layer_blocked;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    BLOCK_ENGINE_INIT_FOR_LOOP(l.weights, weights_valid_range, weights_block_val_ptr, float)
    for(i = 0; i < outputs*inputs; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weights_valid_range, weights_block_val_ptr, true, weights_index, i)
        *(weights_block_val_ptr+weights_index-weights_valid_range.block_requested_ind) = scale*rand_uniform(-1, 1);
        //l.weights[i] = scale*rand_uniform(-1, 1);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weights, weights_valid_range)

    BLOCK_ENGINE_INIT_FOR_LOOP(l.biases, biases_valid_range, biases_block_val_ptr, float)
    for(i = 0; i < outputs; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.biases, biases_valid_range, biases_block_val_ptr, true, biases_index, i)
        *(biases_block_val_ptr+biases_index-biases_valid_range.block_requested_ind) = 0.0;
        //l.biases[i] = 0;
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.biases, biases_valid_range)

    if(adam){
        //l.m = (float*)calloc(l.inputs*l.outputs, sizeof(float));
        l.m = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({l.inputs*l.outputs});
        //l.v = (float*)calloc(l.inputs*l.outputs, sizeof(float));
        l.v = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({l.inputs*l.outputs});
        //l.bias_m = (float*)calloc(l.outputs, sizeof(float));
        l.bias_m = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({l.outputs});
        //l.scale_m = (float*)calloc(l.outputs, sizeof(float));
        l.scale_m = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({l.outputs});
        //l.bias_v = (float*)calloc(l.outputs, sizeof(float));
        l.bias_v = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({l.outputs});
        //l.scale_v = (float*)calloc(l.outputs, sizeof(float));
        l.scale_v = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({l.outputs});
    }
    if(batch_normalize){
        //l.scales = (float*)calloc(outputs, sizeof(float));
        l.scales = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});
        //l.scale_updates = (float*)calloc(outputs, sizeof(float));
        l.scale_updates = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});
        BLOCK_ENGINE_INIT_FOR_LOOP(l.scales, scales_valid_range, scales_block_val_ptr, float)
        for(i = 0; i < outputs; ++i){
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.scales, scales_valid_range, scales_block_val_ptr, true, scales_index_var, i)
            *(scales_block_val_ptr+scales_index_var-scales_valid_range.block_requested_ind) = 1.0;
            //l.scales[i] = 1;
        }
        BLOCK_ENGINE_LAST_UNLOCK(l.scales, scales_valid_range)

        //l.mean = (float*)calloc(outputs, sizeof(float));
        l.mean = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});
        //l.mean_delta = (float*)calloc(outputs, sizeof(float));
        l.mean_delta = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});
        //l.variance = (float*)calloc(outputs, sizeof(float));
        l.variance = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});
        //l.variance_delta = (float*)calloc(outputs, sizeof(float));
        l.variance_delta = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});

        //l.rolling_mean = (float*)calloc(outputs, sizeof(float));
        l.rolling_mean = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});
        //l.rolling_variance = (float*)calloc(outputs, sizeof(float));
        l.rolling_variance = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({outputs});

        //l.x = (float*)calloc(batch*outputs, sizeof(float));
        l.x = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({batch*outputs});
        //l.x_norm = (float*)calloc(batch*outputs, sizeof(float));
        l.x_norm = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({batch*outputs});
    }
    l.activation = activation;
    //fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}
void update_connected_layer_blocked(layer_blocked l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu_blocked(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu_blocked(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu_blocked(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu_blocked(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu_blocked(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu_blocked(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu_blocked(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

void forward_connected_layer_blocked(layer_blocked l, network_blocked net)
{
    fill_cpu_blocked(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    //float *a = net.input;
    //float *b = l.weights;
    //float *c = l.output;
    gemm_blocked(0,1,m,n,k,1,net.input,0,k,l.weights,0,k,1,l.output,0,n);
    if(l.batch_normalize){
        forward_batchnorm_layer_blocked(l, net);
    } else {
        add_bias_blocked(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array_blocked(l.output, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_blocked(layer_blocked l, network_blocked net)
{
    gradient_array_blocked(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer_blocked(l, net);
    } else {
        backward_bias_blocked(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    //float *a = l.delta;
    //float *b = net.input;
    //float *c = l.weight_updates;
    gemm_blocked(1,0,m,n,k,1,l.delta,0,m,net.input,0,n,1,l.weight_updates,0,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    //a = l.delta;
    //b = l.weights;
    //c = net.delta;

    if(net.delta) gemm_blocked(0,0,m,n,k,1,l.delta,0,k,l.weights,0,n,1,net.delta,0,n);
}
#endif