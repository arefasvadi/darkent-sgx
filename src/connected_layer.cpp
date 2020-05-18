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
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam,PRNG& net_layer_rng_deriver)
{
    int i;
    layer l = {};
    l.learning_rate_scale = 1;
    l.layer_rng = std::make_shared<PRNG>(generate_random_seed_from(net_layer_rng_deriver));
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
    l.nbiases = outputs;

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
        // l.weights[i] = scale*rand_uniform(-1, 1);
        l.weights[i] = rand_uniform(*(l.layer_rng),-1, 1);
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
#if defined(SGX_VERIFIES) && defined(GPU)
    l.forward_gpu_sgx_verifies = forward_connected_gpu_sgx_verifies_; 
    l.backward_gpu_sgx_verifies = backward_connected_gpu_sgx_verifies_;
    l.update_gpu_sgx_verifies = update_connected_gpu_sgx_verifies_;
    l.create_snapshot_for_sgx = create_connected_snapshot_for_sgx_;
#endif 

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
    float clip = a.grad_clip;
    // this is where it should clip the weight updates!
    if (clip != 0) {
       constrain_cpu(l.outputs,clip,l.bias_updates,1);    
    }
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        if (clip != 0) {
            constrain_cpu(l.outputs,clip,l.scale_updates,1);    
        }
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }
    // this is where it should clip the weight updates!
    if (clip != 0) {
        constrain_cpu(l.inputs*l.outputs,clip,l.weight_updates,1);    
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
        cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.outputs);
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
        cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.outputs);
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
    float clip = a.grad_clip;

    // if(a.adam){
    if(0){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        if (clip != 0) {
            constrain_gpu(l.outputs,clip,l.bias_updates_gpu,1);    
        }
        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if(l.batch_normalize){
            if (clip != 0) {
                constrain_gpu(l.outputs,clip,l.scale_updates_gpu,1);    
            }
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }
        if (clip != 0) {
            constrain_gpu(l.inputs*l.outputs,clip,l.weight_updates_gpu,1);    
        }
        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}

void forward_connected_layer_gpu(layer l, network net)
{   
    // cuda_pull_array(net.input_gpu,net.input,l.inputs*l.batch);
    // print_array(net.input,100,0,"GPU before connected forward input");
    // cuda_pull_array(l.weights_gpu,l.weights,l.nweights);
    // print_array(l.weights,l.nweights,0,"GPU before connected forward weights");
    // cuda_pull_array(l.biases_gpu,l.biases,l.outputs*l.batch);
    // print_array(l.biases,l.nbiases,0,"GPU before connected forward input");
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

    // cuda_pull_array(l.output_gpu,l.output,l.outputs*l.batch);
    // print_array(l.output,100,0,"GPU connected forward input before bias or batchnorm");
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_gpu(layer l, network net)
{   
    // commented to make things alike
    // constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    
    // cuda_pull_array(l.delta_gpu, l.delta, l.outputs*l.batch);
    // print_array(l.delta, l.outputs*l.batch/10, 0, "before connected layer delta");
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    // cuda_pull_array(l.delta_gpu, l.delta, l.outputs*l.batch);
    // print_array(l.delta, l.outputs*l.batch/10, 0, "after connected layer delta");
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }
    // cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.nbiases);
    // print_array(l.bias_updates, l.nbiases, 0, "GPU connected layer bias updates");

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
    // cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    // print_array(l.weight_updates, 100,0, "before connected layer weight updates");
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);
    // cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    // print_array(l.weight_updates, l.nweights, 0, "GPU after connected layer weight updates");

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
    // if (c) {
    //    cuda_pull_array(net.delta_gpu, net.delta, l.batch*l.inputs);
    //    print_array(net.delta, l.batch*l.inputs, 0, "GPU after connected layer net delta"); 
    // }
}
#endif

#if defined(SGX_VERIFIES) && defined(GPU)
    
void forward_connected_layer_gpu_frbmmv(layer l, network net) {
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    // store the output of MM
    if (train_iterations_snapshots_frbmmv.step_net_reports.count(gpu_iteration) == 0) {
        train_iterations_snapshots_frbmmv.step_net_reports[gpu_iteration] = std::move(network_batch_step_snapshot_frbmmv_t());
    }
    auto& net_report = train_iterations_snapshots_frbmmv.step_net_reports[gpu_iteration];
    if (net_report.net_layers_reports.count(net.index) == 0) {
        net_report.net_layers_reports[net.index] = std::move(layer_batch_step_snapshot_frbmmv_t());
        net_report.net_layers_reports[net.index].layer_forward_MM_outputs = std::vector<uint8_t>();
        net_report.net_layers_reports[net.index].layer_backward_MM_prev_delta = std::vector<uint8_t>();
    }
    auto& layer_report = net_report.net_layers_reports[net.index];
    auto curr_out_size = layer_report.layer_forward_MM_outputs.size();
    layer_report.layer_forward_MM_outputs.resize(curr_out_size+(l.outputs*l.batch*sizeof(float)));
    cuda_pull_array(l.output_gpu, (float*)(layer_report.layer_forward_MM_outputs.data()+curr_out_size),l.outputs*l.batch);
    // if (net.index == 13) {
    //     cuda_pull_array(l.output_gpu, l.output,l.outputs*l.batch);
    //     print_array(l.output,l.batch*l.outputs,0,"GPU FRBMMV connected forward input before bias or batchnorm");
    // }
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    // if (net.index == 13) {
    //     cuda_pull_array(l.output_gpu, l.output,l.outputs*l.batch);
    //     print_array(l.output,l.batch*l.outputs,0,"GPU FRBMMV connected forward input after bias or batchnorm before activation");
    // }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_gpu_frbmmv(layer l, network net) {
    // commented to make things alike
    // constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    // if (net.index == 13) {
    //     cuda_pull_array(l.delta_gpu, l.delta, l.outputs*l.batch);
    //     print_array(l.delta, l.outputs*l.batch/10, 0, "before connected layer delta");
    // }
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

    if(c) {
        gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        // store the output of MM
        if (train_iterations_snapshots_frbmmv.step_net_reports.count(gpu_iteration) == 0) {
            train_iterations_snapshots_frbmmv.step_net_reports[gpu_iteration] = std::move(network_batch_step_snapshot_frbmmv_t());
        }
        auto& net_report = train_iterations_snapshots_frbmmv.step_net_reports[gpu_iteration];
        if (net_report.net_layers_reports.count(net.index) == 0) {
            net_report.net_layers_reports[net.index] = std::move(layer_batch_step_snapshot_frbmmv_t());
            net_report.net_layers_reports[net.index].layer_backward_MM_prev_delta = std::vector<uint8_t>();
        }
        auto& layer_report = net_report.net_layers_reports[net.index];
        auto curr_out_size = layer_report.layer_backward_MM_prev_delta.size();
        layer_report.layer_backward_MM_prev_delta.resize(curr_out_size+(l.inputs*l.batch*sizeof(float)));
        cuda_pull_array(net.delta_gpu, (float*)(layer_report.layer_backward_MM_prev_delta.data()+curr_out_size),l.inputs*l.batch);
    }
}

void forward_connected_gpu_sgx_verifies_     (struct layer l, struct network net) {
    if (*main_verf_task_variation_ == verf_variations_t::FRBV) {
        forward_connected_layer_gpu(l,net);
        return;
    }
    forward_connected_layer_gpu_frbmmv(l, net);
    return;
}
void backward_connected_gpu_sgx_verifies_(struct layer l, struct network net) {
    if (*main_verf_task_variation_ == verf_variations_t::FRBV) {
        backward_connected_layer_gpu(l,net);
        return;
    }
    backward_connected_layer_gpu_frbmmv(l,net);
    return;
}
void update_connected_gpu_sgx_verifies_(struct layer l, update_args a) {
    update_connected_layer_gpu(l,a);
    return;
}

// takes overall weight updates,bias updates
// in case of BN, also take rolling mean,var and scales
void create_connected_snapshot_for_sgx_(struct layer& l, struct network& net, uint8_t** out, uint8_t** sha256_out) {
    if (gpu_index >= 0) {
        pull_connected_layer(l);
    }
    size_t total_bytes = count_layer_paramas_bytes(l);
    size_t buff_ind = 0;
    *out = new uint8_t[total_bytes];
    *sha256_out = new uint8_t[SHA256_DIGEST_LENGTH];
    
    std::memcpy((*out+buff_ind),l.bias_updates,l.nbiases*sizeof(float));
    buff_ind += l.nbiases*sizeof(float);
    std::memcpy((*out+buff_ind),l.weight_updates,l.nweights*sizeof(float));
    buff_ind += l.nweights*sizeof(float);
    
    if (l.batch_normalize) {
        std::memcpy((*out+buff_ind),l.scale_updates,l.nbiases*sizeof(float));
        buff_ind += l.nbiases*sizeof(float);
        std::memcpy((*out+buff_ind),l.rolling_mean,l.nbiases*sizeof(float));
        buff_ind += l.nbiases*sizeof(float);
        std::memcpy((*out+buff_ind),l.rolling_variance,l.nbiases*sizeof(float));
        buff_ind += l.nbiases*sizeof(float);
    }
    if (buff_ind != total_bytes) {
            LOG_ERROR("size mismatch\n")
            abort();
    }
    gen_sha256(*out,total_bytes,*sha256_out);
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