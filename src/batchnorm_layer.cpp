#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>

#ifndef USE_SGX_LAYERWISE
layer make_batchnorm_layer(int batch, int w, int h, int c)
{
#ifndef USE_SGX
  fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
#else
#endif
    layer l = {};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = (float*)calloc(h * w * c * batch, sizeof(float));
    l.delta  = (float*)calloc(h * w * c * batch, sizeof(float));
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = (float*)calloc(c, sizeof(float));
    l.scale_updates = (float*)calloc(c, sizeof(float));
    l.biases = (float*)calloc(c, sizeof(float));
    l.bias_updates = (float*)calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        l.scales[i] = 1;
    }

    l.mean = (float*)calloc(c, sizeof(float));
    l.variance = (float*)calloc(c, sizeof(float));

    l.rolling_mean = (float*)calloc(c, sizeof(float));
    l.rolling_variance = (float*)calloc(c, sizeof(float));

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;

#if defined(SGX_VERIFIES) && defined(GPU)
    l.forward_gpu_sgx_verifies = forward_batchnorm_gpu_sgx_verifies_fbv; 
    l.backward_gpu_sgx_verifies = backward_batchnorm_gpu_sgx_verifies_fbv;
    l.create_snapshot_for_sgx = create_batchnorm_snapshot_for_sgx_fbv;
#endif 

#ifdef GPU
    l.forward_gpu = forward_batchnorm_layer_gpu;
    l.backward_gpu = backward_batchnorm_layer_gpu;

    l.output_gpu =  cuda_make_array(l.output, h * w * c * batch);
    l.delta_gpu =   cuda_make_array(l.delta, h * w * c * batch);

    l.biases_gpu = cuda_make_array(l.biases, c);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

    l.scales_gpu = cuda_make_array(l.scales, c);
    l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

    l.mean_gpu = cuda_make_array(l.mean, c);
    l.variance_gpu = cuda_make_array(l.variance, c);

    l.rolling_mean_gpu = cuda_make_array(l.mean, c);
    l.rolling_variance_gpu = cuda_make_array(l.variance, c);

    l.mean_delta_gpu = cuda_make_array(l.mean, c);
    l.variance_delta_gpu = cuda_make_array(l.variance, c);

    l.x_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    #ifdef CUDNN
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
    cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 

    #endif
#endif
    return l;
}
#endif

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./std::sqrt(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * std::pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(std::sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

#ifndef USE_SGX
void resize_batchnorm_layer(layer *layer, int w, int h)
{
  fprintf(stderr, "Not implemented\n");
}
#else
#endif

#ifndef USE_SGX_LAYERWISE
void forward_batchnorm_layer(layer& l, network& net)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    if(net.train){
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_batchnorm_layer(layer& l, network& net)
{
    if(!net.train){
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
layer make_batchnorm_layer(int batch, int w, int h, int c)
{
#ifndef USE_SGX
  fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
#else
#endif
    layer l = {};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    //l.output = (float*)calloc(h * w * c * batch, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(h * w * c * batch);
    //l.delta  = (float*)calloc(h * w * c * batch, sizeof(float));
    l.delta  = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(h * w * c * batch);
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    //l.scales = (float*)calloc(c, sizeof(float));
    l.scales = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c);
    //l.scale_updates = (float*)calloc(c, sizeof(float));
    l.scale_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c);
    //l.biases = (float*)calloc(c, sizeof(float));
    l.biases = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c);
    //l.bias_updates = (float*)calloc(c, sizeof(float));
    l.bias_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c);
    int i;
    {
        auto vec_sclaes = l.scales->getItemsInRange(0, c);
        for(i = 0; i < c; ++i){
            //l.scales[i] = 1;
            vec_sclaes[i] = 1;
        }
        l.scales->setItemsInRange(0, c, vec_sclaes);
    }
    
    //vec_sclaes.clear();

    //l.mean = (float*)calloc(c, sizeof(float));
    l.mean = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c);
    //l.variance = (float*)calloc(c, sizeof(float));
    l.variance = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c);

    //l.rolling_mean = (float*)calloc(c, sizeof(float));
    l.rolling_mean = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c);
    //l.rolling_variance = (float*)calloc(c, sizeof(float));
    l.rolling_variance = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c);

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;
    return l;
}

void forward_batchnorm_layer(layer& l, network& net)
{
    auto l_out = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto l_rolling_mean = l.rolling_mean->getItemsInRange(0,l.rolling_mean->getBufferSize());
    auto l_rolling_variance = l.rolling_variance->getItemsInRange(0,l.rolling_variance->getBufferSize());
    if(l.type == BATCHNORM) {
        auto net_inp = net.input->getItemsInRange(0, net.input->getBufferSize());
        copy_cpu(l.outputs*l.batch, &net_inp[0], 1, &l_out[0], 1);
    }
    {
        auto l_x = l.x->getItemsInRange(0, l.x->getBufferSize());
        copy_cpu(l.outputs*l.batch, &l_out[0], 1, &l_x[0], 1);
        l.x->setItemsInRange(0,l.x->getBufferSize() , l_x);
    }
    if(net.train){
        auto l_mean = l.mean->getItemsInRange(0,l.mean->getBufferSize());
        auto l_variance = l.variance->getItemsInRange(0,l.variance->getBufferSize());
        auto l_xnorm = l.x_norm->getItemsInRange(0, l.x_norm->getBufferSize());
        mean_cpu(&l_out[0], l.batch, l.out_c, l.out_h*l.out_w, &l_mean[0]);
        variance_cpu(&l_out[0], &l_mean[0], l.batch, l.out_c, l.out_h*l.out_w, &l_variance[0]);

        scal_cpu(l.out_c, .99, &l_rolling_mean[0], 1);
        axpy_cpu(l.out_c, .01, &l_mean[0], 1, &l_rolling_mean[0], 1);
        scal_cpu(l.out_c, .99, &l_rolling_variance[0], 1);
        axpy_cpu(l.out_c, .01, &l_variance[0], 1, &l_rolling_variance[0], 1);

        normalize_cpu(&l_out[0], &l_mean[0], &l_variance[0], l.batch, l.out_c, l.out_h*l.out_w);   
        copy_cpu(l.outputs*l.batch, &l_out[0], 1, &l_xnorm[0], 1);
        l.mean->setItemsInRange(0, l.mean->getBufferSize(), l_mean);
        l.variance->setItemsInRange(0, l.variance->getBufferSize(), l_variance);
        l.x_norm->setItemsInRange(0, l.x_norm->getBufferSize(), l_xnorm);
        l.rolling_mean->setItemsInRange(0, l.rolling_mean->getBufferSize(), l_rolling_mean);
        l.rolling_variance->setItemsInRange(0, l.rolling_variance->getBufferSize(), l_rolling_variance);
    } else {
        normalize_cpu(&l_out[0], &l_rolling_mean[0], &l_rolling_variance[0], l.batch, l.out_c, l.out_h*l.out_w);
    }
    auto l_scales = l.scales->getItemsInRange(0,l.scales->getBufferSize());
    auto l_biases = l.biases->getItemsInRange(0,l.biases->getBufferSize());
    scale_bias(&l_out[0], &l_scales[0], l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(&l_out[0], &l_biases[0], l.batch, l.out_c, l.out_h*l.out_w);
    l.output->setItemsInRange(0, l.output->getBufferSize(), l_out);
}

void backward_batchnorm_layer(layer& l, network& net)
{
    if(!net.train){
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }

    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());

   { 
        auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.biases->getBufferSize());
        // LOG_DEBUG("SGX batchnorm back batch:%d,out_c:%d,out_w:%d,out_h:%d\n", l.batch, l.out_c, l.out_w,l.out_h)
        // if (net.index == 7) {
        //     std::string text = std::string("SGX bactnorm delta after gradient on activation layer ") + std::to_string(net.index);
        //     print_array(&l_delta[(l.batch/2)*l.outputs],1*l.outputs,(l.batch/2)*l.outputs,text.c_str());
        //     print_array(&l_bias_updates[0], l.nbiases, 0, "SGX bactnorm forw bias updates before");
        // }
        backward_bias(&l_bias_updates[0], &l_delta[0], l.batch, l.out_c, l.out_w*l.out_h);
        // if (net.index == 7) {
        //     print_array(&l_bias_updates[0], l.nbiases, 0, "SGX bactnorm forw bias updates after");
        // }
        l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(), l_bias_updates);
    }
    {
        auto l_x_norm = l.x_norm->getItemsInRange(0, l.x_norm->getBufferSize());
        auto l_scale_updates = l.scale_updates->getItemsInRange(0, l.scale_updates->getBufferSize());
        // print_array(&l_scale_updates[0], l.nbiases, 0, "SGX: batchnorm scale updates before");
        backward_scale_cpu(&l_x_norm[0], &l_delta[0], l.batch, l.out_c, l.out_w*l.out_h, &l_scale_updates[0]);
        // print_array(&l_scale_updates[0], l.nbiases, 0, "SGX: batchnorm scale updates after");
        l.scale_updates->setItemsInRange(0, l.scale_updates->getBufferSize(), l_scale_updates);
    }

    {
        auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
        scale_bias(&l_delta[0], &l_scales[0], l.batch, l.out_c, l.out_h*l.out_w);
    }

    {
        auto l_mean_delta = l.mean_delta->getItemsInRange(0, l.mean_delta->getBufferSize());
        auto l_variance = l.variance->getItemsInRange(0, l.variance->getBufferSize());
        mean_delta_cpu(&l_delta[0], &l_variance[0], l.batch, l.out_c, l.out_w*l.out_h, &l_mean_delta[0]);

        auto l_x = l.x->getItemsInRange(0, l.x->getBufferSize());
        auto l_mean = l.mean->getItemsInRange(0, l.mean->getBufferSize());
        auto l_variance_delta = l.variance_delta->getItemsInRange(0, l.variance_delta->getBufferSize());
        variance_delta_cpu(&l_x[0], &l_delta[0], &l_mean[0], &l_variance[0], l.batch, l.out_c, l.out_w*l.out_h, &l_variance_delta[0]);
        normalize_delta_cpu(&l_x[0], &l_mean[0], &l_variance[0], &l_mean_delta[0], &l_variance_delta[0], l.batch, l.out_c, l.out_w*l.out_h, &l_delta[0]);

        l.mean_delta->setItemsInRange(0, l.mean_delta->getBufferSize(), l_mean_delta);
        l.variance_delta->setItemsInRange(0, l.variance_delta->getBufferSize(), l_variance_delta);

    }
    if(l.type == BATCHNORM) {
        auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
        copy_cpu(l.outputs*l.batch, &l_delta[0], 1, &net_delta[0], 1);
        net.delta->setItemsInRange(0, net.delta->getBufferSize(), net_delta);
    }
    l.delta->setItemsInRange(0, l.delta->getBufferSize(), l_delta);
}
#endif

#if defined(SGX_VERIFIES) && defined(GPU)
    void forward_batchnorm_gpu_sgx_verifies_fbv     (struct layer l, struct network net) {
        forward_batchnorm_layer_gpu(l,net);
    }
    void backward_batchnorm_gpu_sgx_verifies_fbv(struct layer l, struct network net) {
        backward_batchnorm_layer_gpu(l,net);
    }
    // void update_batchnorm_gpu_sgx_verifies_fbv(struct layer l, update_args a) {
    //     update_batchnorm_layer_gpu(l,a);
    // }

    void create_batchnorm_snapshot_for_sgx_fbv(struct layer& l, struct network& net, uint8_t** out, uint8_t** sha256_out) {
        if (gpu_index >= 0) {
            pull_batchnorm_layer(l);
        }
        int total_bytes = count_layer_paramas_bytes(l);
        size_t buff_ind = 0;
        *out = new uint8_t[total_bytes];
        *sha256_out = new uint8_t[SHA256_DIGEST_LENGTH];
        
        std::memcpy((*out+buff_ind),l.scale_updates,l.c*sizeof(float));
        buff_ind += l.c*sizeof(float);
        std::memcpy((*out+buff_ind),l.rolling_mean,l.c*sizeof(float));
        buff_ind += l.c*sizeof(float);
        std::memcpy((*out+buff_ind),l.rolling_variance,l.c*sizeof(float));
        buff_ind += l.c*sizeof(float);
        if (buff_ind != total_bytes) {
                LOG_ERROR("size mismatch\n")
                abort();
        }
        gen_sha256(*out,total_bytes,*sha256_out);
    }
#endif

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

void forward_batchnorm_layer_gpu(layer l, network net)
{
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
    if (net.train) {
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                l.dstTensorDesc,
                l.x_gpu,
                l.dstTensorDesc,
                l.output_gpu,
                l.normTensorDesc,
                l.scales_gpu,
                l.biases_gpu,
                .01,
                l.rolling_mean_gpu,
                l.rolling_variance_gpu,
                .00001,
                l.mean_gpu,
                l.variance_gpu);
#else
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

        scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
        axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
        axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);

        // scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        // add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
#endif
    } else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        // scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        // add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    }
    scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_batchnorm_layer_gpu(layer l, network net)
{
    if(!net.train){
        l.mean_gpu = l.rolling_mean_gpu;
        l.variance_gpu = l.rolling_variance_gpu;
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            &one,
            &one,
            l.dstTensorDesc,
            l.x_gpu,
            l.dstTensorDesc,
            l.delta_gpu,
            l.dstTensorDesc,
            l.x_norm_gpu,
            l.normTensorDesc,
            l.scales_gpu,
            l.scale_updates_gpu,
            l.bias_updates_gpu,
            .00001,
            l.mean_gpu,
            l.variance_gpu);
    copy_gpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
#else
    // LOG_DEBUG("GPU batchnorm back batch:%d,out_c:%d,out_w:%d,out_h:%d\n", l.batch, l.out_c, l.out_w,l.out_h)
    // if (net.index == 7) {
    //     cuda_pull_array(l.delta_gpu,l.delta,l.outputs*l.batch);
    //     std::string text = std::string("GPU bactnorm delta after gradient on activation layer ") + std::to_string(net.index);
    //     print_array(l.delta+(l.batch/2)*l.outputs,1*l.outputs,(l.batch/2)*l.outputs,text.c_str());
    //     cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.nbiases);
    //     print_array(l.bias_updates, l.nbiases, 0, "GPU bactnorm forw bias updates before");
    // }

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    // if (net.index == 7) {
    //     cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.nbiases);
    //     print_array(l.bias_updates, l.nbiases, 0, "GPU bactnorm forw bias updates after");
    // }
    
    // cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.nbiases);
    // print_array(l.scale_updates, l.nbiases, 0, "GPU: batchnorm scale updates before");
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);
    // cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.nbiases);
    // print_array(l.scale_updates, l.nbiases, 0, "GPU: batchnorm scale updates after");

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
#endif
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
layer_blocked make_batchnorm_layer_blocked(int batch, int w, int h, int c) {
    layer_blocked l = {};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    // l.output = (float*)calloc(h * w * c * batch, sizeof(float));
    l.output = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({h * w * c * batch});
    // l.delta  = (float*)calloc(h * w * c * batch, sizeof(float));
    l.delta  = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({h * w * c * batch});
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    // l.scales = (float*)calloc(c, sizeof(float));
    l.scales = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({c});
    // l.scale_updates = (float*)calloc(c, sizeof(float));
    l.scale_updates = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({c});
    // l.biases = (float*)calloc(c, sizeof(float));
    l.biases = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({c});
    // l.bias_updates = (float*)calloc(c, sizeof(float));
    l.bias_updates = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({c});
    int i;
    BLOCK_ENGINE_INIT_FOR_LOOP(l.scales, scales_range, scales_ptr, float)
    for(i = 0; i < c; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.scales, scales_range, scales_ptr, true, scales_ind, i)
        *(scales_ptr + scales_ind - scales_range.block_requested_ind) = 1.0;
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.scales, scales_range)
    
    // l.mean = (float*)calloc(c, sizeof(float));
    l.mean = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({c});
    // l.variance = (float*)calloc(c, sizeof(float));
    l.variance = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({c});

    // l.rolling_mean = (float*)calloc(c, sizeof(float));
    l.rolling_mean = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({c});
    // l.rolling_variance = (float*)calloc(c, sizeof(float));
    l.rolling_variance = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({c});

    l.forward_blocked = forward_batchnorm_layer_blocked;
    l.backward_blocked = backward_batchnorm_layer_blocked;
}

void forward_batchnorm_layer_blocked(layer_blocked l, network_blocked net)
{
    if(l.type == BATCHNORM) copy_cpu_blocked(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu_blocked(l.outputs*l.batch, l.output, 1, l.x, 1);
    if(net.train){
        mean_cpu_blocked(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu_blocked(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu_blocked(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu_blocked(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu_blocked(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu_blocked(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu_blocked(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   
        copy_cpu_blocked(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu_blocked(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias_blocked(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias_blocked(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer_blocked(layer_blocked l, network_blocked net)
{
    if(!net.train){
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    backward_bias_blocked(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_cpu_blocked(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias_blocked(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu_blocked(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu_blocked(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu_blocked(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM) copy_cpu_blocked(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

void backward_scale_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x_norm, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, int batch, int n, int size, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &scale_updates) {
    int i,b,f;
    //BLOCK_ENGINE_INIT_FOR_LOOP(x_norm, x_norm_valid_range, x_norm_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(x_norm, x_norm_valid_range, x_norm_block_val_ptr,x_norm_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(delta, delta_valid_range, delta_block_val_ptr,delta_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(scale_updates, scale_updates_valid_range, scale_updates_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(scale_updates, scale_updates_valid_range, scale_updates_block_val_ptr,scale_updates_index_var,true, float)
    for(f = 0; f < n; ++f){
        float sum = 0;
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(scale_updates, scale_updates_valid_range, scale_updates_block_val_ptr, true, scale_updates_index_var, f)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(scale_updates, scale_updates_valid_range, scale_updates_block_val_ptr, true, scale_updates_index_var, f)
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, index)
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(x_norm, x_norm_valid_range, x_norm_block_val_ptr, false, x_norm_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(x_norm, x_norm_valid_range, x_norm_block_val_ptr, false, x_norm_index_var, index)
                sum += (*(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind)) * (*(x_norm_block_val_ptr+x_norm_index_var-x_norm_valid_range.block_requested_ind));
                // sum += delta[index] * x_norm[index];
            }
        }
        *(scale_updates_block_val_ptr+scale_updates_index_var-scale_updates_valid_range.block_requested_ind) += sum;
        //scale_updates[f] += sum;
    }
    BLOCK_ENGINE_LAST_UNLOCK(x_norm, x_norm_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(scale_updates, scale_updates_valid_range)
}

void mean_delta_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean_delta) {
    int i,j,k;
    //BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(delta, delta_valid_range, delta_block_val_ptr, delta_index_var, false,float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(variance, variance_valid_range, variance_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(variance, variance_valid_range, variance_block_val_ptr, variance_index_var, false,float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(mean_delta, mean_delta_valid_range, mean_delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(mean_delta, mean_delta_valid_range, mean_delta_block_val_ptr, mean_delta_index_var, true,float)
    for(i = 0; i < filters; ++i){
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(mean_delta, mean_delta_valid_range, mean_delta_block_val_ptr, true, mean_delta_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(mean_delta, mean_delta_valid_range, mean_delta_block_val_ptr, true, mean_delta_index_var, i)
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(variance, variance_valid_range, variance_block_val_ptr, false,variance_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(variance, variance_valid_range, variance_block_val_ptr, false,variance_index_var, i)
        *(mean_delta_block_val_ptr+mean_delta_index_var-mean_delta_valid_range.block_requested_ind) = 0.0;
        //mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, index)
                *(mean_delta_block_val_ptr+mean_delta_index_var-mean_delta_valid_range.block_requested_ind) += *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind);
                // mean_delta[i] += delta[index];
            }
        }
        *(mean_delta_block_val_ptr+mean_delta_index_var-mean_delta_valid_range.block_requested_ind) *= (-1./sqrt(*(variance_block_val_ptr+variance_index_var-variance_valid_range.block_requested_ind) + .00001f));
        // mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(variance, variance_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(mean_delta, mean_delta_valid_range)
}

void  variance_delta_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance_delta) {
    int i,j,k;
    //BLOCK_ENGINE_INIT_FOR_LOOP(x, x_valid_range, x_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(x, x_valid_range, x_block_val_ptr,x_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(delta, delta_valid_range, delta_block_val_ptr,delta_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(mean, mean_valid_range, mean_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(mean, mean_valid_range, mean_block_val_ptr,mean_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(variance, variance_valid_range, variance_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(variance, variance_valid_range, variance_block_val_ptr,variance_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(variance_delta, variance_delta_valid_range, variance_delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(variance_delta, variance_delta_valid_range, variance_delta_block_val_ptr,variance_delta_index_var,true, float)
    for(i = 0; i < filters; ++i){
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(variance_delta, variance_delta_valid_range, variance_delta_block_val_ptr,true, variance_delta_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(variance_delta, variance_delta_valid_range, variance_delta_block_val_ptr,true, variance_delta_index_var, i)
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(variance, variance_valid_range, variance_block_val_ptr, false, variance_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(variance, variance_valid_range, variance_block_val_ptr, false, variance_index_var, i)
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(mean, mean_valid_range, mean_block_val_ptr, false, mean_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(mean, mean_valid_range, mean_block_val_ptr, false, mean_index_var, i)
        *(variance_delta_block_val_ptr+variance_delta_index_var-variance_delta_valid_range.block_requested_ind) = 0.0;
        //variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(x, x_valid_range, x_block_val_ptr, false, x_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(x, x_valid_range, x_block_val_ptr, false, x_index_var, index)
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(delta, delta_valid_range, delta_block_val_ptr, false, delta_index_var, index)
                *(variance_delta_block_val_ptr+variance_delta_index_var-variance_delta_valid_range.block_requested_ind) += *(delta_block_val_ptr + delta_index_var-delta_valid_range.block_requested_ind)*(*(x_block_val_ptr+x_index_var-x_valid_range.block_requested_ind) - *(mean_block_val_ptr+mean_index_var-mean_valid_range.block_requested_ind));
                //variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        *(variance_delta_block_val_ptr+variance_delta_index_var-variance_delta_valid_range.block_requested_ind) *= -.5 * pow(*(variance_block_val_ptr+variance_index_var-variance_valid_range.block_requested_ind) + .00001f, (float)(-3./2.));
        //variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
    BLOCK_ENGINE_LAST_UNLOCK(x, x_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(mean, mean_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(variance, variance_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(variance_delta, variance_delta_valid_range)
}

void normalize_delta_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean_delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance_delta, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta) {
    int f, j, k;
    //BLOCK_ENGINE_INIT_FOR_LOOP(x, x_valid_range, x_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(x, x_valid_range, x_block_val_ptr,x_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(delta, delta_valid_range, delta_block_val_ptr,delta_index_var,true, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(mean, mean_valid_range, mean_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(mean, mean_valid_range, mean_block_val_ptr,mean_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(variance, variance_valid_range, variance_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(variance, variance_valid_range, variance_block_val_ptr,variance_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(variance_delta, variance_delta_valid_range, variance_delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(variance_delta, variance_delta_valid_range, variance_delta_block_val_ptr,variance_delta_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(mean_delta, mean_delta_valid_range, mean_delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(mean_delta, mean_delta_valid_range, mean_delta_block_val_ptr,mean_delta_index_var,false, float)
    for(j = 0; j < batch; ++j) {
        for(f = 0; f < filters; ++f){
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(variance, variance_valid_range, variance_block_val_ptr, false, variance_index_var, f)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(variance, variance_valid_range, variance_block_val_ptr, false, variance_index_var, f)
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(variance_delta, variance_delta_valid_range, variance_delta_block_val_ptr, false, variance_delta_index_var, f)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(variance_delta, variance_delta_valid_range, variance_delta_block_val_ptr, false, variance_delta_index_var, f)
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(mean, mean_valid_range, mean_block_val_ptr, false, mean_index_var, f)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(mean, mean_valid_range, mean_block_val_ptr, false, mean_index_var, f)
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(mean_delta, mean_delta_valid_range, mean_delta_block_val_ptr, false, mean_delta_index_var, f)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(mean_delta, mean_delta_valid_range, mean_delta_block_val_ptr, false, mean_delta_index_var, f)
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, true, delta_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(delta, delta_valid_range, delta_block_val_ptr, true, delta_index_var, index)
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(x, x_valid_range, x_block_val_ptr, false, x_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(x, x_valid_range, x_block_val_ptr, false, x_index_var, index)
                *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) = 
                    (*(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) * 1./(sqrt(*(variance_block_val_ptr+variance_index_var-variance_valid_range.block_requested_ind) + .00001f)) ) + 
                    (*(variance_delta_block_val_ptr+variance_delta_index_var-variance_delta_valid_range.block_requested_ind) * 
                    2. * (*(x_block_val_ptr+x_index_var-x_valid_range.block_requested_ind) - *(mean_block_val_ptr+mean_index_var-mean_valid_range.block_requested_ind)) / (spatial * batch)) + (*(mean_delta_block_val_ptr+mean_delta_index_var-mean_delta_valid_range.block_requested_ind)/(spatial*batch));
                //delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(x, x_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(mean, mean_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(variance, variance_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(variance_delta, variance_delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(mean_delta, mean_delta_valid_range)
}
#endif