#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifndef USE_SGX_LAYERWISE
void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}
#endif

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

#ifndef USE_SGX_LAYERWISE
image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}
#endif

#ifndef USE_SGX_LAYERWISE
image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}
#endif

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif

#ifndef USE_SGX_LAYERWISE
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c,
                                             int n, int groups, int size,
                                             int stride, int padding,
                                             ACTIVATION activation,
                                             int batch_normalize, int binary,
                                             int xnor, int adam) {
  int i;
  convolutional_layer l = {};
  l.type = CONVOLUTIONAL;

  l.groups = groups;
  l.h = h;
  l.w = w;
  l.c = c;
  l.n = n;
  l.binary = binary;
  l.xnor = xnor;
  l.batch = batch;
  l.stride = stride;
  l.size = size;
  l.pad = padding;
  l.batch_normalize = batch_normalize;

  l.weights = (float *)calloc(c / groups * n * size * size, sizeof(float));
  l.weight_updates =
      (float *)calloc(c / groups * n * size * size, sizeof(float));

  l.biases = (float *)calloc(n, sizeof(float));
  l.bias_updates = (float *)calloc(n, sizeof(float));

  l.nweights = c / groups * n * size * size;
  l.nbiases = n;

  // float scale = 1./sqrt(size*size*c);
  float scale = sqrt(2. / (size * size * c / l.groups));
  // printf("convscale %f\n", scale);
  // scale = .02;
  // for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1,
  // 1);
  for (i = 0; i < l.nweights; ++i)
    l.weights[i] = scale * rand_normal();
  
  //LOG_DEBUG("INIT conv layer 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
  int out_w = convolutional_out_width(l);
  int out_h = convolutional_out_height(l);
  l.out_h = out_h;
  l.out_w = out_w;
  l.out_c = n;
  l.outputs = l.out_h * l.out_w * l.out_c;
  l.inputs = l.w * l.h * l.c;

  l.output = (float *)calloc(l.batch * l.outputs, sizeof(float));
  l.delta = (float *)calloc(l.batch * l.outputs, sizeof(float));

  l.forward = forward_convolutional_layer;
  l.backward = backward_convolutional_layer;
  l.update = update_convolutional_layer;
  if (binary) {
    l.binary_weights = (float *)calloc(l.nweights, sizeof(float));
    l.cweights = (char *)calloc(l.nweights, sizeof(char));
    l.scales = (float *)calloc(n, sizeof(float));
  }
  if (xnor) {
    l.binary_weights = (float *)calloc(l.nweights, sizeof(float));
    l.binary_input = (float *)calloc(l.inputs * l.batch, sizeof(float));
  }

  if (batch_normalize) {
    l.scales = (float *)calloc(n, sizeof(float));
    l.scale_updates = (float *)calloc(n, sizeof(float));
    for (i = 0; i < n; ++i) {
      l.scales[i] = 1;
    }

    l.mean = (float *)calloc(n, sizeof(float));
    l.variance = (float *)calloc(n, sizeof(float));

    l.mean_delta = (float *)calloc(n, sizeof(float));
    l.variance_delta = (float *)calloc(n, sizeof(float));

    l.rolling_mean = (float *)calloc(n, sizeof(float));
    l.rolling_variance = (float *)calloc(n, sizeof(float));
    l.x = (float *)calloc(l.batch * l.outputs, sizeof(float));
    l.x_norm = (float *)calloc(l.batch * l.outputs, sizeof(float));
  }
  if (adam) {
    l.m = (float *)calloc(l.nweights, sizeof(float));
    l.v = (float *)calloc(l.nweights, sizeof(float));
    l.bias_m = (float *)calloc(n, sizeof(float));
    l.scale_m = (float *)calloc(n, sizeof(float));
    l.bias_v = (float *)calloc(n, sizeof(float));
    l.scale_v = (float *)calloc(n, sizeof(float));
  }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}
#endif

#ifndef USE_SGX_LAYERWISE
void denormalize_convolutional_layer(convolutional_layer l) {
  int i, j;
  for (i = 0; i < l.n; ++i) {
    float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .00001);
    for (j = 0; j < l.c / l.groups * l.size * l.size; ++j) {
      l.weights[i * l.c / l.groups * l.size * l.size + j] *= scale;
    }
    l.biases[i] -= l.rolling_mean[i] * scale;
    l.scales[i] = 1;
    l.rolling_mean[i] = 0;
    l.rolling_variance[i] = 1;
  }
}
#endif

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/
#ifndef USE_SGX_LAYERWISE
void resize_convolutional_layer(convolutional_layer *l, int w, int h) {
  l->w = w;
  l->h = h;
  int out_w = convolutional_out_width(*l);
  int out_h = convolutional_out_height(*l);

  l->out_w = out_w;
  l->out_h = out_h;

  l->outputs = l->out_h * l->out_w * l->out_c;
  l->inputs = l->w * l->h * l->c;

  l->output =
      (float *)realloc(l->output, l->batch * l->outputs * sizeof(float));
  l->delta = (float *)realloc(l->delta, l->batch * l->outputs * sizeof(float));
  if (l->batch_normalize) {
    l->x = (float *)realloc(l->x, l->batch * l->outputs * sizeof(float));
    l->x_norm =
        (float *)realloc(l->x_norm, l->batch * l->outputs * sizeof(float));
  }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}
#endif

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

#ifndef USE_SGX_LAYERWISE
void forward_convolutional_layer(convolutional_layer l, network net)
{
  //LOG_DEBUG("before forward 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
  int i, j;

  fill_cpu(l.outputs * l.batch, 0, l.output, 1);

  if (l.xnor) {
    binarize_weights(l.weights, l.n, l.c / l.groups * l.size * l.size,
                     l.binary_weights);
    swap_binary(&l);
    binarize_cpu(net.input, l.c * l.h * l.w * l.batch, l.binary_input);
    net.input = l.binary_input;
  }

  int m = l.n / l.groups;
  int k = l.size * l.size * l.c / l.groups;
  int n = l.out_w * l.out_h;
  //LOG_DEBUG("begining conv with parameters outputs:%d, batch:%d,groups:%d, m:%d, k:%d, n:%d, out_w:%d,out_h:%d,out_c:%d\n",l.outputs,l.batch,l.groups,m,k,n,l.out_w,l.out_h,l.out_c);
  for (i = 0; i < l.batch; ++i) {
    for (j = 0; j < l.groups; ++j) {
      float *a = l.weights + j * l.nweights / l.groups;
      float *b = net.workspace;
      float *c = l.output + (i * l.groups + j) * n * m;
      float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

      if (l.size == 1) {
          b = im;
      } else {
          im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
          //LOG_DEBUG("Base start index for C (output) is %d",(i * l.groups + j) * n * m)
      }
      gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
  }
  //LOG_DEBUG("Ready for batch normalize!! goinh to batch nrom? %d",l.batch_normalize)

  if (l.batch_normalize) {
    forward_batchnorm_layer(l, net);
  } else {
    add_bias(l.output, l.biases, l.batch, l.n, l.out_h * l.out_w);
  }

  activate_array(l.output, l.outputs * l.batch, l.activation);
  if (l.binary || l.xnor)
    swap_binary(&l);
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_convolutional_layer(convolutional_layer l, network net)
{
    //LOG_DEBUG("before backward 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("before backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
    //LOG_DEBUG("after backward 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("after backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
}
#endif

#ifndef USE_SGX_LAYERWISE
void update_convolutional_layer(convolutional_layer l, update_args a)
{
    
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    //LOG_DEBUG("lr:%f,moment:%f,decay:%f,batch:%d total weights:%d\n",learning_rate,momentum,decay,batch,l.nweights)
    //LOG_DEBUG("before update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    //LOG_DEBUG("before update 237 and 121 weight updates are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    //LOG_DEBUG("here update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("here update 237 and 121 weight updates are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
    //LOG_DEBUG("after update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
}
#endif

#ifndef USE_SGX_LAYERWISE
image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}
#endif

#ifndef USE_SGX_LAYERWISE
void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}
#endif

#ifndef USE_SGX_LAYERWISE
void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}
#endif

#ifndef USE_SGX_LAYERWISE
image *get_weights(convolutional_layer l) {
  image *weights = (image *)calloc(l.n, sizeof(image));
  int i;
  for (i = 0; i < l.n; ++i) {
    weights[i] = copy_image(get_convolutional_weight(l, i));
    normalize_image(weights[i]);
    /*
       char buff[256];
       sprintf(buff, "filter%d", i);
       save_image(weights[i], buff);
     */
  }
  // error("hey");
  return weights;
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c,
                                             int n, int groups, int size,
                                             int stride, int padding,
                                             ACTIVATION activation,
                                             int batch_normalize, int binary,
                                             int xnor, int adam) {
  int i,j;
  convolutional_layer l = {};
  l.type = CONVOLUTIONAL;

  l.groups = groups;
  l.h = h;
  l.w = w;
  l.c = c;
  l.n = n;
  l.binary = binary;
  l.xnor = xnor;
  l.batch = batch;
  l.stride = stride;
  l.size = size;
  l.pad = padding;
  l.batch_normalize = batch_normalize;

  //l.weights = (float *)calloc(c / groups * n * size * size, sizeof(float));
  l.weights = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c / groups * n * size * size);
  //l.weight_updates = (float *)calloc(c / groups * n * size * size, sizeof(float));
  l.weight_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c / groups * n * size * size);

  //l.biases = (float *)calloc(n, sizeof(float));
  l.biases = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
  //l.bias_updates = (float *)calloc(n, sizeof(float));
  l.bias_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);

  l.nweights = c / groups * n * size * size;
  l.nbiases = n;

  // float scale = 1./sqrt(size*size*c);
  float scale = sqrt(2. / (size * size * c / l.groups));
  // printf("convscale %f\n", scale);
  // scale = .02;
  // for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1,
  // 1);
  {
    auto ws = l.weights->getItemsInRange(0, l.weights->getBufferSize());
    for (i = 0; i < l.nweights; ++i)
      ws[i] = scale * rand_normal();
    l.weights->setItemsInRange(0, l.nweights,ws);
    //LOG_DEBUG("INIT conv layer ID: %u, 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights->getID(),ws[237],ws[121])
  }
  /* {
    // just checking
    auto ws = l.weights->getItemsInRange(0, l.weights->getBufferSize());
    LOG_DEBUG("INIT conv layer ID: %u, 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights->getmID(),ws[237],ws[121])
  } */

  int out_w = convolutional_out_width(l);
  int out_h = convolutional_out_height(l);
  l.out_h = out_h;
  l.out_w = out_w;
  l.out_c = n;
  l.outputs = l.out_h * l.out_w * l.out_c;
  l.inputs = l.w * l.h * l.c;

  //l.output = (float *)calloc(l.batch * l.outputs, sizeof(float));
  l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.batch * l.outputs);
  //l.delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
  l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.batch * l.outputs);

  l.forward = forward_convolutional_layer;
  l.backward = backward_convolutional_layer;
  l.update = update_convolutional_layer;
  if (binary) {
    //l.binary_weights = (float *)calloc(l.nweights, sizeof(float));
    l.binary_weights = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.nweights);
    //l.cweights = (char *)calloc(l.nweights, sizeof(char));
    l.cweights = sgx::trusted::SpecialBuffer<char>::GetNewSpecialBuffer(l.nweights);
    //l.scales = (float *)calloc(n, sizeof(float));
    l.scales = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
  }
  if (xnor) {
    //l.binary_weights = (float *)calloc(l.nweights, sizeof(float));
    l.binary_weights = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.nweights);
    //l.binary_input = (float *)calloc(l.inputs * l.batch, sizeof(float));
    l.binary_input = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.inputs * l.batch);
  }

  if (batch_normalize) {
    //l.scales = (float *)calloc(n, sizeof(float));
    l.scales = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.scale_updates = (float *)calloc(n, sizeof(float));
    l.scale_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);

    {
      auto scs = l.scales->getItemsInRange(0, l.scales->getBufferSize());
      for (i = 0; i < n; ++i)
        scs[i] = 1;
      l.scales->setItemsInRange(0, n,scs);
    }

    //l.mean = (float *)calloc(n, sizeof(float));
    l.mean = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.variance = (float *)calloc(n, sizeof(float));
    l.variance = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);

    //l.mean_delta = (float *)calloc(n, sizeof(float));
    l.mean_delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.variance_delta = (float *)calloc(n, sizeof(float));
    l.variance_delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);

    //l.rolling_mean = (float *)calloc(n, sizeof(float));
    l.rolling_mean = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.rolling_variance = (float *)calloc(n, sizeof(float));
    l.rolling_variance = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.x = (float *)calloc(l.batch * l.outputs, sizeof(float));
    l.x = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.batch * l.outputs);
    //l.x_norm = (float *)calloc(l.batch * l.outputs, sizeof(float));
    l.x_norm = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.batch * l.outputs);
  }
  if (adam) {
    //l.m = (float *)calloc(l.nweights, sizeof(float));
    l.m = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.nweights);
    //l.v = (float *)calloc(l.nweights, sizeof(float));
    l.v = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.nweights);
    //l.bias_m = (float *)calloc(n, sizeof(float));
    l.bias_m = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.scale_m = (float *)calloc(n, sizeof(float));
    l.scale_m = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.bias_v = (float *)calloc(n, sizeof(float));
    l.bias_v = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.scale_v = (float *)calloc(n, sizeof(float));
    l.scale_v = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
  }

  l.workspace_size = get_workspace_size(l);
  l.activation = activation;

  fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
  int i, j;
  auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
  auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
  //auto n_workspace = net.workspace->getItemsInRange(0, net.workspace->getBufferSize());
  auto n_input = net.input->getItemsInRange(0, net.input->getBufferSize());
  //LOG_DEBUG("before forward ID: %u, 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights->getID(),l_weights[237],l_weights[121])

  fill_cpu(l.outputs * l.batch, 0, &l_output[0], 1);

  if (l.xnor) {
    LOG_ERROR("XNOR feature not implemented!\n");
    abort();
    /* binarize_weights(l.weights, l.n, l.c / l.groups * l.size * l.size,
                     l.binary_weights);
    swap_binary(&l);
    binarize_cpu(net.input, l.c * l.h * l.w * l.batch, l.binary_input);
    net.input = l.binary_input; */
  } 

  int m = l.n / l.groups;
  int k = l.size * l.size * l.c / l.groups;
  int n = l.out_w * l.out_h;
  //LOG_DEBUG("begining conv with parameters outputs:%d, batch:%d,groups:%d, m:%d, k:%d, n:%d, out_w:%d,out_h:%d,out_c:%d\n",l.outputs,l.batch,l.groups,m,k,n,l.out_w,l.out_h,l.out_c);
  for (i = 0; i < l.batch; ++i) {
    for (j = 0; j < l.groups; ++j) {
      float *a = &l_weights[0] + j * l.nweights / l.groups;
      float *b = nullptr; //&n_workspace[0];
      float *c = &l_output[0] + (i * l.groups + j) * n * m;
      float *im =  &n_input[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
      if (l.size == 1) {
          b = im;
          gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
      } else {
          auto n_workspace = net.workspace->getItemsInRange(0, 1*l.out_h*l.out_w*l.size*l.size);
          for (int chan = 0; chan < l.c / l.groups; chan++) {
            std::memset(&n_workspace[0], 0, sizeof(float)*n_workspace.size());
            b = &n_workspace[0];
            //im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            im2col_cpu(im+(chan*l.h*l.w), 1, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm(0, 0, m, n, (l.size*l.size), 1, a+(chan*l.size*l.size), k, b, n, 1, c, n);  // k is changed
          }
          //LOG_DEBUG("Base start index for C (output) is %d",(i * l.groups + j) * n * m)
          //gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
      }
    }
  }
  //LOG_DEBUG("Ready for batch normalize!! goinh to batch nrom? %d",l.batch_normalize)

  if (l.batch_normalize) {
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    forward_batchnorm_layer(l, net);
    l_output = l.output->getItemsInRange(0, l.output->getBufferSize());

  } else {
    auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
    add_bias(&l_output[0], &l_biases[0], l.batch, l.n, l.out_h * l.out_w);
  }
  activate_array(&l_output[0], l.outputs * l.batch, l.activation);
  l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
  /* if (l.binary || l.xnor)
    swap_binary(&l); */
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    //LOG_DEBUG("before backward 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("before backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;
    
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto net_delta  = net.delta ? net.delta->getItemsInRange(0, net.delta->getBufferSize()):std::vector<float>();
  
    auto l_weight_updates = l.weight_updates->getItemsInRange(0, l.weight_updates->getBufferSize());
    //auto net_workspace = net.workspace->getItemsInRange(0, net.workspace->getBufferSize());
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());

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
      backward_bias(&l_bias_updates[0], &l_delta[0], l.batch, l.n, k);
      l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(), l_bias_updates);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = &l_delta[0] + (i*l.groups + j)*m*k;
            float *b = nullptr;   //&net_workspace[0];
            float *c = &l_weight_updates[0] + j*l.nweights/l.groups;
            float *im  = &net_input[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
                gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
            } else {
                auto net_workspace = net.workspace->getItemsInRange(0, l.size*l.size*l.out_h*l.out_w);
                for (int chan = 0; chan < l.c/l.groups;++chan) {
                  std::memset(&net_workspace[0], 0, sizeof(float)*net_workspace.size());
                  b = &net_workspace[0];
                  im2col_cpu(im+chan*(l.h*l.w), 1, l.h, l.w, l.size, l.stride, l.pad, b);
                  //im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
                  gemm(0,1,m,(l.size*l.size),k,1,a,k,b,k,1,c+(chan*l.size*l.size),n);
                  //gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
                }
            }
            //gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta != nullptr) {
                auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
                a = &l_weights[0] + j*l.nweights/l.groups;
                b = &l_delta[0] + (i*l.groups + j)*m*k;
                float *imd = &net_delta[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
                c = nullptr;  // &net_workspace[0];
                if (l.size == 1) {
                    c = imd;
                    gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);
                }
                else {
                    auto net_workspace = net.workspace->getItemsInRange(0, l.size*l.size*l.out_h*l.out_w);
                    for (int chan=0;chan < l.c/l.groups;chan++) {
                        std::memset(&net_workspace[0], 0, sizeof(float)*net_workspace.size());
                        c = &net_workspace[0];
                        gemm(1,0,l.size*l.size,k,m,1,a+(chan*l.size*l.size),n,b,k,0,c,k);
                        col2im_cpu(c, 1, l.h, l.w, l.size, l.stride, l.pad, imd+(chan*l.h*l.w));
                    }
                    //gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);
                    //col2im_cpu(&net_workspace[0], l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                    
                }
            }
        }
    }
    l.weight_updates->setItemsInRange(0, l.weight_updates->getBufferSize(),l_weight_updates);
    l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
    if (net.delta != nullptr) {
      net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
    }
    //LOG_DEBUG("after backward 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("after backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    //LOG_DEBUG("lr:%f,moment:%f,decay:%f,batch:%d total weights:%d\n",learning_rate,momentum,decay,batch,l.nweights)
    //LOG_DEBUG("before update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    {
      auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
      auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
      axpy_cpu(l.n, learning_rate/batch, &l_bias_updates[0], 1, &l_biases[0], 1);
      scal_cpu(l.n, momentum, &l_bias_updates[0], 1);
      l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(),l_bias_updates);
      l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
    }

    if(l.scales){
        auto l_scale_updates = l.scale_updates->getItemsInRange(0, l.scale_updates->getBufferSize());
        auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
        axpy_cpu(l.n, learning_rate/batch, &l_scale_updates[0], 1, &l_scales[0], 1);
        scal_cpu(l.n, momentum, &l_scale_updates[0], 1);
        l.scale_updates->setItemsInRange(0, l.scale_updates->getBufferSize(),l_scale_updates);
        l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
    }

    //LOG_DEBUG("before update 237 and 121 weight updates are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
    auto l_weight_updates = l.weight_updates->getItemsInRange(0, l.weight_updates->getBufferSize());
    axpy_cpu(l.nweights, -decay*batch, &l_weights[0], 1, &l_weight_updates[0], 1);
    //LOG_DEBUG("here update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("here update 237 and 121 weight updates are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    axpy_cpu(l.nweights, learning_rate/batch, &l_weight_updates[0], 1, &l_weights[0], 1);
    scal_cpu(l.nweights, momentum, &l_weight_updates[0], 1);
    l.weights->setItemsInRange(0, l.weights->getBufferSize(),l_weights);
    l.weight_updates->setItemsInRange(0, l.weight_updates->getBufferSize(),l_weight_updates);

    //LOG_DEBUG("after update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
}
#endif

#ifndef USE_SGX
image *visualize_convolutional_layer(convolutional_layer l, char *window,
                                     image *prev_weights) {
  image *single_weights = get_weights(l);
  show_images(single_weights, l.n, window);

  image delta = get_convolutional_image(l);
  image dc = collapse_image_layers(delta, 1);
  char buff[256];
  sprintf(buff, "%s: Output", window);
  // show_image(dc, buff);
  // save_image(dc, buff);
  free_image(dc);
  return single_weights;
}
#else
#endif

#if defined(USE_SGX) && defined(USE_SGX_BLOCKING)
int convolutional_out_height_blocked(convolutional_layer_blocked l) {
  return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width_blocked(convolutional_layer_blocked l) {
  return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}
static size_t get_workspace_size(layer_blocked l) {
  return (size_t)l.out_h * l.out_w * l.size * l.size * l.c / l.groups *
         sizeof(float);
}
void scale_bias_blocked(
    const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output,
    const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &scales,
    int batch, int n, int size) {
  int i, j, b;
  //BLOCK_ENGINE_INIT_FOR_LOOP(output, output_valid_range, output_block_val_ptr,float)
  BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(output, output_valid_range, output_block_val_ptr,output_index_var,true,float)
  //BLOCK_ENGINE_INIT_FOR_LOOP(scales, scales_valid_range, scales_block_val_ptr,float)
  BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(scales, scales_valid_range, scales_block_val_ptr,scales_current_index,false,float)
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(scales, scales_valid_range, scales_block_val_ptr, false, scales_current_index, i)
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(scales, scales_valid_range, scales_block_val_ptr, false, scales_current_index, i)
      for (j = 0; j < size; ++j) {
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(output, output_valid_range, output_block_val_ptr, true, output_index_var, (b * n + i) * size + j)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(output, output_valid_range, output_block_val_ptr, true, output_index_var, (b * n + i) * size + j)
        *(output_block_val_ptr + output_index_var -
          output_valid_range.block_requested_ind) *=
            *(scales_block_val_ptr + scales_current_index -
              scales_valid_range.block_requested_ind);
        // output[(b*n + i)*size + j] *= scales[i];
      }
    }
  }
  BLOCK_ENGINE_LAST_UNLOCK(output, output_valid_range)
  BLOCK_ENGINE_LAST_UNLOCK(scales, scales_valid_range)
}

void add_bias_blocked(
    const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output,
    const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &biases,
    int batch, int n, int size) {
  int i, j, b;
  //BLOCK_ENGINE_INIT_FOR_LOOP(output, output_valid_range, output_block_val_ptr, float)
  BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(output, output_valid_range, output_block_val_ptr,output_index_var, true, float)
  //BLOCK_ENGINE_INIT_FOR_LOOP(biases, biases_valid_range, biases_block_val_ptr, float)
  BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(biases, biases_valid_range, biases_block_val_ptr,biases_current_index, false, float)
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(biases, biases_valid_range, biases_block_val_ptr, false, biases_current_index, i)
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(biases, biases_valid_range, biases_block_val_ptr, false, biases_current_index, i)
      for (j = 0; j < size; ++j) {
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(output, output_valid_range, output_block_val_ptr, true, output_index_var, (b * n + i) * size + j)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(output, output_valid_range, output_block_val_ptr, true, output_index_var, (b * n + i) * size + j)
        *(output_block_val_ptr + output_index_var -
          output_valid_range.block_requested_ind) +=
            *(biases_block_val_ptr + biases_current_index -
              biases_valid_range.block_requested_ind);
        // output[(b*n + i)*size + j] += biases[i];
      }
    }
  }
  BLOCK_ENGINE_LAST_UNLOCK(output, output_valid_range)
  BLOCK_ENGINE_LAST_UNLOCK(biases, biases_valid_range)
}

void backward_bias_blocked(
    const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &bias_updates,
    const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta,
    int batch, int n, int size) {
  int i, b;
  //BLOCK_ENGINE_INIT_FOR_LOOP(bias_updates, bias_updates_valid_range, bias_updates_block_val_ptr, float)
  BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(bias_updates, bias_updates_valid_range, bias_updates_block_val_ptr,bias_updates_index_var,true, float)
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(bias_updates, bias_updates_valid_range, bias_updates_block_val_ptr,true, bias_updates_index_var, i)
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(bias_updates, bias_updates_valid_range, bias_updates_block_val_ptr,true, bias_updates_index_var, i)
      *(bias_updates_block_val_ptr + bias_updates_index_var -
        bias_updates_valid_range.block_requested_ind) +=
          sum_array_blocked(delta, size, size * (i + b * n));
      // bias_updates[i] += sum_array(delta+size*(i+b*n), size);
    }
  }
  BLOCK_ENGINE_LAST_UNLOCK(bias_updates, bias_updates_valid_range)
}

convolutional_layer_blocked
make_convolutional_layer_blocked(int batch, int h, int w, int c, int n,
                                 int groups, int size, int stride, int padding,
                                 ACTIVATION activation, int batch_normalize,
                                 int binary, int xnor, int adam) {
  int i;
  convolutional_layer_blocked l = {};
  l.type = CONVOLUTIONAL;

  l.groups = groups;
  l.h = h;
  l.w = w;
  l.c = c;
  l.n = n;
  l.binary = binary;
  l.xnor = xnor;
  l.batch = batch;
  l.stride = stride;
  l.size = size;
  l.pad = padding;
  l.batch_normalize = batch_normalize;

  // l.weights = (float*)calloc(c/groups*n*size*size, sizeof(float));
  l.weights = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer(
      {c / groups * n * size * size});
  // l.weight_updates = (float*)calloc(c/groups*n*size*size, sizeof(float));
  //LOG_OUT("total weight updates for conv layer is: %d",c / groups * n * size * size);
  l.weight_updates = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer(
      {c / groups * n * size * size});
  // l.biases = (float*)calloc(n, sizeof(float));
  l.biases = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
  // l.bias_updates = (float*)calloc(n, sizeof(float));
  l.bias_updates =
      sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});

  l.nweights = c / groups * n * size * size;
  l.nbiases = n;

  // float scale = 1./sqrt(size*size*c);
  float scale = sqrt(2. / (size * size * c / l.groups));
  // printf("convscale %f\n", scale);
  // scale = .02;
  // for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1,
  // 1);
  BLOCK_ENGINE_INIT_FOR_LOOP(l.weights, weight_range, weight_range_ptr, float)
  for (i = 0; i < l.nweights; ++i) {
    BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                        weight_range_ptr, true, weight_index, i)
    *(weight_range_ptr + weight_index - weight_range.block_requested_ind) =
        scale * rand_normal();
  }
  
  BLOCK_ENGINE_LAST_UNLOCK(l.weights, weight_range);

  int out_w = convolutional_out_width_blocked(l);
  int out_h = convolutional_out_height_blocked(l);
  l.out_h = out_h;
  l.out_w = out_w;
  l.out_c = n;
  l.outputs = l.out_h * l.out_w * l.out_c;
  l.inputs = l.w * l.h * l.c;

  // l.output = (float*)calloc(l.batch*l.outputs, sizeof(float));
  l.output = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer(
      {l.batch * l.outputs});
  // l.delta  = (float*)calloc(l.batch*l.outputs, sizeof(float));
  l.delta = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer(
      {l.batch * l.outputs});

  l.forward_blocked = forward_convolutional_layer_blocked;
  l.backward_blocked = backward_convolutional_layer_blocked;
  l.update_blocked = update_convolutional_layer_blocked;

  /* if(binary){
      l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
      l.cweights = (char*)calloc(l.nweights, sizeof(char));
      l.scales = (float*)calloc(n, sizeof(float));
  }
  if(xnor){
      l.binary_weights = (float*)calloc(l.nweights, sizeof(float));
      l.binary_input = (float*)calloc(l.inputs*l.batch, sizeof(float));
  } */

  if (batch_normalize) {
    // l.scales = (float*)calloc(n, sizeof(float));
    l.scales = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    // l.scale_updates = (float*)calloc(n, sizeof(float));
    l.scale_updates =
        sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    BLOCK_ENGINE_INIT_FOR_LOOP(l.scales, scales_range, scales_ptr, float)
    for (i = 0; i < n; ++i) {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.scales, scales_range, scales_ptr,
                                          true, scales_ind, i)
      *(scales_ptr + scales_ind - scales_range.block_requested_ind) = 1.0;
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.scales, scales_range)
    // l.mean = (float*)calloc(n, sizeof(float));
    l.mean = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    // l.variance = (float*)calloc(n, sizeof(float));
    l.variance = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});

    // l.mean_delta = (float*)calloc(n, sizeof(float));
    l.mean_delta =
        sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    // l.variance_delta = (float*)calloc(n, sizeof(float));
    l.variance_delta =
        sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});

    // l.rolling_mean = (float*)calloc(n, sizeof(float));
    l.rolling_mean =
        sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    // l.rolling_variance = (float*)calloc(n, sizeof(float));
    l.rolling_variance =
        sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    // l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
    l.x = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer(
        {l.batch * l.outputs});
    // l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
    l.x_norm = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer(
        {l.batch * l.outputs});
  }
  if (adam) {
    // l.m = (float*)calloc(l.nweights, sizeof(float));
    l.m =
        sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({l.nweights});
    // l.v = (float*)calloc(l.nweights, sizeof(float));
    l.v =
        sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({l.nweights});
    // l.bias_m = (float*)calloc(n, sizeof(float));
    l.bias_m = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    // l.scale_m = (float*)calloc(n, sizeof(float));
    l.scale_m = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    // l.bias_v = (float*)calloc(n, sizeof(float));
    l.bias_v = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
    // l.scale_v = (float*)calloc(n, sizeof(float));
    l.scale_v = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({n});
  }

  l.workspace_size = get_workspace_size(l);
  l.activation = activation;

  return l;
}

void forward_convolutional_layer_blocked(convolutional_layer_blocked l,
                                         network_blocked net) {
  /* BLOCK_ENGINE_INIT_FOR_LOOP(l.weights, weight_range, weight_range_ptr, float)
  float i237_,i121_;
  {
    BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                        weight_range_ptr, false, weight_index, 237)
    i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
  }
  {
    BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                        weight_range_ptr, false, weight_index, 121)
    i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
  }
  BLOCK_ENGINE_LAST_UNLOCK(l.weights, weight_range);
  LOG_DEBUG("237 and 121 weights are: %0.10e, .. %0.10e\n",i237_,i121_); */
  int i, j;

  fill_cpu_blocked(l.outputs * l.batch, 0, l.output, 1);

  /* if(l.xnor){
      binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size,
  l.binary_weights); swap_binary(&l); binarize_cpu(net.input,
  l.c*l.h*l.w*l.batch, l.binary_input); net.input = l.binary_input;
  } */

  int m = l.n / l.groups;
  int k = l.size * l.size * l.c / l.groups;
  int n = l.out_w * l.out_h;
  //LOG_DEBUG("begining conv with parameters outputs:%d, batch:%d,groups:%d, m:%d, k:%d, n:%d, out_w:%d,out_h:%d,out_c:%d\n",l.outputs,l.batch,l.groups,m,k,n,l.out_w,l.out_h,l.out_c);
  for (i = 0; i < l.batch; ++i) {
    for (j = 0; j < l.groups; ++j) {
      //LOG_DEBUG("conv forward i: %d, j:%d",i,j);
      // float *a = l.weights + j*l.nweights/l.groups;
      // float *b = net.workspace;
      // float *c = l.output + (i*l.groups + j)*n*m;
      // float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

      if (l.size == 1) {
        // b=im;
        gemm_blocked(0, 0, m, n, k, 1, l.weights, j * l.nweights / l.groups, k,
                   net.input, (i*l.groups + j)*l.c/l.groups*l.h*l.w, n, 1, l.output, (i * l.groups + j) * n * m,
                   n);
      }
      else {
        im2col_cpu_blocked(
          net.input, (i * l.groups + j) * l.c / l.groups * l.h * l.w,
          l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, net.workspace, 0);
        //LOG_DEBUG("Base start index for C (output) is %d",(i * l.groups + j) * n * m)
        gemm_blocked(0, 0, m, n, k, 1, l.weights, j * l.nweights / l.groups, k,
                   net.workspace, 0, n, 1, l.output, (i * l.groups + j) * n * m,
                   n);
      }
      
      // gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }
  }
  //LOG_DEBUG("Ready for batch normalize!! goinh to batch nrom? %d",l.batch_normalize)

  if (l.batch_normalize) {
    forward_batchnorm_layer_blocked(l, net);
  } else {
    add_bias_blocked(l.output, l.biases, l.batch, l.n, l.out_h * l.out_w);
  }

  activate_array_blocked(l.output, l.outputs * l.batch, l.activation);
  // if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer_blocked(convolutional_layer_blocked l,
                                          network_blocked net) {
  
  /* {
    float i237_,i121_;
    BLOCK_ENGINE_INIT_FOR_LOOP(l.weights, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weights, weight_range);
    LOG_DEBUG("before backward 237 and 121 weights are: %0.10e, .. %0.10e\n",i237_,i121_);
  } */
  /* {
    float i237_,i121_;
    BLOCK_ENGINE_INIT_FOR_LOOP(l.weight_updates, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weight_updates, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weight_updates, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weight_updates, weight_range);
    LOG_DEBUG("before backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",i237_,i121_);
  } */
  int i, j;
  int m = l.n / l.groups;
  int n = l.size * l.size * l.c / l.groups;
  int k = l.out_w * l.out_h;
  
  gradient_array_blocked(l.output, l.outputs * l.batch, l.activation, l.delta);

  if (l.batch_normalize) {
    backward_batchnorm_layer_blocked(l, net);
  } else {
    backward_bias_blocked(l.bias_updates, l.delta, l.batch, l.n, k);
  }
  for (i = 0; i < l.batch; ++i) {
    for (j = 0; j < l.groups; ++j) {
      // float *a = l.delta + (i*l.groups + j)*m*k;
      // float *b = net.workspace;
      // float *c = l.weight_updates + j*l.nweights/l.groups;

      // float *im = net.input+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
      // float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
      //LOG_DEBUG("backward conv i:%d,j:%d\n",i,j)
      if(l.size == 1) {
        //b = im;
        gemm_blocked(0, 1, m, n, k, 1, l.delta, (i * l.groups + j) * m * k, k,
                   net.input, (i*l.groups + j)*l.c/l.groups*l.h*l.w, k, 1, l.weight_updates,
                   j * l.nweights / l.groups, n);
      }
      else {
        im2col_cpu_blocked(
          net.input, (i * l.groups + j) * l.c / l.groups * l.h * l.w,
          l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, net.workspace, 0);
        gemm_blocked(0, 1, m, n, k, 1, l.delta, (i * l.groups + j) * m * k, k,
                   net.workspace, 0, k, 1, l.weight_updates,
                   j * l.nweights / l.groups, n);
      }
      // gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

      if (net.delta) {
        // a = l.weights + j*l.nweights/l.groups;
        // b = l.delta + (i*l.groups + j)*m*k;
        // c = net.workspace;
        //LOG_DEBUG("Conv back net delta\n")
        if (l.size == 1) {
          // c = imd;
          gemm_blocked(1, 0, n, k, m, 1, l.weights, j * l.nweights / l.groups, n,
                     l.delta, (i * l.groups + j) * m * k, k, 0, net.delta,
                     (i*l.groups + j)*l.c/l.groups*l.h*l.w, k);
        }
        else {
          gemm_blocked(1, 0, n, k, m, 1, l.weights, j * l.nweights / l.groups, n,
                     l.delta, (i * l.groups + j) * m * k, k, 0, net.workspace,
                     0, k);
        }
        
        //LOG_DEBUG("Done with Conv back net delta\n")
        if (l.size != 1) {
          col2im_cpu_blocked(net.workspace, 0, l.c / l.groups, l.h, l.w, l.size,
                           l.stride, l.pad, net.delta,
                           (i * l.groups + j) * l.c / l.groups * l.h * l.w);
          //LOG_DEBUG("Done with Col2im_cpu_blocked back net delta\n")
        }
      }
    }
  }
  /* {
    float i237_,i121_;
    BLOCK_ENGINE_INIT_FOR_LOOP(l.weights, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weights, weight_range);
    LOG_DEBUG("after backward 237 and 121 weights are: %0.10e, .. %0.10e\n",i237_,i121_);
    } */
  /* {
    float i237_,i121_;
    BLOCK_ENGINE_INIT_FOR_LOOP(l.weight_updates, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weight_updates, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weight_updates, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weight_updates, weight_range);
    LOG_DEBUG("after backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",i237_,i121_);
  } */
}

void update_convolutional_layer_blocked(convolutional_layer_blocked l,
                                        update_args a) {

  float learning_rate = a.learning_rate * l.learning_rate_scale;
  float momentum = a.momentum;
  float decay = a.decay;
  int batch = a.batch;
  /* LOG_DEBUG("lr:%f,moment:%f,decay:%f,batch:%d total weights:%d\n",learning_rate,momentum,decay,batch,l.nweights)
  {
      float i237_,i121_;
      BLOCK_ENGINE_INIT_FOR_LOOP(l.weights, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weights, weight_range);
    LOG_DEBUG("before update 237 and 121 weights are: %0.10e, .. %0.10e\n",i237_,i121_);
  } */

  axpy_cpu_blocked(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
  scal_cpu_blocked(l.n, momentum, l.bias_updates, 1);

  if (l.scales) {
    axpy_cpu_blocked(l.n, learning_rate / batch, l.scale_updates, 1, l.scales,
                     1);
    scal_cpu_blocked(l.n, momentum, l.scale_updates, 1);
  }

  /* {
      float i237_,i121_;
      BLOCK_ENGINE_INIT_FOR_LOOP(l.weight_updates, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weight_updates, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weight_updates, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weight_updates, weight_range);
    LOG_DEBUG("before update 237 and 121 weight updates are: %0.10e, .. %0.10e\n",i237_,i121_);
  } */
  axpy_cpu_blocked(l.nweights, -decay * batch, l.weights, 1, l.weight_updates,1);
  /* {
      float i237_,i121_;
      BLOCK_ENGINE_INIT_FOR_LOOP(l.weights, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weights, weight_range);
    LOG_DEBUG("here update 237 and 121 weights are: %0.10e, .. %0.10e\n",i237_,i121_);
  } */
  /* {
      float i237_,i121_;
      BLOCK_ENGINE_INIT_FOR_LOOP(l.weight_updates, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weight_updates, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weight_updates, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weight_updates, weight_range);
    LOG_DEBUG("here update 237 and 121 weight updates are: %0.10e, .. %0.10e\n",i237_,i121_);
  } */
  axpy_cpu_blocked(l.nweights, learning_rate / batch, l.weight_updates, 1,
                   l.weights, 1);
  scal_cpu_blocked(l.nweights, momentum, l.weight_updates, 1);

  /* {
    float i237_,i121_;
    BLOCK_ENGINE_INIT_FOR_LOOP(l.weights, weight_range, weight_range_ptr, float)
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 237)
      i237_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    {
      BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(l.weights, weight_range,
                                          weight_range_ptr, false, weight_index, 121)
      i121_ = *(weight_range_ptr + weight_index - weight_range.block_requested_ind);
    }
    BLOCK_ENGINE_LAST_UNLOCK(l.weights, weight_range);
    LOG_DEBUG("after update 237 and 121 weights are: %0.10e, .. %0.10e\n",i237_,i121_);
  } */
}
#endif
