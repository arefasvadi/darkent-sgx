#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

//extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
//}

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    #if !defined(SGX_VERIFIES)
    cuda_random(layer.rand_gpu, size);
    #else
    // fill gpu
    for(int i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform(*(layer.layer_rng),0, 1);
    }
    // if (net.index == 12) {
    //     print_array(layer.rand,size,0,"GPU dropout rand vals");
    // }
    
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    #endif
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}
