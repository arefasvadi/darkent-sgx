#include "cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "seg")==0) return SEG;
    if (strcmp(s, "sse")==0) return SSE;
    if (strcmp(s, "masked")==0) return MASKED;
    if (strcmp(s, "smooth")==0) return SMOOTH;
    if (strcmp(s, "L1")==0) return L1;
    if (strcmp(s, "wgan")==0) return WGAN;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SEG:
            return "seg";
        case SSE:
            return "sse";
        case MASKED:
            return "masked";
        case SMOOTH:
            return "smooth";
        case L1:
            return "L1";
        case WGAN:
            return "wgan";
    }
    return "sse";
}

#ifndef USE_SGX_LAYERWISE
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    fprintf(stderr, "cost                                           %4d\n",  inputs);
    cost_layer l = {};
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward = forward_cost_layer;
    l.backward = backward_cost_layer;
    #ifdef GPU
    l.forward_gpu = forward_cost_layer_gpu;
    l.backward_gpu = backward_cost_layer_gpu;

    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
    l.output_gpu = cuda_make_array(l.delta, inputs*batch);
    #endif
    return l;
}
#endif

#ifndef USE_SGX_LAYERWISE
void resize_cost_layer(cost_layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;
    l->delta = (float*)realloc(l->delta, inputs*l->batch*sizeof(float));
    l->output = (float*)realloc(l->output, inputs*l->batch*sizeof(float));
#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
#endif
}
#endif

#ifndef USE_SGX_LAYERWISE
void forward_cost_layer(cost_layer l, network net)
{
    if (!net.truth) return;
    if(l.cost_type == MASKED){
        int i;
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
        }
    }
    if(l.cost_type == SMOOTH){
        smooth_l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }else if(l.cost_type == L1){
        l1_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    } else {
        l2_cpu(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_cost_layer(const cost_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}
#endif

#ifdef GPU

void pull_cost_layer(cost_layer l)
{
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void push_cost_layer(cost_layer l)
{
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

int float_abs_compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    if(fa < 0) fa = -fa;
    float fb = *(const float*) b;
    if(fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}

void forward_cost_layer_gpu(cost_layer l, network net)
{
    if (!net.truth) return;
    if(l.smooth){
        scal_gpu(l.batch*l.inputs, (1-l.smooth), net.truth_gpu, 1);
        add_gpu(l.batch*l.inputs, l.smooth * 1./l.inputs, net.truth_gpu, 1);
    }

    if(l.cost_type == SMOOTH){
        smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == L1){
        l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == WGAN){
        wgan_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else {
        l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    }

    if (l.cost_type == SEG && l.noobject_scale != 1) {
        scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
        scale_mask_gpu(l.batch*l.inputs, l.output_gpu, 0, net.truth_gpu, l.noobject_scale);
    }
    if (l.cost_type == MASKED) {
        mask_gpu(l.batch*l.inputs, net.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
    }

    if(l.ratio){
        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
        qsort(l.delta, l.batch*l.inputs, sizeof(float), float_abs_compare);
        int n = (1-l.ratio) * l.batch*l.inputs;
        float thresh = l.delta[n];
        thresh = 0;
        printf("%f\n", thresh);
        supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
    }

    if(l.thresh){
        supp_gpu(l.batch*l.inputs, l.thresh*1./l.inputs, l.delta_gpu, 1);
    }

    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

void backward_cost_layer_gpu(const cost_layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    //fprintf(stderr, "cost                                           %4d\n",  inputs);
    cost_layer l = {};
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    //l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*batch);
    //l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.output = l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*batch);
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward = forward_cost_layer;
    l.backward = backward_cost_layer;
    return l;
}

void forward_cost_layer(cost_layer l, network net)
{
    if (!net.truth) return;
    if(l.cost_type == MASKED){
        LOG_ERROR("Should not reach here!\n");
        abort();
        /* int i;
        for(i = 0; i < l.batch*l.inputs; ++i){
            if(net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
        } */
    }
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto net_truth = net.truth->getItemsInRange(0, net.truth->getBufferSize());
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());

    if(l.cost_type == SMOOTH){
        smooth_l1_cpu(l.batch*l.inputs, &net_input[0], &net_truth[0], &l_delta[0], &l_output[0]);
    }else if(l.cost_type == L1){
        l1_cpu(l.batch*l.inputs, &net_input[0], &net_truth[0], &l_delta[0], &l_output[0]);
    } else {
        l2_cpu(l.batch*l.inputs, &net_input[0], &net_truth[0], &l_delta[0], &l_output[0]);
    }
    l.cost[0] = sum_array(&l_output[0], l.batch*l.inputs);
    l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
}

void backward_cost_layer(const cost_layer l, network net)
{
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    axpy_cpu(l.batch*l.inputs, l.scale, &l_delta[0], 1, &net_delta[0], 1);
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
cost_layer_blocked make_cost_layer_blocked(int batch, int inputs, COST_TYPE type, float scale) {
    cost_layer_blocked l = {};
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = type;
    // l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({inputs*batch});
    // l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.output = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({inputs*batch});
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward_blocked = forward_cost_layer_blocked;
    l.backward_blocked = backward_cost_layer_blocked;
    
    return l;
}
void forward_cost_layer_blocked(const cost_layer_blocked l, network_blocked net) {
    if (!net.truth) return;
    if(l.cost_type == MASKED){
        int i;
        BLOCK_ENGINE_INIT_FOR_LOOP(net.truth, net_truth_valid_range, net_truth_block_val_ptr, float)
        BLOCK_ENGINE_INIT_FOR_LOOP(net.input, net_input_valid_range, net_input_block_val_ptr, float)
        for(i = 0; i < l.batch*l.inputs; ++i){
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.truth, net_truth_valid_range, net_truth_block_val_ptr, false, net_truth_current_index, i)
            // if(net.truth[i] == SECRET_NUM) {
            if(*(net_truth_block_val_ptr+net_truth_current_index-net_truth_valid_range.block_requested_ind) == SECRET_NUM) {
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(net.input, net_input_valid_range, net_input_block_val_ptr, true, net_input_current_index, i)
                // net.input[i] = SECRET_NUM;
                *(net_input_block_val_ptr+net_input_current_index-net_input_valid_range.block_requested_ind) = SECRET_NUM;
            }
        }
        BLOCK_ENGINE_LAST_UNLOCK(net.truth, net_truth_valid_range)
        BLOCK_ENGINE_LAST_UNLOCK(net.input, net_input_valid_range)
    }
    if(l.cost_type == SMOOTH){
        smooth_l1_cpu_blocked(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }else if(l.cost_type == L1){
        l1_cpu_blocked(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    } else {
        l2_cpu_blocked(l.batch*l.inputs, net.input, net.truth, l.delta, l.output);
    }
    l.cost[0] = sum_array_blocked(l.output, l.batch*l.inputs,0);
}
void backward_cost_layer_blocked(const cost_layer_blocked l, network_blocked net) {
    axpy_cpu_blocked(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}
#endif

