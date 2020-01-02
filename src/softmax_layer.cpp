#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string>

#ifndef USE_SGX_LAYERWISE
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float*)calloc(inputs*batch, sizeof(float));
    l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    #if defined(SGX_VERIFIES) && defined(GPU)
    l.forward_gpu_sgx_verifies = forward_softmax_gpu_sgx_verifies_fbv; 
    l.backward_gpu_sgx_verifies = backward_softmax_gpu_sgx_verifies_fbv;
    // l.update_gpu_sgx_verifies = update_softmax_gpu_sgx_verifies_fbv;
    // l.create_snapshot_for_sgx = create_softmax_snapshot_for_sgx_fbv;
    #endif 
    #ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}
#endif

#ifndef USE_SGX_LAYERWISE
void forward_softmax_layer(softmax_layer& l, network& net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
        /* std::string temp_str = "PURE SGX Softmax Vals\n";
        for (int i=0;i<l.batch;++i) {
            temp_str = temp_str + "batch " + std::to_string(i) + " ->: ";
            for (int j=0;j<l.outputs;++j) {
                temp_str = temp_str + std::to_string(l.output[i*l.outputs + j]) +  ", ";
            }
            temp_str = temp_str + "\n";
        }
        LOG_DEBUG("%s",temp_str.c_str()); */
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
#endif

#ifndef USE_SGX_LAYERWISE
void backward_softmax_layer(softmax_layer& l, network& net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}
#endif

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
        /*
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
            count += group_size;
        }
        */
    } else {
        cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
        print_array(net.input, 100, 0, "GPU: before softamx input");
        if(l.spatial){
            softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
        }else{
            softmax_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
        }
    }
    if(net.truth && !l.noloss){
        softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
        if(l.softmax_tree){
            mask_gpu(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
            mask_gpu(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
        }
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
    pull_softmax_layer_output(l);
    print_array(l.output, l.batch*l.outputs, 0, "GPU Softmax Vals");
    // std::string temp_str = std::string("GPU Softmax Vals with batch size: ")+std::to_string(l.batch)+std::string("\n");
    // for (int i=0;i<l.batch;++i) {
    //     temp_str = temp_str + "batch " + std::to_string(i) + " ->: ";
    //     for (int j=0;j<l.outputs;++j) {
    //         temp_str = temp_str + std::to_string(l.output[i*l.outputs + j]) +  ", ";
    //     }
    //     temp_str = temp_str + "\n";
    // }
    // LOG_DEBUG("%s",temp_str.c_str());
}

void backward_softmax_layer_gpu(const softmax_layer layer, network net)
{
    axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif

#if defined(SGX_VERIFIES) && defined(GPU)
void
forward_softmax_gpu_sgx_verifies_fbv(softmax_layer l, network net) {
  forward_softmax_layer_gpu(l, net);
}
void
backward_softmax_gpu_sgx_verifies_fbv(softmax_layer l, network net) {
  backward_softmax_layer_gpu(l, net);
}

// void
// update_softmax_gpu_sgx_verifies_fbv(softmax_layer l, update_args a) {
//   update_softmax_layer_gpu(l, a);
// }

// void
// create_softmax_snapshot_for_sgx_fbv(struct layer &  l,
//                                           struct network &net,
//                                           uint8_t **      out,
//                                           uint8_t **      sha256_out) {
//   if (gpu_index >= 0) {
//     pull_convolutional_layer(l);
//   }
//   int total_bytes = (l.nbiases + l.nweights) * sizeof(float);
//   if (l.batch_normalize) {
//     total_bytes += (3 * l.nbiases) * sizeof(float);
//   }
//   size_t buff_ind = 0;
//   *out            = new uint8_t[total_bytes];
//   *sha256_out     = new uint8_t[SHA256_DIGEST_LENGTH];

//   std::memcpy((*out + buff_ind), l.biases, l.nbiases * sizeof(float));
//   buff_ind += l.nbiases * sizeof(float);
//   std::memcpy((*out + buff_ind), l.weights, l.nweights * sizeof(float));
//   buff_ind += l.nweights * sizeof(float);

//   if (l.batch_normalize) {
//     std::memcpy((*out + buff_ind), l.scales, l.nbiases * sizeof(float));
//     buff_ind += l.nbiases * sizeof(float);
//     std::memcpy((*out + buff_ind), l.rolling_mean, l.nbiases * sizeof(float));
//     buff_ind += l.nbiases * sizeof(float);
//     std::memcpy(
//         (*out + buff_ind), l.rolling_variance, l.nbiases * sizeof(float));
//     buff_ind += l.nbiases * sizeof(float);
//   }
//   if (buff_ind != total_bytes) {
//     LOG_ERROR("size mismatch\n")
//     abort();
//   }
//   gen_sha256(*out, total_bytes, *sha256_out);
// }
#endif

#if defined (USE_SGX) && defined (USE_SGX_LAYERWISE)
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    //l.loss = (float*)calloc(inputs*batch, sizeof(float));
    l.loss = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*batch);
    //l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*batch);
    //l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*batch);
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    return l;
}

void forward_softmax_layer(softmax_layer& l, network& net)
{
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto l_loss = l.loss->getItemsInRange(0, l.loss->getBufferSize());
    if(l.softmax_tree){
        auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(&net_input[0] + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, &l_output[0] + count);
            count += group_size;
        }
    } else {
        auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
        print_array(&net_input[0],100,0,"SGX before softmax input");
        softmax_cpu(&net_input[0], l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, &l_output[0]);
    }

    if(net.truth && !l.noloss){
        auto net_truth = net.truth->getItemsInRange(0, net.truth->getBufferSize());
        softmax_x_ent_cpu(l.batch*l.inputs, &l_output[0], &net_truth[0], &l_delta[0], &l_loss[0]);
        l.cost[0] = sum_array(&l_loss[0], l.batch*l.inputs);
    }
    print_array(&l_output[0], l.batch*l.outputs, 0, "SGX Softmax Vals");
    // std::string temp_str = std::string("LAYERWISE Softmax Vals with batch size: ")+std::to_string(l.batch)+std::string("\n");
    // for (int i=0;i<l.batch;++i) {
    //     temp_str = temp_str + "batch " + std::to_string(i) + " ->: ";
    //     for (int j=0;j<l.outputs;++j) {
    //         temp_str = temp_str + std::to_string(l_output[i*l.outputs + j]) +  ", ";
    //     }
    //     temp_str = temp_str + "\n";
    // }
    // LOG_DEBUG("%s",temp_str.c_str());
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
    l.loss->setItemsInRange(0, l.loss->getBufferSize(),l_loss);
}

void backward_softmax_layer(softmax_layer& l, network& net)
{
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    axpy_cpu(l.inputs*l.batch, 1, &l_delta[0], 1, &net_delta[0], 1);
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
}
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
softmax_layer_blocked make_softmax_layer_blocked(int batch, int inputs, int groups) {
    assert(inputs%groups == 0);
    // fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer_blocked l = {};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    // l.loss = (float*)calloc(inputs*batch, sizeof(float));
    l.loss = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({inputs*batch});
    // l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.output = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({inputs*batch});
    // l.delta = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({inputs*batch});
    l.cost = (float*)calloc(1, sizeof(float));

    l.forward_blocked = forward_softmax_layer_blocked;
    l.backward_blocked = backward_softmax_layer_blocked;
    return l;
}
void forward_softmax_layer_blocked(const softmax_layer_blocked l, network_blocked net) {
    if(l.softmax_tree){
        /* int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        } */
        LOG_ERROR("Softmax tree with blocking has not been implemented!")
        abort();
    } else {
        softmax_cpu_blocked(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu_blocked(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array_blocked(l.loss, l.batch*l.inputs,0);
    }
}
void backward_softmax_layer_blocked(const softmax_layer_blocked l, network_blocked net) {
    axpy_cpu_blocked(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}
#endif