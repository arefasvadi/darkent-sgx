#include "sgxlwfit.h"
#include "rand/PRNG.h"
#include "../activations.h"
#include "../gemm.h"
#include "../col2im.h"
#include "../im2col.h"
#include "../blas.h"
#include "../option_list.h"


convolutional_layer make_convolutional_layer(int batch, int h, int w, int c,
                                             int n, int groups, int size,
                                             int stride, int padding,
                                             ACTIVATION activation,
                                             int batch_normalize, int binary,
                                             int xnor, int adam,PRNG& net_layer_rng_deriver) {
  int i,j;
  convolutional_layer l = {};
  l.type = CONVOLUTIONAL;
  l.layer_rng = std::make_shared<PRNG>(generate_random_seed_from(net_layer_rng_deriver));
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
  if (global_training) {
    l.weight_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(c / groups * n * size * size);
    l.bkwrd_weight_delta_rhs = (double*)calloc(l.n/l.groups,sizeof(double));
    l.bkwrd_weight_delta_rand = (float*)calloc(c / groups * size * size,sizeof(float));
    l.frwd_outs_rand = (float*)calloc(l.n/l.groups,sizeof(float));
    l.frwd_outs_rhs = (float*)calloc(c / groups * size * size,sizeof(float));
    l.bkwrd_input_delta_rand = (float*)calloc(c / groups * size * size,sizeof(float));
    l.bkwrd_input_delta_rhs = (float*)calloc(l.n/l.groups,sizeof(float));
  }

  //l.biases = (float *)calloc(n, sizeof(float));
  l.biases = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
  //l.bias_updates = (float *)calloc(n, sizeof(float));
  if (global_training) l.bias_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);

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
    for (i = 0; i < l.nweights; ++i) {
      // ws[i] = scale * rand_normal();
      ws[i] = scale * rand_normal(*(l.layer_rng));
    }
      
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
  if (global_training) l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.batch * l.outputs);

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
    if (global_training) l.scale_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);

    if (global_training) {
      auto scs = l.scales->getItemsInRange(0, l.scales->getBufferSize());
      for (i = 0; i < n; ++i) {
        scs[i] = 1;
      }
      l.scales->setItemsInRange(0, n,scs);
    }

    if (global_training) {
      //l.mean = (float *)calloc(n, sizeof(float));
      l.mean = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
      //l.variance = (float *)calloc(n, sizeof(float));
      l.variance = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);

      //l.mean_delta = (float *)calloc(n, sizeof(float));
      l.mean_delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
      //l.variance_delta = (float *)calloc(n, sizeof(float));
      l.variance_delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);

      //l.x = (float *)calloc(l.batch * l.outputs, sizeof(float));
      l.x = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.batch * l.outputs);
      //l.x_norm = (float *)calloc(l.batch * l.outputs, sizeof(float));
      l.x_norm = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(l.batch * l.outputs);
    }
    //l.rolling_mean = (float *)calloc(n, sizeof(float));
    l.rolling_mean = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    //l.rolling_variance = (float *)calloc(n, sizeof(float));
    l.rolling_variance = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(n);
    
  }
   if (global_training) {
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
  }

  l.workspace_size = get_workspace_size(l);
  l.activation = activation;

  uint64_t consumed_space_wo_workspace_bytes = (l.inputs + l.outputs + l.nweights +l.nbiases)*sizeof(float);
  l.enclave_layered_batch = (SGX_LAYERWISE_MAX_LAYER_SIZE - (consumed_space_wo_workspace_bytes));
  if (l.enclave_layered_batch <=0) {
      LOG_ERROR("remaining space is negative!!!!\n");
      abort();
  }
  l.enclave_layered_batch = 
      (l.enclave_layered_batch / (l.workspace_size / ((l.c/l.groups))));

  if (l.enclave_layered_batch <= 0) {
      LOG_ERROR("remaining space is not enough for even a single batch!!!!\n");
      abort();
  }
  if (l.enclave_layered_batch > l.c / l.groups) {
      l.enclave_layered_batch = l.c / l.groups;
  }

  fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void convolutional_get_MM_output_left_compare(layer& l, network& net,float* rand_vec,
                                              float* rand_right,float* result_out) {
  int m = l.n / l.groups;
  int k = l.size * l.size * l.c / l.groups;
  int n = l.out_w * l.out_h;
  std::vector<float> rand_left(n,0);
  gemm
  //primitive_based_sgemv
  (0,0,1,n,m,1,
  rand_vec,m,
  result_out,n,
  1,
  rand_left.data(),n);
  
  //if (rand_right.size()!=rand_left.size()) {
  //      LOG_ERROR("size mismatch\n")
  //      abort();
  //}
  for (int i=0;i<rand_left.size();++i) {
      if (std::isnormal(rand_left[i]) && std::isnormal(rand_right[i]) && std::fabs(rand_right[i]-rand_left[i]) > 0.000001f) {
          LOG_ERROR("rand verify value mismatch at index %d for MM output with values (left,right):(%f,%f)\n",
              i,rand_right[i],rand_left[i])
          abort();
      }
  }
}

void forward_convolutional_layer_verifies_frbmmv(layer& l, network& net) {
  SET_START_TIMING(SGX_TIMING_FORWARD_CONV)
  if(l.binary){
    LOG_ERROR("Binary feature not implemented!\n");
    abort();
  }
  if (l.xnor) {
    LOG_ERROR("XNOR feature not implemented!\n");
    abort();
  }
  
  int q = (l.c/l.groups) / l.enclave_layered_batch;
  int r = (l.c/l.groups) % l.enclave_layered_batch;
  
  auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
  auto n_input = net.input->getItemsInRange(0, net.input->getBufferSize());
  int i, j;
  int m = l.n/l.groups;
  int k = l.size*l.size*l.c/l.groups;
  int n = l.out_w*l.out_h;

  #ifndef SGX_CONV_BATCH_PRECOMPUTE_VERIFY
  auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
  #endif
  
  auto n_workspace = l.size != 1 ? std::unique_ptr<float[]>(
                         new float[l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size])
                                 : std::unique_ptr<float[]>(nullptr);
  std::vector<float> mm_randomized_output_right(n,0);
  for(i = 0; i < l.batch; ++i){
    for(j = 0; j < l.groups; ++j){
      #ifndef SGX_CONV_BATCH_PRECOMPUTE_VERIFY
      for (int jj=0;jj<l.n/l.groups;++jj){
        l.frwd_outs_rand[jj] = sgx_root_rng->getRandomFloat(std::numeric_limits<float>::min(),
                    std::numeric_limits<float>::max());
      }
      std::memset(l.frwd_outs_rhs, 0, l.c/l.groups*l.size*l.size*sizeof(float));
      gemm(0,0,
        1,l.c/l.groups*l.size*l.size,l.n/l.groups,1,
        l.frwd_outs_rand,l.n/l.groups,l_weights.get(),l.c/l.groups*l.size*l.size,1,
        l.frwd_outs_rhs,l.c/l.groups*l.size*l.size);
      #endif
      std::memset(mm_randomized_output_right.data(), 0, 
        mm_randomized_output_right.size()*sizeof(float));
      //float *a = &l_weights[0] + j * l.nweights / l.groups;
      float *b = nullptr; //&n_workspace[0];
      float *c = &l_output[0] + (i * l.groups + j) * n * m;
      float *im =  &n_input[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
      if (l.size == 1) {
        SET_START_TIMING(SGX_TIMING_FORWARD_CONV_OUT_KEQ_1)
        b = im;
        #ifndef SGX_USE_BLASFEO_GEMV
        gemm
        //primitive_based_sgemv
        (0,0,1,n,k,1,
        l.frwd_outs_rhs,k,
        b,n,
        1,
        mm_randomized_output_right.data(),n);
        #else
        //blasfeo_gemv_impl(0, k, n, 1.0, b, n, l.frwd_outs_rhs, 1, 1.0, mm_randomized_output_right.data(), 1);
        #endif
        SET_FINISH_TIMING(SGX_TIMING_FORWARD_CONV_OUT_KEQ_1)
      }
      else {
        SET_START_TIMING(SGX_TIMING_FORWARD_CONV_OUT_KGT_1)
        for (int chan = 0; chan < q; chan++) {
          #ifndef SGX_FAST_TWEAKS_NO_MEMSET
          std::memset(&n_workspace[0], 0, sizeof(float)*l.enclave_layered_batch*l.out_h*l.out_w*l.size*l.size);
          #endif
          b = &n_workspace[0];
          im2col_cpu(im+(chan*l.enclave_layered_batch*l.h*l.w), l.enclave_layered_batch, l.h, l.w, l.size, l.stride, l.pad, b);
          gemm
          //primitive_based_sgemv
          (0,0,1,n,l.size*l.size*l.enclave_layered_batch/l.groups,1,
          l.frwd_outs_rhs+(chan*l.size*l.size*l.enclave_layered_batch/l.groups),l.size*l.size*l.enclave_layered_batch/l.groups,
          b,n,
          1,
          mm_randomized_output_right.data(),n);
        }
        if (r > 0) {
          #ifndef SGX_FAST_TWEAKS_NO_MEMSET
          std::memset(&n_workspace[0], 0, 
            sizeof(float)*l.enclave_layered_batch*l.out_h*l.out_w*l.size*l.size);
          #endif
          b = &n_workspace[0];
          im2col_cpu(im+(q*l.enclave_layered_batch*l.h*l.w), r, l.h, l.w, l.size, l.stride, l.pad, b);
          gemm
          //primitive_based_sgemv
          (0,0,1,n,l.size*l.size*r/l.groups,1,
          l.frwd_outs_rhs+(q*l.size*l.size*l.enclave_layered_batch),l.size*l.size*r/l.groups,
          b,n,
          1,
          mm_randomized_output_right.data(),n);
        }
      }
      convolutional_get_MM_output_left_compare(l, net,l.frwd_outs_rand, 
        mm_randomized_output_right.data(),c);
      SET_FINISH_TIMING(SGX_TIMING_FORWARD_CONV_OUT_KGT_1)
    }
  }
  //LOG_DEBUG("Ready for batch normalize!! goinh to batch nrom? %d",l.batch_normalize)
  // print_array(&l_output[0],100,0,"sgx after mult, before batchnorm forward input");
  if (l.batch_normalize) {
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    forward_batchnorm_layer(l, net);
    l_output = l.output->getItemsInRange(0, l.output->getBufferSize());

  } else {
    auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
    add_bias(&l_output[0], &l_biases[0], l.batch, l.n, l.out_h * l.out_w);
  }
  // print_array(&l_output[0],100,0,"sgx after mult, batchnorm before activation forward input");
  activate_array(&l_output[0], l.outputs * l.batch, l.activation);
  l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
  /* if (l.binary || l.xnor)
    swap_binary(&l); */
  SET_FINISH_TIMING(SGX_TIMING_FORWARD_CONV)
}

void forward_convolutional_layer(convolutional_layer& l, network& net)
{
  if (net.verf_type == verf_variations_t::FRBRMMV) {
    forward_convolutional_layer_verifies_frbmmv(l, net);
    return;
  }
  SET_START_TIMING(SGX_TIMING_FORWARD_CONV)
  int i, j;
  SET_START_TIMING("SGX Conv loading weights")
  auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
  SET_FINISH_TIMING("SGX Conv loading weights")
  auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
  //auto n_workspace = net.workspace->getItemsInRange(0, net.workspace->getBufferSize());
  auto n_input = net.input->getItemsInRange(0, net.input->getBufferSize());
  //LOG_DEBUG("before forward ID: %u, 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights->getID(),l_weights[237],l_weights[121])
  // print_array(&n_input[0],100,0,"sgx before conv forward input");
  // print_array(&l_weights[0],l.nweights,0,"sgx conv forward weights");

  fill_cpu(l.outputs * l.batch, 0, &l_output[0], 1);
  int q = (l.c/l.groups) / l.enclave_layered_batch;
  int r = (l.c/l.groups) % l.enclave_layered_batch;
  // LOG_DEBUG("q:%d,r=%d, enclave_channel_limit:%d\n",q,r,l.enclave_layered_batch)

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

  auto n_workspace = l.size != 1 ? std::unique_ptr<float[]>(
                         new float[l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size])
                                 : std::unique_ptr<float[]>(nullptr);
  // LOG_DEBUG("begining conv with parameters outputs:%d, batch:%d,groups:%d,
  // m:%d, k:%d, n:%d,
  // out_w:%d,out_h:%d,out_c:%d\n",l.outputs,l.batch,l.groups,m,k,n,l.out_w,l.out_h,l.out_c);
  for (i = 0; i < l.batch; ++i) {
    for (j = 0; j < l.groups; ++j) {
      float *a = &l_weights[0] + j * l.nweights / l.groups;
      float *b = nullptr; //&n_workspace[0];
      float *c = &l_output[0] + (i * l.groups + j) * n * m;
      float *im =  &n_input[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
      if (l.size == 1) {
          SET_START_TIMING(SGX_TIMING_FORWARD_CONV_OUT_KEQ_1)
          b = im;
          gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
          SET_FINISH_TIMING(SGX_TIMING_FORWARD_CONV_OUT_KEQ_1)
      } else {
          SET_START_TIMING(SGX_TIMING_FORWARD_CONV_OUT_KGT_1)
          for (int chan = 0; chan < q; chan++) {
            #ifndef SGX_FAST_TWEAKS_NO_MEMSET
            std::memset(&n_workspace[0], 0, sizeof(float)*l.enclave_layered_batch*l.out_h*l.out_w*l.size*l.size);
            #endif
            b = &n_workspace[0];
            //im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            im2col_cpu(im+(chan*l.enclave_layered_batch*l.h*l.w), l.enclave_layered_batch, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm(0, 0, m, n, (l.enclave_layered_batch*l.size*l.size), 1, a+(chan*l.enclave_layered_batch*l.size*l.size), k, b, n, 1, c, n);  // k is changed
          }
          if (r > 0) {
            b = &n_workspace[0];
            #ifndef SGX_FAST_TWEAKS_NO_MEMSET
            std::memset(&n_workspace[0], 0, sizeof(float)*l.enclave_layered_batch*l.out_h*l.out_w*l.size*l.size);
            #endif
            im2col_cpu(im+(q*l.enclave_layered_batch*l.h*l.w), r, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm(0, 0, m, n, (r*l.size*l.size), 1, a+(q*l.enclave_layered_batch*l.size*l.size), k, b, n, 1, c, n);  // k is changed
          }
          SET_FINISH_TIMING(SGX_TIMING_FORWARD_CONV_OUT_KGT_1)
          //LOG_DEBUG("Base start index for C (output) is %d",(i * l.groups + j) * n * m)
          //gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
      }
    }
  }
  //LOG_DEBUG("Ready for batch normalize!! goinh to batch nrom? %d",l.batch_normalize)
  // print_array(&l_output[0],100,0,"sgx after mult, before batchnorm forward input");
  if (l.batch_normalize) {
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    forward_batchnorm_layer(l, net);
    l_output = l.output->getItemsInRange(0, l.output->getBufferSize());

  } else {
    auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
    add_bias(&l_output[0], &l_biases[0], l.batch, l.n, l.out_h * l.out_w);
  }
  // print_array(&l_output[0],100,0,"sgx after mult, batchnorm before activation forward input");
  activate_array(&l_output[0], l.outputs * l.batch, l.activation);
  l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
  /* if (l.binary || l.xnor)
    swap_binary(&l); */
  SET_FINISH_TIMING(SGX_TIMING_FORWARD_CONV)
}

void backward_convolutional_layer(convolutional_layer& l, network& net) {
    if (net.verf_type == verf_variations_t::FRBRMMV) {
        backward_convolutional_layer_verifies_frbmmv(l,net);
        return;
    }
    //LOG_DEBUG("before backward 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("before backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    SET_START_TIMING(SGX_TIMING_BACKWARD_CONV)
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;
    
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    auto net_delta  = net.delta ? net.delta->getItemsInRange(0, net.delta->getBufferSize()):std::unique_ptr<float[]>(nullptr);

    int q = (l.c/l.groups) / l.enclave_layered_batch;
    int r = (l.c/l.groups) % l.enclave_layered_batch;

    {
      auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
      gradient_array(&l_output[0], l.outputs*l.batch, l.activation, &l_delta[0]);
    }
    // if (net.index == 7) {
    //   std::string text = std::string("SGX convolution delta after gradient on activation layer ") + std::to_string(net.index);
    //   print_array(&l_delta[0],2*l.outputs,0,text.c_str());
    // }
    if(l.batch_normalize){
        // auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        // print_array(&l_bias_updates[0], l.nbiases, 0, "SGX convolutional layer bias updates bn before");
        l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
        backward_batchnorm_layer(l, net);
        // auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        // print_array(&l_bias_updates[0], l.nbiases, 0, "SGX convolutional layer bias updates");
        l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    } else {
      auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
      // print_array(&l_bias_updates[0], l.nbiases, 0, "SGX convolutional layer bias updates before");
      backward_bias(&l_bias_updates[0], &l_delta[0], l.batch, l.n, k);
      // print_array(&l_bias_updates[0], l.nbiases, 0, "SGX convolutional layer bias updates");
      l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(), l_bias_updates);
    }

    auto net_workspace = l.size != 1 ? std::unique_ptr<float[]>(
                         new float[l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size])
                                 : std::unique_ptr<float[]>(nullptr);
    auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
    auto l_weight_updates = l.weight_updates->getItemsInRange(0, l.weight_updates->getBufferSize());
    //auto net_workspace = net.workspace->getItemsInRange(0, net.workspace->getBufferSize());
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    if (net.delta == nullptr) {
      auto del_ptr = l_weights.release();
      delete[] del_ptr;
    } 
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
          float *a = &l_delta[0] + (i*l.groups + j)*m*k;
          float *b = nullptr;   //&net_workspace[0];
          float *c = &l_weight_updates[0] + j*l.nweights/l.groups;
          float *im  = &net_input[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

          if(l.size == 1){
            SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KEQ_1)
            b = im;
            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
            SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KEQ_1)
          } 
          else {
              SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KGT_1)
              for (int chan = 0; chan < q;++chan) {
                #ifndef SGX_FAST_TWEAKS_NO_MEMSET
                std::memset(&net_workspace[0], 0, sizeof(float)*l.enclave_layered_batch * l.out_h * l.out_w* l.size * l.size);
                #endif
                b = &net_workspace[0];
                im2col_cpu(im+chan*l.enclave_layered_batch*(l.h*l.w), l.enclave_layered_batch, l.h, l.w, l.size, l.stride, l.pad, b);
                //im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
                gemm(0,1,m,(l.enclave_layered_batch*l.size*l.size),k,1,a,k,b,k,1,c+(chan*l.enclave_layered_batch*l.size*l.size),n);
                //gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
              }
              if (r > 0) {
                #ifndef SGX_FAST_TWEAKS_NO_MEMSET
                std::memset(&net_workspace[0], 0, sizeof(float)*l.enclave_layered_batch * l.out_h * l.out_w* l.size * l.size);
                #endif
                b = &net_workspace[0];
                im2col_cpu(im+q*l.enclave_layered_batch*(l.h*l.w), r, l.h, l.w, l.size, l.stride, l.pad, b);
                gemm(0,1,m,(r*l.size*l.size),k,1,a,k,b,k,1,c+(q*l.enclave_layered_batch*l.size*l.size),n);
              }
              SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KGT_1)
          }
          //gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
          if (net.delta != nullptr) {
            // auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
            a = &l_weights[0] + j*l.nweights/l.groups;
            b = &l_delta[0] + (i*l.groups + j)*m*k;
            float *imd = &net_delta[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            c = nullptr;  // &net_workspace[0];
            if (l.size == 1) {
                SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_INGRAD_KEQ_1)
                c = imd;
                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);
                SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_INGRAD_KEQ_1)
            }
            else {
                SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_INGRAD_KGT_1)
                for (int chan=0;chan < q;chan++) {
                  #ifndef SGX_FAST_TWEAKS_NO_MEMSET
                  std::memset(&net_workspace[0], 0, sizeof(float)*l.enclave_layered_batch * l.out_h * l.out_w* l.size * l.size);
                  #endif
                  c = &net_workspace[0];
                  // TODO: potential bug
                  gemm(1,0,l.enclave_layered_batch*l.size*l.size,k,m,1,a+(chan*l.enclave_layered_batch*l.size*l.size),n,b,k,0,c,k);
                  col2im_cpu(c, l.enclave_layered_batch, l.h, l.w, l.size, l.stride, l.pad, imd+(chan*l.enclave_layered_batch*l.h*l.w));
                }
                if (r > 0) {
                  #ifndef SGX_FAST_TWEAKS_NO_MEMSET
                  std::memset(&net_workspace[0], 0, sizeof(float)*l.enclave_layered_batch * l.out_h * l.out_w* l.size * l.size);
                  #endif
                  c = &net_workspace[0];
                  gemm(1,0,r*l.size*l.size,k,m,1,a+(q*l.enclave_layered_batch*l.size*l.size),n,b,k,0,c,k);
                  col2im_cpu(c, r, l.h, l.w, l.size, l.stride, l.pad, imd+(q*l.enclave_layered_batch*l.h*l.w));
                }
                SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_INGRAD_KGT_1)
                //gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);
                //col2im_cpu(&net_workspace[0], l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
            }
          }
        }
    }
    // print_array(&l_weight_updates[0], l.nweights, 0, "SGX after conv layer weight updates");
    l.weight_updates->setItemsInRange(0, l.weight_updates->getBufferSize(),l_weight_updates);
    l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
    if (net.delta != nullptr) {
      // print_array(&net_delta[0], l.batch*l.inputs, 0, "SGX after conv layer net delta");
      net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
    }
    SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV)
    //LOG_DEBUG("after backward 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("after backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
}

void convolutional_get_MM_weight_updates_left_compare(layer& l, network& net,float* l_weight_updates) {
  std::vector<float> rand_left(l.n/l.groups,0);
  //auto l_weight_updates = l.weight_updates->getItemsInRange(0, l.weight_updates->getBufferSize());
  gemm(0,0,l.n/l.groups,1,(l.size*l.size*l.c/l.groups),1,
    l_weight_updates,(l.size*l.size*l.c/l.groups),
    l.bkwrd_weight_delta_rand,1,
    1,
    rand_left.data(),1
  );
  for (int i=0;i<rand_left.size();++i) {
    if (std::abs(l.bkwrd_weight_delta_rhs[i]-rand_left[i]) > 0.00001f) {
      LOG_ERROR("rand verify value mismatch at index %d for MM output with values (left,right):(%f,%f)\n",
          i,l.bkwrd_weight_delta_rhs[i],rand_left[i])
      abort();
    }
  }
}

void convolutional_get_MM_output_prevdelta_left_compare(layer& l, network& net,float* rand_vec,
                                                float* rand_right,float* imd,float* net_workspace,
                                                int iter,
                                                int subdiv, int batch_num) {
  int q = (l.c/l.groups) / l.enclave_layered_batch;
  int r = (l.c/l.groups) % l.enclave_layered_batch;
  int layer_index = net.index;
  std::vector<float> rand_left(l.out_h*l.out_w,0);
  sgx_status_t ret = SGX_ERROR_UNEXPECTED;
  size_t start_prevdelta = ((subdiv*l.batch*(l.size*l.size*l.c/l.groups)*l.out_w*l.out_h)+
                 (batch_num*(l.size*l.size*l.c/l.groups)*l.out_w*l.out_h))*sizeof(float);
  // if (layer_index == 7 && batch_num == 0) {
  // if (batch_num == 0) {
    // LOG_DEBUG("prev delta conv layer %d,iter=%d,subdiv=%d,net_batch=%d,net_enclavesubdiv=%d\nbatch=%d,q=%d,r=%d,l.size=%d\n",
    //   layer_index,iter,subdiv,net.batch,net.enclave_subdivisions,batch_num,q,r,l.size)
  // }
  #if defined(CONV_BACKWRD_INPUT_GRAD_COPY_AFTER_COL2IM)
  if (l.size == 1) {
    gemm(0,0,1,(l.out_w*l.out_h),l.c/l.groups,1,
      rand_vec,l.c/l.groups,
      imd,(l.out_w*l.out_h),
      1,
      rand_left.data(),(l.out_w*l.out_h)
    );
  }
  else {
    for (int chan = 0; chan < q;++chan) {
      #ifndef SGX_FAST_TWEAKS_NO_MEMSET
      std::memset(net_workspace, 0,l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size*sizeof(float));
      #endif
      im2col_cpu(imd+(chan*l.enclave_layered_batch*l.h*l.w), 
        l.enclave_layered_batch, l.h, l.w, l.size, l.stride, l.pad, net_workspace);
      gemm(0,0,1,(l.out_w*l.out_h),(l.size*l.size*l.enclave_layered_batch),1,
        rand_vec+chan*(l.size*l.size*l.enclave_layered_batch),(l.size*l.size*l.enclave_layered_batch),
        net_workspace,(l.out_w*l.out_h),
        1,
        rand_left.data(),(l.out_w*l.out_h)
      );
    }
    if (r > 0) {
      #ifndef SGX_FAST_TWEAKS_NO_MEMSET
      std::memset(net_workspace, 0,l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size*sizeof(float));
      #endif
      im2col_cpu(imd+(q*l.enclave_layered_batch*l.h*l.w), 
        r, l.h, l.w, l.size, l.stride, l.pad, net_workspace);
      gemm(0,0,1,(l.out_w*l.out_h),(l.size*l.size*r),1,
        rand_vec+q*(l.size*l.size*l.enclave_layered_batch),(l.size*l.size*r),
        net_workspace,(l.out_w*l.out_h),
        1,
        rand_left.data(),(l.out_w*l.out_h)
      );
    }
  }
  #elif defined(CONV_BACKWRD_INPUT_GRAD_COPY_BEFORE_COL2IM)
  if (l.size == 1) {
    OCALL_LOAD_LAYER_REPRT_FRBMMV(iter, layer_index,
            0,nullptr,0,nullptr,0,
            0,nullptr,0,nullptr,0,
            start_prevdelta,
            (uint8_t*)imd, (l.c*l.out_w*l.out_h)*sizeof(float),
            nullptr,0);
    gemm(0,0,1,(l.out_w*l.out_h),l.c/l.groups,1,
      rand_vec,l.c/l.groups,
      imd,(l.out_w*l.out_h),
      1,
      rand_left.data(),(l.out_w*l.out_h)
    );
  }
  else {
    for (int chan = 0; chan < q;++chan) {
      #ifndef SGX_FAST_TWEAKS_NO_MEMSET
      std::memset(net_workspace, 0,l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size*sizeof(float));
      #endif
      OCALL_LOAD_LAYER_REPRT_FRBMMV(iter, layer_index,
              0,nullptr,0,nullptr,0,
              0,nullptr,0,nullptr,0,
              start_prevdelta,
              (uint8_t*)net_workspace,l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size*sizeof(float),
              nullptr,0);
      gemm(0,0,1,(l.out_w*l.out_h),(l.size*l.size*l.enclave_layered_batch),1,
        rand_vec+chan*(l.size*l.size*l.enclave_layered_batch),(l.size*l.size*l.enclave_layered_batch),
        net_workspace,(l.out_w*l.out_h),
        1,
        rand_left.data(),(l.out_w*l.out_h)
      );
      col2im_cpu(net_workspace, l.enclave_layered_batch, l.h, l.w, l.size, l.stride, l.pad, imd+(chan*l.enclave_layered_batch*l.h*l.w));
      start_prevdelta += (l.size*l.size*l.enclave_layered_batch*l.out_w*l.out_h)*sizeof(float);
    }
    if (r > 0) {
      #ifndef SGX_FAST_TWEAKS_NO_MEMSET
      std::memset(net_workspace, 0,l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size*sizeof(float));
      #endif
      OCALL_LOAD_LAYER_REPRT_FRBMMV(iter, layer_index,
              0,nullptr,0,nullptr,0,
              0,nullptr,0,nullptr,0,
              start_prevdelta,
              (uint8_t*)net_workspace,r * l.out_h * l.out_w
                                   * l.size * l.size*sizeof(float),
              nullptr,0);
      gemm(0,0,1,(l.out_w*l.out_h),(l.size*l.size*r),1,
        rand_vec+q*(l.size*l.size*l.enclave_layered_batch),(l.size*l.size*r),
        net_workspace,(l.out_w*l.out_h),
        1,
        rand_left.data(),(l.out_w*l.out_h)
      );
      col2im_cpu(net_workspace, r, l.h, l.w, l.size, l.stride, l.pad, imd+(q*l.enclave_layered_batch*l.h*l.w));
      start_prevdelta += (r*l.size*l.size*l.out_w*l.out_h)*sizeof(float);
    }
  }
  #endif
  // iteration, layer_index, subdiv,batch
  // TODO check the validity of report
  // if (rand_right.size()!=rand_left.size()) {
  //  LOG_ERROR("size mismatch\n")
  //  abort();
  // }
  for (int i=0;i<rand_left.size();++i) {
    //LOG_OUT("diff between rand right and rand left %f\n",std::fabs(std::fabs(rand_right[i]-rand_left[i])));
    if (std::fabs(rand_right[i]-rand_left[i]) > 0.00001f) {
      LOG_ERROR("rand verify value mismatch at index %d for MM output with values (left,right):(%f,%f)\n",
          i,rand_left[i],rand_right[i])
      abort();
    }
  }
}

void backward_convolutional_layer_verifies_frbmmv(layer& l, network& net) {
  // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv layer index=%d\n",net.index)
  // LOG_DEBUG("before backward 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
  //LOG_DEBUG("before backward 237 and 121 updates for weights are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
  SET_START_TIMING(SGX_TIMING_BACKWARD_CONV)
  int i, j;
  int m = l.n/l.groups;
  int n = l.size*l.size*l.c/l.groups;
  int k = l.out_w*l.out_h;
  
  auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
  auto net_delta  = net.delta ? net.delta->getItemsInRange(0, net.delta->getBufferSize()):std::unique_ptr<float[]>(nullptr);
  #ifndef SGX_CONV_BATCH_PRECOMPUTE_VERIFY
  auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
  #endif
  int q = (l.c/l.groups) / l.enclave_layered_batch;
  int r = (l.c/l.groups) % l.enclave_layered_batch;

  {
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    gradient_array(&l_output[0], l.outputs*l.batch, l.activation, &l_delta[0]);
  }
  // if (net.index == 7) {
  //   std::string text = std::string("SGX convolution delta after gradient on activation layer ") + std::to_string(net.index);
  //   print_array(&l_delta[0],2*l.outputs,0,text.c_str());
  // }
  if(l.batch_normalize){
      // auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
      // print_array(&l_bias_updates[0], l.nbiases, 0, "SGX convolutional layer bias updates bn before");
      l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
      backward_batchnorm_layer(l, net);
      // auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
      // print_array(&l_bias_updates[0], l.nbiases, 0, "SGX convolutional layer bias updates");
      l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
  } else {
    auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
    // print_array(&l_bias_updates[0], l.nbiases, 0, "SGX convolutional layer bias updates before");
    backward_bias(&l_bias_updates[0], &l_delta[0], l.batch, l.n, k);
    // print_array(&l_bias_updates[0], l.nbiases, 0, "SGX convolutional layer bias updates");
    l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(), l_bias_updates);
  }
  std::vector<float> mm_randomized_output_right(l.n/l.groups,0);
  std::vector<float> mm_randomized_mid_right(l.out_w*l.out_h,0);
  // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv passed 1st point layer index=%d\n",net.index)

  auto net_workspace = l.size != 1 ? std::unique_ptr<float[]>(
                         new float[l.enclave_layered_batch * l.out_h * l.out_w
                                   * l.size * l.size])
                                 : std::unique_ptr<float[]>(nullptr);
  // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv passed 2nd point layer index=%d\n",net.index)
  // precomputed the weight with rand
  //auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
  // if (net.delta == nullptr) {
  //   auto del_ptr = l_weights.release();
  //   delete[] del_ptr;
  // }
  
  // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv passed 3rd point layer index=%d\n",net.index)
  int iter = ((*net.seen-net.batch)/(net.batch*net.enclave_subdivisions)) + 1;
  int subdiv = (((*net.seen-net.batch)/net.batch)%net.enclave_subdivisions);
  auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
  // perform weight_updates rand mult
  for(i = 0; i < l.batch; ++i){
    for(j = 0; j < l.groups; ++j){
      // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv wu mult layer index=%d\n",net.index)
      float *a = &l_delta[0] + (i*l.groups + j)*m*k;
      float *b = nullptr;   //&net_workspace[0];
      //float *c = &l_weight_updates[0] + j*l.nweights/l.groups;
      float *c = nullptr;
      float *im  = &net_input[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
      std::memset(mm_randomized_mid_right.data(),0,mm_randomized_mid_right.size()*sizeof(float));
      std::memset(mm_randomized_output_right.data(),0,mm_randomized_output_right.size()*sizeof(float));
      if (l.size == 1) {
        SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KEQ_1)
        b = im;
        gemm(1,0,(l.out_w*l.out_h),1,l.size*l.size*l.c/l.groups,1,
          b,(l.out_w*l.out_h),
          l.bkwrd_weight_delta_rand,1,
          1,
          mm_randomized_mid_right.data(),1
        );
        // multiply with delta
        gemm(0,0,l.n/l.groups,1,(l.out_w*l.out_h),1,
          a,(l.out_w*l.out_h),
          mm_randomized_mid_right.data(),1,
          1,
          mm_randomized_output_right.data(),1
        );
        // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv sum for wu mult layer index=%d\n",net.index)
        for (int ss=0;ss<mm_randomized_output_right.size();++ss) {
          l.bkwrd_weight_delta_rhs[ss] += mm_randomized_output_right[ss];
        }
        SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KEQ_1)
      }
      else {
        SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KGT_1)
        for (int chan = 0; chan < q;++chan) {
         #ifndef SGX_FAST_TWEAKS_NO_MEMSET
          std::memset(&net_workspace[0], 0, sizeof(float)*l.enclave_layered_batch * l.out_h * l.out_w* l.size * l.size);
          #endif
          b = &net_workspace[0];
          im2col_cpu(im+chan*l.enclave_layered_batch*(l.h*l.w), l.enclave_layered_batch, l.h, l.w, l.size, l.stride, l.pad, b);
          gemm(1,0,(l.out_w*l.out_h),1,l.size*l.size*l.enclave_layered_batch,1,
            b,(l.out_w*l.out_h),
            l.bkwrd_weight_delta_rand+(chan*l.size*l.size*l.enclave_layered_batch),1,
            1,
            mm_randomized_mid_right.data(),1
          );
        }
        if (r > 0) {
          #ifndef SGX_FAST_TWEAKS_NO_MEMSET
          std::memset(&net_workspace[0], 0, sizeof(float)*l.enclave_layered_batch * l.out_h * l.out_w* l.size * l.size);
          #endif
          b = &net_workspace[0];
          im2col_cpu(im+q*l.enclave_layered_batch*(l.h*l.w), r, l.h, l.w, l.size, l.stride, l.pad, b);
          gemm(1,0,(l.out_w*l.out_h),1,l.size*l.size*r,1,
            b,(l.out_w*l.out_h),
            l.bkwrd_weight_delta_rand+(q*l.size*l.size*l.enclave_layered_batch),1,
            1,
            mm_randomized_mid_right.data(),1
          );
        }
        // multiply with delta
        gemm(0,0,l.n/l.groups,1,(l.out_w*l.out_h),1,
          a,(l.out_w*l.out_h),
          mm_randomized_mid_right.data(),1,
          1,
          mm_randomized_output_right.data(),1
        );
        // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv sum for wu mult layer index=%d\n",net.index)
        for (int ss=0;ss<mm_randomized_output_right.size();++ss) {
          l.bkwrd_weight_delta_rhs[ss] += mm_randomized_output_right[ss];
        }
        SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KGT_1)
      }
      // prev delta
      if (net.delta != nullptr) {
        if (l.size == 1) {
          SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_INGRAD_KEQ_1)
        }
        else {
          SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_INGRAD_KGT_1)
        }
        // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv net_delta section layer index=%d\n",net.index)
        #ifndef SGX_CONV_BATCH_PRECOMPUTE_VERIFY
        for (int jj=0;jj<l.c/l.groups*l.size*l.size;++jj){
          l.bkwrd_input_delta_rand[jj] = sgx_root_rng->getRandomFloat(std::numeric_limits<float>::min(),
                      std::numeric_limits<float>::max());
        }
        std::memset(l.bkwrd_input_delta_rhs, 0, l.n/l.groups*sizeof(float));
        gemm(0,1,1,l.n/l.groups,l.c/l.groups*l.size*l.size,1,
          l.bkwrd_input_delta_rand,l.c/l.groups*l.size*l.size,
          l_weights.get(),l.c/l.groups*l.size*l.size,
          1,
          l.bkwrd_input_delta_rhs,l.n/l.groups);
        #endif
        std::vector<float> net_delta_mm_randomized_output_right(l.out_w*l.out_h,0);
        //a = &l_weights[0] + j*l.nweights/l.groups;
        b = &l_delta[0] + (i*l.groups + j)*m*k;
        float *imd = &net_delta[0] + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
        c = nullptr;  // &net_workspace[0];
        
        gemm(0,0,1,(l.out_h*l.out_w),l.n/l.groups,1,
          l.bkwrd_input_delta_rhs,l.n/l.groups,
          l_delta.get(),(l.out_h*l.out_w),
          1,
          net_delta_mm_randomized_output_right.data(),(l.out_h*l.out_w)
        );
        // verify prev delta for this batch element
        // LOG_DEBUG("backward_convolutional_layer_verifies_frbmmv calling to convolutional_get_MM_output_prevdelta_left_compare section layer index=%d\n",net.index)
        convolutional_get_MM_output_prevdelta_left_compare(l, net,l.bkwrd_input_delta_rand,
                                                net_delta_mm_randomized_output_right.data(),imd,net_workspace.get(),
                                                iter,
                                                subdiv, i);
        if (l.size == 1) {
          SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_INGRAD_KEQ_1)
        }
        else {
          SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_INGRAD_KGT_1)
        }
      }
    }
  }
  if (net.delta) {
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
  }
  // check if last backward for this SGD step considering l.batch and l.enclave_subdiv
  if(((*net.seen)/net.batch)%net.enclave_subdivisions == 0) {
    if (l.size == 1) {
      SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KEQ_1)
    }
    else {
      SET_START_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KGT_1)
    }
    auto l_weight_updates = l.weight_updates->getItemsInRange(0, l.weight_updates->getBufferSize());
    convolutional_get_MM_weight_updates_left_compare(l, net,l_weight_updates.get());
    if (l.size == 1) {
      SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KEQ_1)
    }
    else {
      SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV_WGRAD_KGT_1)
    }
  }
  SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONV)
}

void update_convolutional_layer(convolutional_layer& l, update_args a)
{
    
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    float clip = a.grad_clip;
    //LOG_DEBUG("lr:%f,moment:%f,decay:%f,batch:%d total weights:%d\n",learning_rate,momentum,decay,batch,l.nweights)
    //LOG_DEBUG("before update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    {
      auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
      auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
      if (clip != 0) {
        constrain_cpu(l.n,clip,&l_bias_updates[0],1);    
      }
      axpy_cpu(l.n, learning_rate/batch, &l_bias_updates[0], 1, &l_biases[0], 1);
      scal_cpu(l.n, momentum, &l_bias_updates[0], 1);
      l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(),l_bias_updates);
      l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
    }

    if(l.scales){
        auto l_scale_updates = l.scale_updates->getItemsInRange(0, l.scale_updates->getBufferSize());
        auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
        if (clip != 0) {
          constrain_cpu(l.n,clip,&l_scale_updates[0],1);
        }
        axpy_cpu(l.n, learning_rate/batch, &l_scale_updates[0], 1, &l_scales[0], 1);
        scal_cpu(l.n, momentum, &l_scale_updates[0], 1);
        l.scale_updates->setItemsInRange(0, l.scale_updates->getBufferSize(),l_scale_updates);
        l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
    }

    //LOG_DEBUG("before update 237 and 121 weight updates are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    auto l_weights = l.weights->getItemsInRange(0, l.weights->getBufferSize());
    auto l_weight_updates = l.weight_updates->getItemsInRange(0, l.weight_updates->getBufferSize());
    if (clip != 0) {
      constrain_cpu(l.nweights,clip,&l_weight_updates[0],1);
    }
    axpy_cpu(l.nweights, -decay*batch, &l_weights[0], 1, &l_weight_updates[0], 1);
    //LOG_DEBUG("here update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
    //LOG_DEBUG("here update 237 and 121 weight updates are: %0.10e, .. %0.10e\n",l.weight_updates[237],l.weight_updates[121])
    axpy_cpu(l.nweights, learning_rate/batch, &l_weight_updates[0], 1, &l_weights[0], 1);
    scal_cpu(l.nweights, momentum, &l_weight_updates[0], 1);
    l.weights->setItemsInRange(0, l.weights->getBufferSize(),l_weights);
    l.weight_updates->setItemsInRange(0, l.weight_updates->getBufferSize(),l_weight_updates);

    //LOG_DEBUG("after update 237 and 121 weights are: %0.10e, .. %0.10e\n",l.weights[237],l.weights[121])
}

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam,PRNG& net_layer_rng_deriver)
{
    int i,j;
    layer l = {};
    l.layer_rng = std::make_shared<PRNG>(generate_random_seed_from(net_layer_rng_deriver));
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
    l.nbiases = outputs;

    //l.output = (float*)calloc(batch*outputs, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(batch*outputs);
    //l.delta = (float*)calloc(batch*outputs, sizeof(float));
    if (global_training) l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(batch*outputs);

    //l.weight_updates = (float*)calloc(inputs*outputs, sizeof(float));
    if (global_training) {
        l.weight_updates = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*outputs);
        l.bkwrd_weight_delta_rhs = (double*)calloc(l.outputs,sizeof(double));
        l.bkwrd_weight_delta_rand = (float*)calloc(l.inputs,sizeof(float));
        l.frwd_outs_rand = (float*)calloc(l.outputs,sizeof(float));
        l.frwd_outs_rhs = (float*)calloc(l.inputs,sizeof(float));
        l.bkwrd_input_delta_rand = (float*)calloc(l.inputs,sizeof(float));
        l.bkwrd_input_delta_rhs = (float*)calloc(l.outputs,sizeof(float));
    }
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
            // l_weights[j] = scale*rand_uniform(-1, 1);
            l_weights[j] = rand_uniform(*(l.layer_rng),-1, 1);
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
    float clip = a.grad_clip;
    int enclave_update_batch = l.enclave_layered_batch / 2;
    int q = l.outputs / enclave_update_batch;
    int r = l.outputs % enclave_update_batch;
    {
        auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
        auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        if (clip != 0) {
            constrain_cpu(l.outputs,clip,&l_bias_updates[0],1);
        }
        axpy_cpu(l.outputs, learning_rate/batch, &l_bias_updates[0], 1, &l_biases[0], 1);
        scal_cpu(l.outputs, momentum, &l_bias_updates[0], 1);
        l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
        l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(),l_bias_updates);
    }

    if(l.batch_normalize){
        auto l_scale_updates = l.scale_updates->getItemsInRange(0, l.scale_updates->getBufferSize());
        auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
        if (clip != 0) {
            constrain_cpu(l.outputs,clip,&l_scale_updates[0],1);
        }
        axpy_cpu(l.outputs, learning_rate/batch, &l_scale_updates[0], 1, &l_scales[0], 1);
        scal_cpu(l.outputs, momentum, &l_scale_updates[0], 1);
        l.scale_updates->setItemsInRange(0, l.scale_updates->getBufferSize(),l_scale_updates);
        l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
    }

    {
        for (int i=0;i<q;++i) {
            auto l_weights = l.weights->getItemsInRange(i*enclave_update_batch*l.inputs, (i+1)*enclave_update_batch*l.inputs);
            auto l_weight_updates = l.weight_updates->getItemsInRange(i*enclave_update_batch*l.inputs, (i+1)*enclave_update_batch*l.inputs);
            if (clip != 0) {
                constrain_cpu(enclave_update_batch*l.inputs,clip,&l_weight_updates[0],1);
            }
            axpy_cpu(enclave_update_batch*l.inputs, -decay*batch, &l_weights[0], 1, &l_weight_updates[0], 1);
            axpy_cpu(enclave_update_batch*l.inputs, learning_rate/batch, &l_weight_updates[0], 1, &l_weights[0], 1);
            scal_cpu(enclave_update_batch*l.inputs, momentum, &l_weight_updates[0], 1);
            l.weights->setItemsInRange(i*enclave_update_batch*l.inputs, (i+1)*enclave_update_batch*l.inputs,l_weights);
            l.weight_updates->setItemsInRange(i*enclave_update_batch*l.inputs, (i+1)*enclave_update_batch*l.inputs,l_weight_updates);
        }
        if (r > 0) {
            auto l_weights = l.weights->getItemsInRange(q*enclave_update_batch*l.inputs, q*enclave_update_batch*l.inputs + r*l.inputs);
            auto l_weight_updates = l.weight_updates->getItemsInRange(q*enclave_update_batch*l.inputs, q*enclave_update_batch*l.inputs+r*l.inputs);
            if (clip != 0) {
                constrain_cpu(r*l.inputs,clip,&l_weight_updates[0],1);
            }
            axpy_cpu(r*l.inputs, -decay*batch, &l_weights[0], 1, &l_weight_updates[0], 1);
            axpy_cpu(r*l.inputs, learning_rate/batch, &l_weight_updates[0], 1, &l_weights[0], 1);
            scal_cpu(r*l.inputs, momentum, &l_weight_updates[0], 1);
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
    if (net.verf_type == verf_variations_t::FRBRMMV) {
        forward_connected_layer_verifies_frbmmv(l,net);
        return;
    }
    LOG_DEBUG("started lwfit forward connected\n")
    SET_START_TIMING(SGX_TIMING_FORWARD_CONNCTD)
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    LOG_DEBUG("lwfit forward output success connected\n")
    fill_cpu(l.outputs*l.batch, 0, &l_output[0], 1);
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    LOG_DEBUG("lwfit forward net input success connected\n")
    // print_array(&net_input[0],100,0,"SGX before connected forward input");
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    
    int q = l.outputs / l.enclave_layered_batch;
    int r = l.outputs % l.enclave_layered_batch;
    {
        float *a = &net_input[0];
        for (int i=0;i<q;++i) {
            float *c = &l_output[i*l.enclave_layered_batch];
            SET_START_TIMING("SGX Connected Forward loading weights")
            auto l_weights = l.weights->getItemsInRange(i*l.enclave_layered_batch*l.inputs,(i+1)*l.enclave_layered_batch*l.inputs);
            LOG_DEBUG("lwfit forward weights success connected\n")
            SET_FINISH_TIMING("SGX Connected Forward loading weights")
            // print_array(&l_weights[0],l.enclave_layered_batch*l.inputs,i*l.enclave_layered_batch*l.inputs,"SGX before connected forward weights");
            float *b = &l_weights[0];        
            gemm(0,1,m,l.enclave_layered_batch,k,1,a,k,b,k,1,c,n);
        }
        if (r > 0) {
            float *c = &l_output[q*l.enclave_layered_batch];
            SET_START_TIMING("SGX Connected Forward loading weights")
            auto l_weights = l.weights->getItemsInRange(q*l.enclave_layered_batch*l.inputs,q*l.enclave_layered_batch*l.inputs+r*l.inputs);
            LOG_DEBUG("lwfit forward weights success connected\n")
            SET_FINISH_TIMING("SGX Connected Forward loading weights")
            // print_array(&l_weights[0],r*l.inputs,q*l.enclave_layered_batch*l.inputs,"SGX before connected forward weights");
            float *b = &l_weights[0];        
            gemm(0,1,m,r,k,1,a,k,b,k,1,c,n);
        }
        //gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    }
    LOG_DEBUG("finished lwfit multiplication of forward connected\n")
    // print_array(&l_output[0],100,0,"SGX connected forward input before bias or batchnorm");
    if(l.batch_normalize){
        l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
        forward_batchnorm_layer(l, net);
        l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    } else {
        auto l_biases = l.biases->getItemsInRange(0,l.biases->getBufferSize()); 
        add_bias(&l_output[0], &l_biases[0], l.batch, l.outputs, 1);
    }
    // print_array(&l_output[0],100,0,"SGX connected forward input before bias or batchnorm");
    activate_array(&l_output[0], l.outputs*l.batch, l.activation);
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    SET_FINISH_TIMING(SGX_TIMING_FORWARD_CONNCTD)
    LOG_DEBUG("finished lwfit forward connected\n")
}

void backward_connected_layer(layer& l, network& net)
{
    if (net.verf_type == verf_variations_t::FRBRMMV) {
        backward_connected_layer_verifies_frbmmv(l,net);
        return;
    }
    SET_START_TIMING(SGX_TIMING_BACKWARD_CONNCTD)
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    // if (net.index == 13) {
    //     print_array(&l_delta[0], l.outputs*l.batch/10, 0, "before connected layer delta");
    // }
    {
        auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
        gradient_array(&l_output[0], l.outputs*l.batch, l.activation, &l_delta[0]);
    }
    // print_array(&l_delta[0], l.outputs*l.batch/10, 0, "after connected layer delta");

    if(l.batch_normalize){
        l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
        backward_batchnorm_layer(l, net);
        // auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        // print_array(&l_bias_updates[0], l.outputs, 0, "SGX connected layer bias updates");
        l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    } else {
        auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        backward_bias(&l_bias_updates[0], &l_delta[0], l.batch, l.outputs, 1);
        // print_array(&l_bias_updates[0], l.outputs, 0, "SGX connected layer bias updates");
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
        for (int i=0;i<q;++i) {
            float *a = &l_delta[i*l.enclave_layered_batch];
            auto l_weight_updates = l.weight_updates->getItemsInRange(i*l.enclave_layered_batch*n, (i+1)*l.enclave_layered_batch*n);
            float *c = &l_weight_updates[0];
            gemm(1,0,l.enclave_layered_batch,n,k,1,a,m,b,n,1,c,n);
            // print_array(&l_weight_updates[0], l.enclave_layered_batch*n, i*l.enclave_layered_batch*n, "SGX after connected layer weight updates");
            l.weight_updates->setItemsInRange(i*l.enclave_layered_batch*n, (i+1)*l.enclave_layered_batch*n,l_weight_updates);
        }
        if (r > 0) {
            float *a = &l_delta[q*l.enclave_layered_batch];
            auto l_weight_updates = l.weight_updates->getItemsInRange(q*l.enclave_layered_batch*n, q*l.enclave_layered_batch*n + r*n);
            float *c = &l_weight_updates[0];
            gemm(1,0,r,n,k,1,a,m,b,n,1,c,n);
            // print_array(&l_weight_updates[0], r*n, q*l.enclave_layered_batch*n, "SGX after connected layer weight updates");
            l.weight_updates->setItemsInRange(q*l.enclave_layered_batch*n, q*l.enclave_layered_batch*n+r*n,l_weight_updates);
        }
        //gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);
    }

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    if(net.delta) {
        auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
        for (int i=0;i<q;++i) {
            float *a = &l_delta[i*l.enclave_layered_batch];
            auto l_weights = l.weights->getItemsInRange(i*l.enclave_layered_batch*(l.inputs), (i+1)*l.enclave_layered_batch*l.inputs);
            b = &l_weights[0];
            float* c = &net_delta[0];
            gemm(0,0,m,n,l.enclave_layered_batch,1,a,k,b,n,1,c,n);
        }
        if (r > 0) {
            float *a = &l_delta[q*l.enclave_layered_batch];
            auto l_weights = l.weights->getItemsInRange(q*l.enclave_layered_batch*(l.inputs), q*l.enclave_layered_batch*l.inputs+r*l.inputs);
            b = &l_weights[0];
            float* c = &net_delta[0];
            gemm(0,0,m,n,r,1,a,k,b,n,1,c,n);
        }
        //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        // print_array(&net_delta[0], l.batch*l.inputs, 0, "SGX after connected layer net delta");
        net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
    }
    SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONNCTD)
}

void connected_get_MM_output_left_compare(layer& l, network& net,float* rand_vec,float* rand_right,
float* l_output) {
    std::vector<float> rand_left(l.batch,0);
    gemm(0,0,l.batch,1,l.outputs,1,
        l_output,l.outputs,
        rand_vec,1,
        1,
        rand_left.data(),1);
    //if (rand_right.size()!=rand_left.size()) {
    //    LOG_ERROR("size mismatch\n")
    //    abort();
    //}
    for (int i=0;i<rand_left.size();++i) {
        if (std::fabs(rand_right[i]-rand_left[i]) > 0.00001f) {
            LOG_ERROR("rand verify value mismatch at index %d for MM output with values (left,right):(%f,%f)\n",
                i,rand_right[i],rand_left[i])
            abort();
        }
    }
}

void forward_connected_layer_verifies_frbmmv(layer& l, network& net) {
    SET_START_TIMING(SGX_TIMING_FORWARD_CONNCTD)
    std::vector<float> mm_randomized_output_right(l.batch,0);
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    gemm(0,0,l.batch,1,l.inputs,1,
        &net_input[0],l.inputs,
        l.frwd_outs_rhs,1,
        1,
        mm_randomized_output_right.data(),1);
    connected_get_MM_output_left_compare(l,net,
        l.frwd_outs_rand,mm_randomized_output_right.data(),l_output.get());
   
    // if (net.index == 13) {
    //     print_array(&l_output[0],l.batch*l.outputs,0,"SGX connected forward input before bias or batchnorm");
    // }
    if(l.batch_normalize){
        l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
        forward_batchnorm_layer(l, net);
        l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    } else {
        auto l_biases = l.biases->getItemsInRange(0,l.biases->getBufferSize()); 
        add_bias(&l_output[0], &l_biases[0], l.batch, l.outputs, 1);
    }
    // if (net.index == 13) {
    //     print_array(&l_output[0],l.batch*l.outputs,0,"SGX connected forward input after bias or batchnorm before activation");
    // }
    // print_array(&l_output[0],100,0,"SGX connected forward input after bias or batchnorm");
    activate_array(&l_output[0], l.outputs*l.batch, l.activation);
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    SET_FINISH_TIMING(SGX_TIMING_FORWARD_CONNCTD)
}

void connected_get_MM_weight_updates_left_compare(layer& l, network& net) {
    int q = l.outputs / l.enclave_layered_batch;
    int r = l.outputs % l.enclave_layered_batch;
    std::vector<float> rand_left(l.outputs,0);
    for (int i=0;i<q;++i) {
        auto l_weight_updates = l.weight_updates->getItemsInRange(i*l.enclave_layered_batch*l.inputs, (i+1)*l.enclave_layered_batch*l.inputs);
        gemm(0,0,l.enclave_layered_batch,1,l.inputs,1,
        l_weight_updates.get(),l.inputs,
        l.bkwrd_weight_delta_rand,1,
        1,
        rand_left.data()+i*l.enclave_layered_batch,1);
    }
    if (r!=0) {
        auto l_weight_updates = l.weight_updates->getItemsInRange(q*l.enclave_layered_batch*l.inputs, q*l.enclave_layered_batch*l.inputs+r*l.inputs);
        gemm(0,0,r,1,l.inputs,1,
        l_weight_updates.get(),l.inputs,
        l.bkwrd_weight_delta_rand,1,
        1,
        rand_left.data()+q*l.enclave_layered_batch,1);
    }
    for (int i=0;i<rand_left.size();++i) {
        if (std::abs(l.bkwrd_weight_delta_rhs[i]-rand_left[i]) > 0.00001f) {
            LOG_ERROR("rand verify value mismatch at index %d for MM output with values (left,right):(%f,%f)\n",
              i,rand_left[i],l.bkwrd_weight_delta_rhs[i])
            abort();
        }
    }
}

void connected_get_MM_output_prevdelta_left_compare(layer& l, network& net,float* rand_vec,
                                                float* rand_right) {
    std::vector<float> rand_left(l.batch,0);                                                
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    gemm(0,0,l.batch,1,l.inputs,1,
    net_delta.get(),l.inputs,rand_vec,1,
    1,
    rand_left.data(),1);
    //if (rand_right.size()!=rand_left.size()) {
    //    LOG_ERROR("size mismatch\n")
    //    abort();
    //}
    for (int i=0;i<rand_left.size();++i) {
        if (std::fabs(rand_right[i]-rand_left[i]) > 0.00001f) {
            LOG_ERROR("rand verify value mismatch at index %d for MM output with values (left,right):(%f,%f)\n",
                i,rand_left[i],rand_right[i])
            abort();
        }
    }
}

void backward_connected_layer_verifies_frbmmv(layer& l, network& net) {
    SET_START_TIMING(SGX_TIMING_BACKWARD_CONNCTD)
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    // if (net.index == 13) {
    //     print_array(&l_delta[0], l.outputs*l.batch/10, 0, "before connected layer delta");
    // }
    {
        auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
        gradient_array(&l_output[0], l.outputs*l.batch, l.activation, &l_delta[0]);
    }
    // print_array(&l_delta[0], l.outputs*l.batch/10, 0, "after connected layer delta");

    if(l.batch_normalize){
        l.delta->setItemsInRange(0, l.delta->getBufferSize(),l_delta);
        backward_batchnorm_layer(l, net);
        // auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        // print_array(&l_bias_updates[0], l.outputs, 0, "SGX connected layer bias updates");
        l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    } else {
        auto l_bias_updates = l.bias_updates->getItemsInRange(0, l.bias_updates->getBufferSize());
        backward_bias(&l_bias_updates[0], &l_delta[0], l.batch, l.outputs, 1);
        // print_array(&l_bias_updates[0], l.outputs, 0, "SGX connected layer bias updates");
        l.bias_updates->setItemsInRange(0, l.bias_updates->getBufferSize(), l_bias_updates);
    }
    std::vector<float> mm_randomized_output_right(l.outputs,0);
    std::vector<float> mm_randomized_mid_right(l.batch,0);
    {
        auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
        gemm(0,0,l.batch,1,l.inputs,1,
        net_input.get(),l.inputs,
        l.bkwrd_weight_delta_rand,1,
        1,
        mm_randomized_mid_right.data(),1);
    }
    //auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    {
        gemm(1,0,l.outputs,1,l.batch,1,
        l_delta.get(),l.outputs,
        mm_randomized_mid_right.data(),1,
        1,
        mm_randomized_output_right.data(),1);
    }
    for (int i=0;i<mm_randomized_output_right.size();++i) {
        l.bkwrd_weight_delta_rhs[i] += mm_randomized_output_right[i];
    }
    if(((*net.seen)/net.batch)%net.enclave_subdivisions == 0) {
        connected_get_MM_weight_updates_left_compare(l, net);
    }
    if (net.delta) {
        mm_randomized_output_right.resize(l.batch);
        std::memset(mm_randomized_output_right.data(),0,sizeof(float)*l.batch);

        gemm(0,0,l.batch,1,l.outputs,1,
        l_delta.get(),l.outputs,
        l.bkwrd_input_delta_rhs,1,
        1,
        mm_randomized_output_right.data(),1);
        connected_get_MM_output_prevdelta_left_compare(l, 
            net,l.bkwrd_input_delta_rand, mm_randomized_output_right.data());
    }
    SET_FINISH_TIMING(SGX_TIMING_BACKWARD_CONNCTD)
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    //l.indexes = (int*)calloc(output_size, sizeof(int));
    l.indexes = sgx::trusted::SpecialBuffer<int>::GetNewSpecialBuffer(output_size);
    //l.output =  (float*)calloc(output_size, sizeof(float));
    l.output = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(output_size);
    //l.delta =   (float*)calloc(output_size, sizeof(float));
    l.delta = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(output_size);
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    //fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void forward_maxpool_layer(maxpool_layer& l, network& net)
{
    SET_START_TIMING(SGX_TIMING_FORWARD_MAXP)
    // TODO: Should be data oblivious
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    auto l_output = l.output->getItemsInRange(0, l.output->getBufferSize());
    auto l_indexes = l.indexes->getItemsInRange(0, l.indexes->getBufferSize());
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net_input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l_output[out_index] = max;
                    if (max_i == -1) {
                        auto aaa = 0;
                        LOG_ERROR("in image %d for channel %d there was negative index!\n",b,k)
                        abort();
                    }
                    l_indexes[out_index] = max_i;
                }
            }
        }
    }
    l.output->setItemsInRange(0, l.output->getBufferSize(),l_output);
    l.indexes->setItemsInRange(0, l.indexes->getBufferSize(),l_indexes);
    SET_FINISH_TIMING(SGX_TIMING_FORWARD_MAXP)
}

void backward_maxpool_layer(maxpool_layer& l, network& net)
{
    SET_START_TIMING(SGX_TIMING_BACKWARD_MAXP)
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    auto l_indexes = l.indexes->getItemsInRange(0, l.indexes->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    auto l_delta = l.delta->getItemsInRange(0, l.delta->getBufferSize());
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l_indexes[i];
        /* if (index == -1) {
            //auto aaa = 0;
            LOG_ERROR("index -1 seen at i %d out of %d\n",i,h*w*c*l.batch)
            abort();
        } */
        net_delta[index] += l_delta[i];
    }
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
    SET_FINISH_TIMING(SGX_TIMING_BACKWARD_MAXP)
}

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
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

dropout_layer make_dropout_layer(int batch, int inputs, float probability,PRNG& net_layer_rng_deriver) {
    dropout_layer l = {};
    l.layer_rng = std::make_shared<PRNG>(generate_random_seed_from(net_layer_rng_deriver));
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    //l.rand = (float*)calloc(inputs*batch, sizeof(float));
    l.rand = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(inputs*batch);
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    //fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
}

void forward_dropout_layer(dropout_layer& l, network& net)
{
    int i;
    if (!net.train) return;
    auto l_rand = l.rand->getItemsInRange(0, l.rand->getBufferSize());
    auto net_input = net.input->getItemsInRange(0, net.input->getBufferSize());
    for(i = 0; i < l.batch * l.inputs; ++i){
        // float r = rand_uniform(0, 1);
        float r = rand_uniform(*(l.layer_rng),0, 1);
        l_rand[i] = r;
        if(r < l.probability) net_input[i] = 0;
        else net_input[i] *= l.scale;
    }
    // if (net.index == 12) {
    //     print_array(&l_rand[0],l.batch * l.inputs,0,"SGX dropout rand vals");
    // }
    l.rand->setItemsInRange(0, l.rand->getBufferSize(),l_rand);
    net.input->setItemsInRange(0, net.input->getBufferSize(),net_input);
}

void backward_dropout_layer(dropout_layer& l, network& net)
{
    int i;
    if(!net.delta) return;
    auto l_rand = l.rand->getItemsInRange(0, l.rand->getBufferSize());
    auto net_delta = net.delta->getItemsInRange(0, net.delta->getBufferSize());
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l_rand[i];
        if(r < l.probability) net_delta[i] = 0;
        else net_delta[i] *= l.scale;
    }
    net.delta->setItemsInRange(0, net.delta->getBufferSize(),net_delta);
}

network *parse_network_cfg(char *filename,const net_context_variations& context,const verf_variations_t& verf){
    LOG_TRACE("entered in parse network config\n");
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) {
      LOG_ERROR("network is empty!" "\n");
      abort();
    }
    
    network *net = make_network(sections->size - 1);
    net->net_context = context;
    net->verf_type = verf;
    net->gpu_index = gpu_index;
    size_params params;
    set_network_batch_randomness(0,*net);

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) {
      LOG_ERROR("First section must be [net] or [network] \n");
      abort();
    }
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;
    
    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    //LOG_INFO("network type is :%s\n",s->type);
    free_section(s);
    //LOG_DEBUG("free_Section was successful" "\n");

    while(n){
        params.index = count;
        s = (section *)n->val;
        options = s->options;
        layer l = {};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params,*(net->layer_rng_deriver));
        } 
        /*else if(lt == CONVOLUTIONAL1D){
            l = parse_convolutional1D(options, params);
        }*/
        /* else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        } */
        /* else if(lt == LOCAL){
            l = parse_local(options, params);
        } */
        else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }
        /*else if(lt == LOGXENT){
            l = parse_logistic(options, params);
            
        }*/
        /* else if(lt == L2NORM){
            l = parse_l2norm(options, params);
        } */
        /* else if(lt == RNN){
            l = parse_rnn(options, params);
        } */
        /* else if(lt == GRU){
            l = parse_gru(options, params);
        } */
        /* else if (lt == LSTM) {
            l = parse_lstm(options, params);
        } */
        /* else if(lt == CRNN){
            l = parse_crnn(options, params);
        } */
        else if(lt == CONNECTED){
            l = parse_connected(options, params,*(net->layer_rng_deriver));
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }
        /*
        else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == YOLO){
          l = parse_yolo(options, params);
        }else if(lt == ISEG){
            l = parse_iseg(options, params);
        }else if(lt == DETECTION){
          l = parse_detection(options, params);
        }
        */
        else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }
        /*
        else if(lt == NORMALIZATION){
            l = parse_normalization(options, params);
        }*/
        else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }
        /*else if(lt == MAXPOOL1D){
            l = parse_maxpool1D(options, params);
        }*/
        /*
        else if(lt == REORG){
            l = parse_reorg(options, params);
        }*/
        else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }
        else if(lt == AVGPOOLX){
            l = parse_avgpoolx(options, params);
        }
        /*else if(lt == AVGPOOLX1D){
            l = parse_avgpoolx1D(options, params);
        }*/
        else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }
        /* else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        } */
        else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params,*(net->layer_rng_deriver));
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
        }else{
            LOG_ERROR("Type not recognized: %s\n", s->type);
            abort();
        }
        
        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.numload = option_find_int_quiet(options, "numload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    net->input = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(net->inputs*net->batch);
    net->truth = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(net->truths*net->batch);

    if(workspace_size){
        //printf("%ld\n", workspace_size);
        net->workspace = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(workspace_size/sizeof(float));
    }
    LOG_TRACE("finished in parse network config\n");
    return net;
}

network *load_network(char *cfg, char *weights, int clear,const net_context_variations& context,const verf_variations_t& verf){  
  network *net = parse_network_cfg(cfg,context,verf);

  /*if(weights && weights[0] != 0){
    load_weights(net, weights);
  }*/

  if(clear) (*net->seen) = 0;
  return net;
}




