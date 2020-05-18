#pragma once

#ifndef _SGXADL_DARKNET_SGXFFIT_H_
#define _SGXADL_DARKNET_SGXFFIT_H_

#if defined(USE_SGX) && defined(USE_SGX_PURE)
	#include "../layer.h"
	#include "../parser.h"
	#include "global-vars-trusted.h"

	typedef layer convolutional_layer;
	typedef layer maxpool_layer;
	typedef layer dropout_layer;

	convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam,PRNG& net_layer_rng_deriver);
	void forward_convolutional_layer_verifies_frbmmv(layer& l, network& net);
	void backward_convolutional_layer_verifies_frbmmv(layer& l, network& net);
	void forward_convolutional_layer(layer& l, network& net);
	void backward_convolutional_layer(layer& l, network& net);
	void update_convolutional_layer(convolutional_layer& layer, update_args a);
	int convolutional_out_height(const convolutional_layer& layer);
	int convolutional_out_width(const convolutional_layer& layer);
	void add_bias(float *output, float *biases, int batch, int n, int size);
	void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

	layer make_batchnorm_layer(int batch, int w, int h, int c);
	void forward_batchnorm_layer(layer& l, network& net);
	void backward_batchnorm_layer(layer& l, network& net);

	layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam,PRNG& net_layer_rng_deriver);
	void forward_connected_layer(layer& l, network& net);
	void forward_connected_layer_verifies_frbmmv(layer& l, network& net);
	void backward_connected_layer_verifies_frbmmv(layer& l, network& net);
	void backward_connected_layer(layer& l, network& net);
	void update_connected_layer(layer& l, update_args a);

	maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
	void forward_maxpool_layer(maxpool_layer& l, network& net);
	void backward_maxpool_layer(maxpool_layer& l, network& net);

	size_t get_workspace_size(layer l);

	dropout_layer make_dropout_layer(int batch, int inputs, float probability,PRNG& net_layer_rng_deriver);
	void forward_dropout_layer(dropout_layer& l, network& net);
	void backward_dropout_layer(dropout_layer& l, network& net);
	void resize_dropout_layer(dropout_layer *l, int inputs);

	network *make_network(int n);
	network *load_network(char *cfg, char *weights, int clear,const net_context_variations& context,const verf_variations_t& verf);
	network *parse_network_cfg(char *filename,const net_context_variations& context,const verf_variations_t& verf);

	convolutional_layer parse_convolutional(list *options, size_params params,PRNG& net_layer_rng_deriver);
	layer parse_connected(list *options, size_params params,PRNG& net_layer_rng_deriver);
	layer parse_dropout(list *options, size_params params,PRNG& net_layer_rng_deriver);

	//layer parse_convolutional1D(list *options, size_params params);
	layer parse_activation(list *options, size_params params);
	layer parse_crop(list *options, size_params params);
	layer parse_cost(list *options, size_params params);
	layer parse_softmax(list *options, size_params params);
	layer parse_batchnorm(list *options, size_params params);
	layer parse_maxpool(list *options, size_params params);
	layer parse_avgpool(list *options, size_params params);
	layer parse_avgpoolx(list *options, size_params params);
	layer parse_route(list *options, size_params params,network* net);
	layer parse_shortcut(list *options, size_params params,network* net);


	

#else
	#error "USE_SGX and USE_SGX_PURE Required"
#endif

#endif // _SGXADL_DARKNET_SGXFFIT_H_