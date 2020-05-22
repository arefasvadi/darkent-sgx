#include "sgxffit.h"
#include "rand/PRNG.h"
#include "../activations.h"
#include "../gemm.h"
#include "../col2im.h"
#include "../im2col.h"
#include "../blas.h"
#include "../option_list.h"

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
    net->input = (float*)calloc(net->inputs*net->batch, sizeof(float));
    net->truth = (float*)calloc(net->truths*net->batch, sizeof(float));

    if(workspace_size){
        net->workspace = (float*)calloc(1, workspace_size);
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


