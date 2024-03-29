#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "activation_layer.h"
#include "logistic_layer.h"
#include "l2norm_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "avgpoolx_layer.h"
#include "avgpoolx1D_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "convolutional1D_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "maxpool1D_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "iseg_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]")==0) return YOLO;
    if (strcmp(type, "[iseg]")==0) return ISEG;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[conv1D]")==0
            || strcmp(type, "[convolutional1D]")==0) return CONVOLUTIONAL1D;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[logistic]")==0) return LOGXENT;
    if (strcmp(type, "[l2norm]")==0) return L2NORM;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]") == 0) return LSTM;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[max1D]")==0
            || strcmp(type, "[maxpool1D]")==0) return MAXPOOL1D;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[avgx]")==0
            || strcmp(type, "[avgpoolx]")==0) return AVGPOOLX;
    if (strcmp(type, "[avgx1D]")==0
            || strcmp(type, "[avgpoolx1D]")==0) return AVGPOOLX1D;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE;
    return BLANK;
}

void free_section(section *s) {
  free(s->type);
  node *n = s->options->front;
  while (n) {
    kvp *pair = (kvp *)n->val;
    /* printf(ANSI_COLOR_GREEN "KEY is %s" ANSI_COLOR_RESET "\n", pair->key); */
    free(pair->key);
    /* printf(ANSI_COLOR_GREEN "Val is %s" ANSI_COLOR_RESET "\n", pair->val); */
    /* printf(ANSI_COLOR_RED "REACHED!" ANSI_COLOR_RESET "\n"); */
    free(pair);
    node *next = n->next;
    free(n);
    n = next;
  }
  free(s->options);
  free(s);
}

#ifndef USE_SGX
void parse_data(char *data, float *a, int n)
{
  int i;
  if(!data) return;
  char *curr = data;
  char *next = data;
  int done = 0;
  for(i = 0; i < n && !done; ++i){
    while(*++next !='\0' && *next != ',');
    if(*next == '\0') done = 1;
    *next = '\0';
    sscanf(curr, "%g", &a[i]);
    curr = next+1;
  }
}
#else
#endif

// typedef struct size_params{
//     int batch;
//     int inputs;
//     int h;
//     int w;
//     int c;
//     int index;
//     int time_steps;
//     network *net;
// } size_params;

#if defined(USE_SGX) && defined (USE_SGX_BLOCKING)
typedef struct size_params_blocked{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network_blocked *net;
} size_params_blocked;
#endif

/* local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
#ifndef USE_SGX
    if(!(h && w && c)) error("Layer before local layer must output image.");
#else
#endif
    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
} */

/* layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
#ifndef USE_SGX
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");
#else
#endif
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    layer l = make_deconvolutional_layer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, params.net->adam);

    return l;
} */


convolutional_layer parse_convolutional(list *options, size_params params,PRNG& net_layer_rng_deriver)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) {
      LOG_ERROR("Layer before convolutional layer must output image.");
      abort();
    }
    
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, 
    params.net->adam,net_layer_rng_deriver);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
    return layer;
}

convolutional1D_layer parse_convolutional1D(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h == 1 && w && c)) error("Layer before convolutional1D layer must output image with height 1.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional1D_layer layer = make_convolutional1D_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}

/* layer parse_crnn(list *options, size_params params)
{
    int output_filters = option_find_int(options, "output_filters",1);
    int hidden_filters = option_find_int(options, "hidden_filters",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
} */

/* layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
} */

/* layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);
    l.tanh = option_find_int_quiet(options, "tanh", 0);

    return l;
} */

/* layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

    return l;
} */

layer parse_connected(list *options, size_params params,PRNG& net_layer_rng_deriver)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam,net_layer_rng_deriver);
    return l;
}

layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
#if !defined(USE_SGX) && !defined(SGX_VERIFIES)
    if (tree_file) l.softmax_tree = read_tree(tree_file);
#else
#endif
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, "spatial", 0);
    l.noloss =  option_find_int_quiet(options, "noloss", 0);
    return l;
}

#if !defined(USE_SGX) && !defined(SGX_VERIFIES)
int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        mask = (int*)calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}

layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max",90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

layer parse_iseg(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int ids = option_find_int(options, "ids", 32);
    layer l = make_iseg_layer(params.batch, params.w, params.h, classes, ids);
    assert(l.outputs == params.inputs);
    return l;
}

layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.background = option_find_int_quiet(options, "background", 0);
    l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}
#else
#endif


#if !defined(USE_SGX) && !defined(SGX_VERIFIES)
detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, "max",90);
    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    layer.random = option_find_int_quiet(options, "random", 0);
    layer.reorg = option_find_int_quiet(options, "reorg", 0);
    return layer;
}
#else
#endif

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    layer.noobject_scale =  option_find_float_quiet(options, "noobj", 1);
    layer.thresh =  option_find_float_quiet(options, "thresh",0);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) {
      // error("Layer before crop layer must output image.");
      LOG_ERROR("Layer before crop layer must output image.")
      abort();
    }

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

#if !defined(USE_SGX) && !defined(SGX_VERIFIES)
layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);
    int flatten = option_find_int_quiet(options, "flatten",0);
    int extra = option_find_int_quiet(options, "extra",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra);
    return layer;
}
#else 
#endif

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) {
      LOG_ERROR("Layer before maxpool layer must output image.");
      abort();
    }

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    return layer;
}

maxpool_layer parse_maxpool1D(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h == 1 && w && c)) error("Layer before maxpool1D layer must output image with height 1.");

    maxpool1D_layer layer = make_maxpool1D_layer(batch,h,w,c,size,stride,padding);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) {
      LOG_ERROR("Layer before avgpool layer must output image.");
      abort();
    }

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

avgpoolx_layer parse_avgpoolx(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);
    
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpoolx layer must output image.");

    avgpoolx_layer layer = make_avgpoolx_layer(batch,h,w,c,size,stride,padding);
    return layer;
}

avgpoolx_layer parse_avgpoolx1D(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);
    
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h == 1 && w && c)) error("Layer before avgpoolx1D layer must output image.");

    avgpoolx1D_layer layer = make_avgpoolx1D_layer(batch,h,w,c,size,stride,padding);
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params,PRNG& net_layer_rng_deriver)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability,net_layer_rng_deriver);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

#if !defined(USE_SGX_LAYERWISE) && !defined(SGX_VERIFIES)
//#ifndef USE_SGX_LAYERWISE
layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}
#endif

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    return l;
}

layer parse_shortcut(list *options, size_params params, network *net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    s.alpha = option_find_float_quiet(options, "alpha", 1);
    s.beta = option_find_float_quiet(options, "beta", 1);
    return s;
}

/* layer parse_l2norm(list *options, size_params params)
{
    layer l = make_l2norm_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
} */

#if !defined(USE_SGX) && !defined(SGX_VERIFIES)
//#ifndef USE_SGX_LAYERWISE
layer parse_logistic(list *options, size_params params)
{
    layer l = make_logistic_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}
#endif

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}

/* layer parse_upsample(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, "stride",2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
} */

route_layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = (int*)calloc(n, sizeof(int));
    int *sizes = (int*)calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
#ifndef USE_SGX
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
#else
#endif
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    int enclave_subdivs = option_find_int(options, "enclave_subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->gradient_clip = option_find_float_quiet(options, "gradient_update_clip", 0);
    #if defined(SGX_VERIFIES)
    net->batch /= subdivs;
    #elif defined (USE_SGX)
    // if (subdivs != enclave_subdivs) {
    //   LOG_ERROR('subdivisons for gpu and enclave do not match\n');
    //   // enclave_subdivs = subdivs;
    //   //abort();
    // }
    net->batch /= enclave_subdivs;
    #endif
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->enclave_subdivisions= enclave_subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center",0);
    net->clip = option_find_float_quiet(options, "clip", 0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c)) {
      // error("No input parameters supplied");
      LOG_ERROR("No input parameters supplied")
      abort();
    }

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) {
          LOG_ERROR("STEPS policy must have steps and scales in cfg file")
          // error("STEPS policy must have steps and scales in cfg file");
          abort();
        }

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = (int*)calloc(n, sizeof(int));
        float *scales = (float*)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}

#if !defined(USE_SGX_LAYERWISE) && !defined(USE_SGX_PURE)
network *parse_network_cfg(char *filename)
{
  LOG_TRACE("entered in parse network config\n");
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) {
      LOG_ERROR("network is empty!" "\n");
      abort();
    }
    
    network *net = make_network(sections->size - 1);
    net->gpu_index = gpu_index;
    size_params params;
    #if defined(USE_SGX)
    set_network_batch_randomness(0,*net);    
    #elif defined(SGX_VERIFIES)
    net->iter_batch_rng = batch_inp_rng;
    net->layer_rng_deriver = batch_layers_rng;
    #endif

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
#ifndef USE_SGX
        fprintf(stderr, "%5d ", count);
#else
#endif
        s = (section *)n->val;
        options = s->options;
        layer l = {};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params,*(net->layer_rng_deriver));
        } else if(lt == CONVOLUTIONAL1D){
            l = parse_convolutional1D(options, params);
        }
        /* else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        } */
        /* else if(lt == LOCAL){
            l = parse_local(options, params);
        } */
        else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }
        #if !defined(USE_SGX) && !defined (SGX_VERIFIES)
        else if(lt == LOGXENT){
            l = parse_logistic(options, params);
            
        }
        #endif
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
#if !defined(USE_SGX) && !defined (SGX_VERIFIES)
        else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == YOLO){
          l = parse_yolo(options, params);
        }else if(lt == ISEG){
            l = parse_iseg(options, params);
        }else if(lt == DETECTION){
          l = parse_detection(options, params);
        }
#else

#endif
         else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == NORMALIZATION){
          #if !defined(USE_SGX) && !defined (SGX_VERIFIES)
            l = parse_normalization(options, params);
          #endif
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == MAXPOOL1D){
            l = parse_maxpool1D(options, params);
        }
        #if !defined(USE_SGX) && !defined (SGX_VERIFIES)
        else if(lt == REORG){
            l = parse_reorg(options, params);
        }
        #else
        #endif
        else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == AVGPOOLX){
            l = parse_avgpoolx(options, params);
        }else if(lt == AVGPOOLX1D){
            l = parse_avgpoolx1D(options, params);
        }
        else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }
        /* else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        } */
        else if(lt == SHORTCUT){
            //#ifndef USE_SGX
            l = parse_shortcut(options, params, net);
            //#endif
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params,*(net->layer_rng_deriver));
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
#ifdef GPU
            l.output_gpu = net->layers[count-1].output_gpu;
            l.delta_gpu = net->layers[count-1].delta_gpu;
#endif
        }else{
#ifndef USE_SGX
            fprintf(stderr, "Type not recognized: %s\n", s->type);
#else
#endif
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
    #ifndef USE_SGX_LAYERWISE
    net->input = (float*)calloc(net->inputs*net->batch, sizeof(float));
    net->truth = (float*)calloc(net->truths*net->batch, sizeof(float));
    #else
    net->input = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(net->inputs*net->batch);
    net->truth = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(net->truths*net->batch);
    #endif
#ifdef GPU
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
#endif
    if(workspace_size){
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if(gpu_index >= 0){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }else {
            net->workspace = (float*)calloc(1, workspace_size);
        }
#else
        #ifndef USE_SGX_LAYERWISE
        net->workspace = (float*)calloc(1, workspace_size);
        #else
        net->workspace = sgx::trusted::SpecialBuffer<float>::GetNewSpecialBuffer(workspace_size/sizeof(float));
        #endif
#endif
    }
    LOG_TRACE("finished in parse network config\n");
    return net;
}
#endif

#if !defined(USE_SGX) && !defined (SGX_VERIFIES)
list *read_cfg(char *filename)
{
  FILE *file = fopen(filename, "r");
  if(file == 0) file_error(filename);
  char *line;
  int nu = 0;
  list *options = make_list();
  section *current = 0;
  while((line=fgetl(file)) != 0){
    ++ nu;
    strip(line);
    switch(line[0]){
    case '[':
      current = (section*)malloc(sizeof(section));
      list_insert(options, current);
      current->options = make_list();
      current->type = line;
      break;
    case '\0':
    case '#':
    case ';':
      free(line);
      break;
    default:
      if(!read_option(line, current->options)){
        fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
        free(line);
      }
      break;
    }
  }
  fclose(file);
  return options;
}
#else

/* filename is not an actual file name. It's a string containing contents offset
 * of the file.
 */
list *read_cfg(char *filename) {
  LOG_TRACE("entered read cfg\n");
  char file_content[20000];
  memset(file_content, 0, 20000);
  memcpy(file_content, filename, strlen(filename) + 1);
  char *line;
  char *temp;
  /* int nu = 0; */
  list *options = make_list();
  section *current = 0;
  line = strtok(file_content, "\n");
  while (line != NULL) {
    /* ++ nu; */
    strip(line);
    char *line_heap = (char*)calloc((strlen(line) + 1), sizeof(char));
    memcpy(line_heap, line, strlen(line) + 1);
    line_heap[strlen(line)] = '\0';
    /* printf("line is: %s : heap_line is %s\n",line,line_heap); */
    /* switch (line[0]) { */
    switch (line_heap[0]) {
    case '[':
      current = (section*)malloc(sizeof(section));
      list_insert(options, current);
      current->options = make_list();
      current->type = line_heap;
      break;
    case '\0':
    case '#':
    case ';':
      free(line_heap);
      break;
    default:
      /* if (!read_option(line, current->options)) { */
      if (!read_option(line_heap, current->options)) {
        /* fprintf(stderr, "Config file error line %d, could parse: %s\n",
         * nu,
         * line); */
        free(line_heap);
        abort();
      }
      break;
    }
    line = strtok(NULL, "\n");
  }
  LOG_TRACE("finished read cfg\n");
  return options;
}

#endif

#ifndef USE_SGX
void save_convolutional_weights_binary(layer l, FILE *fp) {
#ifdef GPU
  if (gpu_index >= 0) {
    pull_convolutional_layer(l);
  }
#endif
  binarize_weights(l.weights, l.n, l.c * l.size * l.size, l.binary_weights);
  int size = l.c * l.size * l.size;
  int i, j, k;
  fwrite(l.biases, sizeof(float), l.n, fp);
  if (l.batch_normalize) {
    fwrite(l.scales, sizeof(float), l.n, fp);
    fwrite(l.rolling_mean, sizeof(float), l.n, fp);
    fwrite(l.rolling_variance, sizeof(float), l.n, fp);
  }
  for (i = 0; i < l.n; ++i) {
    float mean = l.binary_weights[i * size];
    if (mean < 0)
      mean = -mean;
    fwrite(&mean, sizeof(float), 1, fp);
    for (j = 0; j < size / 8; ++j) {
      int index = i * size + j * 8;
      unsigned char c = 0;
      for (k = 0; k < 8; ++k) {
        if (j * 8 + k >= size)
          break;
        if (l.binary_weights[index + k] > 0)
          c = (c | 1 << k);
      }
      fwrite(&c, sizeof(char), 1, fp);
    }
  }
}
#else
#endif

#if !defined(USE_SGX) && !defined (SGX_VERIFIES)
void save_convolutional_weights(layer l, FILE *fp) {
  if (l.binary) {
    // save_convolutional_weights_binary(l, fp);
    // return;
  }
#ifdef GPU
  if (gpu_index >= 0) {
    pull_convolutional_layer(l);
  }
#endif
  int num = l.nweights;
  fwrite(l.biases, sizeof(float), l.n, fp);
  if (l.batch_normalize) {
    fwrite(l.scales, sizeof(float), l.n, fp);
    fwrite(l.rolling_mean, sizeof(float), l.n, fp);
    fwrite(l.rolling_variance, sizeof(float), l.n, fp);
  }
  fwrite(l.weights, sizeof(float), num, fp);
}

void save_batchnorm_weights(layer l, FILE *fp) {
#ifdef GPU
  if (gpu_index >= 0) {
    pull_batchnorm_layer(l);
  }
#endif
  fwrite(l.scales, sizeof(float), l.c, fp);
  fwrite(l.rolling_mean, sizeof(float), l.c, fp);
  fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp) {
#ifdef GPU
  if (gpu_index >= 0) {
    pull_connected_layer(l);
  }
#endif
  fwrite(l.biases, sizeof(float), l.outputs, fp);
  fwrite(l.weights, sizeof(float), l.outputs * l.inputs, fp);
  if (l.batch_normalize) {
    fwrite(l.scales, sizeof(float), l.outputs, fp);
    fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
    fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
  }
}

void save_weights_upto(network *net, char *filename, int cutoff) {
#ifdef GPU
  if (net->gpu_index >= 0) {
    cuda_set_device(net->gpu_index);
  }
#endif
  fprintf(stderr, "Saving weights to %s\n", filename);
  FILE *fp = fopen(filename, "wb");
  if (!fp)
    file_error(filename);

  int major = 0;
  int minor = 2;
  int revision = 0;
  fwrite(&major, sizeof(int), 1, fp);
  fwrite(&minor, sizeof(int), 1, fp);
  fwrite(&revision, sizeof(int), 1, fp);
  fwrite(net->seen, sizeof(size_t), 1, fp);

  int i;
  for (i = 0; i < net->n && i < cutoff; ++i) {
    layer l = net->layers[i];
    if (l.dontsave)
      continue;
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL || l.type == CONVOLUTIONAL1D) {
      save_convolutional_weights(l, fp);
    }
    if (l.type == CONNECTED) {
      save_connected_weights(l, fp);
    }
    if (l.type == BATCHNORM) {
      save_batchnorm_weights(l, fp);
    }
    if (l.type == RNN) {
      save_connected_weights(*(l.input_layer), fp);
      save_connected_weights(*(l.self_layer), fp);
      save_connected_weights(*(l.output_layer), fp);
    }
    if (l.type == LSTM) {
      save_connected_weights(*(l.wi), fp);
      save_connected_weights(*(l.wf), fp);
      save_connected_weights(*(l.wo), fp);
      save_connected_weights(*(l.wg), fp);
      save_connected_weights(*(l.ui), fp);
      save_connected_weights(*(l.uf), fp);
      save_connected_weights(*(l.uo), fp);
      save_connected_weights(*(l.ug), fp);
    }
    if (l.type == GRU) {
      if (1) {
        save_connected_weights(*(l.wz), fp);
        save_connected_weights(*(l.wr), fp);
        save_connected_weights(*(l.wh), fp);
        save_connected_weights(*(l.uz), fp);
        save_connected_weights(*(l.ur), fp);
        save_connected_weights(*(l.uh), fp);
      } else {
        save_connected_weights(*(l.reset_layer), fp);
        save_connected_weights(*(l.update_layer), fp);
        save_connected_weights(*(l.state_layer), fp);
      }
    }
    if (l.type == CRNN) {
      save_convolutional_weights(*(l.input_layer), fp);
      save_convolutional_weights(*(l.self_layer), fp);
      save_convolutional_weights(*(l.output_layer), fp);
    }
    if (l.type == LOCAL) {
#ifdef GPU
      if (gpu_index >= 0) {
        pull_local_layer(l);
      }
#endif
      int locations = l.out_w * l.out_h;
      int size = l.size * l.size * l.c * l.n * locations;
      fwrite(l.biases, sizeof(float), l.outputs, fp);
      fwrite(l.weights, sizeof(float), size, fp);
    }
  }
  fclose(fp);
}
void save_weights(network *net, char *filename) {
  save_weights_upto(net, filename, net->n);
}
#else
#endif

void transpose_matrix(float *a, int rows, int cols) {
  float *transpose = (float*)calloc(rows * cols, sizeof(float));
  int x, y;
  for (x = 0; x < rows; ++x) {
    for (y = 0; y < cols; ++y) {
      transpose[y * rows + x] = a[x * cols + y];
    }
  }
  memcpy(a, transpose, rows * cols * sizeof(float));
  free(transpose);
}

#if !defined(USE_SGX) && !defined(SGX_VERIFIES)
void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights(layer l, FILE *fp) {
  fread(l.scales, sizeof(float), l.c, fp);
  fread(l.rolling_mean, sizeof(float), l.c, fp);
  fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
  if (gpu_index >= 0) {
    push_batchnorm_layer(l);
  }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp) {
  fread(l.biases, sizeof(float), l.n, fp);
  if (l.batch_normalize && (!l.dontloadscales)) {
    fread(l.scales, sizeof(float), l.n, fp);
    fread(l.rolling_mean, sizeof(float), l.n, fp);
    fread(l.rolling_variance, sizeof(float), l.n, fp);
  }
  int size = l.c * l.size * l.size;
  int i, j, k;
  for (i = 0; i < l.n; ++i) {
    float mean = 0;
    fread(&mean, sizeof(float), 1, fp);
    for (j = 0; j < size / 8; ++j) {
      int index = i * size + j * 8;
      unsigned char c = 0;
      fread(&c, sizeof(char), 1, fp);
      for (k = 0; k < 8; ++k) {
        if (j * 8 + k >= size)
          break;
        l.weights[index + k] = (c & 1 << k) ? mean : -mean;
      }
    }
  }
#ifdef GPU
  if (gpu_index >= 0) {
    push_convolutional_layer(l);
  }
#endif
}

void load_convolutional_weights(layer l, FILE *fp) {
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    if(l.numload) l.n = l.numload;
    int num = l.c/l.groups*l.n*l.size*l.size;
    if (l.type == CONVOLUTIONAL1D) {
        num = l.c/l.groups*l.n*1*l.size;
    }
  fread(l.biases, sizeof(float), l.n, fp);
  if (l.batch_normalize && (!l.dontloadscales)) {
    fread(l.scales, sizeof(float), l.n, fp);
    fread(l.rolling_mean, sizeof(float), l.n, fp);
    fread(l.rolling_variance, sizeof(float), l.n, fp);
    if (0) {
      int i;
      for (i = 0; i < l.n; ++i) {
        printf("%g, ", l.rolling_mean[i]);
      }
      printf("\n");
      for (i = 0; i < l.n; ++i) {
        printf("%g, ", l.rolling_variance[i]);
      }
      printf("\n");
    }
    if (0) {
      fill_cpu(l.n, 0, l.rolling_mean, 1);
      fill_cpu(l.n, 0, l.rolling_variance, 1);
    }
    if (0) {
      int i;
      for (i = 0; i < l.n; ++i) {
        printf("%g, ", l.rolling_mean[i]);
      }
      printf("\n");
      for (i = 0; i < l.n; ++i) {
        printf("%g, ", l.rolling_variance[i]);
      }
      printf("\n");
    }
  }
  fread(l.weights, sizeof(float), num, fp);
  // if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
  if (l.flipped) {
    transpose_matrix(l.weights, l.c * l.size * l.size, l.n);
  }
// if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
#ifdef GPU
  if (gpu_index >= 0) {
    push_convolutional_layer(l);
  }
#endif
}

void load_weights_upto(network *net, char *filename, int start, int cutoff) {
#ifdef GPU
  if (net->gpu_index >= 0) {
    cuda_set_device(net->gpu_index);
  }
#endif
  fprintf(stderr, "Loading weights from %s...", filename);
  fflush(stdout);
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    file_error(filename);

  int major;
  int minor;
  int revision;
  fread(&major, sizeof(int), 1, fp);
  fread(&minor, sizeof(int), 1, fp);
  fread(&revision, sizeof(int), 1, fp);
  if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
    fread(net->seen, sizeof(size_t), 1, fp);
  } else {
    int iseen = 0;
    fread(&iseen, sizeof(int), 1, fp);
    *net->seen = iseen;
  }
  int transpose = (major > 1000) || (minor > 1000);

  int i;
  for (i = start; i < net->n && i < cutoff; ++i) {
    layer l = net->layers[i];
    if (l.dontload)
      continue;
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL  || l.type == CONVOLUTIONAL1D) {
      load_convolutional_weights(l, fp);
    }
    if (l.type == CONNECTED) {
      load_connected_weights(l, fp, transpose);
    }
    if (l.type == BATCHNORM) {
      load_batchnorm_weights(l, fp);
    }
    if (l.type == CRNN) {
      load_convolutional_weights(*(l.input_layer), fp);
      load_convolutional_weights(*(l.self_layer), fp);
      load_convolutional_weights(*(l.output_layer), fp);
    }
    if (l.type == RNN) {
      load_connected_weights(*(l.input_layer), fp, transpose);
      load_connected_weights(*(l.self_layer), fp, transpose);
      load_connected_weights(*(l.output_layer), fp, transpose);
    }
    if (l.type == LSTM) {
      load_connected_weights(*(l.wi), fp, transpose);
      load_connected_weights(*(l.wf), fp, transpose);
      load_connected_weights(*(l.wo), fp, transpose);
      load_connected_weights(*(l.wg), fp, transpose);
      load_connected_weights(*(l.ui), fp, transpose);
      load_connected_weights(*(l.uf), fp, transpose);
      load_connected_weights(*(l.uo), fp, transpose);
      load_connected_weights(*(l.ug), fp, transpose);
    }
    if (l.type == GRU) {
      if (1) {
        load_connected_weights(*(l.wz), fp, transpose);
        load_connected_weights(*(l.wr), fp, transpose);
        load_connected_weights(*(l.wh), fp, transpose);
        load_connected_weights(*(l.uz), fp, transpose);
        load_connected_weights(*(l.ur), fp, transpose);
        load_connected_weights(*(l.uh), fp, transpose);
      } else {
        load_connected_weights(*(l.reset_layer), fp, transpose);
        load_connected_weights(*(l.update_layer), fp, transpose);
        load_connected_weights(*(l.state_layer), fp, transpose);
      }
    }
    if (l.type == LOCAL) {
      int locations = l.out_w * l.out_h;
      int size = l.size * l.size * l.c * l.n * locations;
      fread(l.biases, sizeof(float), l.outputs, fp);
      fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
      if (gpu_index >= 0) {
        push_local_layer(l);
      }
#endif
    }
  }
  fprintf(stderr, "Done!\n");
  fclose(fp);
}

void load_weights(network *net, char *filename) {
  load_weights_upto(net, filename, 0, net->n);
}
#endif

#if defined(USE_SGX)
void load_connected_weights(layer l, size_t *index, int transpose)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    if(transpose){
        //transpose_matrix(l.weights, l.inputs, l.outputs);
        LOG_ERROR("Transpose not implemented yet!\n");
        abort();
    }
    //fread(l.biases, sizeof(float), l.outputs, fp);
    #if defined (USE_SGX)
    #if defined (USE_SGX_LAYERWISE)
    {
      auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_biases[0], sizeof(float)*l.outputs);
      CHECK_SGX_SUCCESS(ret, "could not read biases");
      *index+=sizeof(float)*l.outputs;
      l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
    }
    //fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    {
      for (int i=0;i<l.outputs;++i) {
        auto l_weights = l.weights->getItemsInRange(i*l.inputs, (i+1)*l.inputs);
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_weights[0], sizeof(float)*l.inputs);
        CHECK_SGX_SUCCESS(ret, "could not read weights");
        *index += sizeof(float)*l.inputs;
        l.weights->setItemsInRange(i*l.inputs, (i+1)*l.inputs,l_weights);
      }
    }
    //fread(l.scales, sizeof(float), l.outputs, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
      {
        auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_scales[0], sizeof(float)*l.outputs);
        CHECK_SGX_SUCCESS(ret, "could not read scales");
        *index+=sizeof(float)*l.outputs;
        l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
      }
      //fread(l.rolling_mean, sizeof(float), l.outputs, fp);
      {
        auto l_rolling_mean = l.rolling_mean->getItemsInRange(0, l.rolling_mean->getBufferSize());
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_mean[0], sizeof(float)*l.outputs);
        CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
        *index+=sizeof(float)*l.outputs;
        l.rolling_mean->setItemsInRange(0, l.rolling_mean->getBufferSize(),l_rolling_mean);
      }
      //fread(l.rolling_variance, sizeof(float), l.outputs, fp);
      {
        auto l_rolling_variance = l.rolling_variance->getItemsInRange(0, l.rolling_variance->getBufferSize());
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_variance[0], sizeof(float)*l.outputs);
        CHECK_SGX_SUCCESS(ret, "could not read variance mean");
        *index+=sizeof(float)*l.outputs;
        l.rolling_variance->setItemsInRange(0, l.rolling_variance->getBufferSize(),l_rolling_variance);
      }
    }
    #else
    ret = ocall_load_weights_plain(*index,  (unsigned char*) l.biases, sizeof(float)*l.outputs);
    CHECK_SGX_SUCCESS(ret, "could not read biases");
    *index+=sizeof(float)*l.outputs;

    for (int i=0;i<l.outputs;++i) {
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l.weights[i*l.inputs], sizeof(float)*l.inputs);
      CHECK_SGX_SUCCESS(ret, "could not read weights");
      *index += sizeof(float)*l.inputs;
    }
    if (l.batch_normalize && (!l.dontloadscales)){
      ret = ocall_load_weights_plain(*index,  (unsigned char*) l.scales, sizeof(float)*l.outputs);
      CHECK_SGX_SUCCESS(ret, "could not read scales");
      *index+=sizeof(float)*l.outputs;

      ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_mean, sizeof(float)*l.outputs);
      CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
      *index+=sizeof(float)*l.outputs;

      ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_variance, sizeof(float)*l.outputs);
      CHECK_SGX_SUCCESS(ret, "could not read variance mean");
      *index+=sizeof(float)*l.outputs;
    }
    #endif
    #endif
}

void load_batchnorm_weights(layer l, size_t *index) {
  sgx_status_t ret = SGX_ERROR_UNEXPECTED;
  #if defined (USE_SGX)
  #if defined (USE_SGX_LAYERWISE)
  //fread(l.scales, sizeof(float), l.c, fp);
  {
    auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_scales[0], sizeof(float)*l.c);
    CHECK_SGX_SUCCESS(ret, "could not read scales");
    *index+=sizeof(float)*l.c;
    l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
  }
  //fread(l.rolling_mean, sizeof(float), l.c, fp);
  {
    auto l_rolling_mean = l.rolling_mean->getItemsInRange(0, l.rolling_mean->getBufferSize());
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_mean[0], sizeof(float)*l.c);
    CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
    *index+=sizeof(float)*l.c;
    l.rolling_mean->setItemsInRange(0, l.rolling_mean->getBufferSize(),l_rolling_mean);
  }
  //fread(l.rolling_variance, sizeof(float), l.c, fp);
  {
    auto l_rolling_variance = l.rolling_variance->getItemsInRange(0, l.rolling_variance->getBufferSize());
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_variance[0], sizeof(float)*l.c);
    CHECK_SGX_SUCCESS(ret, "could not read variance mean");
    *index+=sizeof(float)*l.c;
    l.rolling_variance->setItemsInRange(0, l.rolling_variance->getBufferSize(),l_rolling_variance);
  }
  #else
  ret = ocall_load_weights_plain(*index,  (unsigned char*) l.scales, sizeof(float)*l.c);
  CHECK_SGX_SUCCESS(ret, "could not read scales");
  *index+=sizeof(float)*l.c;

  ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_mean, sizeof(float)*l.c);
  CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
  *index+=sizeof(float)*l.c;

  ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_variance, sizeof(float)*l.c);
  CHECK_SGX_SUCCESS(ret, "could not read variance mean");
  *index+=sizeof(float)*l.c;
  #endif
  #endif
}

void load_convolutional_weights_binary(layer l, size_t *index) {
  LOG_ERROR("load conv weights binary not implemented\n")
  abort();
  // fread(l.biases, sizeof(float), l.n, fp);
  // if (l.batch_normalize && (!l.dontloadscales)) {
  //   fread(l.scales, sizeof(float), l.n, fp);
  //   fread(l.rolling_mean, sizeof(float), l.n, fp);
  //   fread(l.rolling_variance, sizeof(float), l.n, fp);
  // }
  // int size = l.c * l.size * l.size;
  // int i, j, k;
  // for (i = 0; i < l.n; ++i) {
  //   float mean = 0;
  //   fread(&mean, sizeof(float), 1, fp);
  //   for (j = 0; j < size / 8; ++j) {
  //     int index = i * size + j * 8;
  //     unsigned char c = 0;
  //     fread(&c, sizeof(char), 1, fp);
  //     for (k = 0; k < 8; ++k) {
  //       if (j * 8 + k >= size)
  //         break;
  //       l.weights[index + k] = (c & 1 << k) ? mean : -mean;
  //     }
  //   }
  // }
}

void load_convolutional_weights(layer l, size_t *index) {
  sgx_status_t ret = SGX_ERROR_UNEXPECTED;
  if(l.binary){
    LOG_ERROR("load binary weights not implemented\n")
    abort();
      //load_convolutional_weights_binary(l, fp);
      //return;
  }
  if(l.numload) l.n = l.numload;
  int num = l.c/l.groups*l.n*l.size*l.size;
  if (l.type == CONVOLUTIONAL1D) {
      num = l.c/l.groups*l.n*1*l.size;
  }
  if (num != l.nweights) {
      LOG_ERROR("nweights is not equal num for loading\n")
      abort();
  }
  #if defined (USE_SGX)
  #if defined (USE_SGX_LAYERWISE)
  //fread(l.biases, sizeof(float), l.n, fp);
  {
    auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_biases[0], sizeof(float)*l.n);
    CHECK_SGX_SUCCESS(ret, "could not read biases");
    *index+=sizeof(float)*l.n;
    l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
  }
  if (l.batch_normalize && (!l.dontloadscales)) {
    //fread(l.scales, sizeof(float), l.n, fp);
    {
      auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_scales[0], sizeof(float)*l.n);
      CHECK_SGX_SUCCESS(ret, "could not read scales");
      *index+=sizeof(float)*l.n;
      l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
    }
    //fread(l.rolling_mean, sizeof(float), l.n, fp);
    {
      auto l_rolling_mean = l.rolling_mean->getItemsInRange(0, l.rolling_mean->getBufferSize());
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_mean[0], sizeof(float)*l.n);
      CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
      *index+=sizeof(float)*l.n;
      l.rolling_mean->setItemsInRange(0, l.rolling_mean->getBufferSize(),l_rolling_mean);
    }
    //fread(l.rolling_variance, sizeof(float), l.n, fp);
    {
      auto l_rolling_variance = l.rolling_variance->getItemsInRange(0, l.rolling_variance->getBufferSize());
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_variance[0], sizeof(float)*l.n);
      CHECK_SGX_SUCCESS(ret, "could not read variance mean");
      *index+=sizeof(float)*l.n;
      l.rolling_variance->setItemsInRange(0, l.rolling_variance->getBufferSize(),l_rolling_variance);
    }
    //fread(l.weights, sizeof(float), num, fp);
    {
      int buffer_size = (num) / (l.c/l.groups);
      for (int i=0;i<l.c/l.groups;++i) {
        auto l_weights = l.weights->getItemsInRange(i*buffer_size, (i+1)*buffer_size);
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_weights[0], sizeof(float)*buffer_size);
        CHECK_SGX_SUCCESS(ret, "could not read conv weights");
        l.weights->setItemsInRange(i*buffer_size, (i+1)*buffer_size,l_weights);
        *index+=sizeof(float)*buffer_size;
      }
    }
  }
  #else
  ret = ocall_load_weights_plain(*index,  (unsigned char*) l.biases, sizeof(float)*l.n);
  CHECK_SGX_SUCCESS(ret, "could not read biases");
  *index+=sizeof(float)*l.n;
  if (l.batch_normalize && (!l.dontloadscales)) {
    ret = ocall_load_weights_plain(*index,  (unsigned char*) l.scales, sizeof(float)*l.n);
    CHECK_SGX_SUCCESS(ret, "could not read scales");
    *index+=sizeof(float)*l.n;

    ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_mean, sizeof(float)*l.n);
    CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
    *index+=sizeof(float)*l.n;

    ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_variance, sizeof(float)*l.n);
    CHECK_SGX_SUCCESS(ret, "could not read variance mean");
    *index+=sizeof(float)*l.n;
  }
  int buffer_size = (num) / (l.c/l.groups);
  for (int i=0;i<l.c/l.groups;++i) {
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l.weights[i*buffer_size], sizeof(float)*buffer_size);
    CHECK_SGX_SUCCESS(ret, "could not read conv weights");
    *index+=sizeof(float)*buffer_size;
  }
  #endif
  #endif
  // if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
  if (l.flipped) {
    LOG_ERROR("loading with l.flipped not implemented\n")
    abort();
    //transpose_matrix(l.weights, l.c * l.size * l.size, l.n);
  }
// if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
}

void load_weights_upto(network *net, size_t *index, int start, int cutoff) {
  int major;
  int minor;
  int revision;
  sgx_status_t ret = SGX_ERROR_UNEXPECTED;
  //fread(&major, sizeof(int), 1, fp);
  ret = ocall_load_weights_plain(*index, (unsigned char*) &major, sizeof(int));
  CHECK_SGX_SUCCESS(ret, "could not read major version");
  *index+=sizeof(int);
  //fread(&minor, sizeof(int), 1, fp);
  ret = ocall_load_weights_plain(*index,  (unsigned char*) &minor, sizeof(int));
  CHECK_SGX_SUCCESS(ret, "could not read minor version");
  *index+=sizeof(int);
  //fread(&revision, sizeof(int), 1, fp);
  ret = ocall_load_weights_plain(*index,  (unsigned char*) &revision, sizeof(int));
  CHECK_SGX_SUCCESS(ret, "could not read revision version");
  *index+=sizeof(int);
  if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
    //fread(net->seen, sizeof(size_t), 1, fp);
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &(net->seen), sizeof(size_t));
    CHECK_SGX_SUCCESS(ret, "could not read net->seen version");
    *index+=sizeof(size_t);
  } else {
    int iseen = 0;
    //fread(&iseen, sizeof(int), 1, fp);
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &(iseen), sizeof(int));
    CHECK_SGX_SUCCESS(ret, "could not read isseen version");
    *index+=sizeof(int);
    *net->seen = iseen;
  }
  int transpose = (major > 1000) || (minor > 1000);

  int i;
  for (i = start; i < net->n && i < cutoff; ++i) {
    layer l = net->layers[i];
    if (l.dontload)
      continue;
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL  || l.type == CONVOLUTIONAL1D) {
      load_convolutional_weights(l, index);
    }
    if (l.type == CONNECTED) {
      load_connected_weights(l, index, transpose);
    }
    if (l.type == BATCHNORM) {
      load_batchnorm_weights(l, index);
    }
    if (l.type == CRNN) {
      load_convolutional_weights(*(l.input_layer), index);
      load_convolutional_weights(*(l.self_layer), index);
      load_convolutional_weights(*(l.output_layer), index);
    }
    if (l.type == RNN) {
      load_connected_weights(*(l.input_layer), index, transpose);
      load_connected_weights(*(l.self_layer), index, transpose);
      load_connected_weights(*(l.output_layer), index, transpose);
    }
    if (l.type == LSTM) {
      load_connected_weights(*(l.wi), index, transpose);
      load_connected_weights(*(l.wf), index, transpose);
      load_connected_weights(*(l.wo), index, transpose);
      load_connected_weights(*(l.wg), index, transpose);
      load_connected_weights(*(l.ui), index, transpose);
      load_connected_weights(*(l.uf), index, transpose);
      load_connected_weights(*(l.uo), index, transpose);
      load_connected_weights(*(l.ug), index, transpose);
    }
    if (l.type == GRU) {
      if (1) {
        load_connected_weights(*(l.wz), index, transpose);
        load_connected_weights(*(l.wr), index, transpose);
        load_connected_weights(*(l.wh), index, transpose);
        load_connected_weights(*(l.uz), index, transpose);
        load_connected_weights(*(l.ur), index, transpose);
        load_connected_weights(*(l.uh), index, transpose);
      } else {
        load_connected_weights(*(l.reset_layer), index, transpose);
        load_connected_weights(*(l.update_layer), index, transpose);
        load_connected_weights(*(l.state_layer), index, transpose);
      }
    }
    if (l.type == LOCAL) {
      LOG_ERROR("LOCAL layer load weights not implemented!\n")
      abort();

      // int locations = l.out_w * l.out_h;
      // int size = l.size * l.size * l.c * l.n * locations;
      // fread(l.biases, sizeof(float), l.outputs, fp);
      // fread(l.weights, sizeof(float), size, fp);
    }
  }
  //fprintf(stderr, "Done!\n");
  //fclose(fp);
}

void load_weights(network *net) {
  size_t * index = new size_t;
  *index = 0;
  load_weights_upto(net, index, 0, net->n);
  delete index;
}

void load_connected_weights_encrypted(layer l, size_t *index, int transpose,uint8_t* enc_key_weights,
                                      uint8_t* iv_weights, uint8_t* tag_weights)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    if(transpose){
        //transpose_matrix(l.weights, l.inputs, l.outputs);
        LOG_ERROR("Transpose not implemented yet!\n");
        abort();
    }
    //fread(l.biases, sizeof(float), l.outputs, fp);
    #if defined (USE_SGX)
    #if defined (USE_SGX_LAYERWISE)
    LOG_ERROR("Layerwise loading encrypted weights not implemented\n")
    abort();
    {
      auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_biases[0], sizeof(float)*l.outputs);
      CHECK_SGX_SUCCESS(ret, "could not read biases");
      *index+=sizeof(float)*l.outputs;
      l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
    }
    //fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    {
      for (int i=0;i<l.outputs;++i) {
        auto l_weights = l.weights->getItemsInRange(i*l.inputs, (i+1)*l.inputs);
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_weights[0], sizeof(float)*l.inputs);
        CHECK_SGX_SUCCESS(ret, "could not read weights");
        *index += sizeof(float)*l.inputs;
        l.weights->setItemsInRange(i*l.inputs, (i+1)*l.inputs,l_weights);
      }
    }
    //fread(l.scales, sizeof(float), l.outputs, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
      {
        auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_scales[0], sizeof(float)*l.outputs);
        CHECK_SGX_SUCCESS(ret, "could not read scales");
        *index+=sizeof(float)*l.outputs;
        l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
      }
      //fread(l.rolling_mean, sizeof(float), l.outputs, fp);
      {
        auto l_rolling_mean = l.rolling_mean->getItemsInRange(0, l.rolling_mean->getBufferSize());
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_mean[0], sizeof(float)*l.outputs);
        CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
        *index+=sizeof(float)*l.outputs;
        l.rolling_mean->setItemsInRange(0, l.rolling_mean->getBufferSize(),l_rolling_mean);
      }
      //fread(l.rolling_variance, sizeof(float), l.outputs, fp);
      {
        auto l_rolling_variance = l.rolling_variance->getItemsInRange(0, l.rolling_variance->getBufferSize());
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_variance[0], sizeof(float)*l.outputs);
        CHECK_SGX_SUCCESS(ret, "could not read variance mean");
        *index+=sizeof(float)*l.outputs;
        l.rolling_variance->setItemsInRange(0, l.rolling_variance->getBufferSize(),l_rolling_variance);
      }
    }
    #else
    uint8_t* temp_space = new uint8_t[l.nweights*sizeof(float)];
    if (!temp_space) {
      LOG_ERROR("Could not allocate temporary space!\n")
      abort();
    }
    //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.biases, sizeof(float)*l.outputs);
    ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.outputs,&iv_weights[0],&tag_weights[0],1);
    CHECK_SGX_SUCCESS(ret, "could not read biases");
    ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.outputs,(uint8_t*)l.biases,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
    CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");

    size_t req_buffer_bytes = l.nweights*sizeof(float);
    size_t interim_buff_bytes = SGX_OCALL_TRANSFER_BLOCK_SIZE;
    int q = req_buffer_bytes / interim_buff_bytes;
    int r = req_buffer_bytes % interim_buff_bytes;
    for (int i=0;i<q;++i) {
      //ret = ocall_load_weights_plain(*index,  (unsigned char*) &l.weights[i*l.inputs], sizeof(float)*l.inputs);      
      if (r == 0 && i == q - 1) {
        ret = ocall_load_weights_encrypted(i*interim_buff_bytes, 
                                          (unsigned char*) &temp_space[i*interim_buff_bytes], 
                                          interim_buff_bytes,&iv_weights[0],&tag_weights[0],1);
      }
      else {
        ret = ocall_load_weights_encrypted(i*interim_buff_bytes, 
                                          (unsigned char*) &temp_space[i*interim_buff_bytes], 
                                          interim_buff_bytes,&iv_weights[0],&tag_weights[0],0);
      }
      CHECK_SGX_SUCCESS(ret, "could not read weights");
    }
    if (r > 0) {
        ret = ocall_load_weights_encrypted(q*interim_buff_bytes, 
                                          (unsigned char*) &temp_space[q*interim_buff_bytes], 
                                          r,&iv_weights[0],&tag_weights[0],1);
        //r = 0;
    }
    ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                temp_space,sizeof(float)*l.nweights,(uint8_t*)l.weights,
                                iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
    CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
    if (l.batch_normalize && (!l.dontloadscales)){
      //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.scales, sizeof(float)*l.outputs);
      ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.outputs,&iv_weights[0],&tag_weights[0],1);
      CHECK_SGX_SUCCESS(ret, "could not read scales");
      ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.outputs,(uint8_t*)l.scales,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
      CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");

      //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_mean, sizeof(float)*l.outputs);
      ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.outputs,&iv_weights[0],&tag_weights[0],1);
      CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
      ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.outputs,(uint8_t*)l.rolling_mean,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
      CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
      
      //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_variance, sizeof(float)*l.outputs);
      ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.outputs,&iv_weights[0],&tag_weights[0],1);
      CHECK_SGX_SUCCESS(ret, "could not read variance mean");
      ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.outputs,(uint8_t*)l.rolling_variance,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
      CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
      
    }
    delete[] temp_space;
    #endif
    #endif
}

void load_batchnorm_weights_encrypted(layer l, size_t *index,
                            uint8_t* enc_key_weights,
                            uint8_t* iv_weights, uint8_t* tag_weights) {
  sgx_status_t ret = SGX_ERROR_UNEXPECTED;
  #if defined (USE_SGX)
  #if defined (USE_SGX_LAYERWISE)
  LOG_ERROR("Layerwise loading encrypted weights not implemented\n")
  abort();
  //fread(l.scales, sizeof(float), l.c, fp);
  {
    auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_scales[0], sizeof(float)*l.c);
    CHECK_SGX_SUCCESS(ret, "could not read scales");
    *index+=sizeof(float)*l.c;
    l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
  }
  //fread(l.rolling_mean, sizeof(float), l.c, fp);
  {
    auto l_rolling_mean = l.rolling_mean->getItemsInRange(0, l.rolling_mean->getBufferSize());
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_mean[0], sizeof(float)*l.c);
    CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
    *index+=sizeof(float)*l.c;
    l.rolling_mean->setItemsInRange(0, l.rolling_mean->getBufferSize(),l_rolling_mean);
  }
  //fread(l.rolling_variance, sizeof(float), l.c, fp);
  {
    auto l_rolling_variance = l.rolling_variance->getItemsInRange(0, l.rolling_variance->getBufferSize());
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_variance[0], sizeof(float)*l.c);
    CHECK_SGX_SUCCESS(ret, "could not read variance mean");
    *index+=sizeof(float)*l.c;
    l.rolling_variance->setItemsInRange(0, l.rolling_variance->getBufferSize(),l_rolling_variance);
  }
  #else
  uint8_t* temp_space = new uint8_t[sizeof(float)*l.c];
  //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.scales, sizeof(float)*l.c);
  ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.c,&iv_weights[0],&tag_weights[0],1);
  CHECK_SGX_SUCCESS(ret, "could not read scales");
  ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.c,(uint8_t*)l.scales,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
  CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");

  //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_mean, sizeof(float)*l.c);
  ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.c,&iv_weights[0],&tag_weights[0],1);
  CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
  ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.c,(uint8_t*)l.rolling_mean,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
  CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
  
  //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_variance, sizeof(float)*l.c);
  ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.c,&iv_weights[0],&tag_weights[0],1);
  CHECK_SGX_SUCCESS(ret, "could not read variance mean");
  ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.c,(uint8_t*)l.rolling_variance,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
  CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
  delete [] temp_space;
  #endif
  #endif
}


void load_convolutional_weights_encrypted(layer l, size_t *index,
                                          uint8_t* enc_key_weights,
                                          uint8_t* iv_weights, uint8_t* tag_weights) {
  sgx_status_t ret = SGX_ERROR_UNEXPECTED;
  if(l.binary) {
    LOG_ERROR("load binary weights not implemented\n")
    abort();
    // load_convolutional_weights_binary(l, fp);
    // return;
  }
  if(l.numload) l.n = l.numload;
  int num = l.c/l.groups*l.n*l.size*l.size;
  if (l.type == CONVOLUTIONAL1D) {
      num = l.c/l.groups*l.n*1*l.size;
  }
  if (num != l.nweights) {
      LOG_ERROR("nweights is not equal num for loading\n")
      abort();
  }
  #if defined (USE_SGX)
  #if defined (USE_SGX_LAYERWISE)
  LOG_ERROR("Layerwise loading encrypted weights not implemented\n")
  abort();
  //fread(l.biases, sizeof(float), l.n, fp);
  {
    auto l_biases = l.biases->getItemsInRange(0, l.biases->getBufferSize());
    ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_biases[0], sizeof(float)*l.n);
    CHECK_SGX_SUCCESS(ret, "could not read biases");
    *index+=sizeof(float)*l.n;
    l.biases->setItemsInRange(0, l.biases->getBufferSize(),l_biases);
  }
  if (l.batch_normalize && (!l.dontloadscales)) {
    //fread(l.scales, sizeof(float), l.n, fp);
    {
      auto l_scales = l.scales->getItemsInRange(0, l.scales->getBufferSize());
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_scales[0], sizeof(float)*l.n);
      CHECK_SGX_SUCCESS(ret, "could not read scales");
      *index+=sizeof(float)*l.n;
      l.scales->setItemsInRange(0, l.scales->getBufferSize(),l_scales);
    }
    //fread(l.rolling_mean, sizeof(float), l.n, fp);
    {
      auto l_rolling_mean = l.rolling_mean->getItemsInRange(0, l.rolling_mean->getBufferSize());
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_mean[0], sizeof(float)*l.n);
      CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
      *index+=sizeof(float)*l.n;
      l.rolling_mean->setItemsInRange(0, l.rolling_mean->getBufferSize(),l_rolling_mean);
    }
    //fread(l.rolling_variance, sizeof(float), l.n, fp);
    {
      auto l_rolling_variance = l.rolling_variance->getItemsInRange(0, l.rolling_variance->getBufferSize());
      ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_rolling_variance[0], sizeof(float)*l.n);
      CHECK_SGX_SUCCESS(ret, "could not read variance mean");
      *index+=sizeof(float)*l.n;
      l.rolling_variance->setItemsInRange(0, l.rolling_variance->getBufferSize(),l_rolling_variance);
    }
    //fread(l.weights, sizeof(float), num, fp);
    {
      int buffer_size = (num) / (l.c/l.groups);
      for (int i=0;i<l.c/l.groups;++i) {
        auto l_weights = l.weights->getItemsInRange(i*buffer_size, (i+1)*buffer_size);
        ret = ocall_load_weights_plain(*index,  (unsigned char*) &l_weights[0], sizeof(float)*buffer_size);
        CHECK_SGX_SUCCESS(ret, "could not read conv weights");
        l.weights->setItemsInRange(i*buffer_size, (i+1)*buffer_size,l_weights);
        *index+=sizeof(float)*buffer_size;
      }
    }
  }
  #else
  uint8_t* temp_space = new uint8_t[l.nweights*sizeof(float)];
  //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.biases, sizeof(float)*l.n);
  ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.n,&iv_weights[0],&tag_weights[0],1);
  CHECK_SGX_SUCCESS(ret, "could not read biases");
  ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.n,(uint8_t*)l.biases,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
  CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
  if (l.batch_normalize && (!l.dontloadscales)) {
    //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.scales, sizeof(float)*l.n);
    ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.n,&iv_weights[0],&tag_weights[0],1);
    CHECK_SGX_SUCCESS(ret, "could not read scales");
    ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.n,(uint8_t*)l.scales,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
    CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");

    //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_mean, sizeof(float)*l.n);
    ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.n,&iv_weights[0],&tag_weights[0],1);
    CHECK_SGX_SUCCESS(ret, "could not read rolling mean");
    ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.n,(uint8_t*)l.rolling_mean,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
    CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");

    //ret = ocall_load_weights_plain(*index,  (unsigned char*) l.rolling_variance, sizeof(float)*l.n);
    ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(float)*l.n,&iv_weights[0],&tag_weights[0],1);
    CHECK_SGX_SUCCESS(ret, "could not read variance mean");
    ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(float)*l.n,(uint8_t*)l.rolling_variance,
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
    CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
  }
  size_t req_buffer_bytes = l.nweights*sizeof(float);
  size_t interim_buff_bytes = SGX_OCALL_TRANSFER_BLOCK_SIZE;
  int q = req_buffer_bytes / interim_buff_bytes;
  int r = req_buffer_bytes % interim_buff_bytes;
  //ret = ocall_load_weights_plain(*index,  (unsigned char*) &l.weights[i*buffer_size], sizeof(float)  
  for (int i=0;i<q;++i) {
    if (r == 0 && i == q - 1) {
      ret = ocall_load_weights_encrypted(i*interim_buff_bytes, 
                                        (unsigned char*) &temp_space[i*interim_buff_bytes], 
                                        interim_buff_bytes,&iv_weights[0],&tag_weights[0],1);
    }
    else {
      ret = ocall_load_weights_encrypted(i*interim_buff_bytes, 
                                        (unsigned char*) &temp_space[i*interim_buff_bytes], 
                                        interim_buff_bytes,&iv_weights[0],&tag_weights[0],0);
    }
    CHECK_SGX_SUCCESS(ret, "could not read weights");
  }
  if (r > 0) {
      ret = ocall_load_weights_encrypted(q*interim_buff_bytes, 
                                        (unsigned char*) &temp_space[q*interim_buff_bytes], 
                                        r,&iv_weights[0],&tag_weights[0],1);
      //r = 0;
  }
  ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                temp_space,sizeof(float)*num,(uint8_t*)l.weights,
                                iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
  CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
  delete [] temp_space;
  #endif
  #endif
  // if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
  if (l.flipped) {
    LOG_ERROR("loading with l.flipped not implemented\n")
    abort();
    //transpose_matrix(l.weights, l.c * l.size * l.size, l.n);
  }
// if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
}

void load_weights_upto_encrypted(network *net, size_t *index, int start, int cutoff) {
  int major;
  int minor;
  int revision;

  uint8_t enc_key_weights[AES_GCM_KEY_SIZE] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  uint8_t iv_weights[AES_GCM_IV_SIZE];
  uint8_t tag_weights[AES_GCM_TAG_SIZE];

  uint8_t temp_space[8];

  sgx_status_t ret = SGX_ERROR_UNEXPECTED;
  //fread(&major, sizeof(int), 1, fp);
  ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(int),&iv_weights[0],&tag_weights[0],1);
  CHECK_SGX_SUCCESS(ret, "could not read major version");
  ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(int),(uint8_t*)&major, 
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
  CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");

  //fread(&minor, sizeof(int), 1, fp);
  ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(int),&iv_weights[0],&tag_weights[0],1);
  CHECK_SGX_SUCCESS(ret, "could not read minor version");
  ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(int),(uint8_t*)&minor, 
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
  CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
  //fread(&revision, sizeof(int), 1, fp);
  ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(int),&iv_weights[0],&tag_weights[0],1);
  CHECK_SGX_SUCCESS(ret, "could not read revision version");
  ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(int),(uint8_t*)&revision, 
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
  CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
  if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
    //fread(net->seen, sizeof(size_t), 1, fp);
    ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(size_t),&iv_weights[0],&tag_weights[0],1);
    CHECK_SGX_SUCCESS(ret, "could not read net->seen version");
    ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(size_t),(uint8_t*)net->seen, 
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
    CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
  } else {
    int iseen = 0;
    //fread(&iseen, sizeof(int), 1, fp);
    ret = ocall_load_weights_encrypted(0, (unsigned char*) temp_space, sizeof(int),&iv_weights[0],&tag_weights[0],1);
    CHECK_SGX_SUCCESS(ret, "could not read isseen version");
    ret = sgx_rijndael128GCM_decrypt((const sgx_aes_gcm_128bit_key_t*)enc_key_weights,
                                  temp_space,sizeof(int),(uint8_t*)&iseen, 
                                  iv_weights, AES_GCM_IV_SIZE, NULL, 0, 
                                  (const sgx_aes_gcm_128bit_tag_t*)tag_weights);
    CHECK_SGX_SUCCESS(ret, "aes decrypt caused problem\n");
    *net->seen = iseen;
  }
  int transpose = (major > 1000) || (minor > 1000);

  int i;
  for (i = start; i < net->n && i < cutoff; ++i) {
    layer l = net->layers[i];
    if (l.dontload)
      continue;
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL  || l.type == CONVOLUTIONAL1D) {
      load_convolutional_weights_encrypted(l, index,enc_key_weights,iv_weights,tag_weights);
    }
    if (l.type == CONNECTED) {
      load_connected_weights_encrypted(l, index, transpose,enc_key_weights,iv_weights,tag_weights);
    }
    if (l.type == BATCHNORM) {
      load_batchnorm_weights_encrypted(l, index,enc_key_weights,iv_weights,tag_weights);
    }
    if (l.type == CRNN) {
      load_convolutional_weights_encrypted(*(l.input_layer), index,enc_key_weights,iv_weights,tag_weights);
      load_convolutional_weights_encrypted(*(l.self_layer), index,enc_key_weights,iv_weights,tag_weights);
      load_convolutional_weights_encrypted(*(l.output_layer), index,enc_key_weights,iv_weights,tag_weights);
    }
    if (l.type == RNN) {
      load_connected_weights_encrypted(*(l.input_layer), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.self_layer), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.output_layer), index, transpose,enc_key_weights,iv_weights,tag_weights);
    }
    if (l.type == LSTM) {
      load_connected_weights_encrypted(*(l.wi), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.wf), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.wo), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.wg), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.ui), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.uf), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.uo), index, transpose,enc_key_weights,iv_weights,tag_weights);
      load_connected_weights_encrypted(*(l.ug), index, transpose,enc_key_weights,iv_weights,tag_weights);
    }
    if (l.type == GRU) {
      if (1) {
        load_connected_weights_encrypted(*(l.wz), index, transpose,enc_key_weights,iv_weights,tag_weights);
        load_connected_weights_encrypted(*(l.wr), index, transpose,enc_key_weights,iv_weights,tag_weights);
        load_connected_weights_encrypted(*(l.wh), index, transpose,enc_key_weights,iv_weights,tag_weights);
        load_connected_weights_encrypted(*(l.uz), index, transpose,enc_key_weights,iv_weights,tag_weights);
        load_connected_weights_encrypted(*(l.ur), index, transpose,enc_key_weights,iv_weights,tag_weights);
        load_connected_weights_encrypted(*(l.uh), index, transpose,enc_key_weights,iv_weights,tag_weights);
      } else {
        load_connected_weights_encrypted(*(l.reset_layer), index, transpose,enc_key_weights,iv_weights,tag_weights);
        load_connected_weights_encrypted(*(l.update_layer), index, transpose,enc_key_weights,iv_weights,tag_weights);
        load_connected_weights_encrypted(*(l.state_layer), index, transpose,enc_key_weights,iv_weights,tag_weights);
      }
    }
    if (l.type == LOCAL) {
      LOG_ERROR("LOCAL layer load weights not implemented!\n")
      abort();

      // int locations = l.out_w * l.out_h;
      // int size = l.size * l.size * l.c * l.n * locations;
      // fread(l.biases, sizeof(float), l.outputs, fp);
      // fread(l.weights, sizeof(float), size, fp);
    }
  }
  //fprintf(stderr, "Done!\n");
  //fclose(fp);
}

void load_weights_encrypted(network *net) {
  size_t * index = new size_t;
  *index = 0;
  load_weights_upto_encrypted(net, index, 0, net->n);
  delete index;

}
#endif

#if defined (USE_SGX) && defined(USE_SGX_BLOCKING)
void parse_net_options_blocked(list *options, network_blocked *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center",0);
    net->clip = option_find_float_quiet(options, "clip", 0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c)) {
      // error("No input parameters supplied");
      LOG_ERROR("No input parameters supplied")
      abort();
    }

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) {
          LOG_ERROR("STEPS policy must have steps and scales in cfg file")
          // error("STEPS policy must have steps and scales in cfg file");
          abort();
        }

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = (int*)calloc(n, sizeof(int));
        float *scales = (float*)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

convolutional_layer_blocked parse_convolutional_blocked(list *options, size_params_blocked params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) {
      LOG_ERROR("Layer before convolutional layer must output image.");
      abort();
    }

    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer_blocked layer = make_convolutional_layer_blocked(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}

layer_blocked parse_activation_blocked(list *options, size_params_blocked params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer_blocked l = make_activation_layer_blocked(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}

layer_blocked parse_logistic_blocked(list *options, size_params_blocked params)
{
    layer_blocked l = make_logistic_layer_blocked(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}

layer_blocked parse_connected_blocked(list *options, size_params_blocked params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer_blocked l = make_connected_layer_blocked(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    return l;
}

crop_layer_blocked parse_crop_blocked(list *options, size_params_blocked params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) {
      // error("Layer before crop layer must output image.");
      LOG_ERROR("Layer before crop layer must output image.")
      abort();
    }

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer_blocked l = make_crop_layer_blocked(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

cost_layer_blocked parse_cost_blocked(list *options, size_params_blocked params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer_blocked layer = make_cost_layer_blocked(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    layer.noobject_scale =  option_find_float_quiet(options, "noobj", 1);
    layer.thresh =  option_find_float_quiet(options, "thresh",0);
    return layer;
}

softmax_layer_blocked parse_softmax_blocked(list *options, size_params_blocked params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    softmax_layer_blocked l = make_softmax_layer_blocked(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
#ifndef USE_SGX
    if (tree_file) layer.softmax_tree = read_tree(tree_file);
#else
#endif
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, "spatial", 0);
    l.noloss =  option_find_int_quiet(options, "noloss", 0);
    return l;
}

layer_blocked parse_normalization_blocked(list *options, size_params_blocked params)
{
  LOG_ERROR("blocked normalization layer not yet implemented")
  layer_blocked l = {};
  return l;
  abort();
    /* float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l; */
}

layer_blocked parse_batchnorm_blocked(list *options, size_params_blocked params)
{
    layer_blocked l = make_batchnorm_layer_blocked(params.batch, params.w, params.h, params.c);
    return l;
}

maxpool_layer_blocked parse_maxpool_blocked(list *options, size_params_blocked params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) {
      LOG_ERROR("Layer before maxpool layer must output image.");
      abort();
    }

    maxpool_layer_blocked layer = make_maxpool_layer_blocked(batch,h,w,c,size,stride,padding);
    return layer;
}

avgpool_layer_blocked parse_avgpool_blocked(list *options, size_params_blocked params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) {
      LOG_ERROR("Layer before avgpool layer must output image.");
      abort();
    }

    avgpool_layer_blocked layer = make_avgpool_layer_blocked(batch,w,h,c);
    return layer;
}

dropout_layer_blocked parse_dropout_blocked(list *options, size_params_blocked params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer_blocked layer = make_dropout_layer_blocked(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

network_blocked *parse_network_cfg_blocked(char *filename)
{

  LOG_TRACE("entered in parse network config\n");
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) {
      LOG_ERROR("network is empty!" "\n");
      abort();
    }
    network_blocked *net = make_network_blocked(sections->size - 1);
    net->gpu_index = gpu_index;
    size_params_blocked params;
    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) {
      LOG_ERROR("First section must be [net] or [network] \n");
      abort();
    }
    parse_net_options_blocked(options, net);
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
    LOG_INFO("network type is :%s\n",s->type);
    free_section(s);
    LOG_DEBUG("free_Section was successful" "\n");
    while(n){
        params.index = count;
        s = (section *)n->val;
        options = s->options;
        layer_blocked l = {};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional_blocked(options, params);
        }
        /* else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        } */
        /* else if(lt == LOCAL){
            l = parse_local(options, params);
        } */
        else if(lt == ACTIVE){
            l = parse_activation_blocked(options, params);
        }else if(lt == LOGXENT){
            l = parse_logistic_blocked(options, params);
        }
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
            l = parse_connected_blocked(options, params);
        }else if(lt == CROP){
            l = parse_crop_blocked(options, params);
        }else if(lt == COST){
            l = parse_cost_blocked(options, params);
        }
#ifndef USE_SGX
        else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == YOLO){
          l = parse_yolo(options, params);
        }else if(lt == DETECTION){
          l = parse_detection(options, params);
        }
#else

#endif
         else if(lt == SOFTMAX){
            l = parse_softmax_blocked(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == NORMALIZATION){
            l = parse_normalization_blocked(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm_blocked(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool_blocked(options, params);
        }
        #ifndef USE_SGX
        else if(lt == REORG){
            l = parse_reorg(options, params);
        }
        #else
        #endif
        else if(lt == AVGPOOL){
            l = parse_avgpool_blocked(options, params);
        }
        /* else if(lt == ROUTE){
            l = parse_route(options, params, net);
        } */
        /* else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        } */
        else if(lt == SHORTCUT){
            // l = parse_shortcut(options, params, net);
            LOG_ERROR("Shortcut layer blocked not implemented!")
            abort();
        }else if(lt == DROPOUT){
            l = parse_dropout_blocked(options, params);
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
#ifdef GPU
            l.output_gpu = net->layers[count-1].output_gpu;
            l.delta_gpu = net->layers[count-1].delta_gpu;
#endif
        }else{
#ifndef USE_SGX
            fprintf(stderr, "Type not recognized: %s\n", s->type);
#else
#endif
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
    layer_blocked out = get_network_output_layer_blocked(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    // net->input = (float*)calloc(net->inputs*net->batch, sizeof(float));
    net->input = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({net->inputs*net->batch});
    // net->truth = (float*)calloc(net->truths*net->batch, sizeof(float));
    net->truth = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({net->inputs*net->batch});
#ifdef GPU
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
#endif
    if(workspace_size){
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if(gpu_index >= 0){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }else {
            net->workspace = calloc(1, workspace_size);
        }
#else
        // net->workspace = (float*)calloc(1, workspace_size);
        net->workspace = sgx::trusted::BlockedBuffer<float, 1>::MakeBlockedBuffer({(int64_t)workspace_size/(int64_t)sizeof(float)});
#endif
    }
    LOG_TRACE("finished in parse network config\n");
    return net;
}
#endif

#pragma GCC diagnostic pop
