#pragma once
#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef USE_SGX
#include <pthread.h>
#endif

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
#include "BlockEngine.hpp"
#endif

#define SECRET_NUM -1234
extern int gpu_index;

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifndef __cplusplus
    #ifdef OPENCV
    #include "opencv2/highgui/highgui_c.h"
    #include "opencv2/imgproc/imgproc_c.h"
    #include "opencv2/core/version.hpp"
    #if CV_MAJOR_VERSION == 3
    #include "opencv2/videoio/videoio_c.h"
    #include "opencv2/imgcodecs/imgcodecs_c.h"
    #endif
    #endif
#endif

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
struct network_blocked;
typedef struct network_blocked network_blocked;

struct layer_blocked;
typedef struct layer_blocked layer_blocked;
#endif

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *dc_gpu; 

    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float * binary_input_gpu;
    float * binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_change_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
struct layer_blocked{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward_blocked)   (struct layer_blocked, struct network_blocked);
    void (*backward_blocked)  (struct layer_blocked, struct network_blocked);
    void (*update_blocked)    (struct layer_blocked, update_args);
    
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    //int *mask;
    std::shared_ptr<sgx::trusted::BlockedBuffer<int,1>>  mask;

    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    //char  * cweights;
    std::shared_ptr<sgx::trusted::BlockedBuffer<char,1>>  cweights;
    // int   * indexes;
    std::shared_ptr<sgx::trusted::BlockedBuffer<int,1>>  indexes;
    // int   * input_layers;
    std::shared_ptr<sgx::trusted::BlockedBuffer<int,1>>  input_layers;
    // int   * input_sizes;
    std::shared_ptr<sgx::trusted::BlockedBuffer<int,1>>  input_sizes;
    // int   * map;
    std::shared_ptr<sgx::trusted::BlockedBuffer<int,1>>  map;
    // float * rand;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  rand;
    float * cost;
    // std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  cost;
    // float * state;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  state;
    // float * prev_state;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  prev_state;
    // float * forgot_state;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  forgot_state;
    // float * forgot_delta;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  forgot_delta;
    // float * state_delta;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  state_delta;
    // float * combine_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  combine_cpu;
    // float * combine_delta_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  combine_delta_cpu;
    // float * concat;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  concat;
    //float * concat_delta;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  concat_delta;
    // float * binary_weights;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  binary_weights;

    // float * biases;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  biases;
    //float * bias_updates;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  bias_updates;


    // float * scales;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  scales;
    // float * scale_updates;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  scale_updates;

    // float * weights;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  weights;
    // float * weight_updates;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  weight_updates;

    // float * delta;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  delta;
    // float * output;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  output;

    // float * loss;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  loss;
    // float * squared;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  squared;
    // float * norms;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  norms;

    // float * spatial_mean;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  spatial_mean;
    // float * mean;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  mean;
    // float * variance;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  variance;

    // float * mean_delta;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  mean_delta;
    // float * variance_delta;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  variance_delta;

    // float * rolling_mean;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  rolling_mean;
    // float * rolling_variance;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  rolling_variance;

    // float * x;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  x;
    // float * x_norm;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  x_norm;

    // float * m;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  m;
    // float * v;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  v;
    
    // float * bias_m;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  bias_m;
    // float * bias_v;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  bias_v;
    // float * scale_m;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  scale_m;
    // float * scale_v;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  scale_v;


    // float *z_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  z_cpu;
    // float *r_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  r_cpu;
    // float *h_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  h_cpu;
    // float * prev_state_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  prev_state_cpu;

    // float *temp_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  temp_cpu;
    // float *temp2_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  temp2_cpu;
    // float *temp3_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  temp3_cpu;

    // float *dh_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  dh_cpu;
    // float *hh_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  hh_cpu;
    // float *prev_cell_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  prev_cell_cpu;
    // float *cell_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  cell_cpu;

    // float *f_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  f_cpu;
    // float *i_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  i_cpu;
    // float *g_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  g_cpu;
    // float *o_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  o_cpu;
    // float *c_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  c_cpu;
    // float *dc_cpu;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  dc_cpu;

    // float * binary_input;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  binary_input;

    struct layer_blocked *input_layer;
    struct layer_blocked *self_layer;
    struct layer_blocked *output_layer;

    struct layer_blocked *reset_layer;
    struct layer_blocked *update_layer;
    struct layer_blocked *state_layer;

    struct layer_blocked *input_gate_layer;
    struct layer_blocked *state_gate_layer;
    struct layer_blocked *input_save_layer;
    struct layer_blocked *state_save_layer;
    struct layer_blocked *input_state_layer;
    struct layer_blocked *state_state_layer;

    struct layer_blocked *input_z_layer;
    struct layer_blocked *state_z_layer;

    struct layer_blocked *input_r_layer;
    struct layer_blocked *state_r_layer;

    struct layer_blocked *input_h_layer;
    struct layer_blocked *state_h_layer;
	
    struct layer_blocked *wz;
    struct layer_blocked *uz;
    struct layer_blocked *wr;
    struct layer_blocked *ur;
    struct layer_blocked *wh;
    struct layer_blocked *uh;
    struct layer_blocked *uo;
    struct layer_blocked *wo;
    struct layer_blocked *uf;
    struct layer_blocked *wf;
    struct layer_blocked *ui;
    struct layer_blocked *wi;
    struct layer_blocked *ug;
    struct layer_blocked *wg;
    tree *softmax_tree;
    size_t workspace_size;

};
#endif


void free_layer(layer);

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
void free_layer_blocked(layer_blocked);
#endif

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
#endif

} network;

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
typedef struct network_blocked{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer_blocked *layers;

    // float *output;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    // std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  scales;
    int   *steps;
    // std::shared_ptr<sgx::trusted::BlockedBuffer<int,1>>  steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    // float *input;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  input;
    // float *truth;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  truth;
    // float *delta;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  delta;
    // float *workspace;
    std::shared_ptr<sgx::trusted::BlockedBuffer<float,1>>  workspace;
    int train;
    int index;
    float *cost;
    float clip;
} network_blocked;
#endif

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;


typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;


network *load_network(char *cfg, char *weights, int clear);
#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
network_blocked *load_network_blocked(char *cfg, char *weights, int clear);
#endif
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

#ifndef USE_SGX
pthread_t load_data(load_args args);
#else
#endif
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
void forward_network_blocked(network_blocked *net);
void backward_network_blocked(network_blocked *net);
void update_network_blocked(network_blocked *net);
#endif

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
float dot_cpu_blocked(int N, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, 
                        int INCX, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &Y, int INCY);
void axpy_cpu_blocked(int N, float ALPHA, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &Y, int INCY);
void copy_cpu_blocked(int N, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &Y, int INCY);
void scal_cpu_blocked(int N, float ALPHA, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX);
void fill_cpu_blocked(int N, float ALPHA, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX);
void normalize_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance, int batch, int filters, int spatial);
void softmax_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &input,int input_offset, int n, float temp, int stride, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output,int output_offset);

float train_network_sgd_blocked(network_blocked *net, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &d, int n);
float train_network_datum_blocked(network_blocked *net);
void free_network_blocked(network_blocked *net);
void set_batch_network_blocked(network_blocked *net, int b);
void set_temp_network_blocked(network_blocked *net, float t);
float get_current_rate_blocked(network_blocked *net);
size_t get_current_batch_blocked(network_blocked *net);
layer_blocked get_network_output_layer_blocked(network_blocked *net);
float network_accuracy_blocked(network_blocked *net, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & d);
std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> network_predict_data_blocked(network_blocked *net, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & test);

int network_width_blocked(network_blocked *net);
int network_height_blocked(network_blocked *net);
float train_network_blocked(network_blocked *net);

float sum_array_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> & a, int n,int offset);
network_blocked *parse_network_cfg_blocked(char *filename);
#endif

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

void cuda_set_device(int n);
void cuda_free(float *x_gpu);
float *cuda_make_array(float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
float cuda_mag_array(float *x_gpu, size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);

void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

float train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image_png(image im, const char *name);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
void save_image(image p, const char *name);
void show_image(image p, const char *name);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, float rad);
void visualize_network(network *net);
float box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);
float *network_predict(network *net, float *input);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);

#ifndef __cplusplus
#ifdef OPENCV
image get_image_from_stream(CvCapture *cap);
#endif
#endif
void free_image(image m);
float train_network(network *net, data d);
#ifndef USE_SGX
pthread_t load_data_in_thread(load_args args);
#else
#endif
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
#ifndef USE_SGX
char *fgetl(FILE *fp);
#else
#endif
void strip(char *s);
#ifndef USE_SGX
float sec(clock_t clocks);
void smooth_data(data d);
#else
#endif
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

#endif
