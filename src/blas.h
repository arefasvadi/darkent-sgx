#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void mult_add_into_cpu(int N, float *X, float *Y, float *Z);

void const_cpu(int N, float ALPHA, float *X, int INCX);
void constrain_gpu(int N, float ALPHA, float * X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc);

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

// void flatten(float *x, int size, int layers, int batch, int forward);
// void pm(int M, int N, float *A);
// float *random_matrix(int rows, int cols);
// void time_random_matrix(int TA, int TB, int m, int k, int n);
// void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

// void test_blas();

// void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
// void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
// void mult_add_into_cpu(int N, float *X, float *Y, float *Z);

#ifdef GPU
#include "cuda.h"
#include "tree.h"

void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void add_gpu(int N, float ALPHA, float * X, int INCX);
void supp_gpu(int N, float ALPHA, float * X, int INCX);
void mask_gpu(int N, float * X, float mask_num, float * mask, float val);
void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);
void const_gpu(int N, float ALPHA, float *X, int INCX);
void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);

void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial);

void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);

void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);

void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc);
void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
void mult_add_into_gpu(int num, float *a, float *b, float *c);
void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);

void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t);

void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out);
void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier);
void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#endif
#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
void const_cpu_blocked(int N, float ALPHA, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX);
// void constrain_gpu(int N, float ALPHA, float * X, int INCX);
void pow_cpu_blocked(int N, float ALPHA, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &Y, int INCY);
void mul_cpu_blocked(int N, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &Y, int INCY);

// int test_gpu_blas();
void shortcut_cpu_blocked(int batch, int w1, int h1, int c1, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &add, int w2, int h2, int c2, float s1, float s2, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &out);

void mean_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean);
void variance_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance);

void scale_bias_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &scales, int batch, int n, int size);

void backward_scale_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x_norm, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, int batch, int n, int size, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &scale_updates);

void mean_delta_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean_delta);

void  variance_delta_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance_delta);

void normalize_delta_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean_delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance_delta, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta);

void l2normalize_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &dx, int batch, int filters, int spatial);

void smooth_l1_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error);

void l2_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error);

void l1_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error);

void logistic_x_ent_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error);

void softmax_x_ent_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error);

void weighted_sum_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &a, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &b, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &s, int num, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &c);

void weighted_delta_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &a, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &b, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &s, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &da, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &db, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &ds, int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &dc);

void softmax_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &input,int input_offset, int n, float temp, int stride, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output,int output_offset);

void softmax_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output);

void upsample_cpu(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &in, int w, int h, int c, int batch, int stride, int forward, float scale, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &out);

#endif
#endif
