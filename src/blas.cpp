#include "blas.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = (float*)calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial)
{
    int b,f,i;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < spatial; ++i){
            float sum = 0;
            for(f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                sum += powf(x[index], 2);
            }
            sum = sqrtf(sum);
            for(f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                x[index] /= sum;
                dx[index] = (1 - x[index]) / sum;
            }
        }
    }
}


void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        // TODO: Remember for making it memory oblivious

        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            // TODO: Remember for making it memory oblivious
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        // TODO: Remember for making it memory oblivious
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        // TODO: Remember for making it memory oblivious
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    // TODO: Remember for making it memory oblivious
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}

#if defined (USE_SGX) && defined (USE_SGX_BLOCKING)
void fill_cpu_blocked(int N, float ALPHA, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX){
    int i;
    //BLOCK_ENGINE_INIT_FOR_LOOP(X, X_valid_range, X_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(X, X_valid_range, X_ptr,X_index_var,true, float)
    for(i = 0; i < N; ++i) {
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(X, X_valid_range, X_ptr, true, X_index_var, i*INCX)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(X, X_valid_range, X_ptr, true, X_index_var, i*INCX)
        *(X_ptr+X_index_var-X_valid_range.block_requested_ind) = ALPHA;
    }
    BLOCK_ENGINE_LAST_UNLOCK(X, X_valid_range)
}

void mean_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, int batch, int filters,
                    int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean) {
    float scale = 1./(batch * spatial);
    int i,j,k;
    //BLOCK_ENGINE_INIT_FOR_LOOP(x, x_valid_range, x_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(x, x_valid_range, x_block_val_ptr,x_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(mean, mean_valid_range, mean_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(mean, mean_valid_range, mean_block_val_ptr,mean_index_var,true, float)
    for(i = 0; i < filters; ++i){
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(mean, mean_valid_range, mean_block_val_ptr, true, mean_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(mean, mean_valid_range, mean_block_val_ptr, true, mean_index_var, i)
        *(mean_block_val_ptr+mean_index_var-mean_valid_range.block_requested_ind) = 0.0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(x, x_valid_range, x_block_val_ptr, false, x_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(x, x_valid_range, x_block_val_ptr, false, x_index_var, index)
                *(mean_block_val_ptr+mean_index_var-mean_valid_range.block_requested_ind) += *(x_block_val_ptr+x_index_var-x_valid_range.block_requested_ind);
            }
        }
        *(mean_block_val_ptr+mean_index_var-mean_valid_range.block_requested_ind) *= scale;
    }
    BLOCK_ENGINE_LAST_UNLOCK(x, x_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(mean, mean_valid_range)
}
void variance_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean, int batch, int filters, int spatial, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance) {
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    //BLOCK_ENGINE_INIT_FOR_LOOP(x, x_valid_range, x_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(x, x_valid_range, x_block_val_ptr,x_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(mean, mean_valid_range, mean_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(mean, mean_valid_range, mean_block_val_ptr,mean_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(variance, variance_valid_range, variance_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(variance, variance_valid_range, variance_block_val_ptr,variance_index_var,true, float)
    for(i = 0; i < filters; ++i){
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(variance, variance_valid_range, variance_block_val_ptr, true, variance_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(variance, variance_valid_range, variance_block_val_ptr, true, variance_index_var, i)
        *(variance_block_val_ptr+variance_index_var-variance_valid_range.block_requested_ind) = 0.0;
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(mean, mean_valid_range, mean_block_val_ptr, false, mean_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(mean, mean_valid_range, mean_block_val_ptr, false, mean_index_var, i)
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(x, x_valid_range, x_block_val_ptr, false, x_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(x, x_valid_range, x_block_val_ptr, false, x_index_var, index)
                *(variance_block_val_ptr+variance_index_var-variance_valid_range.block_requested_ind) += pow(
                    (*(x_block_val_ptr+x_index_var-x_valid_range.block_requested_ind) - *(mean_block_val_ptr+mean_index_var-mean_valid_range.block_requested_ind))
                    , 2);
            }
        }
        *(variance_block_val_ptr+variance_index_var-variance_valid_range.block_requested_ind) *= scale;
    }
    BLOCK_ENGINE_LAST_UNLOCK(x, x_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(mean, mean_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(variance, variance_valid_range)
}

void scal_cpu_blocked(int N, float ALPHA, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX)
{
    int i;
    //BLOCK_ENGINE_INIT_FOR_LOOP(X, x_valid_range, x_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(X, x_valid_range, x_block_val_ptr, x_index_var,true,float)
    for(i = 0; i < N; ++i) {
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(X, x_valid_range, x_block_val_ptr, true, x_index_var, i*INCX)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(X, x_valid_range, x_block_val_ptr, true, x_index_var, i*INCX)
        *(x_block_val_ptr+x_index_var-x_valid_range.block_requested_ind) *= ALPHA;
    }
    BLOCK_ENGINE_LAST_UNLOCK(X, x_valid_range)
}

void axpy_cpu_blocked(int N, float ALPHA, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &Y, int INCY) {
    int i;
    //BLOCK_ENGINE_INIT_FOR_LOOP(X, x_valid_range, x_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(X, x_valid_range, x_block_val_ptr,x_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(Y, y_valid_range, y_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(Y, y_valid_range, y_block_val_ptr,y_index_var,true, float)
    for(i = 0; i < N; ++i) {
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(X, x_valid_range, x_block_val_ptr, false, x_index_var, i*INCX)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(Y, y_valid_range, y_block_val_ptr, true, y_index_var, i*INCY)
        *(y_block_val_ptr + y_index_var - y_valid_range.block_requested_ind) += ALPHA *(*(x_block_val_ptr + x_index_var - x_valid_range.block_requested_ind));
    }
    BLOCK_ENGINE_LAST_UNLOCK(X, x_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(Y, y_valid_range)
}

void normalize_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &x, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &mean, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &variance, int batch, int filters, int spatial) {
    int b, f, i;
    //BLOCK_ENGINE_INIT_FOR_LOOP(x, x_valid_range, x_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(x, x_valid_range, x_block_val_ptr,x_index_var,true, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(mean, mean_valid_range, mean_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(mean, mean_valid_range, mean_block_val_ptr,mean_index_var,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(variance, variance_valid_range, variance_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(variance, variance_valid_range, variance_block_val_ptr,variance_index_var,false, float)
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(mean, mean_valid_range, mean_block_val_ptr, false, mean_index_var, f)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(mean, mean_valid_range, mean_block_val_ptr, false, mean_index_var, f)
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(variance, variance_valid_range, variance_block_val_ptr, false, variance_index_var, f)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(variance, variance_valid_range, variance_block_val_ptr, false, variance_index_var, f)
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(x, x_valid_range, x_block_val_ptr, true, x_index_var, index)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(x, x_valid_range, x_block_val_ptr, true, x_index_var, index)
                *(x_block_val_ptr+x_index_var-x_valid_range.block_requested_ind) = (*(x_block_val_ptr+x_index_var-x_valid_range.block_requested_ind) - *(mean_block_val_ptr+mean_index_var-mean_valid_range.block_requested_ind))/(sqrt(*(variance_block_val_ptr+variance_index_var-variance_valid_range.block_requested_ind)) + .000001f);
                // x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(x, x_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(mean, mean_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(variance, variance_valid_range)
}

void copy_cpu_blocked(int N, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &X, int INCX, 
                      const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &Y, int INCY) {
    int i;
    //BLOCK_ENGINE_INIT_FOR_LOOP(X, x_valid_range, x_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(X, x_valid_range, x_block_val_ptr,x_current_index,false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(Y, y_valid_range, y_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(Y, y_valid_range, y_block_val_ptr,y_current_index,true, float)
    for(i = 0; i < N; ++i) {
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(X, x_valid_range, x_block_val_ptr, false, x_current_index, i*INCX)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(X, x_valid_range, x_block_val_ptr, false, x_current_index, i*INCX)
        //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(Y, y_valid_range, y_block_val_ptr, true, y_current_index, i*INCY)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(Y, y_valid_range, y_block_val_ptr, true, y_current_index, i*INCY)
        *(y_block_val_ptr + y_current_index - y_valid_range.block_requested_ind) = *(x_block_val_ptr+x_current_index-x_valid_range.block_requested_ind);

    }
    BLOCK_ENGINE_LAST_UNLOCK(X, x_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(Y, y_valid_range)
}

void logistic_x_ent_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error) {
    int i;
    BLOCK_ENGINE_INIT_FOR_LOOP(pred, pred_valid_range, pred_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(truth, truth_valid_range, truth_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(error, error_valid_range, error_block_val_ptr, float)
    for(i = 0; i < n; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(truth, truth_valid_range, truth_block_val_ptr, false, truth_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(pred, pred_valid_range, pred_block_val_ptr, false, pred_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(error, error_valid_range, error_block_val_ptr, true, error_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, true, delta_index_var, i)
        // float t = truth[i];
        float t = *(truth_block_val_ptr+truth_index_var-truth_valid_range.block_requested_ind);
        // float p = pred[i];
        float p = *(pred_block_val_ptr+pred_index_var-pred_valid_range.block_requested_ind);
        // error[i] = -t*log(p) - (1-t)*log(1-p);
         *(error_block_val_ptr+error_index_var-error_valid_range.block_requested_ind) = -t*log(p) - (1-t)*log(1-p);
        // delta[i] = t-p;
         *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) = t-p;
    }
    BLOCK_ENGINE_LAST_UNLOCK(pred, pred_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(truth, truth_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(error, error_valid_range)
}

void smooth_l1_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error) {
    int i;
    BLOCK_ENGINE_INIT_FOR_LOOP(pred, pred_valid_range, pred_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(truth, truth_valid_range, truth_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(error, error_valid_range, error_block_val_ptr, float)
    for(i = 0; i < n; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(truth, truth_valid_range, truth_block_val_ptr, false, truth_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(pred, pred_valid_range, pred_block_val_ptr, false, pred_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(error, error_valid_range, error_block_val_ptr, true, error_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, true, delta_index_var, i)
        // float diff = truth[i] - pred[i];
        float diff = (*(truth_block_val_ptr+truth_index_var-truth_valid_range.block_requested_ind)) - (*(pred_block_val_ptr+pred_index_var-pred_valid_range.block_requested_ind));
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            // error[i] = diff * diff;
            *(error_block_val_ptr+error_index_var-error_valid_range.block_requested_ind) = diff * diff;
            // delta[i] = diff;
            *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) = diff;
        }
        else {
            // error[i] = 2*abs_val - 1;
            *(error_block_val_ptr+error_index_var-error_valid_range.block_requested_ind) = 2*abs_val - 1;
            // delta[i] = (diff < 0) ? 1 : -1;
            *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) = (diff < 0) ? 1 : -1;
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(pred, pred_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(truth, truth_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(error, error_valid_range)
}

void l1_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error) {
    int i;
    BLOCK_ENGINE_INIT_FOR_LOOP(pred, pred_valid_range, pred_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(truth, truth_valid_range, truth_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(error, error_valid_range, error_block_val_ptr, float)
    for(i = 0; i < n; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(truth, truth_valid_range, truth_block_val_ptr, false, truth_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(pred, pred_valid_range, pred_block_val_ptr, false, pred_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(error, error_valid_range, error_block_val_ptr, true, error_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, true, delta_index_var, i)
        // float diff = truth[i] - pred[i];
        float diff = (*(truth_block_val_ptr+truth_index_var-truth_valid_range.block_requested_ind)) - (*(pred_block_val_ptr+pred_index_var-pred_valid_range.block_requested_ind));
        // error[i] = fabs(diff);
         *(error_block_val_ptr+error_index_var-error_valid_range.block_requested_ind) = fabs(diff);
        // delta[i] = diff > 0 ? 1 : -1;
         *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) = diff > 0 ? 1 : -1;

    }
    BLOCK_ENGINE_LAST_UNLOCK(pred, pred_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(truth, truth_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(error, error_valid_range)
}

void l2_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error){
    int i;
    BLOCK_ENGINE_INIT_FOR_LOOP(pred, pred_valid_range, pred_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(truth, truth_valid_range, truth_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(error, error_valid_range, error_block_val_ptr, float)
    for(i = 0; i < n; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(truth, truth_valid_range, truth_block_val_ptr, false, truth_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(pred, pred_valid_range, pred_block_val_ptr, false, pred_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(error, error_valid_range, error_block_val_ptr, true, error_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, true, delta_index_var, i)
        // float diff = truth[i] - pred[i];
        float diff = (*(truth_block_val_ptr+truth_index_var-truth_valid_range.block_requested_ind)) - (*(pred_block_val_ptr+pred_index_var-pred_valid_range.block_requested_ind));
        // error[i] = diff * diff;
        *(error_block_val_ptr+error_index_var-error_valid_range.block_requested_ind) = diff * diff;
        // delta[i] = diff;
        *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) = diff;
    }
    BLOCK_ENGINE_LAST_UNLOCK(pred, pred_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(truth, truth_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(error, error_valid_range)
}

void softmax_x_ent_cpu_blocked(int n, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &pred, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &truth, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &delta, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &error) {
    int i;
    BLOCK_ENGINE_INIT_FOR_LOOP(pred, pred_valid_range, pred_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(truth, truth_valid_range, truth_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(delta, delta_valid_range, delta_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(error, error_valid_range, error_block_val_ptr, float)
    for(i = 0; i < n; ++i) {
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(truth, truth_valid_range, truth_block_val_ptr, false, truth_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(pred, pred_valid_range, pred_block_val_ptr, false, pred_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(error, error_valid_range, error_block_val_ptr, true, error_index_var, i)
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(delta, delta_valid_range, delta_block_val_ptr, true, delta_index_var, i)
        // float t = truth[i];
        float t = *(truth_block_val_ptr+truth_index_var-truth_valid_range.block_requested_ind);
        // float p = pred[i];
        float p = *(pred_block_val_ptr+pred_index_var-pred_valid_range.block_requested_ind);
        // error[i] = (t) ? -log(p) : 0;
        *(error_block_val_ptr+error_index_var-error_valid_range.block_requested_ind) = (t) ? -log(p) : 0;
        // delta[i] = t-p;
        *(delta_block_val_ptr+delta_index_var-delta_valid_range.block_requested_ind) = t-p;
    }
    BLOCK_ENGINE_LAST_UNLOCK(pred, pred_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(truth, truth_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(delta, delta_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(error, error_valid_range)
}

void softmax_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &input,int input_offset, int n, float temp, int stride, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output,int output_offset) {
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    BLOCK_ENGINE_INIT_FOR_LOOP(input, input_valid_range, input_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(output, output_valid_range, output_block_val_ptr, float)
    for(i = 0; i < n; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(input, input_valid_range, input_block_val_ptr, false, input_index_var, (i*stride + input_offset))
        // if(input[i*stride] > largest) largest = input[i*stride];
        if(*(input_block_val_ptr+input_index_var-input_valid_range.block_requested_ind) > largest) {
            // largest = input[i*stride];
            largest = *(input_block_val_ptr+input_index_var-input_valid_range.block_requested_ind);
        }
    }
    for(i = 0; i < n; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(input, input_valid_range, input_block_val_ptr, false, input_index_var, (i*stride + input_offset))
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(output, output_valid_range, output_block_val_ptr, true, output_index_var, (i*stride + output_offset))
        // float e = exp(input[i*stride]/temp - largest/temp);
        float e = exp((*(input_block_val_ptr+input_index_var-input_valid_range.block_requested_ind))/temp - largest/temp);
        sum += e;
        // output[i*stride] = e;
        *(output_block_val_ptr+output_index_var-output_valid_range.block_requested_ind) = e;
    }
    for(i = 0; i < n; ++i){
        BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(output, output_valid_range, output_block_val_ptr, true, output_index_var, (i*stride + output_offset))
        // output[i*stride] /= sum;
        *(output_block_val_ptr+output_index_var-output_valid_range.block_requested_ind) /= sum;
    }
    BLOCK_ENGINE_LAST_UNLOCK(input, input_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(output, output_valid_range)
}

void softmax_cpu_blocked(const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &output) {
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax_blocked(input, b*batch_offset + g*group_offset, n, temp, stride, output , b*batch_offset + g*group_offset);
        }
    }
}
#endif