#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common-configs.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wregister"

#if defined(USE_SGX) && defined(USE_DNNL_GEMM)
#include "prepare-dnnl.h"
#endif

#if defined(USE_SGX) && defined (USE_GEMM_THREADING_SGX)
gemm_multi_thread_params_t gemm_params = {};
std::vector<gemm_thread_task_t> per_thr_params = {};
namespace {

  void
  reset_gemm_params_thread_context() {
    gemm_params = {};
  }

  void
  set_gemm_params_thread_context(int    TA,
                                 int    TB,
                                 int    M,
                                 int    N,
                                 int    K,
                                 float  ALPHA,
                                 float *A,
                                 int    lda,
                                 float *B,
                                 int    ldb,
                                 float  BETA,
                                 float *C,
                                 int    ldc) {
    gemm_params.TA    = TA;
    gemm_params.TB    = TB;
    gemm_params.M     = M;
    gemm_params.N     = N;
    gemm_params.K     = K;
    gemm_params.ALPHA = ALPHA;
    gemm_params.A     = A;
    gemm_params.lda   = lda;
    gemm_params.B     = B;
    gemm_params.ldb   = ldb;
    gemm_params.BETA  = BETA;
    gemm_params.C     = C;
    gemm_params.ldc   = ldc;
    gemm_params.starterM = 0;
    gemm_params.starterN = 0;
  }

  int
  set_num_threading_gemm_cpu() {
    int available_threads = (AVAIL_THREADS);
    // it seems that accessing per row is more efficient! probably because of
    // caching
    bool more_rows = true;
    // bool more_rows         = (gemm_params.M >= gemm_params.N);
    int q, r;
    if (more_rows) {
      q = gemm_params.M / available_threads;
      r = gemm_params.M % available_threads;
      if (q == 0) {
        available_threads = r;
        q                 = gemm_params.M / available_threads;
        r                 = gemm_params.M % available_threads;
      }
    } else {
      q = gemm_params.N / available_threads;
      r = gemm_params.N % available_threads;
      if (q == 0) {
        available_threads = r;
        q                 = gemm_params.N / available_threads;
        r                 = gemm_params.N % available_threads;
      }
    }
    //per_thr_params.resize(available_threads);

    int currM  = 0;
    int currN  = 0;
    int M_size = 0;
    int N_size = 0;
    for (int i = 0; i < available_threads; ++i) {
      auto thread_gemm_ptrs = gemm_params;
      if (more_rows) {
        M_size = q;
        if (r > 0) {
          M_size += r;
          r = 0;
        }
        thread_gemm_ptrs.starterM = currM;
        currM += M_size;
        thread_gemm_ptrs.M = currM;
      } else {
        N_size = q;
        if (r > 0) {
          N_size += r;
          r = 0;
        }
        thread_gemm_ptrs.starterN = currN;
        currN += N_size;
        thread_gemm_ptrs.N = currN;
      }
      per_thr_params.push_back({thread_gemm_ptrs,{thread_task_status_t::not_started}});
    }
    return available_threads;
  }

  void reset_gemm_per_thread() {
      per_thr_params.resize(0);
  }

  void check_reset_gemm_per_thread() {
      for (int i=0;i<per_thr_params.size();++i) {
          if (per_thr_params[i].second._a.load() != thread_task_status_t::finished) {
              LOG_DEBUG("Some threads task has not yet finished\n");
              abort();
          }
      }
      reset_gemm_per_thread();
  }
}  // namespace
#endif

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = (float*)calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

#ifndef USE_SGX
void time_random_matrix(int TA, int TB, int m, int k, int n)
{
  float *a;
  if(!TA) a = random_matrix(m,k);
  else a = random_matrix(k,m);
  int lda = (!TA)?k:m;
  float *b;
  if(!TB) b = random_matrix(k,n);
  else b = random_matrix(n,k);
  int ldb = (!TB)?n:k;

  float *c = random_matrix(m,n);
  int i;
  clock_t start = clock(), end;
  for(i = 0; i<10; ++i){
    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
  }
  end = clock();
  printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
  free(a);
  free(b);
  free(c);
}
#else
#endif

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    #if defined(USE_DNNL_GEMM) && defined(USE_SGX)
    char transa = TA == 1 ? 'T':'N';
    char transb = TB == 1 ? 'T':'N';
    // dnnl_sgemm(transa,transb, M,  N,  K, ALPHA, A,  lda, B, ldb, BETA, C, ldc);
    primitive_based_sgemm(transa, transb, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
    #else
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
    #endif
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
// #ifdef USE_SGX
// const char* timing_key = "gemm time";
// ocall_set_timing(timing_key,strlen(timing_key)+1 , 1, 0);
// #endif
#if defined(USE_SGX) && defined(USE_GEMM_THREADING_SGX)
  set_gemm_params_thread_context(
      TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
#endif
  // printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda,
  // ldb, BETA, ldc);
  int i, j;
  // LOG_DEBUG("TA:%d, TB:%d, M:%d, N:%d, K:%d, lda:%d, ldb:%d,
  // ldc:%d\n",TA,TB,M,N,K,lda,ldb,ldc) LOG_DEBUG("indexes in range [%d,%d] will
  // be used\n",0,N-1+(M-1)*ldc);
  if (BETA != 1) {
#if defined(USE_SGX) && defined(USE_GEMM_THREADING_SGX)
    auto num_threads = set_num_threading_gemm_cpu();
    sgx_status_t res
        = ocall_handle_gemm_cpu_first_mult(num_threads);
    CHECK_SGX_SUCCESS(
        res, "function ocall_handle_gemm_cpu_first_mult caused problem!")
    check_reset_gemm_per_thread();
#else
    for (i = 0; i < M; ++i) {
      for (j = 0; j < N; ++j) {
        C[i * ldc + j] *= BETA;
      }
    }
#endif
  }
    #if defined(USE_SGX) && defined (USE_GEMM_THREADING_SGX)
    auto num_threads = set_num_threading_gemm_cpu();
    sgx_status_t res = ocall_handle_gemm_all(num_threads);
    CHECK_SGX_SUCCESS(res, "function ocall_handle_gemm_cpu_all caused problem!")
    check_reset_gemm_per_thread();
    reset_gemm_params_thread_context();
    
    #else
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    #endif
// #ifdef USE_SGX
// ocall_set_timing(timing_key,strlen(timing_key)+1 , 0, 1);
// #endif
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = (cudaError_t)cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

#if defined (USE_SGX) && defined(USE_SGX_BLOCKING)
void gemm_nn_blocked(int M, int N, int K, float ALPHA, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &A, int A_offset,int lda, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &B, int B_offset,int ldb,
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &C, int C_offset,int ldc)
{
    int i,j,k;
    //BLOCK_ENGINE_INIT_FOR_LOOP(A, A_valid_range, A_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(A, A_valid_range, A_block_val_ptr, A_current_index_var, false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(B, B_valid_range, B_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(B, B_valid_range, B_block_val_ptr, B_current_index, false, float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(C, C_valid_range, C_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(C, C_valid_range, C_block_val_ptr, C_current_index, true, float)
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(A, A_valid_range, A_block_val_ptr, false, A_current_index_var, i*lda+k + A_offset)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(A, A_valid_range, A_block_val_ptr, false, A_current_index_var, i*lda+k + A_offset)
            register float A_PART = ALPHA* (*(A_block_val_ptr+A_current_index_var-A_valid_range.block_requested_ind));
            //register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(B, B_valid_range, B_block_val_ptr, false, B_current_index, k*ldb+j+B_offset)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(B, B_valid_range, B_block_val_ptr, false, B_current_index, k*ldb+j+B_offset)
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(C, C_valid_range, C_block_val_ptr, true, C_current_index, i*ldc+j+C_offset)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(C, C_valid_range, C_block_val_ptr, true, C_current_index, i*ldc+j+C_offset)
                *(C_block_val_ptr+C_current_index-C_valid_range.block_requested_ind) += A_PART*(*(B_block_val_ptr+B_current_index-B_valid_range.block_requested_ind));
                //C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(A, A_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(B, B_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(C, C_valid_range)
}
void gemm_tn_blocked(int M, int N, int K, float ALPHA, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &A, int A_offset,int lda, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &B, int B_offset,int ldb,
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &C, int C_offset,int ldc)
{
    int i,j,k;
    // BLOCK_ENGINE_INIT_FOR_LOOP(A, A_valid_range, A_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(A, A_valid_range, A_block_val_ptr,A_current_index_var,false,float)
    // BLOCK_ENGINE_INIT_FOR_LOOP(B, B_valid_range, B_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(B, B_valid_range, B_block_val_ptr,B_current_index,false,float)
    // BLOCK_ENGINE_INIT_FOR_LOOP(C, C_valid_range, C_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(C, C_valid_range, C_block_val_ptr,C_current_index,true,float)
    //LOG_DEBUG("started gemm_tn_blocked\n")
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            //LOG_DEBUG("testi 1\n")
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(A, A_valid_range, A_block_val_ptr, false, A_current_index_var, k*lda+i + A_offset)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(A, A_valid_range, A_block_val_ptr, false, A_current_index_var, k*lda+i + A_offset)
            register float A_PART = ALPHA*(*(A_block_val_ptr+A_current_index_var-A_valid_range.block_requested_ind));
            //register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(B, B_valid_range, B_block_val_ptr, false, B_current_index, k*ldb+j+B_offset)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(B, B_valid_range, B_block_val_ptr, false, B_current_index, k*ldb+j+B_offset)
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(C, C_valid_range, C_block_val_ptr, true, C_current_index, i*ldc+j+C_offset)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(C, C_valid_range, C_block_val_ptr, true, C_current_index, i*ldc+j+C_offset)
                *(C_block_val_ptr+C_current_index-C_valid_range.block_requested_ind) += A_PART*(*(B_block_val_ptr+B_current_index-B_valid_range.block_requested_ind));
                //LOG_DEBUG("testi 2 Done\n")
                //C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(A, A_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(B, B_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(C, C_valid_range)
    //LOG_DEBUG("finished gemm_tn_blocked\n")
}
void gemm_nt_blocked(int M, int N, int K, float ALPHA, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &A, int A_offset,int lda, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &B, int B_offset,int ldb,
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &C, int C_offset,int ldc)
{
    //LOG_DEBUG("Entered gemm_nt_blocked!\n")
    int i,j,k;
    //BLOCK_ENGINE_INIT_FOR_LOOP(A, A_valid_range, A_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(A, A_valid_range, A_block_val_ptr,A_current_index,false,float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(B, B_valid_range, B_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(B, B_valid_range, B_block_val_ptr,B_current_index,false,float)
    //BLOCK_ENGINE_INIT_FOR_LOOP(C, C_valid_range, C_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP_NEW_1D(C, C_valid_range, C_block_val_ptr,C_current_index,true,float)
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(A, A_valid_range, A_block_val_ptr, false, A_current_index,i*lda+k+A_offset)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(A, A_valid_range, A_block_val_ptr, false, A_current_index,i*lda+k+A_offset)
                //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(B, B_valid_range, B_block_val_ptr, false, B_current_index, j*ldb + k+B_offset)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(B, B_valid_range, B_block_val_ptr, false, B_current_index, j*ldb + k+B_offset)
                sum += ALPHA*(*(A_block_val_ptr+A_current_index-A_valid_range.block_requested_ind))*(*(B_block_val_ptr+B_current_index-B_valid_range.block_requested_ind));
                //sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            //BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(C, C_valid_range, C_block_val_ptr, true, C_current_index, i*ldc+j+C_offset)
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D_NEW(C, C_valid_range, C_block_val_ptr, true, C_current_index, i*ldc+j+C_offset)
            *(C_block_val_ptr+C_current_index-C_valid_range.block_requested_ind) += sum;
            //C[i*ldc+j] += sum;
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(A, A_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(B, B_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(C, C_valid_range)
    //LOG_DEBUG("Exitted gemm_nt_blocked!\n")
}
void gemm_tt_blocked(int M, int N, int K, float ALPHA, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &A, int A_offset,int lda, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &B, int B_offset,int ldb,
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &C, int C_offset,int ldc)
{
    int i,j,k;
    BLOCK_ENGINE_INIT_FOR_LOOP(A, A_valid_range, A_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(B, B_valid_range, B_block_val_ptr, float)
    BLOCK_ENGINE_INIT_FOR_LOOP(C, C_valid_range, C_block_val_ptr, float)
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(A, A_valid_range, A_block_val_ptr, false, A_current_index,i+k*lda+A_offset)
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(B, B_valid_range, B_block_val_ptr, false, B_current_index, k+j*ldb+B_offset)
                sum += ALPHA*(*(A_block_val_ptr+A_current_index-A_valid_range.block_requested_ind))*(*(B_block_val_ptr+B_current_index-B_valid_range.block_requested_ind));
                //sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(C, C_valid_range, C_block_val_ptr, true, C_current_index, i*ldc+j+C_offset)
            *(C_block_val_ptr+C_current_index-C_valid_range.block_requested_ind) += sum;
            //C[i*ldc+j] += sum;
        }
    }
    BLOCK_ENGINE_LAST_UNLOCK(A, A_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(B, B_valid_range)
    BLOCK_ENGINE_LAST_UNLOCK(C, C_valid_range)
}

void gemm_cpu_blocked(int TA, int TB, int M, int N, int K, float ALPHA, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &A, int A_offset,int lda, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &B, int B_offset,int ldb,
        float BETA,
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &C, int C_offset,int ldc) {

        int i, j;
        //LOG_DEBUG("TA:%d, TB:%d, M:%d, N:%d, K:%d, lda:%d, ldb:%d, ldc:%d\n",TA,TB,M,N,K,lda,ldb,ldc)
        //LOG_DEBUG("C array flattened dim size: %d and indexes in range [%d,%d] will be used\n",C->GetTotalElements(),0,N-1+(M-1)*ldc);
        if (BETA != 1) {
        BLOCK_ENGINE_INIT_FOR_LOOP(C, c_valid_range, c_block_val_ptr, float)
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
                //LOG_DEBUG("gem cpu blocked request for flattened %u max dim size %u",i*ldc + j + C_offset,C->GetDimSize()[0])
                //if (i*ldc + j + C_offset >= C->GetDimSize()[0] || i*ldc + j + C_offset < 0) {
                    //LOG_DEBUG("problem will arise soon i:%d,j:%d\n",i,j)
                    //LOG_DEBUG("TA:%d, TB:%d, M:%d, N:%d, K:%d, lda:%d, ldb:%d, ldc:%d, C_offset:%d\n",TA,TB,M,N,K,lda,ldb,ldc,C_offset)
                    //LOG_DEBUG("C array flattened dim size: %d and indexes in range [%d,%d] will be used\n",C->GetDimSize()[0],C_offset,C_offset+N-1+(M-1)*ldc);
                //}
                BLOCK_ENGINE_COND_CHECK_FOR_LOOP_1D(C, c_valid_range, c_block_val_ptr, true, c_index_var, i*ldc + j + C_offset)
                *(c_block_val_ptr + c_index_var - c_valid_range.block_requested_ind) *= BETA;
                //C[i*ldc + j] *= BETA;
            }
        }
        //LOG_DEBUG("finished dim size: %d and idexes in range [%d,%d] will be used\n",C->GetDimSize()[0],C_offset,C_offset+N-1+(M-1)*ldc);
        BLOCK_ENGINE_LAST_UNLOCK(C, c_valid_range)
        }

        if(!TA && !TB)
            gemm_nn_blocked(M, N, K, ALPHA,A,A_offset,lda, B,B_offset, ldb,C,C_offset,ldc);
        else if(TA && !TB)
            gemm_tn_blocked(M, N, K, ALPHA,A,A_offset ,lda, B,B_offset, ldb,C,C_offset,ldc);
        else if(!TA && TB)
            gemm_nt_blocked(M, N, K, ALPHA,A,A_offset,lda, B,B_offset, ldb,C,C_offset,ldc);
        else
            gemm_tt_blocked(M, N, K, ALPHA,A,A_offset,lda, B,B_offset, ldb,C,C_offset,ldc);

}

void gemm_blocked(int TA, int TB, int M, int N, int K, float ALPHA, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &A, int A_offset,int lda, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &B, int B_offset,int ldb,
        float BETA,
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &C, int C_offset,int ldc) {
            return gemm_cpu_blocked(TA, TB, M, N, K, ALPHA, A, A_offset, lda, B, B_offset, ldb, BETA, C, C_offset, ldc);
        }
#endif
#pragma GCC diagnostic pop
