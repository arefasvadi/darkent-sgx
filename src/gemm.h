#ifndef GEMM_H
#define GEMM_H
#include "./darknet.h"

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);
void gemm_vrf(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_fll(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif

#if defined (USE_SGX) && defined(USE_SGX_BLOCKING)
#include "BlockEngine.hpp"
void gemm_cpu_blocked(int TA, int TB, int M, int N, int K, float ALPHA, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &A, int A_offset,int lda, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &B, int B_offset,int ldb,
        float BETA,
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &C, int C_offset,int ldc);

void gemm_blocked(int TA, int TB, int M, int N, int K, float ALPHA, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &A, int A_offset,int lda, 
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &B, int B_offset,int ldb,
        float BETA,
        const std::shared_ptr<sgx::trusted::BlockedBuffer<float, 1>> &C, int C_offset,int ldc);

#endif
#endif
