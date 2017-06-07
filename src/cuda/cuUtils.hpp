/*
 * utils.hpp
 *
 *  Created on: May 12, 2017
 *      Author: smith
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <cublas_v2.h>


namespace cu{

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define CUBLAS_CHECK_RETURN(value) CheckCublasErrorAux(__FILE__,__LINE__, #value, value)


void project(cublasHandle_t blas,cudaStream_t stream, double *W_d_n,int d,int n, double *X_Xn_d,int Xn, double* mu_d, double * &projected_Xn_n);
void reconstruct(cublasHandle_t blas, double *W,int d,int n, double *Y, double* mu, double * &reconstructed);

void columnMean(cudaStream_t stream, double* X,int rows, int cols, int ldX, double*out );
void columnNorm(cudaStream_t stream, double *W,int rows,int cols,int ldW, double* out);
void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err);
void CheckCublasErrorAux (const char *file, unsigned line, const char *statement, cublasStatus_t err);
void cudaMatdivvect(cudaStream_t stream,double *mat, double *vec,double* out, int rows, int cols, int ldMat, int ldOut);
void cudaMatsubvect(cudaStream_t stream,double *mat, double *vec,double* out, int rows, int cols, int ldMat, int ldOut);
__device__
int getGlobalIdx_2D_2D();
}

#endif /* UTILS_HPP_ */
