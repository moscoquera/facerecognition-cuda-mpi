/*
 * routines.hpp
 *
 *  Created on: May 12, 2017
 *      Author: smith
 */

#ifndef cuROUTINES_HPP_
#define cuROUTINES_HPP_
#include <stdlib.h>
#include "cusolverDn.h"
#include "cublas_v2.h"

namespace cu{
	int pca(cudaStream_t stream, cublasHandle_t blas, cusolverDnHandle_t cusolver, double *X,int num_components,int n, int d, double* &eigenvectors,double* &eigenvalues, double* &mean);
	int fisherfaces(cudaStream_t *streams,int nstreams, cublasHandle_t blas, cusolverDnHandle_t cusolver,double* X,int* y,int num_components,int n, int d, double* &eigenvectors,double* &eigenvalues, double* &mean, int &nfaces);
}


#endif /* cuROUTINES_HPP_ */
