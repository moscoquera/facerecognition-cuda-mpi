#include "cuUtils.hpp"
#include "float.h"
#include "stdlib.h"
#include "stdio.h"

namespace cu {

template<unsigned int blockSize>
__global__ void euclidean_kernel(double *A, double *B, double *out, unsigned int n) {
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < n) {
		sdata[tid] += (A[i]-B[i])*(A[i]-B[i]);
		if (i+blockSize<n){
			sdata[tid] +=  (A[i + blockSize]-B[i + blockSize])*(A[i + blockSize]-B[i + blockSize]);
		}
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	if (tid < 32) {
		if (blockSize >= 64)
			sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32)
			sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16)
			sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8)
			sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4)
			sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2)
			sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0)
		out[blockIdx.x] = sqrt(sdata[0]);
}

void cuEuclideanDistance(cudaStream_t stream,double *d_p, double *d_q,double *out, int n) {
	double* res;
	cudaMalloc((void**)&res,sizeof(double)*n);
	euclidean_kernel<512><<< ceil(n/512.0),512, 512*sizeof(double),stream>>>(d_p,d_q,res,n);
	cudaMemcpy(out,res,sizeof(double),cudaMemcpyDeviceToDevice);
	cudaFree(res);
}

double cuCosineDistance(double *p, double *q, int n) {
	/*double res = 0;
	double* pt;
	double* qt;
	transponer(p, pt, 1, n);
	transponer(q, qt, 1, n);

	double * ptq = matMult(pt, q, 1, n, 1);
	double * ppt = matMult(p, pt, 1, n, 1);
	double * qqt = matMult(q, qt, 1, n, 1);

	res = (-1 * (*ptq)) / sqrt((*ppt) * (*qqt));
	free(pt);
	free(qt);
	free(ptq);
	free(ppt);
	free(qqt);
	return res;*/
	exit(-1);
	//return -1;

}

}

