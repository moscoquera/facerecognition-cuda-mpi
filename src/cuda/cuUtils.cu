#include <stdlib.h>
#include <vector>
#include "cuUtils.hpp"
#include <stdio.h>
#include <cuda.h>




namespace cu{

const int BLOCK_DIM_X=256;


__global__ void cudaMatsubVect_kernel(double *mat, double *vec,double *out, int rows, int cols, int ldMat, int ldOut){


	int c = (blockIdx.x*blockDim.x)+threadIdx.x;
	int r = (blockIdx.y*blockDim.y)+threadIdx.y;
	extern __shared__ double subvec[];
	if (c<cols && r<rows){
		if (threadIdx.x<BLOCK_DIM_X && threadIdx.y==0){
			//los hlos de la primera fila copian los datos a cache
			*(subvec+threadIdx.x)=*(vec+c);
		}
		__syncthreads();
		*(out+(c*ldOut)+r)=*(mat+(c*ldMat)+r)-*(subvec+threadIdx.x);
	}else{
		__syncthreads();
	}

}

__global__ void cudaMatAddVect_kernel(double *mat, double *vec,double *out, int rows, int cols, int ldMat, int ldOut){


	int c = (blockIdx.x*blockDim.x)+threadIdx.x;
	int r = (blockIdx.y*blockDim.y)+threadIdx.y;
	extern __shared__ double subvec[];
	if (c<cols && r<rows){
		if (threadIdx.x<BLOCK_DIM_X && threadIdx.y==0){
			//los hlos de la primera fila copian los datos a cache
			*(subvec+threadIdx.x)=*(vec+c);
		}
		__syncthreads();
		*(out+(c*ldOut)+r)=*(mat+(c*ldMat)+r)+*(subvec+threadIdx.x);
	}else{
		__syncthreads();
	}

}


__global__ void cudaMatDivVect_kernel(double *mat, double *vec,double *out, int rows, int cols, int ldMat, int ldOut){


	int c = (blockIdx.x*blockDim.x)+threadIdx.x;
	int r = (blockIdx.y*blockDim.y)+threadIdx.y;
	extern __shared__ double subvec[];
	if (c<cols && r<rows){
		if (threadIdx.x<BLOCK_DIM_X && threadIdx.y==0){
			//los hlos de la primera fila copian los datos a cache
			*(subvec+threadIdx.x)=*(vec+c);
		}
		__syncthreads();
		*(out+(c*ldOut)+r)=*(mat+(c*ldMat)+r)/ (*(subvec+threadIdx.x));
	}else{
		__syncthreads();
	}

}



void cudaMatsubvect(cudaStream_t stream,double *mat, double *vec,double *out, int rows, int cols, int ldMat, int ldOut){

	dim3 dimBlocks(BLOCK_DIM_X,3,1);
	dim3 dimGrid(ceil(cols*1.0/dimBlocks.x),ceil(rows*1.0/dimBlocks.y),1);
	cudaMatsubVect_kernel<<<dimGrid,dimBlocks,sizeof(double)*BLOCK_DIM_X,stream>>>(mat,vec,out,rows,cols,ldMat,ldOut);
	CUDA_CHECK_RETURN(cudaGetLastError());

}

void cudaMatAddvect(cudaStream_t stream,double *mat, double *vec,double* out, int rows, int cols, int ldMat, int ldOut){

	dim3 dimBlocks(BLOCK_DIM_X,3,1);
	dim3 dimGrid(ceil(cols*1.0/dimBlocks.x),ceil(rows*1.0/dimBlocks.y),1);
	cudaMatAddVect_kernel<<<dimGrid,dimBlocks,sizeof(double)*BLOCK_DIM_X,stream>>>(mat,vec,out,rows,cols,ldMat,ldOut);
	CUDA_CHECK_RETURN(cudaGetLastError());

}

void cudaMatdivvect(cudaStream_t stream,double *mat, double *vec,double *out, int rows, int cols, int ldMat,int ldOut){

	dim3 dimBlocks(BLOCK_DIM_X,3,1);
	dim3 dimGrid(ceil(cols*1.0/dimBlocks.x),ceil(rows*1.0/dimBlocks.y),1);
	cudaMatDivVect_kernel<<<dimGrid,dimBlocks,sizeof(double)*BLOCK_DIM_X,stream>>>(mat,vec,out,rows,cols,ldMat,ldOut);
	CUDA_CHECK_RETURN(cudaGetLastError());

}



void project(cublasHandle_t blas,cudaStream_t stream, double *W_d_n,int d,int n, double *X_Xn_d,int Xn, double* mu_d, double * &projected_Xn_n){
	double *alpha;
	double *beta;
	double a=1,b=0;
	cudaMalloc((void**)&alpha,sizeof(double));
	cudaMalloc((void**)&beta,sizeof(double));
	cudaMemcpyAsync(alpha,&a,sizeof(double),cudaMemcpyHostToDevice,stream);
	cudaMemcpyAsync(beta,&b,sizeof(double),cudaMemcpyHostToDevice,stream);


	if (mu_d!=NULL){
		cudaMatsubvect(stream,X_Xn_d,mu_d,X_Xn_d,Xn,d,Xn,Xn);
	}
	cublasSetStream_v2(blas,stream);
	CUBLAS_CHECK_RETURN(cublasDgemm_v2(blas,CUBLAS_OP_N,CUBLAS_OP_N,Xn,n,d,alpha,X_Xn_d,Xn,W_d_n,d,beta,projected_Xn_n,Xn));
}


void reconstruct(cublasHandle_t blas,double *W,int d,int n, double *Y, double* mu, double * &reconstructed){
	cudaMalloc((void**)&reconstructed,sizeof(double)*1*d);
	double *alpha;
	double *beta;
	cudaMalloc((void**)&alpha,sizeof(double));
	cudaMalloc((void**)&beta,sizeof(double));
	cudaMemset(alpha,1,sizeof(double));
	cudaMemset(beta,0,sizeof(double));
	CUBLAS_CHECK_RETURN(cublasDgemm_v2(blas,CUBLAS_OP_N,CUBLAS_OP_T,1,d,n,alpha,Y,1,W,d,beta,reconstructed,1));

	if (mu!=NULL){
		cudaMatAddvect(0,reconstructed,mu,reconstructed,1,d,1,1);
	}



}

__global__ void columnMean_kernel(double* in, double *out, int n, int d,int ldx){

	int i=threadIdx.x+(blockIdx.x*blockDim.x);

	if (i<d){
		double* start=in+(i*ldx);
		double tmp=0;
		for(int j=0;j<n;j++){
			tmp+=*(start+j);
		}
		tmp/=n;
		*(out+i)=tmp;
	}

}

void columnMean(cudaStream_t stream, double* X,int rows, int cols, int ldX, double*out){
	columnMean_kernel<<<ceil((cols*1.0)/256),256,0,stream>>>(X,out,rows,cols,ldX);
	CUDA_CHECK_RETURN(cudaGetLastError());

}


__global__ void columnNorm_kernel(double* in, double *out, int n, int d, int ldIn){

	int c=threadIdx.x+(blockIdx.x*blockDim.x);

	if (c>d-1){
		return;
	}
	double norm=0;
	for(int r=0;r<n;r++){
		norm+=(*(in+(c*ldIn)+r))*(*(in+(c*n)+r));
	}
	*(out+c)=sqrt(norm);
}


void columnNorm(cudaStream_t stream, double *W,int rows,int cols,int ldW, double* out){
	columnNorm_kernel<<<ceil((cols*1.0)/256),256,0,stream>>>(W,out,rows,cols,ldW);
	CUDA_CHECK_RETURN(cudaGetLastError());
}

void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

void CheckCublasErrorAux (const char *file, unsigned line, const char *statement, cublasStatus_t err)
{
	if (err == CUBLAS_STATUS_SUCCESS)
		return;
	std::cerr << statement<<" returned " << " at "<<file<<":"<<line << std::endl;
	exit (1);
}

__device__
int getGlobalIdx_2D_2D(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
	 + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}
}

