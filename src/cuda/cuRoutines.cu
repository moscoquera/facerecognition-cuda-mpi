#include <stdlib.h>
#include <iostream>
#include <vector>
#include "cuUtils.hpp"
#include <lapacke.h>
#include <set>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

using namespace std;
namespace cu{

const int BLOCK_DIM_X = 256;

__global__ void getLastColumns_kernel(double* in, double *out, int num_components, int rows, int cols){
	int k = (blockIdx.x*blockDim.x)+threadIdx.x;

	if (k<num_components){
		//*(out+(r*num_components)+k)=*(in+(r*cols)+(cols-k-1));
		memcpy(out+(k*rows),in+((cols-k-1)*rows),sizeof(double)*rows);
	}


}


int pca(cudaStream_t stream,cublasHandle_t blas, cusolverDnHandle_t cusolver, double *d_X_n_d,int num_components,int n, int d, double* &eigenvectors_d_num_components,double* &eigenvalues_num_components, double* &mean_d){

	double* eigenvalues;
	double* eigenvectors;

	if ((num_components <= 0) || (num_components>n)){
			num_components = n;
	}

	//todas las matrices se deben trabajar en colum-major


	cudaMalloc((void**)&mean_d,sizeof(double)*d);

	columnMean(stream,d_X_n_d,n,d,n,mean_d);

	cudaMatsubvect(stream,d_X_n_d,mean_d,d_X_n_d,n,d,n,n);



	//cudaMalloc((void**)&Xmt,sizeof(double)*n*d);
	double *alpha;
	double *beta;
	double a=1,b=0;
	cudaMalloc((void**)&alpha,sizeof(double));
	cudaMalloc((void**)&beta,sizeof(double));
	cudaMemcpyAsync(alpha,&a,sizeof(double),cudaMemcpyHostToDevice,stream);
	cudaMemcpyAsync(beta,&b,sizeof(double),cudaMemcpyHostToDevice,stream);
	if (n>d){

		cudaMalloc((void**)&eigenvalues,sizeof(double)*d);
		cudaMalloc((void**)&eigenvectors,sizeof(double)*d*d);
		cudaMemsetAsync(eigenvalues,0,d*sizeof(double),stream);
		//C = matMult(blas,Xmt,Xm,d,n,d);
		cublasDgemm_v2(blas,CUBLAS_OP_N,CUBLAS_OP_T,d,d,n,alpha,d_X_n_d,d,d_X_n_d,d,beta,eigenvectors,d);

		int lwork=0;
		cusolverDnDsyevd_bufferSize(cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,d,eigenvectors,d,eigenvalues,&lwork);

		double *work;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&work,lwork*sizeof(double)));
		CUDA_CHECK_RETURN(cudaMemsetAsync(work,0,lwork*sizeof(double),stream));
		int *d_info;
		int info=0;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_info,sizeof(int)));
		cusolverStatus_t res =cusolverDnDsyevd(cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,eigenvectors,d,eigenvalues,work,lwork,d_info);
		if (res != CUSOLVER_STATUS_SUCCESS){
			printf("Error eigen %d\n",res);
			exit(-1);
		}
		CUDA_CHECK_RETURN(cudaMemcpy((void**)&info,d_info,sizeof(int),cudaMemcpyDeviceToHost));
		if (info!=0){
			printf("Error eigen %d\n",info);
			exit(-1);
		}
		cudaStreamSynchronize(stream);
		cudaFree(d_info);
		cudaFree(work);
	}else{
		CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenvalues,sizeof(double)*n));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenvectors,sizeof(double)*n*n));
		CUDA_CHECK_RETURN(cudaMemsetAsync(eigenvalues,0,n*sizeof(double),stream));

		cublasSetStream_v2(blas,stream);
		cublasDgemm_v2(blas,CUBLAS_OP_N,CUBLAS_OP_T,n,n,d,alpha,d_X_n_d,n,d_X_n_d,n,beta,eigenvectors,n);
		int lwork=0;


		cusolverDnSetStream(cusolver,stream);
		cusolverDnDsyevd_bufferSize(cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,eigenvectors,n,eigenvalues,&lwork);

		double *work;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&work,lwork*sizeof(double)));
		CUDA_CHECK_RETURN(cudaMemsetAsync(work,0,lwork*sizeof(double),stream));
		int *d_info;
		int info=0;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_info,sizeof(int)));
		cusolverDnSetStream(cusolver,stream);
		cusolverStatus_t res =cusolverDnDsyevd(cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,eigenvectors,n,eigenvalues,work,lwork,d_info);
		if (res != CUSOLVER_STATUS_SUCCESS){
			printf("Error eigen %d\n",res);
			exit(-1);
		}
		CUDA_CHECK_RETURN(cudaMemcpy((void**)&info,d_info,sizeof(int),cudaMemcpyDeviceToHost));
		if (info!=0){
			printf("Error eigen %d\n",info);
			exit(-1);
		}

		double* eigenvectorsTmp = eigenvectors;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenvectors,sizeof(double)*d*n));
		cublasSetStream_v2(blas,stream);
		cublasDgemm_v2(blas,CUBLAS_OP_T,CUBLAS_OP_N,d,n,n,alpha,d_X_n_d,n,eigenvectorsTmp,n,beta,eigenvectors,d);
		cudaStreamSynchronize(stream);
		cudaFree(eigenvectorsTmp);


		double* eigennorm;
		cudaMalloc((void**)&eigennorm,sizeof(double)*n);
		columnNorm(stream,eigenvectors,d,n,d,eigennorm);
		cudaMatdivvect(stream,eigenvectors,eigennorm,eigenvectors,d,n,d,d);

		cudaStreamSynchronize(stream);

		cudaFree(eigennorm);
		cudaFree(work);
		cudaFree(d_info);

	}
	CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenvectors_d_num_components,sizeof(double)*d*num_components));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenvalues_num_components,(sizeof(double)*num_components)));

	cudaStreamSynchronize(stream);

	getLastColumns_kernel<<<ceil(num_components*1.0/BLOCK_DIM_X),BLOCK_DIM_X,0,stream>>>(eigenvectors,eigenvectors_d_num_components,num_components,d,n);
	cublasSetStream_v2(blas,stream);
	CUBLAS_CHECK_RETURN(cublasDcopy_v2(blas,num_components,eigenvalues+n-1,-1,eigenvalues_num_components,1)); //invierto los ultimos k values
	cudaStreamSynchronize(stream);
	cudaFree(eigenvalues);
	cudaFree(eigenvectors);
	return 0;
}


int lda(cudaStream_t *streams,int nstreams,cublasHandle_t blas, cusolverDnHandle_t cusolver,int *clases,int nclases,double *X_n_d,int* y_n,int n,int &num_components, int d, double* &eigenvectors_d_num,double* &eigenvalues_num){

	double *alpha;
	double *beta;
	double *beta1;
	double a=1,b=0;
	cudaMalloc((void**)&alpha,sizeof(double));
	cudaMalloc((void**)&beta,sizeof(double));
	cudaMalloc((void**)&beta1,sizeof(double));
	cudaMemcpyAsync(alpha,&a,sizeof(double),cudaMemcpyHostToDevice,*streams);
	cudaMemcpyAsync(beta,&b,sizeof(double),cudaMemcpyHostToDevice,*streams);
	cudaMemcpyAsync(beta1,&a,sizeof(double),cudaMemcpyHostToDevice,*streams);
	if (num_components<=0 || num_components>nclases-1){
		num_components=nclases-1;
	}
	double* meanTotal;
	cudaMalloc((void**)&meanTotal,sizeof(double)*d);
	columnMean(*streams,X_n_d,n,d,n,meanTotal);

	double * Sw;
	CUDA_CHECK_RETURN(cudaMalloc(&Sw,sizeof(double)*d*d));
	double * Sb;
	CUDA_CHECK_RETURN(cudaMalloc(&Sb,sizeof(double)*d*d));
	double * SwSb;
	CUDA_CHECK_RETURN(cudaMalloc(&SwSb,sizeof(double)*d*d));

	CUDA_CHECK_RETURN(cudaMemsetAsync(Sw,0,sizeof(double)*d*d,*streams));
	CUDA_CHECK_RETURN(cudaMemsetAsync(Sb,0,sizeof(double)*d*d,*streams));

	double* meanClass;
	cudaMalloc((void**)&meanClass,sizeof(double)*nclases*d);

	int base=0;
	double* d_XiSize;
	cudaMalloc((void**)&d_XiSize,sizeof(double));
	for(int i=0;i<nclases;i++){
		double XiSize=0;
		for(int yIterator=base;yIterator<n;yIterator++){
			if (*(clases+i) == *(y_n+yIterator)){
				XiSize++;
			}else if (XiSize>0){
				break; //ya no hay mas instancias de la clase
			}
		}
		cudaMemsetAsync(d_XiSize,XiSize,sizeof(double),*(streams+(i%nstreams)));
		columnMean(*(streams+(i%nstreams)),X_n_d+base,XiSize,d,n,meanClass+(i*d));
		cudaMatsubvect(*(streams+(i%nstreams)),X_n_d+base,meanClass+(i*d),X_n_d+base,XiSize,d,n,n);
		cudaMatsubvect(*(streams+(i%nstreams)),meanClass+(i*d),meanTotal,meanClass+(i*d),1,d,1,1);
		cublasSetStream_v2(blas,*(streams+(i%nstreams)));
		CUBLAS_CHECK_RETURN(cublasDgemm_v2(blas,CUBLAS_OP_T,CUBLAS_OP_N,d,d,XiSize,alpha,X_n_d+base,n,X_n_d+base,n,beta1,Sw,d));
		cublasSetStream_v2(blas,*(streams+(i%nstreams)));
		CUBLAS_CHECK_RETURN(cublasDgemm_v2(blas,CUBLAS_OP_T,CUBLAS_OP_N,d,d,1,d_XiSize,meanClass+(i*d),1,meanClass+(i*d),1,beta1,Sb,d));

		base+=XiSize;
	}

	for(int st=0;st<nstreams;st++){
		cudaStreamSynchronize(*(streams+st));
	}


	int lwork=0;
	cusolverDnSetStream(cusolver,*streams);
	cusolverStatus_t cures= cusolverDnDgetrf_bufferSize(cusolver,d,d,Sw,d,&lwork);
	if (cures!=CUSOLVER_STATUS_SUCCESS){
		printf("buffersize Error\n");
		exit(1);
	}

	double *work;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&work,lwork*sizeof(double)));
	CUDA_CHECK_RETURN(cudaMemsetAsync(work,0,lwork*sizeof(double),*streams));

	int *devIpiv,*devInfo,info;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&devIpiv,sizeof(int)*d));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&devInfo,sizeof(int)));

	cusolverDnSetStream(cusolver,*streams);
	cusolverStatus_t res = cusolverDnDgetrf(cusolver,d,d,Sw,d,work,devIpiv,devInfo);
	if (res != CUSOLVER_STATUS_SUCCESS){
		printf("Error dgetrf %d\n",res);
		exit(-1);
	}
	CUDA_CHECK_RETURN(cudaMemcpy(&info,devInfo,sizeof(int),cudaMemcpyDeviceToHost));
	if (info!=0){
		printf("Error dgetrf: %d\n",info);
		exit(-1);
	}

	double *SwI;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&SwI,sizeof(double)*d*d));
	cusolverDnSetStream(cusolver,*streams);
	res = cusolverDnDgetrs(cusolver,CUBLAS_OP_N,d,d,Sw,d,devIpiv,SwI,d,devInfo);
	if (res != CUSOLVER_STATUS_SUCCESS){
		printf("Error eigen %d\n",res);
		exit(-1);
	}
	CUDA_CHECK_RETURN(cudaMemcpy(&info,devInfo,sizeof(int),cudaMemcpyDeviceToHost));
	if (info!=0){
		printf("Error eigen %d\n",info);
		exit(-1);
	}
	cublasSetStream_v2(blas,*streams);
	CUBLAS_CHECK_RETURN(cublasDgemm_v2(blas,CUBLAS_OP_N,CUBLAS_OP_N,d,d,d,alpha,SwI,d,Sb,d,beta,SwSb,d));





	double *eigenvalues,*eigenvectors;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenvalues,sizeof(double)*d));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenvectors,sizeof(double)*d*d));
	CUDA_CHECK_RETURN(cudaMemsetAsync(eigenvalues,0,d*sizeof(double),*streams));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(eigenvectors,SwSb,sizeof(double)*d*d,cudaMemcpyDeviceToDevice,*streams));


	lwork=0;
	cusolverDnSetStream(cusolver,*streams);
	cusolverDnDsyevd_bufferSize(cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,d,eigenvectors,d,eigenvalues,&lwork);
	cudaStreamSynchronize(*streams);
	CUDA_CHECK_RETURN(cudaFree(work)); //limpio el previo work
	CUDA_CHECK_RETURN(cudaMalloc((void**)&work,lwork*sizeof(double)));
	CUDA_CHECK_RETURN(cudaMemsetAsync(work,0,lwork*sizeof(double),*streams));
	int *d_info;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_info,sizeof(int)));
	cusolverDnSetStream(cusolver,*streams);
	res =cusolverDnDsyevd(cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,d,eigenvectors,d,eigenvalues,work,lwork,d_info);
	if (res != CUSOLVER_STATUS_SUCCESS){
		printf("Error eigen %d\n",res);
		exit(-1);
	}
	CUDA_CHECK_RETURN(cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost));
	if (info!=0){
		printf("Error eigen info %d\n",info);
		exit(-1);
	}
	cudaStreamSynchronize(*streams);
	CUDA_CHECK_RETURN(cudaFree(d_info));
	CUDA_CHECK_RETURN(cudaFree(work));


	double *bestfaces;
	double *bestvalues;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&bestfaces,sizeof(double)*d*num_components));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&bestvalues,(sizeof(double)*num_components)));

	getLastColumns_kernel<<<ceil(num_components*1.0/BLOCK_DIM_X),BLOCK_DIM_X,0,*streams>>>(eigenvectors,bestfaces,num_components,d,d);

	//por el momento no amerita un kernel
	/*for(int k=0;k<num_components;k++){
		CUDA_CHECK_RETURN(cudaMemcpy(bestvalues+k,eigenvalues+(d-k-1),sizeof(double),cudaMemcpyDeviceToDevice));
	}*/
	cublasSetStream_v2(blas,*streams);
	CUBLAS_CHECK_RETURN(cublasDcopy_v2(blas,num_components,eigenvalues+d-1,-1,bestvalues,1)); //invierto los ultimos k values

	eigenvectors_d_num = bestfaces;
	eigenvalues_num = bestvalues;

	cudaStreamSynchronize(*streams);
	CUDA_CHECK_RETURN(cudaFree(meanTotal));
	CUDA_CHECK_RETURN(cudaFree(Sw));
	CUDA_CHECK_RETURN(cudaFree(Sb));
	CUDA_CHECK_RETURN(cudaFree(SwSb));
	CUDA_CHECK_RETURN(cudaFree(eigenvalues));
	CUDA_CHECK_RETURN(cudaFree(eigenvectors));
	CUDA_CHECK_RETURN(cudaFree(meanClass));
	return 0;
}

int fisherfaces(cudaStream_t *streams,int nstreams,cublasHandle_t blas, cusolverDnHandle_t cusolver, double* X_n_d,int* y_n,int num_components,int n, int d, double* &eigenvectors,double* &eigenvalues, double* &mean, int &nfaces){

	int *cclases=(int*)malloc(sizeof(int)*n);
	int nclases=0;
	int prevclass=-1;
	//y_n está en el host
	for(int i=0;i<n;i++){
		if (*(y_n+i)!=prevclass){
			*(cclases+nclases)=*(y_n+i);
			prevclass=*(y_n+i);
			nclases++;
		}
	}



	double *eigenvalues_ncpca_pca,*eigenvectors_d_ncpca_pca,*mu_d_pca;
	double *eigenvalues_lda_num,*eigenvectors_lda_ncpca_num;
	double *eigenpca_project_n_ncpca;

	int ncpca=(n-nclases);


	pca(*streams,blas,cusolver,X_n_d,ncpca,n,d,eigenvectors_d_ncpca_pca,eigenvalues_ncpca_pca,mu_d_pca);
	CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenpca_project_n_ncpca,sizeof(double)*ncpca*n));



	//X_n_d ya es X_n_d-mu por lo que no se vuelve a hacer
	project(blas,*streams,eigenvectors_d_ncpca_pca,d,ncpca,X_n_d,n,0,eigenpca_project_n_ncpca);

	//project presenta diferencias, pero se presume buena

	int lda_num=0;



	lda(streams,nstreams,blas,cusolver,cclases,nclases,eigenpca_project_n_ncpca,y_n,n,lda_num,ncpca,eigenvectors_lda_ncpca_num,eigenvalues_lda_num);


	//printf("%f %f\n",*(eigenvalues_lda_num),*(eigenvectors_lda_ncpca_num));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&eigenvectors,sizeof(double)*d*lda_num));
	double *alpha;
	double *beta;
	double a=1,b=0;
	cudaMalloc((void**)&alpha,sizeof(double));
	cudaMalloc((void**)&beta,sizeof(double));
	cudaMemcpyAsync(alpha,&a,sizeof(double),cudaMemcpyHostToDevice,*streams);
	cudaMemcpyAsync(beta,&b,sizeof(double),cudaMemcpyHostToDevice,*streams);

	cublasSetStream_v2(blas,*streams);
	CUBLAS_CHECK_RETURN(cublasDgemm_v2(blas,CUBLAS_OP_N,CUBLAS_OP_N,d,lda_num,ncpca,alpha,eigenvectors_d_ncpca_pca,d,eigenvectors_lda_ncpca_num,ncpca,beta,eigenvectors,d));
	//matMult(0,eigenvectors_d_ncpca_pca,eigenvectors_lda_ncpca_num,d,ncpca,lda_num);
	eigenvalues=eigenvalues_lda_num;
	mean=mu_d_pca;
	nfaces=lda_num;

	cudaStreamSynchronize(*streams);
	cudaFree(eigenvalues_ncpca_pca);
	cudaFree(eigenvectors_d_ncpca_pca);
	cudaFree(eigenvectors_lda_ncpca_num);
	cudaFree(eigenpca_project_n_ncpca);
	free(cclases);
	return 0;
}

}
