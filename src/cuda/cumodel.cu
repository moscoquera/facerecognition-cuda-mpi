#include <stdlib.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <float.h>
#include "cuUtils.hpp"
#include "cuRoutines.hpp"
#include "cumodel.hpp"
#include "cublas_v2.h"
#include "cusolverDn.h"


using namespace std;

const int MAX_STREAMS = 3;

double *alpha,*beta;

namespace cu{



EigenfacesModel::EigenfacesModel(int d,void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int)):
								BaseModel(d,dist_metric){
								cublasCreate_v2(&this->blasHandler);
								cusolverDnCreate(&this->cusolverHandler);
								cublasSetPointerMode_v2(this->blasHandler,CUBLAS_POINTER_MODE_DEVICE);
								double a=1,b=0;
								cudaMalloc((void**)&alpha,sizeof(double));
								cudaMalloc((void**)&beta,sizeof(double));
								cudaMemcpy(alpha,&a,sizeof(double),cudaMemcpyHostToDevice);
								cudaMemcpy(beta,&b,sizeof(double),cudaMemcpyHostToDevice);

	}
EigenfacesModel::~EigenfacesModel(){
	cublasDestroy_v2(this->blasHandler);
	cusolverDnDestroy(this->cusolverHandler);
	cudaDeviceReset();
}


void EigenfacesModel::compute(double* X, int* y, int n){
		double *d_V;
		double* cuX,*cuXC;
		int *cuY;

		this->streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*MAX_STREAMS);
		for(int xi=0;xi<MAX_STREAMS;xi++){
			cudaStreamCreate(this->streams+xi);
		}

		cudaMalloc((void**)&cuX,sizeof(double)*n*d);
		cudaMalloc((void**)&cuY,sizeof(int)*n);
		cudaMalloc((void**)&cuXC,sizeof(double)*n*d);
		cudaMemcpyAsync(cuX,X,sizeof(double)*n*d,cudaMemcpyHostToDevice,*this->streams);
		cudaMemcpyAsync(cuY,y,sizeof(int)*n,cudaMemcpyHostToDevice,*this->streams);


		//cambio cuX a column-major, m y n las invierto porque cuX originalmente es como si fuese la transpuesta
		CUBLAS_CHECK_RETURN(cublasSetStream_v2(this->blasHandler,*this->streams));
		CUBLAS_CHECK_RETURN(cublasDgeam(this->blasHandler,CUBLAS_OP_T,CUBLAS_OP_N,n,d,alpha,cuX,d,beta,cuXC,n,cuXC,n));



		pca(*this->streams,this->blasHandler,this->cusolverHandler,cuXC,this->num_components,n,this->d,this->W,d_V,this->mu);
		//this->y = cuY;
		this->y = y;


		this->projections=(double**)malloc(sizeof(double*)*n);
		this->nprojections=n;
		double *projections;
		cudaMalloc((void**)&projections,sizeof(double)*n*this->num_components);

		//cuXC ya es cuXC-mu
		cublasSetStream_v2(this->blasHandler,*this->streams);
		CUBLAS_CHECK_RETURN(cublasDgeam(this->blasHandler,CUBLAS_OP_T,CUBLAS_OP_N,d,n,alpha,cuXC,n,beta,cuX,d,cuX,d));

		for(int xi=0;xi<n;xi++){
			double * proj=projections+(this->num_components*xi);
			*(this->projections+xi)=proj;
		}
		int step=ceil(n*1.0/MAX_STREAMS);
		for (int t=0;t<MAX_STREAMS;t++){
			cublasSetStream_v2(this->blasHandler,*(this->streams+t));
			cublasDgemmStridedBatched(this->blasHandler,CUBLAS_OP_N,CUBLAS_OP_N,1,this->num_components,this->d,alpha,cuX+(d*step*t),1,this->d,this->W,this->d,0,beta,projections+(this->num_components*step*t),1,this->num_components,min(n-(step*t),step));
		}
		for (int t=0;t<MAX_STREAMS;t++){
			cudaStreamSynchronize(*(this->streams+t));
		}
		cudaFree(d_V);
		cudaFree(cuX);
		cudaFree(cuXC);

}


void EigenfacesModel::predict(double *X, int &faceClass, int &faceIdx){

	double* d_X;
	cudaMalloc((void**)&d_X,sizeof(double)*this->d);
	cudaMemcpy(d_X,X,sizeof(double)*this->d,cudaMemcpyHostToDevice);

	double *d_Q;
	double* d_mins=0,*mins;
	cudaMalloc((void**)&d_mins,sizeof(double)*this->nprojections);
	mins=(double*)malloc(sizeof(double)*this->nprojections);
	cudaMalloc((void**)&d_Q,sizeof(double)*this->num_components*1);
	project(this->blasHandler,0,this->W,this->d,this->num_components,d_X,1,this->mu,d_Q);

	for(int i=0;i<this->nprojections;i++){
		double *pi = *(this->projections+i);
		this->dist_metric(0,pi,d_Q,d_mins+i,this->num_components);

	}
	cudaDeviceSynchronize();
	cudaMemcpy(mins,d_mins,sizeof(double)*this->nprojections,cudaMemcpyDeviceToHost);
	double minDist= DBL_MAX;
	int minIdx=-1;

	for(int i=0;i<this->nprojections;i++){
		if (*(mins+i)<minDist){
			minDist=*(mins+i);
			minIdx=i;
		}
	}

	faceClass=(minIdx<0)?-1:this->y[minIdx];
	faceIdx=minIdx;
	cudaFree(d_Q);
	cudaFree(d_X);
	cudaFree(d_mins);
}


double *EigenfacesModel::reconstructProjection(int data){

	double *d_R;
	reconstruct(this->blasHandler,this->W,this->d,this->num_components,*(this->projections+data),this->mu,d_R);
	double* R = (double*)malloc(sizeof(double)*this->d);
	cudaMemcpy(R,d_R,sizeof(double)*this->d,cudaMemcpyDeviceToHost);
	cudaFree(d_R);
	return R;
}


FisherfacesModel::FisherfacesModel(int d,
							void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int)):
								BaseModel(d,dist_metric){
								cublasCreate_v2(&this->blasHandler);
								cusolverDnCreate(&this->cusolverHandler);
								cublasSetPointerMode_v2(this->blasHandler,CUBLAS_POINTER_MODE_DEVICE);
								double a=1,b=0;
								cudaMalloc((void**)&alpha,sizeof(double));
								cudaMalloc((void**)&beta,sizeof(double));
								cudaMemcpy(alpha,&a,sizeof(double),cudaMemcpyHostToDevice);
								cudaMemcpy(beta,&b,sizeof(double),cudaMemcpyHostToDevice);
	}

FisherfacesModel::~FisherfacesModel(){
		cublasDestroy_v2(this->blasHandler);
		cusolverDnDestroy(this->cusolverHandler);
		cudaDeviceReset();
}

void FisherfacesModel::compute(double* X, int* y, int n){
		double *d_V;
		double* cuX,*cuXC;

		this->streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*MAX_STREAMS);
		for(int xi=0;xi<MAX_STREAMS;xi++){
			cudaStreamCreate(this->streams+xi);
		}

		cudaMalloc((void**)&cuX,sizeof(double)*n*d);
		cudaMalloc((void**)&cuXC,sizeof(double)*n*d);
		cudaMemcpyAsync(cuX,X,sizeof(double)*n*d,cudaMemcpyHostToDevice,*this->streams);
		//cambio cuX a column-major, m y n las invierto porque cuX originalmente es como si fuese la transpuesta
		cublasSetStream_v2(this->blasHandler,*this->streams);
		if (cublasDgeam(this->blasHandler,CUBLAS_OP_T,CUBLAS_OP_N,n,d,alpha,cuX,d,beta,cuXC,n,cuXC,n)!=CUBLAS_STATUS_SUCCESS){
			printf("holy shi!\n");
			exit(-1);
		}





		fisherfaces(this->streams,MAX_STREAMS,this->blasHandler,this->cusolverHandler, cuXC,y,this->num_components,n,this->d,this->W,d_V,this->mu,this->nfaces);


		this->y = y;



		this->projections=(double**)malloc(sizeof(double*)*n);
		this->nprojections=n;
		double *projections;
		cudaMalloc((void**)&projections,sizeof(double)*n*this->nfaces);

		//cuXC ya es cuXC-mu
		cublasSetStream_v2(this->blasHandler,*streams);
		CUBLAS_CHECK_RETURN(cublasDgeam(this->blasHandler,CUBLAS_OP_T,CUBLAS_OP_N,d,n,alpha,cuXC,n,beta,cuX,d,cuX,d));

		for(int xi=0;xi<n;xi++){
			double * proj=projections+(this->nfaces*xi);
			*(this->projections+xi)=proj;
		}


		int step=ceil(n*1.0/MAX_STREAMS);
		for (int t=0;t<MAX_STREAMS;t++){
			cublasSetStream_v2(this->blasHandler,*(this->streams+t));
			cublasDgemmStridedBatched(this->blasHandler,CUBLAS_OP_N,CUBLAS_OP_N,1,this->nfaces,this->d,alpha,cuX+(d*step*t),1,this->d,this->W,this->d,0,beta,projections+(this->nfaces*step*t),1,this->nfaces,min(n-(step*t),step));
		}
		for (int t=0;t<MAX_STREAMS;t++){
			cudaStreamSynchronize(*(this->streams+t));
		}


		cudaFree(d_V);
		cudaFree(cuX);
		cudaFree(cuXC);

}


void FisherfacesModel::predict(double *X, int &faceClass, int &faceIdx){

	double* d_X;
	cudaMalloc((void**)&d_X,sizeof(double)*this->d);
	cudaMemcpyAsync(d_X,X,sizeof(double)*this->d,cudaMemcpyHostToDevice,*this->streams);

	double *d_Q;
	double* d_mins=0,*mins;
	cudaMalloc((void**)&d_mins,sizeof(double)*this->nprojections);
	mins=(double*)malloc(sizeof(double)*this->nprojections);
	cudaMalloc((void**)&d_Q,sizeof(double)*this->nfaces*1);
	project(this->blasHandler,0,this->W,this->d,this->nfaces,d_X,1,this->mu,d_Q);
	for(int i=0;i<this->nprojections;i++){
		double *pi = *(this->projections+i);
		this->dist_metric(*(this->streams+(i%MAX_STREAMS)),pi,d_Q,d_mins+i,this->nfaces);

	}
	cudaDeviceSynchronize();
	cudaMemcpy(mins,d_mins,sizeof(double)*this->nprojections,cudaMemcpyDeviceToHost);

	double minDist= DBL_MAX;
	int minIdx=-1;

	for(int i=0;i<this->nprojections;i++){
		if (*(mins+i)<minDist){
			minDist=*(mins+i);
			minIdx=i;
		}
	}

	faceClass=(minIdx<0)?-1:this->y[minIdx];
	faceIdx=minIdx;
	cudaFree(d_Q);
	cudaFree(d_X);
	cudaFree(d_mins);

}

double *FisherfacesModel::reconstructProjection(int data){

	double *d_R;
	reconstruct(this->blasHandler,this->W,this->d,this->nfaces,*(this->projections+data),this->mu,d_R);
	double* R = (double*)malloc(sizeof(double)*this->d);
	cudaMemcpy(R,d_R,sizeof(double)*this->d,cudaMemcpyDeviceToHost);
	cudaFree(d_R);
	return R;
}

}
