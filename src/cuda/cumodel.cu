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
								this->streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*MAX_STREAMS);
								for(int xi=0;xi<MAX_STREAMS;xi++){
									cudaStreamCreate(this->streams+xi);
								}

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



		this->nprojections=n;
		cudaMalloc((void**)&this->projections,sizeof(double)*n*this->num_components);

		//cuXC ya es cuXC-mu
		cublasSetStream_v2(this->blasHandler,*this->streams);
		CUBLAS_CHECK_RETURN(cublasDgeam(this->blasHandler,CUBLAS_OP_T,CUBLAS_OP_N,d,n,alpha,cuXC,n,beta,cuX,d,cuX,d));

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

void EigenfacesModel::save(char* path){
	FILE *out = fopen(path, "wb");
	  if(out != NULL)
	  {
		  fwrite(&this->d,sizeof(int),1,out);
		  fwrite(&this->num_components,sizeof(int),1,out);
		  fwrite(&this->nprojections,sizeof(int),1,out);

			size_t to_go = this->nprojections;
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(this->y,sizeof(int),to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}

			to_go=this->d;
			double* hmu =(double*)malloc(sizeof(double)*to_go);
			cudaMemcpy(hmu,this->mu,sizeof(double)*to_go,cudaMemcpyDeviceToHost);
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(hmu,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			free(hmu);


			to_go=this->d*this->num_components;
			double* hw =(double*)malloc(sizeof(double)*to_go);
			cudaMemcpy(hw,this->W,sizeof(double)*to_go,cudaMemcpyDeviceToHost);

			while(to_go > 0)
			{
			  const size_t wrote = fwrite(hw,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			free(hw);

			to_go=this->nprojections*this->num_components;
			double* hpro =(double*)malloc(sizeof(double)*to_go);
			cudaMemcpy(hpro,this->projections,sizeof(double)*to_go,cudaMemcpyDeviceToHost);
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(hpro,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			free(hpro);

		fclose(out);
	  }

}

void EigenfacesModel::load(char* path){
	FILE *in = fopen(path, "rb");
	  if(in != NULL)
	  {
		  fread(&this->d,sizeof(int),1,in);
		  fread(&this->num_components,sizeof(int),1,in);
		  fread(&this->nprojections,sizeof(int),1,in);
		  this->y=(int*)malloc(sizeof(int)*this->nprojections);
		  double *dmu = (double*)malloc(sizeof(double)*this->d);
		  double *dW = (double*)malloc(sizeof(double)*this->d*this->num_components);
		  double *dpro = (double*)malloc(sizeof(double)*this->nprojections*this->num_components);
		  CUDA_CHECK_RETURN(cudaMalloc((void**)&this->mu,sizeof(double)*this->d));
		  CUDA_CHECK_RETURN(cudaMalloc((void**)&this->W,sizeof(double)*this->d*this->num_components));
		  CUDA_CHECK_RETURN(cudaMalloc((void**)&this->projections,sizeof(double)*this->nprojections*this->num_components));

		  fread(this->y,sizeof(int),this->nprojections,in);
		  fread(dmu,sizeof(double), this->d, in);
		  fread(dW,sizeof(double), this->d*this->num_components, in);
		  fread(dpro,sizeof(double), this->nprojections*this->num_components, in);

		  CUDA_CHECK_RETURN(cudaMemcpy(this->mu,dmu,sizeof(double)*this->d,cudaMemcpyHostToDevice));
		  CUDA_CHECK_RETURN(cudaMemcpy(this->W,dW,sizeof(double)*this->d*this->num_components,cudaMemcpyHostToDevice));
		  CUDA_CHECK_RETURN(cudaMemcpy(this->projections,dpro,sizeof(double)*this->nprojections*this->num_components,cudaMemcpyHostToDevice));

		  free(dW);
		  free(dmu);
		  free(dpro);

		  fclose(in);
	  }else{
		  printf("Load Error\n");
		  exit(-1);
	  }
}



double EigenfacesModel::predict(double *X, int &faceClass, int &faceIdx){

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
		double *pi = this->projections+(i*this->num_components);
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
	return minDist;
}


double *EigenfacesModel::reconstructProjection(int data){

	double *d_R;
	reconstruct(this->blasHandler,this->W,this->d,this->num_components,this->projections+(data*this->num_components),this->mu,d_R);
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

								this->streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*MAX_STREAMS);
								for(int xi=0;xi<MAX_STREAMS;xi++){
									cudaStreamCreate(this->streams+xi);
								}
	}

FisherfacesModel::~FisherfacesModel(){
		cublasDestroy_v2(this->blasHandler);
		cusolverDnDestroy(this->cusolverHandler);
		cudaDeviceReset();
}

void FisherfacesModel::compute(double* X, int* y, int n){
		double *d_V;
		double* cuX,*cuXC;


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




		this->nprojections=n;
		cudaMalloc((void**)&this->projections,sizeof(double)*n*this->nfaces);

		//cuXC ya es cuXC-mu
		cublasSetStream_v2(this->blasHandler,*streams);
		CUBLAS_CHECK_RETURN(cublasDgeam(this->blasHandler,CUBLAS_OP_T,CUBLAS_OP_N,d,n,alpha,cuXC,n,beta,cuX,d,cuX,d));


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


double FisherfacesModel::predict(double *X, int &faceClass, int &faceIdx){

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
		double *pi = this->projections+(this->nfaces*i);
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
	return minDist;

}

double *FisherfacesModel::reconstructProjection(int data){

	double *d_R;
	reconstruct(this->blasHandler,this->W,this->d,this->nfaces,this->projections+(data*this->nfaces),this->mu,d_R);
	double* R = (double*)malloc(sizeof(double)*this->d);
	cudaMemcpy(R,d_R,sizeof(double)*this->d,cudaMemcpyDeviceToHost);
	cudaFree(d_R);
	return R;
}

void FisherfacesModel::save(char* path){
	FILE *out = fopen(path, "wb");
	  if(out != NULL)
	  {
		  fwrite(&this->d,sizeof(int),1,out);
		  fwrite(&this->num_components,sizeof(int),1,out);
		  fwrite(&this->nprojections,sizeof(int),1,out);
		  fwrite(&this->nfaces,sizeof(int),1,out);

			size_t to_go = this->nprojections;
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(this->y,sizeof(int),to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}

			to_go=this->d;
			double* hmu =(double*)malloc(sizeof(double)*to_go);
			cudaMemcpy(hmu,this->mu,sizeof(double)*to_go,cudaMemcpyDeviceToHost);
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(hmu,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			free(hmu);


			to_go=this->d*this->nfaces;
			double* hw =(double*)malloc(sizeof(double)*to_go);
			cudaMemcpy(hw,this->W,sizeof(double)*to_go,cudaMemcpyDeviceToHost);

			while(to_go > 0)
			{
			  const size_t wrote = fwrite(hw,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			free(hw);

			to_go=this->nprojections*this->nfaces;
			double* hpro =(double*)malloc(sizeof(double)*to_go);
			cudaMemcpy(hpro,this->projections,sizeof(double)*to_go,cudaMemcpyDeviceToHost);
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(hpro,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			free(hpro);

		fclose(out);
	  }

}

void FisherfacesModel::load(char* path){
	FILE *in = fopen(path, "rb");
	  if(in != NULL)
	  {
		  fread(&this->d,sizeof(int),1,in);
		  fread(&this->num_components,sizeof(int),1,in);
		  fread(&this->nprojections,sizeof(int),1,in);
		  fread(&this->nfaces,sizeof(int),1,in);
		  this->y=(int*)malloc(sizeof(int)*this->nprojections);
		  double *dmu = (double*)malloc(sizeof(double)*this->d);
		  double *dW = (double*)malloc(sizeof(double)*this->d*this->nfaces);
		  double *dpro = (double*)malloc(sizeof(double)*this->nprojections*this->nfaces);
		  cudaMalloc((void**)&this->mu,sizeof(double)*this->d);
		  cudaMalloc((void**)&this->W,sizeof(double)*this->d*this->nfaces);
		  cudaMalloc((void**)&this->projections,sizeof(double)*this->nprojections*this->nfaces);

		  fread(this->y,sizeof(int),this->nprojections,in);
		  fread(dmu,sizeof(double), this->d, in);
		  fread(dW,sizeof(double), this->d*this->nfaces, in);
		  fread(dpro,sizeof(double), this->nprojections*this->nfaces, in);

		  cudaMemcpy(this->mu,dmu,sizeof(double)*this->d,cudaMemcpyHostToDevice);
		  cudaMemcpy(this->W,dW,sizeof(double)*this->d*this->nfaces,cudaMemcpyHostToDevice);
		  cudaMemcpy(this->projections,dpro,sizeof(double)*this->nprojections*this->nfaces,cudaMemcpyHostToDevice);

		  free(dW);
		  free(dmu);
		  free(dpro);

		  fclose(in);
	  }else{
		  printf("Load Error\n");
		  exit(-1);
	  }
}

}
