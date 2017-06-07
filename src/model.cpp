#include "distance.hpp"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <float.h>
#include "utils.hpp"
#include "routines.hpp"
#include "model.hpp"

using namespace std;


void BaseModel::compute(double*  X, int* y, int n){

}

BaseModel::BaseModel(int d, void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int)){
		this->dist_metric=dist_metric;
		this->W=NULL;
		this->mu=NULL;
		this->d=d;
		this->y=NULL;

	}


double BaseModel::predict(double *X, int &faceClass, int &faceIdx){
	double minDist = DBL_MAX;
	int minIdx=-1;
	double *Q = (double*)malloc(sizeof(double)*1*this->num_components);
	project(this->W,this->d,this->num_components,X,1,this->mu,Q);
	for(int i=0;i<this->nprojections;i++){
		double *pi = this->projections+(i*this->num_components);
		double dist;
		this->dist_metric(0,pi,Q,&dist,this->num_components);
		if (dist<minDist){
			minDist=dist;
			minIdx=i;
		}
	}

	faceClass=(minIdx<0)?-1:this->y[minIdx];
	faceIdx=minIdx;
	free(Q);
	return minDist;
}


double *BaseModel::reconstructProjection(int data){

	double *R;
	reconstruct(this->W,this->d,this->num_components,this->projections+(data*this->num_components),this->mu,R);
	return R;
}

void BaseModel::load(char*){

}

void BaseModel::save(char*){

}

BaseModel::~BaseModel(){

}

EigenfacesModel::EigenfacesModel(int d,void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int)):
								BaseModel(d,dist_metric){



	}

EigenfacesModel::~EigenfacesModel(){

}

void EigenfacesModel::compute(double* X, int* y, int n){
		double *V;
		pca(X,this->num_components,n,this->d,this->W,V,this->mu);
		this->y = y;
		double *imgdatadouble = (double*)malloc(sizeof(double)*this->d);
		this->projections= (double*)malloc(sizeof(double)*this->num_components*n);
		this->nprojections=n;
		for(int xi=0;xi<n;xi++){
			double * proj = this->projections+(this->num_components*xi);
			double* imgdata = X+(xi*this->d);
			for(int pix=0;pix<this->d;pix++){
				*(imgdatadouble+pix)=*(imgdata+pix);
			}
			project(this->W,this->d,this->num_components,imgdatadouble,1,this->mu,proj);

		}

		free(V);
		free(imgdatadouble);
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
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(this->mu,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			to_go=this->d*this->num_components;
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(this->W,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			to_go=this->nprojections*this->num_components;
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(this->projections,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}

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
		  this->mu=(double*)malloc(sizeof(double)*this->d);
		  this->W=(double*)malloc(sizeof(double)*this->d*this->num_components);
		  this->projections=(double*)malloc(sizeof(double)*this->nprojections*this->num_components);
		  fread(this->y,sizeof(int),this->nprojections,in);
		  fread(this->mu,sizeof(double), this->d, in);
		  fread(this->W,sizeof(double), this->d*this->num_components, in);
		  fread(this->projections,sizeof(double), this->nprojections*this->num_components, in);
		  fclose(in);
	  }else{
		  printf("Load Error\n");
		  exit(-1);
	  }
}

FisherfacesModel::FisherfacesModel(int d,
							void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int)):
								BaseModel(d,dist_metric){


	}

FisherfacesModel::~FisherfacesModel(){

}

void FisherfacesModel::compute(double* X, int* y, int n){
		double *V;
		fisherfaces(X,y,this->num_components,n,this->d,this->W,V,this->mu,this->nfaces);
		this->y = y;
		double *imgdatadouble = (double*)malloc(sizeof(double)*this->d);
		this->projections= (double*)malloc(sizeof(double)*this->nfaces*n);
		this->nprojections=n;
		for(int xi=0;xi<n;xi++){
			double * proj = this->projections+(this->nfaces*xi);
			double* imgdata = X+(xi*this->d);
			for(int pix=0;pix<this->d;pix++){
				*(imgdatadouble+pix)=*(imgdata+pix);
			}
			project(this->W,this->d,this->nfaces,imgdatadouble,1,this->mu,proj);
			//this->projections.push_back(proj);
		}

		free(V);
		free(imgdatadouble);
}


double FisherfacesModel::predict(double *X, int &faceClass, int &faceIdx){
	double minDist = DBL_MAX;
	int minIdx=-1;
	double *Q = (double*)malloc(sizeof(double)*1*this->nfaces);
	project(this->W,this->d,this->nfaces,X,1,this->mu,Q);
	for(int i=0;i<this->nprojections;i++){
		double *pi = this->projections+(i*this->nfaces);
		double dist;
		this->dist_metric(0,pi,Q,&dist,this->nfaces);
		if (dist<minDist){
			minDist=dist;
			minIdx=i;
		}
	}

	faceClass=(minIdx<0)?-1:this->y[minIdx];
	faceIdx=minIdx;
	free(Q);
	return minDist;
}

double *FisherfacesModel::reconstructProjection(int data){

	double *R;
	reconstruct(this->W,this->d,this->nfaces,this->projections+(data*this->nfaces),this->mu,R);
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
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(this->mu,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			to_go=this->d*this->nfaces;
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(this->W,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}
			to_go=this->nprojections*this->nfaces;
			while(to_go > 0)
			{
			  const size_t wrote = fwrite(this->projections,sizeof(double), to_go, out);
			  if(wrote == 0)
				break;
			  to_go -= wrote;
			}

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
		  this->mu=(double*)malloc(sizeof(double)*this->d);
		  this->W=(double*)malloc(sizeof(double)*this->d*this->nfaces);
		  this->projections=(double*)malloc(sizeof(double)*this->nprojections*this->nfaces);
		  fread(this->y,sizeof(int),this->nprojections,in);
		  fread(this->mu,sizeof(double), this->d, in);
		  fread(this->W,sizeof(double), this->d*this->nfaces, in);
		  fread(this->projections,sizeof(double), this->nprojections*this->nfaces, in);
		  fclose(in);
	  }else{
		  printf("Load Error\n");
		  exit(-1);
	  }
}
