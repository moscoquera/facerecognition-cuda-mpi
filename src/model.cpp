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


void BaseModel::predict(double *X, int &faceClass, int &faceIdx){
	double minDist = DBL_MAX;
	int minIdx=-1;
	double *Q;
	project(this->W,this->d,this->num_components,X,1,this->mu,Q);
	for(int i=0;i<this->nprojections;i++){
		double *pi = *(this->projections+i);
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

}


double *BaseModel::reconstructProjection(int data){

	double *R;
	reconstruct(this->W,this->d,this->num_components,*(this->projections+data),this->mu,R);
	return R;
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
		for(int xi=0;xi<n;xi++){
			double * proj;
			double* imgdata = X+(xi*this->d);
			for(int pix=0;pix<this->d;pix++){
				*(imgdatadouble+pix)=*(imgdata+pix);
			}
			project(this->W,this->d,this->num_components,imgdatadouble,1,this->mu,proj);
			*(this->projections+xi)=proj;
		}

		free(V);
		free(imgdatadouble);
}


FisherfacesModel::FisherfacesModel(int d,
							void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int)):
								BaseModel(d,dist_metric){


	}

FisherfacesModel::~FisherfacesModel(){

}

void FisherfacesModel::compute(double* X, int* y, int n){
		/*double *V;
		fisherfaces(X,y,this->num_components,n,this->d,this->W,V,this->mu,this->nfaces);
		this->y = y;
		double *imgdatadouble = (double*)malloc(sizeof(double)*this->d);
		for(int xi=0;xi<n;xi++){
			double * proj;
			double* imgdata = X+(xi*this->d);
			for(int pix=0;pix<this->d;pix++){
				*(imgdatadouble+pix)=*(imgdata+pix);
			}
			project(this->W,this->d,this->nfaces,imgdatadouble,1,this->mu,proj);
			this->projections.push_back(proj);
		}

		free(V);
		free(imgdatadouble);*/
}


void FisherfacesModel::predict(double *X, int &faceClass, int &faceIdx){
	/*double minDist = DBL_MAX;
	int minIdx=-1;
	double *Q;
	project(this->W,this->d,this->nfaces,X,1,this->mu,Q);
	for(int i=0;i<this->projections.size();i++){
		double *pi = this->projections.at(i);
		double dist = this->dist_metric(pi,Q,this->nfaces);
		if (dist<minDist){
			minDist=dist;
			minIdx=i;
		}
	}

	faceClass=(minIdx<0)?-1:this->y[minIdx];
	faceIdx=minIdx;
	free(Q);*/
}

double *FisherfacesModel::reconstructProjection(int data){
/*
	double *R;
	reconstruct(this->W,this->d,this->nfaces,this->projections.at(data),this->mu,R);
	return R;*/
}
