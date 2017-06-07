/*
 * cumodel.hpp
 *
 *  Created on: May 12, 2017
 *      Author: smith
 */

#ifndef cu_MODEL_HPP_
#define cu_MODEL_HPP_

#include <stdlib.h>
#include "../model.hpp"
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace cu{



class EigenfacesModel: public BaseModel{

public:
EigenfacesModel(int d, void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int));
~EigenfacesModel();
virtual void compute(double*  X, int* y, int n);
virtual double predict(double *X, int &faceClass,int &faceIdx);
virtual double * reconstructProjection(int i);
virtual void save(char*);
	virtual void load(char*);
protected:
	cublasHandle_t blasHandler;
	cusolverDnHandle_t cusolverHandler;
	cudaStream_t* streams;
};


class FisherfacesModel: public BaseModel{

int nfaces=0;

public:
	FisherfacesModel(int d, void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int));
	~FisherfacesModel();
	virtual void compute(double*  X, int* y, int n);
		virtual double predict(double *X, int &faceClass,int &faceIdx);
		virtual double * reconstructProjection(int i);
		virtual void save(char*);
			virtual void load(char*);
protected:
	cublasHandle_t blasHandler;
	cusolverDnHandle_t cusolverHandler;
	cudaStream_t* streams;

};

}

#endif /* MODEL_HPP_ */
