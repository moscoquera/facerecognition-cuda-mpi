/*
 * model.hpp
 *
 *  Created on: May 12, 2017
 *      Author: smith
 */

#ifndef MODEL_HPP_
#define MODEL_HPP_

#include "distance.hpp"
#include <stdlib.h>

class BaseModel{

public:
	int num_components;
protected:
	void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int);
	double* projections;
	int nprojections;
	double* W;
	double* mu;
	int* y;
	int d;


public:
	BaseModel(int d, void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int));
	virtual ~BaseModel();
	virtual void compute(double* X, int* y, int n);
	virtual double predict(double *X, int &faceClass,int &faceIdx);
	virtual double * reconstructProjection(int i);
	virtual void save(char*);
	virtual void load(char*);

};

class EigenfacesModel: public BaseModel{

public:
	EigenfacesModel(int d, void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int));
	~EigenfacesModel();
	virtual void compute(double*  X, int* y, int n);
	virtual void save(char*);
	virtual void load(char*);
};


class FisherfacesModel: public BaseModel{

	int nfaces;

public:
	FisherfacesModel(int d, void (*dist_metric)(cudaStream_t stream,double *, double *,double*, int));
	~FisherfacesModel();
	virtual void compute(double*  X, int* y, int n);
	virtual double predict(double *X, int &faceClass,int &faceIdx);
	virtual double * reconstructProjection(int i);
	virtual void save(char*);
	virtual void load(char*);
};



#endif /* MODEL_HPP_ */
