/*
 * utils.hpp
 *
 *  Created on: May 12, 2017
 *      Author: smith
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <cstring>


template <class T>
void subMatrix(T* source,int rows, int cols, int rfrom, int rto, int cfrom, int cto, T* &dst){
	int dstrows = rto-rfrom;
	int dstcols = cto-cfrom;
	dst = (T*)malloc(sizeof(T)*dstrows*dstcols);
	for(int r=0;r<dstrows;r++){
		for(int c=0;c<dstcols;c++){
			*(dst+(r*dstcols)+c)=*(source+((r+rfrom)*cols)+c+cfrom);
		}
	}

}

template <class T>
void transponer(T* data,T * &out,int rows,int cols){
	/**
	 *  está bien, si se hace la suma en diferente orden da diferentes valores culpa de la presicion
	 *
	 */

	out = (T*)malloc(sizeof(T)*rows*cols);
	memset(out,0,sizeof(T)*rows*cols);
	for(int r=0;r<rows;r++){
		for(int c=0;c<cols;c++){
			*(out+(c*rows)+r)=*(data+(r*cols)+c);

		}
	}
}


double* matMult(double* A,double* B, int n, int m, int p);
void project(double *W,int d,int n, double *X,int Xn, double* mu, double * &projected);
void reconstruct(double *W,int d,int n, double *Y, double* mu, double * &reconstructed);

double* columnMean(double* X,int rows, int cols );
double* picksubcolumns(double *data, int rows, int cols,int* idxs, int k);
double columnNorm(double *W,int rows,int cols,int c);

#endif /* UTILS_HPP_ */
