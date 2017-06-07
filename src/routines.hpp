/*
 * routines.hpp
 *
 *  Created on: May 12, 2017
 *      Author: smith
 */

#ifndef ROUTINES_HPP_
#define ROUTINES_HPP_
#include <stdlib.h>


int pca(double *X,int num_components,int n, int d, double* &eigenvectors,double* &eigenvalues, double* &mean);
int fisherfaces(double* X,int* y,int num_components,int n, int d, double* &eigenvectors,double* &eigenvalues, double* &mean, int &nfaces);



#endif /* ROUTINES_HPP_ */
