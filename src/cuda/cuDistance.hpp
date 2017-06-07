/*
 * distance.hpp
 *
 *  Created on: May 11, 2017
 *      Author: smith
 */

#ifndef cuDISTANCE_HPP_
#define cuDISTANCE_HPP_

#include <cuda.h>

namespace cu{

void cuEuclideanDistance(cudaStream_t stream,double *p, double *q,double* out, int n);
double cuCosineDistance(double *p, double *q, int n);
}


#endif /* DISTANCE_HPP_ */
