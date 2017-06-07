/*
 * distance.hpp
 *
 *  Created on: May 11, 2017
 *      Author: smith
 */

#ifndef DISTANCE_HPP_
#define DISTANCE_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

void EuclideanDistance(cudaStream_t stream,double *p, double *q,double*res, int n);
void CosineDistance(double *p, double *q,double*, int n);



#endif /* DISTANCE_HPP_ */
