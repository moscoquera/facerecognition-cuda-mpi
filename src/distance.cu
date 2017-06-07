#include "utils.hpp"
#include "float.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include <cuda.h>

void EuclideanDistance(cudaStream_t stream, double *p, double *q,double* res, int n){
		long double maxPq=DBL_MIN;
		double tmp;

		//se implementa si n es muy grande y res se desborda
		/*for(int i=0;i<n;i++){
			tmp=abs((*(p+i)-*(q+i)));
			if (tmp>maxPq){
				maxPq=tmp;
			}
		}*/

		for(int i=0;i<n;i++){
			tmp=(*(p+i)-*(q+i));
			*(res)+=(tmp*tmp);
			//printf("%d %.15f %.15f %.15f\n",i,tmp,tmp*tmp,res);
		}

		*(res)=sqrt(*(res));

}

void CosineDistance(double *p, double *q,double *res, int n){
		double* pt;
		double* qt;
		transponer(p,pt,1,n);
		transponer(q,qt,1,n);

		double * ptq = matMult(pt,q,1,n,1);
		double * ppt = matMult(p,pt,1,n,1);
		double * qqt = matMult(q,qt,1,n,1);

		*(res) = (-1*(*ptq))/sqrt((*ppt)*(*qqt));
		free(pt);
		free(qt);
		free(ptq);
		free(ppt);
		free(qqt);


}

