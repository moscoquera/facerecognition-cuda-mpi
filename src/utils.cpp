#include <stdlib.h>
#include <vector>
#include "utils.hpp"
#include <stdio.h>
#include <math.h>



void matMult(double* A_n_m,double* B_m_p, int n, int m, int p, double* AB){


	for(int r=0;r<n;r++){
		for(int c=0;c<p;c++){
			*(AB+(r*p)+c)=0;
			for(int k=0;k<m;k++){
				*(AB+(r*p)+c)+=(*(A_n_m+(r*m)+k))*(*(B_m_p+(k*p)+c));
			}
		}
	}


}


void project(double *W_d_n,int d,int n, double *X_Xn_d,int Xn, double* mu_d, double * projected_Xn_n){
	if (mu_d==NULL){
		matMult(X_Xn_d,W_d_n,Xn,d,n,projected_Xn_n);
		return;
	}
	double* Xmu_Xn_d = (double*)malloc(sizeof(double)*d*Xn);
	for(int r=0;r<Xn;r++){
		for(int c=0;c<d;c++){
			*(Xmu_Xn_d+(r*d)+c)=*(X_Xn_d+(r*d)+c)-*(mu_d+c);
		}
	}
	//Xmu difiere cuando se suman las columnas posiblemente por la larga cantidad de sumas, porque los valores
	//de las celdas coinciden

	matMult(Xmu_Xn_d,W_d_n,Xn,d,n,projected_Xn_n);
	free(Xmu_Xn_d);
}


void reconstruct(double *W,int d,int n, double *Y, double* mu, double * &reconstructed){
	double *Wt;
	transponer(W,Wt,d,n);
	reconstructed=(double*)malloc(sizeof(double)*1*d);
	matMult(Y,Wt,1,n,d,reconstructed);
	if (mu!=NULL){
		for(int i=0;i<d;i++){
			*(reconstructed+i)=*(reconstructed+i)+*(mu+i);
		}
	}
	free(Wt);

}



double* columnMean(double* x,int n, int d){
	double *tmp = (double*)malloc(sizeof(double)*d);
	memset(tmp,0,d*sizeof(double));
	double* cx;
	for(int i=0;i<n;i++){
		cx = x+(i*d);
		for(int j=0;j<d;j++){
			*(tmp+j)+=*(cx+j);
		}
	}

	for(int j=0;j<d;j++){
		*(tmp+j)=*(tmp+j)/n;
	}
	return tmp;
}




double* picksubcolumns(double *data, int rows, int cols,int* idxs, int k){
	double* out = (double*)malloc(sizeof(double)*rows*k);
	int idx;
	for(int c=0;c<k;c++){
			idx=*(idxs+c);
			for(int r=0;r<rows;r++){
				*(out+(k*r)+c)=*(data+(cols*r)+idx);
			}

	}
	return out;
}

double columnNorm(double *W,int rows,int cols,int c){
	double norm=0;
	for(int r=0;r<rows;r++){
		norm+=(*(W+(r*cols)+c))*(*(W+(r*cols)+c));
	}
	return sqrt(norm);
}




