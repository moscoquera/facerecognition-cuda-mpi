#include <stdlib.h>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include <lapacke.h>
#include <set>
#include <stdio.h>


extern void dsyevd_( char* jobz, char* uplo, lapack_int* n, double* a,
        lapack_int* lda, double* w, double* work, lapack_int* lwork,
        lapack_int* iwork, lapack_int* liwork, lapack_int *info );

extern void dsyevd2_( char* jobz, char* uplo, lapack_int* n, double* a,
        lapack_int* lda, double* w, double* work, lapack_int* lwork,
        lapack_int* iwork, lapack_int* liwork, lapack_int *info );



using namespace std;

int pca(double *X_n_d,int num_components,int n, int d, double* &eigenvectors_d_num_components,double* &eigenvalues_num_components, double* &mean_d){
	double* Xi;
	double* C;
	double* eigenvalues;
	double* eigenvectors;


	if ((num_components <= 0) || (num_components>n)){
			num_components = n;
	}


	double* mu = columnMean(X_n_d,n,d);
	//mu está bien calculado


	double* Xm = (double*)malloc(sizeof(double)*n*d);
	for(int r=0;r<n;r++){
		Xi=X_n_d+(r*d);
		for(int c=0;c<d;c++){
			*(Xm+(r*d)+c)=*(Xi+c)-*(mu+c);
		}
	}


	char jobz ='V';
	char uplo = 'L';
	double* Xmt;
	transponer(Xm,Xmt,n,d);

	if (n>d){
		eigenvalues = (double*)malloc(sizeof(double)*d);
		eigenvectors  =(double*)malloc(sizeof(double)*d*d);
		memset(eigenvalues,0,d*sizeof(double));
		memset(eigenvectors,0,d*d*sizeof(double));
		C = (double*)malloc(sizeof(double)*d*d);
		matMult(Xmt,Xm,d,n,d,C);

		memcpy(eigenvectors,C,sizeof(double)*d*d);
		LAPACKE_dsyevd(LAPACK_ROW_MAJOR,jobz,uplo,d,eigenvectors,d,eigenvalues);
	}else{
		eigenvalues = (double*)malloc(sizeof(double)*n);
		eigenvectors  =(double*)malloc(sizeof(double)*n*n);
		memset(eigenvalues,0,n*sizeof(double));
		memset(eigenvectors,0,n*n*sizeof(double));
		C = (double*)malloc(sizeof(double)*n*n);
		matMult(Xm,Xmt,n,d,n,C);

		memcpy(eigenvectors,C,sizeof(double)*n*n);
		int res =LAPACKE_dsyevd(LAPACK_ROW_MAJOR,jobz,uplo,n,eigenvectors,n,eigenvalues);



		double* eigenvectorsTmp = eigenvectors;
		eigenvectors = (double*)malloc(sizeof(double)*d*n);
		matMult(Xmt,eigenvectorsTmp,d,n,n,eigenvectors);
		free(eigenvectorsTmp);



		for(int c=0;c<n;c++){
			double eigennorm = columnNorm(eigenvectors,d,n,c);
			for(int r=0;r<d;r++){
				*(eigenvectors+(r*n)+c)=*(eigenvectors+(r*n)+c)/eigennorm;
			}

		}
	}

	double *bestfaces = (double*)malloc(sizeof(double)*d*num_components);
	double *bestvalues = (double*)malloc(sizeof(double)*num_components);

	for(int k=0;k<num_components;k++){
		*(bestvalues+k)=*(eigenvalues+(n-k-1));
		for(int r=0;r<d;r++){
			*(bestfaces+(r*num_components)+k)=*(eigenvectors+(r*n)+(n-k-1));
		}

	}

		eigenvectors_d_num_components = bestfaces;
	eigenvalues_num_components = bestvalues;
	mean_d=mu;


	free(eigenvalues);
	free(eigenvectors);
	free(Xmt);
	free(Xm);
	free(C);


	return 0;
}


int lda(double *X_n_d,int* y_n,int n,int &num_components, int d, double* &eigenvectors_d_num,double* &eigenvalues_num){
	std::set<int> cset;
	for(int yIterator=0;yIterator<n;yIterator++){
		if (cset.find(*(y_n+yIterator))==cset.end()){
			cset.insert(*(y_n+yIterator));
		}
	}


	if (num_components<=0 || num_components>cset.size()-1){
		num_components=cset.size()-1;
	}

	/*double fun;

	for(int r=0;r<n;r++){
		double max=-1000000;
		double min=1000000000;
		for (int col=0;col<d;col++){
			fun=*(X_n_d+(r*d)+col);
			if (fun>max){
				max=fun;
			}

			if (fun<min){
				min=fun;
			}

		}
		printf("$.. %d %.15f %.15f\n",r,max,min);
	}*/


	double* meanTotal = columnMean(X_n_d,n,d);




	double * Sw = (double*)malloc(sizeof(double)*d*d);
	double * Sb = (double*)malloc(sizeof(double)*d*d);
	double * SwSb;

	memset(Sw,0,sizeof(double)*d*d);
	memset(Sb,0,sizeof(double)*d*d);

	set<int>::iterator i;
	double *Xi;

	for(i=cset.begin();i!=cset.end();i++){
		int XiSize=0;
		for(int yIterator=0;yIterator<n;yIterator++){
			if (*(y_n+yIterator) == *i){
				XiSize++;
			}
		}
		Xi = (double*)malloc(sizeof(double)*XiSize*d);
		int idx=0;
		for(int yIterator=0;idx<XiSize;yIterator++){
					if (*(y_n+yIterator) == *i){
						memcpy(Xi+(idx*d),X_n_d+(yIterator*d),sizeof(double)*d);
						idx++;
					}
		}




		double* meanClass = columnMean(Xi,XiSize,d);
		double* Ximean = (double*)malloc(sizeof(double)*XiSize*d);
		memset(Ximean,0,sizeof(double)*XiSize*d);
		double* XimeanT_Ximean,*XimeanT;
		double* meanClassTotal = (double*)malloc(sizeof(double)*d);
		memset(meanClassTotal,0,sizeof(double)*d);
		double* meansDot;
		for(int r = 0;r<XiSize;r++){
			for(int c=0;c<d;c++){
				*(Ximean+(d*r)+c)=*(Xi+(r*d)+c)-*(meanClass+c);
			}
		}
		for(int c=0;c<d;c++){
			*(meanClassTotal+c)=(*(meanClass+c))-(*(meanTotal+c));
		}


		transponer(Ximean,XimeanT,XiSize,d);
		XimeanT_Ximean=(double*)malloc(sizeof(double)*d*d);
		matMult(XimeanT,Ximean,d,XiSize,d,XimeanT_Ximean);
/*
		double fun;
		for(int r=0;r<XiSize;r++){
			for(int ib=0;ib<d;ib++){
				fun=*(Ximean+(r*d)+ib);
				printf("%.6f ",fun);
			}
			printf("\n");

		}*/


		meansDot=(double*)malloc(sizeof(double)*d*d);
		matMult(meanClassTotal,meanClassTotal,d,1,d,meansDot);
		//meansDot=matMult(meanClassTotal,meanClassTotal,1,d,1); //debe dar un escalar
		//printf(" %.15f\n",*(meansDot));
		for(int Si=0;Si<d*d;Si++){
			*(Sw+Si)=*(Sw+Si)+*(XimeanT_Ximean+Si);
			*(Sb+Si)=*(Sb+Si)+n*(*(meansDot));
		}




		//printf("%d %.15f :: %.15f\n",*i,*(meanClassTotal),*(meansDot));

		free(Ximean);
		free(XimeanT);
		free(meanClass);
		free(XimeanT_Ximean);
		free(meanClassTotal);
		free(meansDot);
		free(Xi);
	}

	char jobz ='V';
	char uplo = 'L';


	lapack_int* SwIpiv = (lapack_int*)malloc(sizeof(lapack_int)*d);
	memset(SwIpiv,0,sizeof(lapack_int)*d);
	int resdgetrf=LAPACKE_dgetrf(LAPACK_ROW_MAJOR,d,d,Sw,d,SwIpiv);
	int resdgetri=LAPACKE_dgetri(LAPACK_ROW_MAJOR,d,Sw,d,SwIpiv);



	SwSb=(double*)malloc(sizeof(double)*d*d);
	matMult(Sw,Sb,d,d,d,SwSb);


	double* eigenvalues = (double*)malloc(sizeof(double)*d);
	double* eigenvectors  =(double*)malloc(sizeof(double)*d*d);
	memset(eigenvalues,0,d*sizeof(double));
	memcpy(eigenvectors,SwSb,sizeof(double)*d*d);

	LAPACKE_dsyevd(LAPACK_ROW_MAJOR,jobz,uplo,d,eigenvectors,d,eigenvalues);



	double *bestfaces = (double*)malloc(sizeof(double)*d*num_components);
	double *bestvalues = (double*)malloc(sizeof(double)*num_components);

	for(int k=0;k<num_components;k++){
		*(bestvalues+k)=*(eigenvalues+(d-k-1));
		for(int r=0;r<d;r++){
			*(bestfaces+(r*num_components)+k)=*(eigenvectors+(r*d)+(d-k-1));
		}

	}


	eigenvectors_d_num = bestfaces;
	eigenvalues_num = bestvalues;


	free(meanTotal);
	free(Sw);
	free(Sb);
	free(SwSb);
	free(eigenvalues);
	free(eigenvectors);
	return 0;
}

int fisherfaces(double* X_n_d,int* y_n,int num_components,int n, int d, double* &eigenvectors,double* &eigenvalues, double* &mean, int &nfaces){

	std::set<int> c;
	vector<int>::iterator yIterator;
	for(int yIterator=0;yIterator<n;yIterator++){
		if (c.find(y_n[yIterator])==c.end()){
			c.insert(y_n[yIterator]);
		}
	}


	int ncpca=(n-c.size());


	double *eigenvalues_ncpca_pca,*eigenvectors_d_ncpca_pca,*mu_d_pca;
	double *eigenvalues_lda_num,*eigenvectors_lda_ncpca_num;
	double *eigenpca_project_n_ncpca = (double*)malloc(sizeof(double)*n*ncpca);


	pca(X_n_d,ncpca,n,d,eigenvectors_d_ncpca_pca,eigenvalues_ncpca_pca,mu_d_pca);

	//hasta PCA está posiblemente bien

	project(eigenvectors_d_ncpca_pca,d,ncpca,X_n_d,n,mu_d_pca,eigenpca_project_n_ncpca);
	//project presenta diferencias, pero se presume buena






	int lda_num=0;

	lda(eigenpca_project_n_ncpca,y_n,n,lda_num,ncpca,eigenvectors_lda_ncpca_num,eigenvalues_lda_num);

	//printf("%f %f\n",*(eigenvalues_lda_num),*(eigenvectors_lda_ncpca_num));

	eigenvectors=(double*)malloc(sizeof(double)*d*lda_num);
	matMult(eigenvectors_d_ncpca_pca,eigenvectors_lda_ncpca_num,d,ncpca,lda_num,eigenvectors);
	eigenvalues=eigenvalues_lda_num;
	mean=mu_d_pca;
	nfaces=lda_num;

	free(eigenvalues_ncpca_pca);
	free(eigenvectors_d_ncpca_pca);
	free(eigenvectors_lda_ncpca_num);
	free(eigenpca_project_n_ncpca);
	return 0;
}

