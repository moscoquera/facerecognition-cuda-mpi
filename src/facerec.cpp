/*
 ============================================================================
 Name        : facerec.cu
 Author      : smith
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <cerrno>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "utils.hpp"
#include "distance.hpp"
#include "cuda/cuDistance.hpp"
#include "model.hpp"
#include "cuda/cumodel.hpp"
#include "routines.hpp"
#include <cstdio>
#include "openmpi-x86_64/mpi.h"


using namespace std;
using namespace cv;

const int width=92,height=112;

static void CheckErrorAux (const char *, unsigned, const char *, int);

#define CHECK_RETURN(value) CheckErrorAux(__FILE__,__LINE__, #value, value)

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalcatface_extended.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);




/**
    * @brief makeCanvas Makes composite image from the given images
    * @param vecMat Vector of Images.
    * @param windowHeight The height of the new composite image to be formed.
    * @param nRows Number of rows of images. (Number of columns will be calculated
    *              depending on the value of total number of images).
    * @return new composite image.
    */
   cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
           int N = vecMat.size();
           nRows  = nRows > N ? N : nRows;
           int edgeThickness = 10;
           int imagesPerRow = ceil(double(N) / nRows);
           int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
           int maxRowLength = 0;

           std::vector<int> resizeWidth;
           for (int i = 0; i < N;) {
                   int thisRowLen = 0;
                   for (int k = 0; k < imagesPerRow; k++) {
                           double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
                           int temp = int( ceil(resizeHeight * aspectRatio));
                           resizeWidth.push_back(temp);
                           thisRowLen += temp;
                           if (++i == N) break;
                   }
                   if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
                           maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
                   }
           }
           int windowWidth = maxRowLength;
           cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

           for (int k = 0, i = 0; i < nRows; i++) {
                   int y = i * resizeHeight + (i + 1) * edgeThickness;
                   int x_end = edgeThickness;
                   for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
                           int x = x_end;
                           cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
                           cv::Size s = canvasImage(roi).size();
                           // change the number of channels to three
                           cv::Mat target_ROI(s, CV_8UC3);
                           if (vecMat[k].channels() != canvasImage.channels()) {
                               if (vecMat[k].channels() == 1) {
                                   cv::cvtColor(vecMat[k], target_ROI, CV_GRAY2BGR);
                               }
                           } else {
                               vecMat[k].copyTo(target_ROI);
                           }
                           cv::resize(target_ROI, target_ROI, s);
                           if (target_ROI.type() != canvasImage.type()) {
                               target_ROI.convertTo(target_ROI, canvasImage.type());
                           }
                           target_ROI.copyTo(canvasImage(roi));
                           x_end += resizeWidth[k] + edgeThickness;
                   }
           }
           return canvasImage;
   }







//antes era la mitad,pero ya me dio pereza organizarla, igual el tiempo de computo es el mismo.
static int list_files ( char* path,dirent* &names, int &count_out){


		DIR *root;
		dirent* dp;
		errno=0;
		if ((root=opendir(path))==NULL){
			return errno;
		}
		int counting=1;
		int count=0;
		while (counting) {
		    errno = 0;
		    if ((dp = readdir(root)) != NULL) {
		        if (strcmp(dp->d_name,".")!=0 && strcmp(dp->d_name,"..")!=0){
		        	count++;
		       }

		    } else {
		    	if (errno != 0) {
		    		closedir(root);
		        	return errno;
		        }
		    	//llegó al final del directorio
		    	if (counting){
		    		counting=0;
		    	}
		    }
		}
		rewinddir(root);
		names = (dirent*)malloc(sizeof(dirent)*count);
		int index=0;
		while(index<count){
			//printf("%d- %s\n",index,*(names+index));
			errno = 0;
			if ((dp = readdir(root)) != NULL) {
				if (strcmp(dp->d_name,".")!=0 && strcmp(dp->d_name,"..")!=0){
					*(names+index)=*dp;
					index++;
			   }

			} else {
				if (errno != 0) {
					closedir(root);
					return errno;
				}
				//llegó al final del directorio
				if (counting){
					break;
				}
			}
		}

		closedir(root);
		count_out= count;
		return 0;
}


static int read_images(char* path,double* &outputImagesData,int* &outputImagesLabels,std::vector<char*> &names, int Width, int Height, int &k){
	dirent* folder,*subject,*images,*image;
	int folderlength=0,imagesLength=0;
	CHECK_RETURN(list_files(path,folder,folderlength));
	char fullname[512];
	int classIdx=0;
	int idxImg=0;

	k=0;
	for(int kt=0;kt<folderlength;kt++){
		subject = (dirent*)(folder+kt);
		if (subject->d_type!=DT_DIR){
			continue; //ignoro los que no son directorios
		}
		fullname[0]=0;
		strcat(fullname,path);
		strcat(fullname,"/");
		strcat(fullname,subject->d_name);
		CHECK_RETURN(list_files(fullname,images,imagesLength));
		k+=imagesLength;
	}

	outputImagesData=(double*)malloc(sizeof(double)*k*Width*Height);
	outputImagesLabels=(int*)malloc(sizeof(int)*k);


	for(int i=0;i<folderlength;i++){
		subject = (dirent*)(folder+i);
		if (subject->d_type!=DT_DIR){
			continue; //ignoro los que no son directorios
		}
		fullname[0]=0;
		strcat(fullname,path);
		strcat(fullname,"/");
		strcat(fullname,subject->d_name);
		CHECK_RETURN(list_files(fullname,images,imagesLength));
		for(int j=0;j<imagesLength;j++){
			image = (dirent*)(images+j);
			fullname[0]=0;
			strcat(fullname,path);
			strcat(fullname,"/");
			strcat(fullname,subject->d_name);
			strcat(fullname,"/");
			strcat(fullname,image->d_name);
			Mat imageData;
			imageData = imread(fullname, CV_LOAD_IMAGE_GRAYSCALE);

			if(! imageData.data )
			{
				cout <<  "Could not open or find the image" << std::endl ;
				return -1;
			}

			Mat face(Height,Width,CV_8UC1);
			cv::resize(imageData,face,face.size(),0,0,INTER_LINEAR);
			cv::Mat f(Height,Width,DataType<double>::type);
			face.convertTo(f,DataType<double>::type,1,0);
			memcpy(outputImagesData+(idxImg*Width*Height),f.data,sizeof(double)*Width*Height);
			*(outputImagesLabels+idxImg)= classIdx;
			idxImg++;
		}
		char* nameo = (char*)malloc(sizeof(char)*256);
		memcpy(nameo,subject->d_name,sizeof(char)*256);
		names.push_back(nameo);
		classIdx++;
	}
	return 0;
}




void normalize(double *X,int Xlen,int low,int high,uchar * &out){
	double minX = DBL_MAX, maxX = DBL_MIN;
	out = (uchar*)malloc(sizeof(uchar)*Xlen);
	for(int i=0;i<Xlen;i++){
		if (*(X+i)<minX){
			minX=*(X+i);
		}

		if (*(X+i)>maxX){
			maxX=*(X+i);
		}

	}

	double scale = maxX-minX;
	int nscale = high-low;
	for(int i=0;i<Xlen;i++){
		*(out+i)=(uchar)((((*(X+i)-minX)/scale)*nscale)+low);
		//printf("%d\n",*(out+i));
	}

}


Mat labelImage(Mat source, char** labels, int l, int ls){
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;

	int baseline=0;
	Size textSize = getTextSize((char*)labels, fontFace,fontScale, thickness, &baseline);
	baseline+=thickness;

	int destHeight=textSize.height*l;
	Mat dest(destHeight+source.rows,source.cols,CV_8UC1,Scalar::all(255));


	for(int i=0;i<l;i++){
		Point labelPos(1,(i+1)*textSize.height);
		putText(dest,(char*)labels+(i*ls),labelPos,fontFace,fontScale,Scalar::all(0),thickness,8,false);
	}

	Rect roi(0,textSize.height*l,source.cols,source.rows);

	source.copyTo(dest(roi));

	return dest;
}


void pca_test(char* imgPath){
		double* ImagesData;
		int* ImagesLabels;
		std::vector<char*> Subjects;
		int width=92,height=112,samples;
		double* W,*V,* mu;

		CHECK_RETURN(read_images(imgPath,ImagesData,ImagesLabels,Subjects,width,height,samples));
		int k= samples;

		CHECK_RETURN(pca(ImagesData,k,samples,width*height,W,V,mu));
		printf("pca done\n");


		double *Wt;
		transponer(W,Wt,width*height,k);

		std::vector<Mat> images;
		for(int i=0;i<16;i++){
			uchar* imgdata;
			normalize(Wt+(width*height*i),width*height,0,255,imgdata);
			Mat face(height,width,CV_8UC1);
			face.data= imgdata;
			Mat facejet;
			cv::applyColorMap(face,facejet,COLORMAP_JET);
			images.push_back(facejet);
		}

		Mat faces = makeCanvas(images,640,4);
		namedWindow( "EigenMap", WINDOW_AUTOSIZE );// Create a window for display.
		imshow( "EigenMap", faces );

		images.clear();
		for(int samples=10;samples<k;samples+=20){
			double* Wsamples;
			subMatrix(W,width*height,k,0,width*height,0,samples,Wsamples);
			double* P,*R;
			double* imgdata = ImagesData+(50*width*height);
			project(Wsamples,width*height,samples,imgdata,1,mu,P);
			reconstruct(Wsamples,width*height,samples,P,mu,R);

			uchar *reconstructed;
			normalize(R,width*height,0,255,reconstructed);
			Mat face(height,width,CV_8UC1);
			face.data= reconstructed;
			images.push_back(face);

			free(Wsamples);
			free(P);
			free(R);
		}

		faces = makeCanvas(images,640,4);
		namedWindow( "Reconstruct", WINDOW_AUTOSIZE );// Create a window for display.
		imshow( "Reconstruct", faces );
		waitKey(0);

}

void pca_prediction(char* path){

	double* ImagesData;
	int* ImagesLabels;
	std::vector<char*> Subjects;
	int width=92,height=112,samples;


	CHECK_RETURN(read_images(path,ImagesData,ImagesLabels,Subjects,width,height,samples));


	EigenfacesModel mod= EigenfacesModel(width*height,EuclideanDistance);

	std::vector<Mat> images;

	for(int i=0;i<50;i++){
		int test = rand()%samples;
		double* imgdata = ImagesData+(test*width*height);


		int pred;
		int predIdx;
		mod.predict(imgdata,pred,predIdx);

		printf("Prediction: %d expected: %d subject: %s\n",pred,*(ImagesLabels+test),Subjects.at(pred));

		double* R = mod.reconstructProjection(predIdx);
		uchar *reconstructed;
		normalize(R,width*height,0,255,reconstructed);
		Mat face(height,width,CV_8UC1);
		face.data= reconstructed;
		char labels[3][20];
		sprintf(labels[0],"pred: %d",pred);
		sprintf(labels[1],"exp: %d",*(ImagesLabels+test));
		sprintf(labels[2],"sub: %s",Subjects.at(pred));
		images.push_back(labelImage(face,(char**)labels,3,20));

	}

	Mat faces = makeCanvas(images,640,640/images.at(0).rows);
	namedWindow( "Reconstruct", WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Reconstruct", faces );
	waitKey(0);

}

void fisherfaces_test(char* imgPath){
		double* ImagesData;
		int* ImagesLabels;
		std::vector<char*> Subjects;
		int width=92,height=112,samples;
		double* W,*V,* mu;

		CHECK_RETURN(read_images(imgPath,ImagesData,ImagesLabels,Subjects,width,height,samples));
		int k= samples;
		int nfaces=0;
		CHECK_RETURN(fisherfaces(ImagesData,ImagesLabels,k,samples,width*height,W,V,mu,nfaces));
		printf("fisher done %.15f %.15f %.15f\n",*(V+0),*(V+1),*(V+2));


		double *Wt;
		transponer(W,Wt,width*height,nfaces);

		std::vector<Mat> images;
		for(int i=0;i<16;i++){
			uchar* imgdata;
			normalize(Wt+(width*height*i),width*height,0,255,imgdata);
			Mat face(height,width,CV_8UC1);
			face.data= imgdata;
			Mat facejet;
			cv::applyColorMap(face,facejet,COLORMAP_JET);
			images.push_back(facejet);
		}

		Mat faces = makeCanvas(images,640,4);
		namedWindow( "EigenMap", WINDOW_AUTOSIZE );// Create a window for display.
		imshow( "EigenMap", faces );

		images.clear();
		for(int i=0;i<nfaces;i++){
			double* Wsamples;
			subMatrix(W,width*height,nfaces,0,width*height,i,i+1,Wsamples);
			double* P,*R;
			double* imgdata = ImagesData+(0*width*height);
			project(Wsamples,width*height,1,imgdata,1,mu,P);
			reconstruct(Wsamples,width*height,1,P,mu,R);

			uchar *reconstructed;
			normalize(R,width*height,0,255,reconstructed);
			Mat face(height,width,CV_8UC1);
			face.data= reconstructed;
			images.push_back(face);

			free(Wsamples);
			free(P);
			free(R);
		}

		faces = makeCanvas(images,640,4);
		namedWindow( "Reconstruct", WINDOW_AUTOSIZE );// Create a window for display.
		imshow( "Reconstruct", faces );
		waitKey(0);

}

void fisherfaces_prediction(char* path){

	double* ImagesData;
	int* ImagesLabels;
	std::vector<char*> Subjects;
	int width=92,height=112,samples;


	CHECK_RETURN(read_images(path,ImagesData,ImagesLabels,Subjects,width,height,samples));


	FisherfacesModel mod= FisherfacesModel(width*height,EuclideanDistance);

	std::vector<Mat> images;

	for(int i=0;i<50;i++){
		int test = rand()%samples;
		double* imgdata = ImagesData+(test*width*height);


		int pred;
		int predIdx;
		mod.predict(imgdata,pred,predIdx);
		printf("pred:%d\n",pred);
		printf("Prediction: %d expected: %d subject: %s\n",pred,*(ImagesLabels+test),Subjects.at(pred));

		double* R = mod.reconstructProjection(predIdx);
		uchar *reconstructed;
		normalize(R,width*height,0,255,reconstructed);
		Mat face(height,width,CV_8UC1);
		face.data= reconstructed;
		char labels[3][20];
		sprintf(labels[0],"pred: %d",pred);
		sprintf(labels[1],"exp: %d",*(ImagesLabels+test));
		sprintf(labels[2],"sub: %s",Subjects.at(pred));
		images.push_back(labelImage(face,(char**)labels,3,20));

	}

	Mat faces = makeCanvas(images,640,640/images.at(0).rows);
	namedWindow( "Reconstruct", WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Reconstruct", faces );
	waitKey(0);

}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, BaseModel* model, std::vector<char*> Subjects)
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );


  std::vector<char*> names;
  for (size_t i=0;i<faces.size();i++){

	Mat imageData = frame(faces[i]);
	Mat face(height,width,CV_8UC1);
	cv::resize(imageData,face,face.size(),0,0,INTER_LINEAR);
	cv::Mat f(height,width,DataType<double>::type);
	face.convertTo(f,DataType<double>::type,1,0);
	int preclass=0,preid=0;
	model->predict((double*)f.data,preclass,preid);
	//printf("Pred: %d\n",preclass);
	char* name = Subjects.at(preclass);
	names.push_back(name);

  }

  int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	Scalar txtColor(0,0,255);

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    int baseline=0;
	Size textSize = getTextSize(names.at((int)i), fontFace,fontScale, thickness, &baseline);
	baseline+=thickness;
	putText(frame,names.at((int)i),center,fontFace,fontScale,txtColor,thickness,8,false);

  }
  //-- Show what you got
  imshow( window_name, frame );
 }

void detectAndDisplayMPI( Mat frame, int nodes)
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  MPI_Status status;
  std::vector<char*> names;
  for (size_t i=0;i<faces.size();i++){

	Mat imageData = frame(faces[i]);
	Mat face(height,width,CV_8UC1);
	cv::resize(imageData,face,face.size(),0,0,INTER_LINEAR);
	cv::Mat f(height,width,DataType<double>::type);
	face.convertTo(f,DataType<double>::type,1,0);

	int size=width*height;
	for(int node=1;node<nodes;node++){
		MPI_Send(&size,1,MPI_INT,node,1,MPI_COMM_WORLD);
		MPI_Send(f.data,width*height,MPI_DOUBLE,node,2,MPI_COMM_WORLD);
	}
	double best=DBL_MAX;
	double predval;
	char* predClass,*bestclass;
	for(int node=1;node<nodes;node++){
		MPI_Recv(&predval,1,MPI_DOUBLE,MPI_ANY_SOURCE,3,MPI_COMM_WORLD,&status);
		predClass=(char*)malloc(sizeof(char)*100);
		MPI_Recv(predClass,100,MPI_CHAR,status.MPI_SOURCE,3,MPI_COMM_WORLD,&status);

		if (predval<best){
			best=predval;
			bestclass=predClass;
		}

	}

	names.push_back(bestclass);

  }

  int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	Scalar txtColor(0,0,255);

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    int baseline=0;
	Size textSize = getTextSize(names.at((int)i), fontFace,fontScale, thickness, &baseline);
	baseline+=thickness;
	putText(frame,names.at((int)i),center,fontFace,fontScale,txtColor,thickness,8,false);

  }
  //-- Show what you got
  imshow( window_name, frame );
 }

int main(int argc, char* argv[])
{
	bool useCuda=false;
	bool test=false;
	bool camera=false;
	bool save=false;
	bool load=false;
	int mpirank=0;
	int mpi=0;
	char* method="pca";
	bool train=false;
	char* dataset=0;
	for(int i=0;i<argc;i++){
		if (strcmp("-cuda",argv[i])==0){
			useCuda=true;
		}else if (strcasecmp("-m",argv[i])==0){
			if (argc-i==1){
				printf("Error -m\n");
				return -1;
			}
			i++;
			if (strcmp("fisher",argv[i])==0){
				method=(char*)"fisher";
			}
		}else if (strcmp("-train",argv[i])==0){
			train=true;
		}else if (strcmp("-test",argv[i])==0){
			test=true;
		}else if (strcmp("-camera",argv[i])==0){
			camera=true;
		}else if (strcmp("-load",argv[i])==0){
			load=true;
		}else if (strcmp("-save",argv[i])==0){
			save=true;
		}else if (strcmp("-mpi",argv[i])==0){
			MPI_Init(&argc,&argv);
			MPI_Comm_size(MPI_COMM_WORLD, &mpi);
			MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
			for(int j=0;j<mpi;j++){
				if (j+1==mpirank){
					dataset=argv[i+j+1];
					break;
				}
			}
			i+=mpi;

			char processor_name[MPI_MAX_PROCESSOR_NAME];
			int name_len;
			MPI_Get_processor_name(processor_name, &name_len);

			// Print off a hello world message
			printf("processor %s, rank %d"
				   " out of %d processors in %s\n",
				   processor_name, mpirank, mpi,dataset);


		}else{
			dataset=argv[i];
		}
	}

	int samples=0;
	double* ImagesData;
	int* ImagesLabels;
	std::vector<char*> Subjects;


	if (mpi>0 && mpirank==0){ //primer nodo de hace nada
		if (train){
			int toreceive=mpi-1;
			int samples;
			int totalsamples=0;
			MPI_Status status;
			while(toreceive>0){
				MPI_Recv(&samples,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
				toreceive--;
				printf("node %d, samples: %d\n",status.MPI_SOURCE,samples);
				totalsamples+=samples;
			}
			printf("total samples: %d\n",totalsamples);
		}

		if (test){
			CHECK_RETURN(read_images(dataset,ImagesData,ImagesLabels,Subjects,width,height,samples));
			MPI_Status status;
			for(int i=0;i<50;i++){
				int test = rand()%samples;
				double* imgdata = ImagesData+(test*width*height);
				int size=width*height;
				for(int node=1;node<mpi;node++){
					MPI_Send(&size,1,MPI_INT,node,1,MPI_COMM_WORLD);
					MPI_Send(imgdata,width*height,MPI_DOUBLE,node,2,MPI_COMM_WORLD);
				}
				double best=DBL_MAX;
				double predval;
				char* predClass,*bestclass;
				for(int node=1;node<mpi;node++){
					MPI_Recv(&predval,1,MPI_DOUBLE,MPI_ANY_SOURCE,3,MPI_COMM_WORLD,&status);
					predClass=(char*)malloc(sizeof(char)*100);
					MPI_Recv(predClass,100,MPI_CHAR,status.MPI_SOURCE,3,MPI_COMM_WORLD,&status);

					if (predval<best){
						best=predval;
						bestclass=predClass;
					}

				}


				printf("Prediction: %s expected: %s\n",bestclass,Subjects.at(*(ImagesLabels+test)));

			}
			for(int node=1;node<mpi;node++){
				int size=0;
				MPI_Send(&size,1,MPI_INT,node,1,MPI_COMM_WORLD);
			}
		}

		if (camera){
				VideoCapture cap(0);
			   cap.open(-1);
			   Mat frame;

			   //-- 1. Load the cascades
			   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
			   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

			   //-- 2. Read the video stream
			   if( cap.isOpened() )
			   {
				 while( true )
				 {
				 cap >> frame;

			   //-- 3. Apply the classifier to the frame
				   if(!frame.empty() )
				   {
					   detectAndDisplayMPI(frame,mpi);
				   }
				   else
				   { printf(" --(!) No captured frame -- Break!"); break; }

				   int c = waitKey(10);
				   if( (char)c == 'c' ) { break; }
				  }
				 for(int node=1;node<mpi;node++){
					int size=0;
					MPI_Send(&size,1,MPI_INT,node,1,MPI_COMM_WORLD);
				}
			   }else{
				   printf("Error opening camera\n");
			   }
			}

	}



	if ((mpi>0 && mpirank>0) || mpi==0){ //si no es mpi o un nodo diferente al primero
		BaseModel* model;
		if (useCuda){
			if (strcmp(method,"fisher")==0){
				model=new cu::FisherfacesModel(width*height,&cu::cuEuclideanDistance);
			}else{
				model= new cu::EigenfacesModel(width*height,&cu::cuEuclideanDistance);
			}

		}else{
			if (strcmp(method,"fisher")==0){
				model=new FisherfacesModel(width*height,&EuclideanDistance);
			}else{
				model= new EigenfacesModel(width*height,&EuclideanDistance);
			}
		}


		char dataname[512];
		sprintf(dataname,"/tmp/facerec_%s_%d.bin",method,mpirank);

		if (load){
			model->load(dataname);
			CHECK_RETURN(read_images(dataset,ImagesData,ImagesLabels,Subjects,width,height,samples));

		}


		if (train){
			if (dataset==0){
				printf("invalid dataset\n");
				return -1;
			}

			CHECK_RETURN(read_images(dataset,ImagesData,ImagesLabels,Subjects,width,height,samples));
			model->num_components=samples;
			model->compute(ImagesData,ImagesLabels,samples);

			printf("done %d\n",mpirank);

			if (mpi){
				MPI_Send(&samples,1,MPI_INT,0,0,MPI_COMM_WORLD);
			}


		}

		if ((test || camera) && mpi>0){
			int size=0;
			MPI_Status status;
			double* imgdata;
			while(true){
				MPI_Recv(&size,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
				if (size==0){
					break;
				}
				imgdata = (double*)malloc(sizeof(double)*size);
				MPI_Recv(imgdata,size,MPI_DOUBLE,0,2,MPI_COMM_WORLD,&status);
				int pred;
				int predIdx;
				double dist =model->predict(imgdata,pred,predIdx);
				MPI_Send(&dist,1,MPI_DOUBLE,0,3,MPI_COMM_WORLD);
				MPI_Send(Subjects.at(pred),100,MPI_CHAR,0,3,MPI_COMM_WORLD);
				free(imgdata);

			}
		}

		if (test && mpi==0){
			std::vector<Mat> images;
			for(int i=0;i<50;i++){
				int test = rand()%samples;
				double* imgdata = ImagesData+(test*width*height);


				int pred;
				int predIdx;
				model->predict(imgdata,pred,predIdx);

				printf("Prediction: %d expected: %d subject: %s\n",pred,*(ImagesLabels+test),Subjects.at(pred));

				double* R = model->reconstructProjection(predIdx);
				uchar *reconstructed;
				normalize(R,width*height,0,255,reconstructed);
				Mat face(height,width,CV_8UC1);
				face.data= reconstructed;
				char labels[3][20];
				sprintf(labels[0],"pred: %d",pred);
				sprintf(labels[1],"exp: %d",*(ImagesLabels+test));
				sprintf(labels[2],"sub: %s",Subjects.at(pred));
				images.push_back(labelImage(face,(char**)labels,3,20));

			}
			/*Mat faces = makeCanvas(images,640,640/images.at(0).rows);
			namedWindow( "Reconstruct", WINDOW_AUTOSIZE );// Create a window for display.
			imshow( "Reconstruct", faces );
			waitKey(0);*/
		}

		if (save){
			model->save(dataname);
		}


	if (camera && mpi==0){
		VideoCapture cap(0);
	   cap.open(-1);
	   Mat frame;

	   //-- 1. Load the cascades
	   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	   //-- 2. Read the video stream
	   if( cap.isOpened() )
	   {
		 while( true )
		 {
		 cap >> frame;

	   //-- 3. Apply the classifier to the frame
		   if(!frame.empty() )
		   {
			   detectAndDisplay(frame,model,Subjects );
		   }
		   else
		   { printf(" --(!) No captured frame -- Break!"); break; }

		   int c = waitKey(10);
		   if( (char)c == 'c' ) { break; }
		  }
	   }else{
		   printf("Error opening camera\n");
	   }
	}

	delete model;
	}

	if (mpi){
		MPI_Finalize();
	}

	return 0;
}

static void CheckErrorAux (const char *file, unsigned line, const char *statement, int err)
{
	if (err == 0)
		return;
	std::cerr << statement<<" returned " << strerror(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}


