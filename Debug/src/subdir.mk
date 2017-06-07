################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/facerec.cpp \
../src/model.cpp \
../src/routines.cpp \
../src/utils.cpp 

CU_SRCS += \
../src/distance.cu 

CU_DEPS += \
./src/distance.d 

OBJS += \
./src/distance.o \
./src/facerec.o \
./src/model.o \
./src/routines.o \
./src/utils.o 

CPP_DEPS += \
./src/facerec.d \
./src/model.d \
./src/routines.d \
./src/utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/src/kernels/4.10.9-200.fc25.x86_64/include -I/usr/include/openmpi-x86_64 -I/home/smith/Downloads/delete/lapack-3.7.0 -I/home/smith/Downloads/delete/lapack-3.7.0/LAPACKE/include -G -g -pg -O0 -Xcompiler `pkg-config --cflags opencv --libs` -Xcompiler \"-Wl,-rpath,/usr/lib64/openmpi/lib\" -gencode arch=compute_35,code=sm_35  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc --compiler-bindir /opt/rh/devtoolset-3/root/usr/bin/gcc -I/usr/src/kernels/4.10.9-200.fc25.x86_64/include -I/usr/include/openmpi-x86_64 -I/home/smith/Downloads/delete/lapack-3.7.0 -I/home/smith/Downloads/delete/lapack-3.7.0/LAPACKE/include -G -g -pg -O0 -Xcompiler `pkg-config --cflags opencv --libs` -Xcompiler \"-Wl,-rpath,/usr/lib64/openmpi/lib\" --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/src/kernels/4.10.9-200.fc25.x86_64/include -I/usr/include/openmpi-x86_64 -I/home/smith/Downloads/delete/lapack-3.7.0 -I/home/smith/Downloads/delete/lapack-3.7.0/LAPACKE/include -G -g -pg -O0 -Xcompiler `pkg-config --cflags opencv --libs` -Xcompiler \"-Wl,-rpath,/usr/lib64/openmpi/lib\" -gencode arch=compute_35,code=sm_35  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc --compiler-bindir /opt/rh/devtoolset-3/root/usr/bin/gcc -I/usr/src/kernels/4.10.9-200.fc25.x86_64/include -I/usr/include/openmpi-x86_64 -I/home/smith/Downloads/delete/lapack-3.7.0 -I/home/smith/Downloads/delete/lapack-3.7.0/LAPACKE/include -G -g -pg -O0 -Xcompiler `pkg-config --cflags opencv --libs` -Xcompiler \"-Wl,-rpath,/usr/lib64/openmpi/lib\" --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


