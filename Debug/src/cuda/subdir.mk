################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cuda/cuDistance.cu \
../src/cuda/cuRoutines.cu \
../src/cuda/cuUtils.cu \
../src/cuda/cumodel.cu 

CU_DEPS += \
./src/cuda/cuDistance.d \
./src/cuda/cuRoutines.d \
./src/cuda/cuUtils.d \
./src/cuda/cumodel.d 

OBJS += \
./src/cuda/cuDistance.o \
./src/cuda/cuRoutines.o \
./src/cuda/cuUtils.o \
./src/cuda/cumodel.o 


# Each subdirectory must supply rules for building sources it contributes
src/cuda/%.o: ../src/cuda/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/src/kernels/4.10.9-200.fc25.x86_64/include -I/usr/include/openmpi-x86_64 -I/home/smith/Downloads/delete/lapack-3.7.0 -I/home/smith/Downloads/delete/lapack-3.7.0/LAPACKE/include -G -g -pg -O0 -Xcompiler `pkg-config --cflags opencv --libs` -Xcompiler \"-Wl,-rpath,/usr/lib64/openmpi/lib\" -gencode arch=compute_35,code=sm_35  -odir "src/cuda" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc --compiler-bindir /opt/rh/devtoolset-3/root/usr/bin/gcc -I/usr/src/kernels/4.10.9-200.fc25.x86_64/include -I/usr/include/openmpi-x86_64 -I/home/smith/Downloads/delete/lapack-3.7.0 -I/home/smith/Downloads/delete/lapack-3.7.0/LAPACKE/include -G -g -pg -O0 -Xcompiler `pkg-config --cflags opencv --libs` -Xcompiler \"-Wl,-rpath,/usr/lib64/openmpi/lib\" --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


