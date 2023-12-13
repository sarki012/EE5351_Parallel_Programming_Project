/* Vector Addition: C = A + B.
 * Device code.
 */

 #include <stdio.h>
 #include "image.h"
 
 #include <cuda.h>
 
 
 void VectorAddOnDevice(float* A, float* B, float* C);
 
 // 3. launch kernel to compute C_d = A_d + B_d
 
 
 // Vector addition kernel thread specification
 __global__ void VectorAddKernel(float* A, float* B, float* C)
 {
   // INSERT CODE to add the two vectors
     // Get our global thread ID
     int id = blockIdx.x*blockDim.x+threadIdx.x;
  
     // Make sure we do not go out of bounds
     if (id < VSIZE)
         C[id] = A[id] + B[id];
 
 }
 
 
 ////////////////////////////////////////////////////////////////////////////////
 //! Run a simple test for CUDA
 ////////////////////////////////////////////////////////////////////////////////
 void VectorAddOnDevice(float* A_h, float* B_h, float* C_h)
 {
     // Vectors for the program
     float *A_d, *B_d, *C_d;
     //Interface host call to the device kernel code and invoke the kernel
 
     // steps:
     // 1. declare and allocate device vectors A_d, B_d and C_d with length same as input vectors
     // ALLOCATE DEVICE MEMORY
     cudaMalloc((void**)&A_d,VSIZE*sizeof(float));
     cudaMalloc((void**)&B_d,VSIZE*sizeof(float));
     cudaMalloc((void**)&C_d,VSIZE*sizeof(float));
 
     
 
     // 2. copy A_h to A_d, B_h to B_d
     // Copy host vectors to device
     cudaMemcpy( A_d, A_h, VSIZE*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy( B_d, B_h, VSIZE*sizeof(float), cudaMemcpyHostToDevice);
 
     // SET GRID AND BLOCK DIMENSIONS
     dim3 blockDim(VSIZE,1,1);
     dim3 gridDim(1, 1, 1);
 
     // LAUNCH THE KERNEL!
     VectorAddKernel<<<gridDim,blockDim>>>(A_d,B_d,C_d);
 
     // 4. copy C_d back to host vector C
     cudaMemcpy(C_h,C_d, VSIZE*sizeof(float), cudaMemcpyDeviceToHost);
 
     // 5. synchronize host and device to ensure that transfer is finished
     cudaDeviceSynchronize();
 
     // 6. free device vectors A_d, B_d, C_d
     cudaFree(A_d);
     cudaFree(B_d);
     cudaFree(C_d);
 
 }
