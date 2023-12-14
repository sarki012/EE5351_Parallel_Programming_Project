/* Histogram Kernel with Privatization
 * Device code.
 */

 #include <stdio.h>
 #include "histogram.h"
 
 #include <cuda.h>
 using namespace std; 
 
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include <iomanip>

void convert_to_gray(const cv::Mat& input, cv::Mat& output);

__global__ void bgr_to_gray_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep, unsigned int *bins_d) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int bdimx = blockDim.x;
    
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int bdimy = blockDim.y;
    
    __syncthreads();
    
    //2D Index of current thread
    const int xIndex = bx * bdimx + tx;
    const int yIndex = by * bdimy + ty;
	
    const int out_tid  = yIndex * grayWidthStep + (3*xIndex);
    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
    

   // unsigned int i = bx*bdimx + tx;
    if((xIndex<width) && (yIndex<height)){
        uint32_t value = input[color_tid];
        atomicAdd(&bins_d[(bx + gridDim.x*by)*1024 + value], 1);
    }
    if(bx > 0){
        __syncthreads();
        for(unsigned int bin_index = (tx + bdimx*by); bin_index < 1024; bin_index+=bdimx){
            unsigned int bin_amount = bins_d[(bx + gridDim.x*by)*1024 + bin_index];
            if(bin_amount > 0){
                atomicAdd(&bins_d[bin_index], bin_amount);
            }
        }
    }

}
 
 ////////////////////////////////////////////////////////////////////////////////
void convert_to_gray(const cv::Mat& input, cv::Mat& output) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *input_d, *output_d;
	unsigned int *bins_d;
	unsigned int *kernel_bins;
	
	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);
	
	kernel_bins = (unsigned int *)malloc(256*sizeof(unsigned int));

	// Allocate device memory
	cudaMalloc<unsigned char>(&input_d,colorBytes);
	cudaMalloc<unsigned char>(&output_d,grayBytes);
	
	int gridSize = ((input.cols + block.x - 1)/block.x)*((input.rows + block.y - 1)/block.y);
	
	cudaMalloc<unsigned int>(&bins_d,gridSize*sizeof(unsigned int));
	
	
	//SAFE_CALL(cudaMalloc(void**)(&bins_d,256*sizeof(int),"CUDA Malloc Failed");
	cudaMemset(bins_d, 0, 256*sizeof(int));

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(input_d,input.ptr(),colorBytes,cudaMemcpyHostToDevice);


	
	//printf("%d ", (input.cols + block.x - 1)/block.x);
	//printf("%d ", (input.rows + block.y - 1)/block.y);


	cudaEventRecord(start);
	// Launch the color conversion kernel
	bgr_to_gray_kernel<<<grid,block>>>(input_d,output_d,input.cols,input.rows,input.step,output.step, bins_d);
	
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

    
	
	cudaMemcpy(kernel_bins, bins_d, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//Synchronie host and device to ensure that transfer is finished
	cudaDeviceSynchronize();
	
	
	for(int i = 0; i < 255; i++){
		printf(" %u", kernel_bins[i]);
	}
	
	cout << "Time taken by program is : "<< fixed << setprecision(10) << milliseconds;
    	cout << " milliseconds " << endl;
	
	
    	cudaFree(input_d);

	// Synchronize to check for any kernel launch errors
	cudaDeviceSynchronize();

	// Copy back data from destination device meory to OpenCV output image
	cudaMemcpy(output.ptr(),output_d,grayBytes,cudaMemcpyDeviceToHost);

	// Free the device memory
	cudaFree(input_d);
	cudaFree(output_d);
}
