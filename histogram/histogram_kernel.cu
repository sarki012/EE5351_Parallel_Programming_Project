/* Vector Addition: C = A + B.
 * Device code.
 */

 #include <stdio.h>
 #include "histogram.h"
 
 #include <cuda.h>
 
 
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

void convert_to_gray(const cv::Mat& input, cv::Mat& output);

__global__ void bgr_to_gray_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep, float *bins_d) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int bdimx = blockDim.x;
    
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int bdimy = blockDim.y;
    
    //2D Index of current thread
    const int xIndex = bx * bdimx + tx;
    const int yIndex = by * bdimy + ty;
	
    const int out_tid  = yIndex * grayWidthStep + (3*xIndex);
    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
    if((xIndex<width) && (yIndex<height)){

        int value = input[color_tid];		//Blue first
        atomicAdd(&bins_d[value], 0.01);
        if(bins_d[value] > 255){
            bins_d[value] = 255;
        }
    }
    output[out_tid] = input[color_tid];
}
 
 ////////////////////////////////////////////////////////////////////////////////
void convert_to_gray(const cv::Mat& input, cv::Mat& output) {
	// Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *input_d, *output_d;
	float *bins_d;
	float *kernel_bins;
	
	kernel_bins = (float *)malloc(256*sizeof(float));

	// Allocate device memory
	cudaMalloc<unsigned char>(&input_d,colorBytes);
	cudaMalloc<unsigned char>(&output_d,grayBytes);
	
	cudaMalloc<float>(&bins_d,256*sizeof(float));
	//SAFE_CALL(cudaMalloc(void**)(&bins_d,256*sizeof(int),"CUDA Malloc Failed");
	cudaMemset(bins_d, 0, 256*sizeof(float));

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(input_d,input.ptr(),colorBytes,cudaMemcpyHostToDevice);

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	// Launch the color conversion kernel
	bgr_to_gray_kernel<<<grid,block>>>(input_d,output_d,input.cols,input.rows,input.step,output.step, bins_d);
	
	cudaMemcpy(kernel_bins, bins_d, 256*sizeof(float), cudaMemcpyDeviceToHost);
	//Synchronie host and device to ensure that transfer is finished
	cudaDeviceSynchronize();
	
	
	for(int i = 0; i < 255; i++){
		printf(" %.1f", kernel_bins[i]);
	}
	
	
    	cudaFree(input_d);

	// Synchronize to check for any kernel launch errors
	cudaDeviceSynchronize();

	// Copy back data from destination device meory to OpenCV output image
	cudaMemcpy(output.ptr(),output_d,grayBytes,cudaMemcpyDeviceToHost);

	// Free the device memory
	cudaFree(input_d);
	cudaFree(output_d);
}
