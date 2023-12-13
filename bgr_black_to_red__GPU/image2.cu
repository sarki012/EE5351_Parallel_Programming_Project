/* Vector Addition: C = A + B.
 * Device code.
 */

 #include <stdio.h>
 #include "image.h"
 
 #include <cuda.h>
 
 
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

void convert_to_gray(const cv::Mat& input, cv::Mat& output);
/**
 * @brief      CUDA safe call.
 *
 * @param[in]  err          The error
 * @param[in]  msg          The message
 * @param[in]  file_name    The file name
 * @param[in]  line_number  The line number
 */
 
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number) {
	if(err!=cudaSuccess) {
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

/// Safe call macro.
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

/**
 * @brief      BGR to Gray Kernel
 *
 *             This is a simple image processing kernel that converts color
 *             images to black and white by iterating over the individual
 *             pixels.
 *
 * @param      input           The input
 * @param      output          The output
 * @param[in]  width           The width
 * @param[in]  height          The height
 * @param[in]  colorWidthStep  The color width step
 * @param[in]  grayWidthStep   The gray width step
 */
__global__ void bgr_to_gray_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep) {
	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
		
		//Location of gray pixel in output
		const int gray_tid  = yIndex * grayWidthStep + (3*xIndex);

		const unsigned char blue	= input[color_tid];
		const unsigned char green	= input[color_tid + 1];
		const unsigned char red		= input[color_tid + 2];
		
		if(blue < 50 && green < 50 && red < 50){
			output[gray_tid + 2] = 255;
		}
		else{
			output[gray_tid] = input[color_tid];
			output[gray_tid + 1] = input[color_tid + 1];
		}
		/*
		else if(green < 50){
			output[gray_tid + 1] = 75;
		}
		else if(red < 50){
			output[gray_tid + 2] = 75;
		}
	*/

		//const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		//output[gray_tid] = static_cast<unsigned char>(gray);
	}
}
 
 
 ////////////////////////////////////////////////////////////////////////////////
void convert_to_gray(const cv::Mat& input, cv::Mat& output) {
	// Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,colorBytes),"CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,grayBytes),"CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	// Launch the color conversion kernel
	bgr_to_gray_kernel<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step,output.step);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
}
