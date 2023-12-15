/* Image Histogram .cu
 * Device code.
 */

 #include <stdio.h>
 #include "blur.h"
 
 #include <cuda.h>
 using namespace std; 
 
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include <iomanip>
#include <math.h>

void blur(const cv::Mat& input, cv::Mat& output);

__constant__ float Mc[KERNEL_SIZE][KERNEL_SIZE];

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int inputWidthStep, int outputWidthStep, unsigned int *bins_d) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int bdimx = blockDim.x;
    
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int bdimy = blockDim.y;
    

    int o_col = bx*OUT_TILE_SIZE + tx;     //col_o
    int o_row = by*OUT_TILE_SIZE + ty;     //row_o

    //Loading input tile
    __shared__ float N_s[IN_TILE_SIZE][IN_TILE_SIZE];

    //Calculating output elements
    int i_col = o_col - KS_DIV_2;    //col_i
    int i_row = o_row - KS_DIV_2;    //row_i

    if(i_row >= 0 && i_row < height && i_col >= 0 && i_col < width){
        N_s[ty][tx] = input[i_row*width + i_col];
    }
    else{
        N_s[ty][tx] = 0.0f;      //Address out of bounds problem here ///////////////////////////////////////////////////////////////////////////
    }
    __syncthreads();

  
    //Turning off the threads at the edges of the block
    if(o_col >= 0 && o_col < width && o_row >=0 && o_row < height){     //col_o & row_o
        if(tx >= 0 && tx < OUT_TILE_SIZE && ty >= 0 && ty < OUT_TILE_SIZE){     //row_i & col_i
            float Pvalue = 0.0f;
            for(int fRow = 0; fRow < KERNEL_SIZE; fRow++){
                for(int fCol = 0; fCol < KERNEL_SIZE; fCol++){
                    Pvalue += Mc[fRow][fCol]*N_s[ty + fRow][tx + fCol];   
                }   
            }
            output[o_row*width + o_col] = Pvalue;     //P.width or N.width? 
        }      
    }
}
 
 ////////////////////////////////////////////////////////////////////////////////
void blur(const cv::Mat& input, cv::Mat& output) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *input_d, *output_d;
	unsigned int *bins_d;
	unsigned int *kernel_bins;
	
	kernel_bins = (unsigned int *)malloc(256*sizeof(unsigned int));

	// Allocate device memory
	cudaMalloc<unsigned char>(&input_d,colorBytes);
	cudaMalloc<unsigned char>(&output_d,grayBytes);
	
	cudaMalloc<unsigned int>(&bins_d,256*sizeof(unsigned int));
	//SAFE_CALL(cudaMalloc(void**)(&bins_d,256*sizeof(int),"CUDA Malloc Failed");
	cudaMemset(bins_d, 0, 256*sizeof(unsigned int));

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(input_d,input.ptr(),colorBytes,cudaMemcpyHostToDevice);

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);
	
	float M[5][5] = {{1,1,1,1,1}, {1,1,1,1,1}, {1,1,1,1,1}, {1,1,1,1,1}, {1,1,1,1,1}};
	
	cudaMemcpyToSymbol(Mc, M, ((KERNEL_SIZE * KERNEL_SIZE)*sizeof(float)));

	cudaEventRecord(start);
	// Launch the color conversion kernel
	blur_kernel<<<grid,block>>>(input_d,output_d,input.cols,input.rows,input.step,output.step, bins_d);
	
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop); 
	
	cudaMemcpy(kernel_bins, bins_d, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//Synchronie host and device to ensure that transfer is finished
	cudaDeviceSynchronize();
	
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
