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

void histogram_private(const cv::Mat& input, cv::Mat& output);
int histo_gold(const cv::Mat& input, int height, int width, unsigned int *gold_bins);

__global__ void histogram_private_kernel(unsigned char* input_1d, unsigned char* output, int width, int height, int inputWidthStep, int outputWidthStep, unsigned int *bins_d) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int bdimx = blockDim.x;
  
    __syncthreads();
    
    //2D Index of current thread
    const int xIndex = bx * bdimx + tx;
	
   // unsigned int i = bx*bdimx + tx;
    if(xIndex<width*height){
        uint32_t value = input_2d[xIndex];
        atomicAdd(&bins_d[bx*256 + value], 1);
    }
    if(bx > 0){
        __syncthreads();
        for(unsigned int bin_index = tx; bin_index < 256; bin_index+=bdimx){
            unsigned int bin_amount = bins_d[bx*256 + bin_index];
            if(bin_amount > 0){
                atomicAdd(&bins_d[bin_index], bin_amount);
            }
        }
    }

}
 
 ////////////////////////////////////////////////////////////////////////////////
void histogram_private(const cv::Mat& input, cv::Mat& output) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *input_d, *output_d, *input_1dd;
	unsigned char *input_1dh;
	unsigned int *bins_d;
	unsigned int *kernel_bins;
	unsigned int *gold_bins;
	
	// Specify a reasonable block size
	const dim3 block(1024, 1, 1);

	// Calculate grid size to cover the whole image
	const dim3 grid(((input.cols + block.x - 1)/block.x)*((input.rows + block.y - 1)/block.y), 1, 1);
	
	gold_bins = (unsigned int *)malloc(256*sizeof(unsigned int));
	kernel_bins = (unsigned int *)malloc(256*sizeof(unsigned int));
	
	input_1dh = (unsigned char *)malloc(colorBytes*sizeof(unsigned char));

	// Allocate device memory
	//cudaMalloc<unsigned char>(&input_d,colorBytes);
	cudaMalloc<unsigned char>(&output_d,grayBytes);
	cudaMalloc<unsigned char>(&input_1dd,colorBytes);
	
	int gridSize = ((input.cols + block.x - 1)/block.x)*((input.rows + block.y - 1)/block.y);
	
	cudaMalloc<unsigned int>(&bins_d,gridSize*sizeof(unsigned int));
	
	
	//SAFE_CALL(cudaMalloc(void**)(&bins_d,256*sizeof(int),"CUDA Malloc Failed");
	cudaMemset(bins_d, 0, 256*sizeof(int));
	
	cudaMemset(input_1dd, 0, colorBytes*input.cols*sizeof(unsigned char));

	// Copy data from OpenCV input image to device memory
	//cudaMemcpy(input_d,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
	
//	unsigned int m = 0;

  //      for (int j = 0; j < input.rows; ++j)
    //    {
        //	for (int i = 0; i < input.cols; ++i)
	///        {
	   // 		input_1d[m] = input.at<uchar>(j, i);
	    //		m++;
		//}
         //}



	unsigned int m = 0;

        for (int j = 0; j < input.rows; ++j)
        {
        	for (int i = 0; i < input.cols; ++i)
	        {
	    		input_1dh[m] = input.at<uchar>(j, i);
	    		m++;
		}
         }
	
	//// Copy data from OpenCV input image to device memory
	//cudaMemcpy(input_1d,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(input_1dd,input_1dh,colorBytes,cudaMemcpyHostToDevice);
	
	//printf("%d ", (input.cols + block.x - 1)/block.x);
	//printf("%d ", (input.rows + block.y - 1)/block.y);


	cudaEventRecord(start);
	// Launch the color conversion kernel
	histogram_private_kernel<<<grid,block>>>(input_1dd,output_d,input.cols,input.rows,input.step,output.step, bins_d);
	
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

    
	
	cudaMemcpy(kernel_bins, bins_d, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//Synchronie host and device to ensure that transfer is finished
	cudaDeviceSynchronize();
	
	
	//for(int i = 0; i < 256; i++){
	//	printf(" K: %u", kernel_bins[i]);
	//	printf(" G: %u", gold_bins[i]);
	//}
	
	cout << "Time taken by program is : "<< fixed << setprecision(10) << milliseconds;
    	cout << " milliseconds " << endl;
    	
    	histo_gold(input, input.rows, input.cols, gold_bins);
    	cout << "\n";
    	
    	for(int i = 0; i < 256; i++){
    		printf(" K: ");
		printf("%u", kernel_bins[i]);
		printf(" G: ");
		printf("%u", gold_bins[i]);
	}
    	
	int passed=1;
        for (int i=0; i < 256; i++){
        	if (gold_bins[i] != kernel_bins[i]){
	       	       passed = 0;
	       	       break;
	        }
         }
         (passed) ? printf("\n    Test PASSED\n") : printf("\n    Test FAILED\n");
	
	free(input_1dh);
    	cudaFree(input_1dd);

	// Synchronize to check for any kernel launch errors
	cudaDeviceSynchronize();

	// Copy back data from destination device meory to OpenCV output image
	cudaMemcpy(output.ptr(),output_d,grayBytes,cudaMemcpyDeviceToHost);

	free(kernel_bins);
	free(gold_bins);
	// Free the device memory
	//cudaFree(input_d);
	cudaFree(output_d);
}



int histo_gold(const cv::Mat& input, int height, int width, unsigned int *gold_bins)
{

    // Zero out all the bins
    memset(gold_bins, 0, 256*sizeof(unsigned int));

    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            unsigned int value = input.at<uchar>(j, i);

            gold_bins[value]++;
            //unsigned int *p = (unsigned int*)gold_bins;
            //++p[value];
        }
    }

    return 0;
}


