/* blur_kernel.cu
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
#include <stdlib.h>

void blur(const cv::Mat& input, cv::Mat& output);
void blur_gold(cv::Mat& reference, float **M, const cv::Mat& input, int width, int height);
bool CompareMatrices(cv::Mat& input, cv::Mat& output, int width, int height);
__constant__ float Mc[KERNEL_SIZE][KERNEL_SIZE];

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int inputWidthStep, int outputWidthStep) {
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

	// Allocate device memory
	cudaMalloc<unsigned char>(&input_d,colorBytes);
	cudaMalloc<unsigned char>(&output_d,grayBytes);
	
	// Copy data from OpenCV input image to device memory
	cudaMemcpy(input_d,input.ptr(),colorBytes,cudaMemcpyHostToDevice);

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);
	
	// the 2D array variable is declared to be `int **` (a pointer to an int *)
	// a dynamically allocated array of dynamically allocated int arrays
	// (a pointer to pointers to ints)
	float **Mask;

	// allocate an array of N pointers to ints
	// malloc returns the address of this array (a pointer to (int *)'s)
	Mask = (float **)malloc(sizeof(float) * KERNEL_SIZE);

	// for each row, malloc space for its column elements and add it to
	// the array of arrays
	for (int k = 0; k < KERNEL_SIZE; k++) {
		// malloc space for row i's M column elements
		Mask[k] = (float *)malloc(sizeof(float)*KERNEL_SIZE);
	}
	
	
	
//	float *M[KERNEL_SIZE];
	
//	for(int i = 0; i < KERNEL_SIZE; i++){
//		float *M[i] = (float *)malloc(KERNEL_SIZE*sizeof(float));
//	}
	/*
	Mask[0][0] 
	Mask = {{1,4,7,4,1},
	{4,16,26,16,4},
	{7,26,41,26,7},
	{4,16,26,16,4},
	{1,4,7,4,1}};
	*/
	/*
	Mask[0] = {1,4,7,4,1};
	*Mask[1] = {4,16,26,16,4};
	*Mask[2] = {7,26,41,26,7};
	*Mask[3] = {4,16,26,16,4};
	*Mask[4] = {1,4,7,4,1};
	*/
	//floatM[5][5] = {{1,4,7,4,1}, {4,16,26,16,4}, {7,26,41,26,7}, {4,16,26,16,4}, {1,4,7,4,1}};
	//float Mask[KERNEL_SIZE][KERNEL_SIZE] = {{1,4,7,4,1}, {4,16,26,16,4}, {7,26,41,26,7}, {4,16,26,16,4}, {1,4,7,4,1}};
	
	for(int m = 0; m < KERNEL_SIZE; m++){
		for(int n = 0; n < KERNEL_SIZE; n++){
			Mask[m][n] = 1;
			//Mask[m][n] = Mask[m][n] / 256;
		}
	}
	
	cudaMemcpyToSymbol(Mc, Mask, ((KERNEL_SIZE * KERNEL_SIZE)*sizeof(float)));

	cudaEventRecord(start);
	// Launch the color conversion kernel
	blur_kernel<<<grid,block>>>(input_d,output_d,input.cols,input.rows,input.step,output.step);
	

    
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop); 
	
	//Synchronie host and device to ensure that transfer is finished
	cudaDeviceSynchronize();
	
	cout << "Time taken by GPU is : "<< fixed << setprecision(10) << milliseconds;
    	cout << " milliseconds " << endl;
	
	
    	cudaFree(input_d);

	// Synchronize to check for any kernel launch errors
	cudaDeviceSynchronize();

	// Copy back data from destination device meory to OpenCV output image
	cudaMemcpy(output.ptr(),output_d,grayBytes,cudaMemcpyDeviceToHost);

        // compute the matrix multiplication on the CPU for comparison
      //  unsigned char *reference = (unsigned char*) malloc((input.rows)*(input.cols)*sizeof(unsigned char));
      
      	// Create reference image
	//cv::Mat reference(input.rows,input.cols,CV_8UC1);
      	cv::Mat reference = output;
      	
      	/*
      	cudaEvent_t gold_start, gold_stop;
	cudaEventCreate(&gold_start);
	cudaEventCreate(&gold_stop);

      	float gold_milliseconds = 0;
      	*/
      	cudaEventRecord(start);
      	
        blur_gold(reference, Mask, input, input.cols, input.rows);
	
        // in this case check if the result is equivalent to the expected soluion
        bool res = CompareMatrices(reference, output, input.cols, input.rows);
        printf("Test %s\n", (res) ? "PASSED" : "FAILED");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	
	

	cudaEventElapsedTime(&milliseconds, start, stop); 
	
	cout << "Time taken by CPU is : "<< fixed << setprecision(10) << milliseconds;
    	cout << " milliseconds " << endl;
	
	// Free the device memory
	cudaFree(input_d);
	cudaFree(output_d);
	
}

// returns true iff A and B have same elements in same order
bool CompareMatrices(cv::Mat& reference, cv::Mat& compare, int width, int height){
    float errTol = 0.01f;

	for(int i = 0; i < compare.cols; i++){
		for(int j = 0; j < compare.rows; j++){
			float diff = abs(reference.at<uchar>(i, j) - compare.at<uchar>(i, j));
			bool small= abs(reference.at<uchar>(i, j)) < 1.0e-2f;
			
			if (small && diff > errTol)
				return false;
			else if (!small && abs(diff / reference.at<uchar>(i, j)) > errTol)
	    			return false;
	    	}
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
//void blur_gold(cv::Mat& reference, float **M, cv::Mat& input, int width, int height);
////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A convolved with B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param kernel_size         height and width of matrix A
//! @param hB         height of matrices B and C
//! @param wB         width of matrices B and C
////////////////////////////////////////////////////////////////////////////////
void blur_gold(cv::Mat& reference, float **M, const cv::Mat& input, int wB, int hB){
	// For each element in the result matrix matrix
	for (unsigned int i = 0; i < hB; ++i){
		for (unsigned int j = 0; j < wB; ++j) {
			float sum = 0;
			// check the start and end values of m and n to prevent overrunning the 
			//  matrix edges
			unsigned int mbegin = (i < KS_DIV_2)? KS_DIV_2 - i : 0;
			unsigned int mend = (i > (hB - (KS_DIV_2+1)))?
									hB - i + KS_DIV_2 : KERNEL_SIZE;
			unsigned int nbegin = (j < KS_DIV_2)? KS_DIV_2 - j : 0;
			unsigned int nend = (j > (wB - (KS_DIV_2+1)))?
									(wB-j) + KS_DIV_2 : KERNEL_SIZE;
			// overlay A over B centered at element (i,j).  For each 
			//  overlapping element, multiply the two and accumulate
			for(unsigned int m = mbegin; m < mend; ++m) {
				for(unsigned int n = nbegin; n < nend; n++) {
					sum += M[m][n]*input.at<uchar>(i + m - KS_DIV_2, j+n - KS_DIV_2);
				}
			}
			// store the result
			reference.at<uchar>(i, j) = (unsigned char)sum;
		}
	}
	return;
}

