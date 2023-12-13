
#include <iostream> 
//#include <opencv4/opencv.hpp> 
#include <opencv4/opencv2/opencv.hpp> 
using namespace cv; 
using namespace std; 
#include <cstdio>
#include <stdio.h>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/core.hpp>
#include <cuda_runtime.h>

void convert_to_gray(const cv::Mat& input, cv::Mat& output);
 
int main() {
	clock_t start, end;
	start = clock();
	//time_t start, end;
	//time(&start);
        // unsync the I/O of C and C++. 
   	 ///ios_base::sync_with_stdio(false); 
 
	
	std::string imagePath = "snowboard.png";

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath);

	if(input.empty()){
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}

	// Create output image
	cv::Mat output(input.rows,input.cols,CV_8UC3);

	// Call the wrapper function
	convert_to_gray(input,output);
	end = clock();
	double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    	cout << "Time taken by program is : " << fixed 
         << time_taken << setprecision(10);
    	cout << " sec " << endl;
	/*
	time(&end); 
	    // Calculating total time taken by the program. 
    	double time_taken = double(end - start); 
   	 cout << "Time taken by program is : " << fixed 
        << time_taken << setprecision(5); 
   	 cout << " sec " << endl; 
   	 */
	// Show the input and output
	cv::imshow("Input",input);
	cv::imshow("Output",output);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
