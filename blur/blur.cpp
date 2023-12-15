
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

void blur(const cv::Mat& input, cv::Mat& output);
 
int main() {
	std::string imagePath = "snowboard.png";

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, IMREAD_GRAYSCALE);

	if(input.empty()){
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}

	// Create output image
	cv::Mat output(input.rows,input.cols,CV_8UC1);

	// Call the wrapper function
	blur(input,output);

	// Show the input and output
	cv::imshow("Input",input);
	cv::imshow("Output",output);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
