
#include <iostream> 
#include <opencv2/opencv.hpp> 
using namespace cv; 
using namespace std; 
#include <cstdio>
#include <stdio.h>


void VectorAddOnDevice(float* A, float* B, float* C);

int main()
{
   // Mat image = imread("snowboard.png", IMREAD_GRAYSCALE); 
   Mat image2 = imread("snowboard.png"); 
   Mat image1 = image2;
   Mat channel[3];
    // Error Handling 
    if (image1.empty()) { 
        cout << "Image File "
             << "Not Found" << endl; 
  
        // wait for any key press 
        cin.get(); 
        return -1; 
    } 
    int up_width = 1200;
    int up_height = 800;
    Mat image;
    //resize up
    resize(image1, image, Size(up_width, up_height), INTER_LINEAR);
    
  //  for(int i=0; i<image.rows; i++){
    //    for(int j=0; j<image.cols; j++){
      for(int i=0; i<1200; i++){
        for(int j=0; j<800; j++){
            // get pixel
            Vec3b & color = image.at<Vec3b>(j, i);
            if(color[0] < 50 && color[1] < 50 && color[2] < 50){
                color[0] = 0;
                color[1] = 0;
                color[2] = 255;
                cout << "Pixel >200 :" << i << "," << j << endl;
            }
            else if(color[0] > 250 && color[1] > 250 && color[2] > 250){
                color.val[0] = 0;
                color.val[1] = 255;
                color.val[2] = 0;
            }  
            //image.at<Vec3b>(Point(i, j)) = color;
        }
    }  
    float A_h[5] = {1, 2, 3, 4, 5};
    float B_h[5] = {5, 4, 3, 2, 2};
    float C_h[5] = {0, 0, 0, 0, 0};

    VectorAddOnDevice(A_h, B_h, C_h);
    for(int i = 0; i < 5; i++){
        cout << "C_h: " << C_h[i] << endl;
    }
    // Show Image inside a window with 
    // the name provided 
    imshow("Esark is a boss", image); 
  
    // Wait for any keystroke 
    waitKey(0); 
    return 0; 
}
