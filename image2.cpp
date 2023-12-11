
#include <iostream> 
#include <opencv2/opencv.hpp> 
using namespace cv; 
using namespace std; 
#include <cstdio>
#include <stdio.h>

int main()
{
   // Mat image = imread("snowboard.png", IMREAD_GRAYSCALE); 
   Mat image2 = imread("snowboard.png"); 
   Mat image1 = image2;
   Mat channel[3];
  // vector<Mat> three_channels = split(image);
   //split(image, channel);
  // vector<Mat> channel = split(image);
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
    


          //  Vec3b & color = image.at<Vec3b>(i, j);
        //    channel[0].at<uchar>(i, j) = 0;
        //    image.at<Vec3b>(i, j)[0] == 0;
           // channel[0] = Mat::zeros(image.rows, image.cols, CV_8UC1);//Set blue channel to 0
          //  image.at<Vec3b>(i, j) = Vec3b(0,0,255);
            // get pixel
      /*
            

            // ... do something to the color ....
            color[0] = 255;
            color[1] = 255;
            color[2] = 255;
*/
            // set pixel
        //    image.at<Vec3b>(Point(i, j)) = color;
            //if you copy value
          
          
          //  Vec3b color = image.at<Vec3b>(Point(i, j));
            // You can now access the pixel value with cv::Vec3b
            /*
            if(image.at<cv::Vec3b>(i,j)[0] > 200){
                image.at<cv::Vec3b>(i,j)[0] == 0;
            } 
            if(image.at<cv::Vec3b>(i,j)[1] > 200){
                image.at<cv::Vec3b>(i,j)[1] == 0;
            } 
            if(image.at<cv::Vec3b>(i,j)[2] > 200){
                image.at<cv::Vec3b>(i,j)[2] == 0;
            } 
            */
    //    }
    //}
      //  std::cout << image.at<cv::Vec3b>(i,j)[0] << " " << image.at<cv::Vec3b>(i,j)[1] << " " << image.at<cv::Vec3b>(i,j)[2] << std::endl;
    // Show Image inside a window with 
    // the name provided 
    imshow("Esark is a boss", image); 
  
    // Wait for any keystroke 
    waitKey(0); 
    return 0; 
/*
    cv::Mat image1,image2;

    image1=cv::imread("Snowboard.jpg");
    //reads the input image
    cv::namedWindow("SB",cv::WINDOW_AUTOSIZE);

    cv::imshow("SB",image1);

    cv::waitKey(0);

    cv::destroyWindow("SB");

    return 0;
    */
}

/*
// C++ program for the above approach 
#include <iostream> 
#include <opencv2/opencv.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace cv; 
using namespace std; 
#include <cstdio>
  
// Driver code 
int main(int argc, char** argv) 
{ 
    
    for(int i = 0; i < 1000; i++){
        printf("Esark is the Man!!!");
        for(int j = 0; j < 10000; j++);
    }
        // Read the image file as 
    // imread("default.jpg"); 
    Mat image = imread("Snowboard.jpg", IMREAD_GRAYSCALE); 
  
    // Error Handling 
    if (image.empty()) { 
        cout << "Image File "
             << "Not Found" << endl; 
  
        // wait for any key press 
        cin.get(); 
        return -1; 
    } 
  
    // Show Image inside a window with 
    // the name provided 
    imshow("Window Name", image); 
  
    // Wait for any keystroke 
    waitKey(0); 
    return 0; 
    
}
*/