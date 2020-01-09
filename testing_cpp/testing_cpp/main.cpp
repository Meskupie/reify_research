//
//  main.cpp
//  testing_cpp
//
//  Created by Michael on 2020-01-09.
//  Copyright © 2020 Michael. All rights reserved.
//

#include <iostream>


int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World !\n";
    return 0;
}

//#include <opencv4/opencv2/opencv.hpp>
//#using namespace cv;
//int main(int argc, char** argv)
//{
//
//    VideoCapture cap;
//    // open the default camera, use something different from 0 otherwise;
//    // Check VideoCapture documentation.
//    if(!cap.open(0))
//        return 0;
//    for(;;)
//    {
//          Mat frame;
//          cap >> frame;
//          if( frame.empty() ) break; // end of video stream
//          imshow("this is you, smile! :)", frame);
//          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
//    }
//    // the camera will be closed automatically upon exit
//    // cap.close();
//    return 0;
//}
