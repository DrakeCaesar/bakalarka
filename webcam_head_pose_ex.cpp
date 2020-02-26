// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/
#define CURL_STATICLIB
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <curl/curl.h>
#include <vector>
#include <iostream>
#include <thread>
#include <opencv2\imgproc.hpp>
#pragma comment (lib, "Normaliz.lib")
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Wldap32.lib")
#pragma comment (lib, "Crypt32.lib")
#pragma comment (lib, "advapi32.lib")


using namespace dlib;
using namespace cv;

using namespace std;




bool stop = false;
const std::string videoStreamAddress = "http://192.168.1.20:8080/video";
int capture(cv::Mat * image)
{
    cv::VideoCapture cap;
    if (!cap.open(videoStreamAddress)) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    while (true) {
        cap >> *image;
        if ((*image).empty())
            cap.set(CAP_PROP_POS_FRAMES, 0);
    }

}
int min(int x, int y) {
    if (x < y)
        return x;
    return y;
}
int max(int x, int y) {
    if (x > y)
        return x;
    return y;
}
void detectface(cv::Mat* image)
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
    image_window win;
    while (!(*image).empty() && !win.is_closed()) {

        cv::Mat dst;
        cv::resize(* image, dst, cv::Size((*image).cols/4,(*image).rows/4));
        cv::flip(dst, dst, +1);

        cv_image<bgr_pixel> cimg(dst);

        //rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
        //cv::resize(image, image, cv::Size(), 0.5, 0.5);

        std::vector<dlib::rectangle> faces = detector(cimg);
        std::vector<full_object_detection> shapes;
        cv::Rect rect[2];
        for (unsigned long i = 0; i < faces.size(); ++i)
        {
            shapes.push_back(pose_model(cimg, faces[i]));
            for (int j = 0; j <= 1; j++) {
                int x1 = INT_MAX, y1 = INT_MAX;
                int x2 = 0, y2 = 0;
                for (int k = 0; k < 6; k++) {
                    x1 = min(x1, shapes[i].part(36 + j*6 + k).x());
                    x2 = max(x2, shapes[i].part(36 + j*6 + k).x());
                    y1 = min(y1, shapes[i].part(36 + j*6 + k).y());
                    y2 = max(y2, shapes[i].part(36 + j*6 + k).y());
                }
                rect[j] = cv::Rect(x1, y1 -10, x2 - x1, y2 - y1 + 20);
                cv::rectangle(dst, rect[j], cv::Scalar(0, 255, 0));
            }
        }
        //win.clear_overlay();
        win.set_image(cimg);
        if (faces.size() > 0)
        Sleep(5000);
        //win.add_overlay(render_face_detections(shapes));
    }

}


void detectfaceresize(cv::Mat* image)
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
    image_window win;
    while (!(*image).empty() && !win.is_closed()) {

        cv::Mat dst;
        cv::flip(*image, dst, +1);

        cv_image<bgr_pixel> cimg(dst);

        //rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
        //cv::resize(image, image, cv::Size(), 0.5, 0.5);

        std::vector<dlib::rectangle> faces = detector(cimg);
        std::vector<full_object_detection> shapes;

        for (unsigned long i = 0; i < faces.size(); ++i)
            shapes.push_back(pose_model(cimg, faces[i]));


        

        //cv::resize(dst, dst, cv::Size((*image).cols / 4, (*image).rows / 4));
        win.clear_overlay();
        win.set_image(cimg);
        win.add_overlay(render_face_detections(shapes));

    }

}
void justrender(cv::Mat* image)
{
    image_window win;
    while (!(*image).empty() && !win.is_closed()) {

        cv::Mat dst;
        cv::resize(*image, dst, cv::Size((*image).cols / 4, (*image).rows / 4));
        cv::flip(dst, dst, +1);
        cv_image<bgr_pixel> cimg(dst);
        win.set_image(cimg);
    }

}

int main()
{   
    cv::Mat image;
    std::thread capture(capture,&image);
    Sleep(2000);
    std::thread detectface(detectface, &image);
    //std::thread detectfaceresize(detectfaceresize, &image);

    //std::thread justrender(justrender, &image);


    while (true)
    {
        Sleep(1000);
    }
    //t1.join();
}

