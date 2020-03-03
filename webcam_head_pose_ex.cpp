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
#include <chrono>
#include< opencv2/features2d.hpp >
#pragma comment (lib, "Normaliz.lib")
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Wldap32.lib")
#pragma comment (lib, "Crypt32.lib")
#pragma comment (lib, "advapi32.lib")

using namespace dlib;
using namespace cv;
using namespace std;

int captureold(cv::Mat * image)
{
    const int targetFramerate = 32;
    const int targetFrameTime = 1000000 / targetFramerate;
    const std::string videoStreamAddress = "http://192.168.1.34:8080/video";
    cv::VideoCapture cap;
    if (!cap.open(videoStreamAddress)) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    while (true) {
        cap >> *image;
        if ((*image).empty())
            cap.set(CAP_PROP_POS_FRAMES, 0);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        int microsec = std::chrono::duration_cast<std::chrono::microseconds>  (end - begin).count();
        int sleep = 0;
        if (1000000 / microsec > targetFramerate) {
            sleep = targetFrameTime - microsec;
            std::this_thread::sleep_for(std::chrono::microseconds(sleep));
        }
        /*
        end = std::chrono::steady_clock::now();
        microsec = std::chrono::duration_cast<std::chrono::microseconds>  (end - begin).count();
        std::cout << 1000000/microsec << " fps; sleep "<< sleep/1000 << std::endl;
        */
        begin = std::chrono::steady_clock::now();
    }
}
int capture(cv::Mat* image)
{
    const unsigned int targetFramerate = 32;
    const unsigned int second = 1000000;
    const unsigned int targetFrameTime = second / targetFramerate;
    const std::string videoStreamAddress = "http://192.168.1.20:8080/video";
    cv::VideoCapture cap;
    if (!cap.open(videoStreamAddress)) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    while (true) {
        cv::Mat temp;
        cap >> temp;
        cv::flip(temp, *image, +1);
        //if ((*image).empty())
        //    cap.set(CAP_PROP_POS_FRAMES, 0);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        unsigned int microsec = std::chrono::duration_cast<std::chrono::microseconds>  (end - begin).count();
        if (second / microsec > targetFramerate)
            std::this_thread::sleep_for(std::chrono::microseconds(targetFrameTime - microsec));
        begin = std::chrono::steady_clock::now();
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

void detectface(cv::Mat* image, cv::Rect rect[])
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
    image_window win;
    while (!(*image).empty() && !win.is_closed()) {
        cv::Mat dst;
        cv::resize(* image, dst, cv::Size((*image).cols/4,(*image).rows/4));
        //cv::flip(dst, dst, +1);
        cv_image<bgr_pixel> cimg(dst);
        std::vector<dlib::rectangle> faces = detector(cimg);
        std::vector<full_object_detection> shapes;
        //cv::Rect rect[2];
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
                rect[j] = cv::Rect(x1, y1 - .5*(y2 - y1), x2 - x1, 2*(y2 - y1));
                
                cv::rectangle(dst, rect[j], cv::Scalar(0, 255, 0));
            }
        }
        win.set_image(cimg);
        //cout << rect[0];
    }
}
void normalize(cv::Rect* rect, int x, int y) {
    if ((*rect).x < 0)
        (*rect).x = 0;
    else if ((*rect).x + (*rect).width > x) {
        (*rect).x = x;
        (*rect).width = rect->height = 0;
    }
    if ((*rect).y < 0)
        (*rect).y = 0;
    else if ((*rect).y + (*rect).width > y) {
        (*rect).y = y;
        (*rect).width = rect->height = 0;
    }
}


void detectEye(cv::Mat * image, cv::Rect * rect){
    image_window win;
    int factor = 4;
    cv::Rect scaledRect;
    image_window win1;
    //namedWindow("My Window", 1);
    int sliderPos = 150;
    //createTrackbar("threshold", "My Window", &sliderPos, 255);
    while (true) {
    
       // Sleep(10);
        if ((*rect).width == 0)
            continue;
        scaledRect.x = (*rect).x * factor;
        scaledRect.y = (*rect).y * factor;
        scaledRect.width = (*rect).width * factor;
        scaledRect.height = (*rect).height * factor;
        normalize(&scaledRect, (*image).cols, (*image).rows);
        cv::Mat croppedImage = (*image)(scaledRect);
        cv_image<bgr_pixel> cimg(croppedImage);
        win.set_image(cimg);
        //imshow("My Window", croppedImage);

        //createTrackbar("threshold", "result", &sliderPos, 255, croppedImage);
        //croppedImage = imread("media/circles.png");
        cvtColor(croppedImage, croppedImage, COLOR_BGR2GRAY);

        std::vector<std::vector<Point> > contours;
        //Mat bimage = croppedImage;
        Mat bimage = croppedImage >= sliderPos;
        findContours(bimage, contours, RETR_LIST, CHAIN_APPROX_NONE);

        Mat cimage = Mat::zeros(bimage.size(), CV_8UC3);

        for (size_t i = 0; i < contours.size(); i++)
        {
            size_t count = contours[i].size();
            if (count < 6)
                continue;

            Mat pointsf;
            Mat(contours[i]).convertTo(pointsf, CV_32F);
            RotatedRect box = fitEllipse(pointsf);

            if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 30)
                continue;
            drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);

            ellipse(cimage, box, Scalar(0, 0, 255), 1, LINE_AA);
            ellipse(cimage, box.center, box.size * 0.5f, box.angle, 0, 360, Scalar(0, 255, 255), 1, LINE_AA);
            Point2f vtx[4];
            box.points(vtx);
            for (int j = 0; j < 4; j++)
                cv::line(cimage, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
        }



        cv_image<bgr_pixel> cimg1(cimage);
        win1.set_image(cimg1);





        //Sleep(10);
        /*
        Mat src = croppedImage;
        Mat src_gray;

        src = imread("media/circles.png", 1);

        //Ptr<FeatureDetector> blobsDetector = FeatureDetector::create("SimpleBlob");
            // Read image

        Mat im = imread("media/circles.png", IMREAD_GRAYSCALE);
        im = croppedImage.clone();

        // Setup SimpleBlobDetector parameters.
        SimpleBlobDetector::Params params;

        // Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 200;

        //Filter by Area.
        params.filterByArea = true;
        params.minArea = 150;

        // Filter by Circularity
        params.filterByCircularity = false;
        params.minCircularity = 0.1;

        // Filter by Convexity
        params.filterByConvexity = false;
        params.minConvexity = 0.87;

        // Filter by Inertia
        params.filterByInertia = false;
        params.minInertiaRatio = 0.01;

        // Storage for blobs
        std::vector<KeyPoint> keypoints;


        // Set up detector with params
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        // Detect blobs
        detector->detect(im, keypoints);


        // Draw detected blobs as red circles.
        // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
        // the size of the circle corresponds to the size of blob

        Mat im_with_keypoints;
        drawKeypoints(im, keypoints, im_with_keypoints, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //for (size_t i = 0; i < keypoints.size(); ++i)
        //    circle(im_with_keypoints, keypoints[i].pt, 4, Scalar(255, 0, 255), -1);

        // Show blobs
        cv_image<bgr_pixel> cimg1(im_with_keypoints);
        win1.set_image(cimg1);
       */
        /*
        Mat drawImage = src.clone();
        for (size_t i = 0; i < keypoints.size(); ++i)
            circle(drawImage, keypoints[i].pt, 4, Scalar(255, 0, 255), -1);
        cv_image<bgr_pixel> cimg1(drawImage);
        win1.set_image(cimg1);
        */





        /*/
        /// Convert it to gray
        cvtColor(src, src_gray, COLOR_BGR2GRAY);

        /// Reduce the noise so we avoid false circle detection
        GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

        std::vector<Vec3f> circles;

        /// Apply the Hough Transform to find the circles
        HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0);

        /// Draw the circles detected
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            // circle outline
            circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
        }

        /// Show your results
        //Sleep(1000);

        cv_image<bgr_pixel> cimg1(src);
        win1.set_image(cimg1);
        */
    }
}

int main()
{   
    cv::Mat image;
    cv::Rect rect[2];
    std::thread capture(capture,&image);
    Sleep(2000);
    std::thread detectface(detectface, &image ,rect);
    Sleep(1000);
    //std::thread detectEyeLeft(detectEye, &image, rect+0);
    //Sleep(1000);
    std::thread detectEyeRight(detectEye, &image, rect+1);
    Sleep(1000);
    while (true) {
        Sleep(1000);
    }
    //std::thread detectfaceresize(detectfaceresize, &image);
    //std::thread justrender(justrender, &image);

    
}

