#include "ellipseDetector.h"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <sys/ioctl.h>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

#define LINE_AA 16

#pragma comment (lib, "Normaliz.lib")
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Wldap32.lib")
#pragma comment (lib, "Crypt32.lib")
#pragma comment (lib, "advapi32.lib") 

using namespace dlib;
using namespace cv;
using namespace std;

std::mutex mtxCam;
std::atomic<bool> grabOn; //this is lock free
std::queue<Mat> buffer;

frontal_face_detector detector = get_frontal_face_detector();
shape_predictor pose_model;
image_window win;
image_window win1;

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

int detectface(cv::Mat * image, cv::Rect rect[])
{
    bool debug = false;
    cv::Mat resize, flip, grey, test;
    cv::cvtColor(* image, grey, COLOR_BGR2GRAY);
    cv::resize(grey, resize, cv::Size(grey.cols/8,grey.rows/8));
    //cv::flip(resize, flip, +1);
    dlib::array2d<unsigned char> cimg, cimg1;
    dlib::assign_image(cimg, dlib::cv_image<unsigned char>(resize));
    if (debug){
        cv::cvtColor(resize, test, COLOR_GRAY2BGR);
    }
    std::vector<dlib::rectangle> faces = detector(cimg);
    std::vector<full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
    {
        shapes.push_back(pose_model(cimg, faces[i]));
        for (int j = 0; j < 2; j++) {
            int x1 = INT_MAX, y1 = INT_MAX;
            int x2 = 0, y2 = 0;
            for (int k = 0; k < 6; k++) {
                x1 = min(x1, shapes[i].part(36 + j*6 + k).x());
                x2 = max(x2, shapes[i].part(36 + j*6 + k).x());
                y1 = min(y1, shapes[i].part(36 + j*6 + k).y());
                y2 = max(y2, shapes[i].part(36 + j*6 + k).y());
            }
            rect[j] = cv::Rect(x1, y1 - .5*(y2 - y1), x2 - x1, 2*(y2 - y1));
            if (debug) {
                cv::rectangle(test, rect[j], cv::Scalar(0, 255, 0));
            }
        }
        if (debug) {
            win.clear_overlay();
            win.add_overlay(render_face_detections(shapes));
        }
    }
    if (debug) {
        cv_image<bgr_pixel> cimg1(test);
        win.set_image(cimg1);
    }
    return faces.size();
}
void normalize(cv::Rect* rect, int x, int y) {
    if ((*rect).x < 0)
        (*rect).x = 0;
    else if ((*rect).x + (*rect).width >= x) {
        (*rect).x = x;
        (*rect).width = rect->height = 0;
    }
    if ((*rect).y < 0)
        (*rect).y = 0;
    else if ((*rect).y + (*rect).width >= y) {
        (*rect).y = y;
        (*rect).width = rect->height = 0;
    }
}

int sliderPos = 50;// 80;
int alpha = 40; /*< Simple contrast control Enter the alpha value [1.0-3.0]: 2.2 */
int betaa = 0;       /*< Simple brightness control Enter the beta value [0-100]: 50 */



void detectEye(cv::Mat * image, cv::Rect * rect){
    int factor1 = 8;
    int factor2 = 2;
    cv::Rect scaledRect;

    scaledRect.x = (*rect).x * factor1 * factor2;
    scaledRect.y = (*rect).y * factor1 * factor2;
    scaledRect.width = (*rect).width * factor1 * factor2;
    scaledRect.height = (*rect).height * factor1 * factor2;
    cv::Mat croppedImage = (*image).clone();
    cv::resize(croppedImage, croppedImage, cv::Size((*image).cols * factor2, (*image).rows * factor2));
    normalize(&scaledRect, (croppedImage).cols, (croppedImage).rows);

    croppedImage = croppedImage(scaledRect).clone();
    /*
    scaledRect.height = (*rect).height * factor;
    cv::Mat croppedImage = (*image)(scaledRect).clone();
    cv::resize(croppedImage, croppedImage, cv::Size((*image).cols * 4, (*image).rows * 4));
    normalize(&scaledRect, (croppedImage).cols, (croppedImage).rows);
    croppedImage = (croppedImage)(scaledRect).clone();
    */

    /*
    for (int y = 0; y < croppedImage.rows; y++) {
        for (int x = 0; x < croppedImage.cols; x++) {
            for (int c = 0; c < croppedImage.channels(); c++) {
                croppedImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(.1 * alpha * croppedImage.at<Vec3b>(y, x)[c] + beta - 100);
            }
        }
    }
    */
    /*
    cv_image<bgr_pixel> cimg(croppedImage);
    win.set_image(cimg);
    cvtColor(croppedImage, croppedImage, COLOR_BGR2GRAY);
    medianBlur(croppedImage, croppedImage, 5);
    equalizeHist(croppedImage, croppedImage);
    //Mat render = croppedImage.clone();
    inRange(croppedImage, 0, sliderPos, croppedImage);

    auto kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    //distanceTransform(croppedImage,croppedImage, DIST_L2, DIST_MASK_PRECISE);
    Mat render = croppedImage.clone();

    dilate(croppedImage, kernel, 2);
    erode(croppedImage, kernel, 3);
    //morphologyEx(croppedImage, croppedImage, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));

    //morphologyEx(croppedImage, croppedImage, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(25, 25)));


    cvtColor(render, render, COLOR_GRAY2BGR);

    cv_image<bgr_pixel> cimg1(render);
    win1.set_image(cimg1);
    */
}


cv::Mat thresh(cv::Mat * image, int rmin, int rmax) {
    cv::Mat I = (*image).clone();
    int scale = 1;
    rmin = rmin * scale;
    rmax = rmax * scale;
    cv::resize(I, I, cv::Size((*image).cols * scale, (*image).rows * scale));
    return I;
}
int random(int top) {
    return (std::rand() % top - top)*2;
}

int totalFrames = 0;

void GrabThread(VideoCapture *cap)
{
    Mat tmp;

    //To know how many memory blocks will be allocated to store frames in the queue.
    //Even if you grab N frames and create N x Mat in the queue
    //only few real memory blocks will be allocated
    //thanks to std::queue and cv::Mat memory recycling
    std::map<unsigned char*, int> matMemoryCounter;
    uchar * frameMemoryAddr;
    while (grabOn.load() == true) //this is lock free
    {
        //grab will wait for cam FPS
        //keep grab out of lock so that
        //idle time can be used by other threads
        *cap >> tmp; //this will wait for cam FPS

        if (tmp.empty()) continue;

        //get lock only when we have a frame
        mtxCam.lock();
        totalFrames++;
        //buffer.push(tmp) stores item by reference than avoid
        //this will create a new cv::Mat for each grab
        buffer.push(Mat(tmp.size(), tmp.type()));
        tmp.copyTo(buffer.back());
        frameMemoryAddr = buffer.front().data;
        mtxCam.unlock();
        //count how many times this memory block has been used
        matMemoryCounter[frameMemoryAddr]++;
    }
    std::cout << std::endl << "Number of Mat in memory: " << matMemoryCounter.size();
}

void ProcessFrame(Mat &src,int bufSize)
{
    Rect eyes[2];
    if(bufSize > 8 ) return;
    if(src.empty()) return;
    //cout << "test" << endl;
    //cv::resize(src, src, cv::Size((src).cols/8,(src).rows/8));
    //putText( const_cast<const decltype(src)>(src) , "PROC FRAME", Point(10, 10), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
    if (detectface(const_cast<const decltype(&src)>(&src),eyes)){
        detectEye(const_cast<const decltype(&src)>(&src),eyes);
    }
    //imshow("Image main", src);
}


void framerate(std::string msg)
{
    while(1)
    {
    int old = totalFrames;
    sleep(1u);
    int curr = totalFrames;
    std::cout << curr - old << " fps" << endl;
    }
    std::cout << "task1 says: " << msg;
}

int main() {
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
    Mat frame;
    VideoCapture cap;

    //const std::string videoStreamAddress = "http://192.168.1.31:8080/video";
    //const std::string videoStreamAddress = "http://192.168.1.100:8080/video";
    const std::string videoStreamAddress = "http://192.168.42.129:8080/video";

    cap.open(videoStreamAddress);
    if (!cap.isOpened()) //check if we succeeded
        return -1;

    grabOn.store(true);                //set the grabbing control variable
    thread t(GrabThread, &cap);          //start the grabbing thread
    int bufSize;
    //std::thread t1(framerate,"hmm");
    while (true)
    {
        mtxCam.lock();                //lock memory for exclusive access
        bufSize = buffer.size();      //check how many frames are waiting
        if (bufSize > 0)              //if some
        {
            //reference to buffer.front() should be valid after
            //pop because of Mat memory reference counting
            //but if content can change after unlock is better to keep a copy
            //an alternative is to unlock after processing (this will lock grabbing)
            buffer.front().copyTo(frame);   //get the oldest grabbed frame (queue=FIFO)
            buffer.pop();            //release the queue item
        }
        mtxCam.unlock();            //unlock the memory

        if (bufSize > 0)            //if a new frame is available
        {
            ProcessFrame(frame, bufSize);    //process it
            bufSize--;
        }

        //if bufSize is increasing means that process time is too slow regards to grab time
        //may be you will have out of memory soon
        if (bufSize)
            cout << endl << "frames to process:" << bufSize;

        if (waitKey(1) >= 0)        //press any key to terminate
        {
            grabOn.store(false);    //stop the grab loop
            t.join();               //wait for the grab loop

            cout << endl << "Flushing buffer of:" << bufSize << " frames...";
            while (!buffer.empty())    //flushing the buffer
            {
                frame = buffer.front();
                ProcessFrame(frame, bufSize);
                buffer.pop();
            }
            cout << "done"<<endl;
            break; //exit from process loop
        }
    }
    cout << endl << "Press Enter to terminate"; cin.get();
    return 0;
}


/*
int main(){




    //ellipseDetector((char *)"media/circles.jpg");

    cv::Mat image;
    cv::Rect rect[2];


    const std::string videoStreamAddress = "http://192.168.1.31:8080/video";
    cv::VideoCapture cap;
    const unsigned int targetFramerate = 10;
    const unsigned int second = 1000000;
    const unsigned int targetFrameTime = second / targetFramerate;

    if (!cap.open(videoStreamAddress)) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cout << "running" << endl;
    int i = 0;
    //cap.set(CV_CAP_PROP_POS_FRAMES, 0);

    while (cap.isOpened()) {
        //cout << "iteration " << ++ i << endl;
        //cap.set(CV_CAP_PROP_POS_FRAMES, cap.get(CV_CAP_PROP_FRAME_COUNT));
        //cap.release();
        cv::Mat temp;
        //cap.open("cap.open(videoStreamAddress)");//Insert own url
        cap >> temp;
        cv::flip(temp, image, +1);
        //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //unsigned int microsec = std::chrono::duration_cast<std::chrono::microseconds>  (end - begin).count();
        //if (second / microsec > targetFramerate)
        //    std::this_thread::sleep_for(std::chrono::microseconds(targetFrameTime - microsec));
        //begin = std::chrono::steady_clock::now();
        imshow("TestImage", image);
        //cout << cap.get(CV_CAP_PROP_FRAME_COUNT) << endl;
        waitKey(1);
    }
    /*
    {
        VideoCapture cap;
        // open the default camera, use something different from 0 otherwise;
        // Check VideoCapture documentation.
        if(!cap.open(0))
            return 0;
        for(;;)
        {
            Mat frame;
            cap >> frame;
            if( frame.empty() ) break; // end of video stream
            imshow("this is you, smile! :)", frame);
            if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
        }
        // the camera will be closed automatically upon exit
        // cap.close();
        return 0;
    }
    */






    /*


    image = imread("media/circles.png");
    cvtColor(image, image, COLOR_BGR2GRAY);
    medianBlur(image, image, 5);
    equalizeHist(image, image);
    inRange(image, 0, sliderPos, image);
    threshold(image, image, 100, 255, THRESH_BINARY | THRESH_OTSU);
    cvtColor(image, image, COLOR_GRAY2BGR);
    auto start = chrono::steady_clock::now();
    cv::Mat resultImage = OnImage(image);
    auto end = chrono::steady_clock::now();
    cout << "Elapsed time in milliseconds : "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;


    //return 0;

    //imshow("Annotated Image", resultImage);
    /*
    //waitKey();

    cv::namedWindow("out", WINDOW_AUTOSIZE);

    cv_image<bgr_pixel> cimg(image);

    cvtColor(image, image, COLOR_BGR2GRAY);

    medianBlur(image, image, 5);

    equalizeHist(image, image);
    inRange(image, 0, sliderPos, image);
    threshold(image, image, 100, 255, THRESH_BINARY | THRESH_OTSU);
    //morphologyEx(image, image, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(25, 25)));

    //distanceTransform(image, image, DIST_L2, 3);

    //normalize(image, image, 0, 1.0, NORM_MINMAX);
    Mat dst;


    //Mat render = croppedImage.clone();

    auto kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    //dilate(image, kernel, 2);
    //erode(image, kernel, 3);

    //morphologyEx(croppedImage, croppedImage, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(25, 25)));
    Mat imagec;
    cvtColor(image, imagec, COLOR_GRAY2BGR);
    */

    


    //sleep(2u);
    //std::thread Detectface(detectface, &image ,rect);
    //sleep(2u);
    /*
    std::thread detectEyeLeft(detectEye, &image, rect+1);
    sleep(1u);
    std::thread detectEyeRight(detectEye, &image, rect+0);
    sleep(1u);
    */
    /*
    //Mat frame = imread("media/circles.png");
    //cv::namedWindow("frame", WINDOW_AUTOSIZE);

    cv::createTrackbar("threshold", "frame", &sliderPos, 255);
    cv::createTrackbar("alpha", "frame", &alpha, 100);
    cv::createTrackbar("beta", "frame", &betaa, 200);

}
*/
