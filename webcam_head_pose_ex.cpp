
#include "ellipseDetector.h"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <vector>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include "render_eye_detections.h"

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
shape_predictor pose_model, eyes_model;
image_window win;
//image_window eyeL;
//image_window eyeR;
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


int detectEyes(cv::Mat * image, cv::Rect rect[])
{
    (*image) = imread("face.jpeg");
    bool debug = true;
    cv::Mat resize, flip, grey, test, transpose;
    cv::cvtColor(* image, grey, COLOR_BGR2GRAY);
    cv::resize(grey, resize, cv::Size(grey.cols/2,grey.rows/2));
    //cv::transpose(resize, transpose);
    //cv::flip(transpose, flip, +1);
    dlib::array2d<unsigned char> cimg, cimg1;
    dlib::assign_image(cimg, dlib::cv_image<unsigned char>(resize));
    if (debug){
        cv::cvtColor(resize, test, COLOR_GRAY2BGR);
    }
    std::vector<dlib::rectangle> faces = detector(cimg);
    std::vector<full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
    {
        shapes.push_back(eyes_model(cimg, faces[i]));
        for (int j = 0; j < 2; j++) {
            int x1 = INT_MAX, y1 = INT_MAX;
            int x2 = 0, y2 = 0;
            for (int k = 0; k < 6; k++) {
                x1 = min(x1, shapes[i].part(0 + j*6 + k).x());
                x2 = max(x2, shapes[i].part(0 + j*6 + k).x());
                y1 = min(y1, shapes[i].part(0 + j*6 + k).y());
                y2 = max(y2, shapes[i].part(0 + j*6 + k).y());
            }
            rect[j] = cv::Rect(x1, y1 - .5*(y2 - y1), x2 - x1, 2*(y2 - y1));
            //rect[j] = cv::Rect(x1, y1, x2 - x1, y2 - y1);

            if (debug) {
                //cv::rectangle(test, rect[j], cv::Scalar(0, 0, 255));
            }
        }
        full_object_detection remapped_image;
        if (debug) {
            win1.clear_overlay();
            win1.add_overlay(render_eye_detections(shapes));
        }
    }
    if (debug) {
        cv_image<bgr_pixel> cimg1(test);
        win1.set_image(cimg1);
    }
    return faces.size();
}

int detectface(cv::Mat * image, cv::Rect rect[])
{
    bool debug = true;
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
            //win.add_overlay(render_face_detections(shapes));
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

void detectEye(cv::Mat * image, cv::Rect * rect, image_window * window){
    int factor1 = 8;
    int factor2 = 2;
    cv::Rect scaledRect;

    scaledRect.x = (*rect).x * factor1;
    scaledRect.y = (*rect).y * factor1;
    scaledRect.width = (*rect).width * factor1;
    scaledRect.height = (*rect).height * factor1;
    normalize(&scaledRect, (* image).cols, (* image).rows);

    cv::Mat croppedImage = (* image)(scaledRect);
    if (croppedImage.cols * croppedImage.rows == 0)
        return;

    cv::cvtColor( croppedImage, croppedImage, COLOR_BGR2GRAY );
    cv::equalizeHist( croppedImage, croppedImage );
    cv::medianBlur(croppedImage, croppedImage,(3,3));

    cv::Mat frame;

    cv::GaussianBlur(croppedImage, frame, cv::Size(0, 0), 3);
    cv::addWeighted(croppedImage, 1.5, frame, -0.5, 0, frame);

    cv::cvtColor( croppedImage, croppedImage, COLOR_GRAY2RGB );
    cv::Mat result = OnImage(croppedImage);
    cv_image<bgr_pixel> cimg1(result);
    (*window).set_image(cimg1);

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
        //tmp = imread("face.jpeg");
        if (tmp.empty()) continue;
        //if (tmp.empty())
        //    (*cap).set(CAP_PROP_POS_FRAMES, 0);


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
int duration = 0;
int duration1 = 0;
void ProcessFrame(Mat &src,int bufSize)
{
    cv::transpose(src,src);
    //src = imread("face.jpeg");
    //imshow("Image main", src);
    Rect eyes[2];
    //src = imread("face.jpeg");
    if(bufSize > 8 ) return;
    if(src.empty()) return;
    int faces = detectEyes(const_cast<const decltype(&src)>(&src),eyes);
    //if (faces)
    //detectEye( &src, eyes+1, &eyeL);

/*
    for (int i = 0; i < faces; i++ ){
        std::thread detectEyeLeft(detectEye, &src, eyes+1, &eyeL);
        sleep(1u);
        std::thread detectEyeRight(detectEye, &src, eyes+0, &eyeR);
    }
    */
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
    deserialize("eye_predictor.dat") >> eyes_model;

    Mat frame;
    VideoCapture cap;

    const std::string videoStreamAddress = "http://192.168.0.103:8080/video";
    //const std::string videoStreamAddress = "VID_20201023_163951.mp4";
    cap.open(0);
    if (!cap.isOpened()) //check if we succeeded
        return -1;

    grabOn.store(true);                //set the grabbing control variable
    thread t(GrabThread, &cap);          //start the grabbing thread
    int bufSize;
    std::thread t1(framerate,"hmm");
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
            cout  << "frames to process:" << bufSize << endl;

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

