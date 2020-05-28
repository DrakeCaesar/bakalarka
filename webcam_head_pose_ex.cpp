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

#define LINE_AA 16

#pragma comment (lib, "Normaliz.lib")
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Wldap32.lib")
#pragma comment (lib, "Crypt32.lib")
#pragma comment (lib, "advapi32.lib") 

using namespace dlib;
using namespace cv;
using namespace std;

#define NUM_THREADS 254

int handleError( int status, const char* func_name,
                 const char* err_msg, const char* file_name,
                 int line, void* userdata )
{
    //Do nothing -- will suppress console output
    return 0;   //Return value is not used
}
cv::VideoCapture capup;
void *wait(void *t) {
    int i;
    long tid;
    tid = (long)t;

    const std::string leftAddr = "http://192.168.1.";
    const std::string rightAddr = ":8080/video";

    std::string Addr = leftAddr + to_string(2+(long)t) + rightAddr;
    cv::VideoCapture cap;
    //cout << Addr + "\n";

    //cv::redirectError(handleError);
    //try {
        if (cap.open(Addr)) {
            std::cout << "Opened video stream at " << Addr << std::endl;
            capup = cap;
        }
    //}catch (...) {  }

    pthread_exit(NULL);

}

int capture(cv::Mat* image)
{
    const std::string videoStreamAddress = "http://192.168.1.13:8080/video";
    cv::VideoCapture cap;
    /*
    int rc;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    void *status;

    // Initialize and set thread joinable
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(long i = 0; i < NUM_THREADS; i++ ) {
        rc = pthread_create(&threads[i], &attr, wait, (void *)i );
        if (rc) {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }
    }
    int seconds = 0;
    while (!capup.isOpened()){
        sleep(1u);
        if (++seconds ==5)
            break;
    }
    pthread_attr_destroy(&attr);
    for(int i = 0; i < NUM_THREADS; i++ ) {
        pthread_cancel(threads[i]);
    }
    if (capup.isOpened())
        cout << "capture opened\n";
    else {
        cout << "capture not opened\n";
        pthread_exit(NULL);
    }
    cout << "Main: program exiting." << endl;
    //pthread_exit(NULL);

    cap = capup;
    */
    const unsigned int targetFramerate = 30;
    const unsigned int second = 1000000;
    const unsigned int targetFrameTime = second / targetFramerate;

    if (!cap.open(videoStreamAddress)) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat temp;

        //temp = imread("media/circles.png",1);
        cap >> temp;
        cv::flip(temp, *image, +1);
        if ((*image).empty())
            cap.set(CV_CAP_PROP_POS_FRAMES, 0);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        unsigned int microsec = std::chrono::duration_cast<std::chrono::microseconds>  (end - begin).count();
        if (second / microsec > targetFramerate)
            std::this_thread::sleep_for(std::chrono::microseconds(targetFrameTime - microsec));
        begin = std::chrono::steady_clock::now();
        imshow("Image", *image);
        //waitKey(0);
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

void detectface(cv::Mat* image, cv::Rect rect[], RotatedRect* box)
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
            win.clear_overlay();

            win.add_overlay(render_face_detections(shapes));
        }

        /*
        for (int i = 0; i < 2; i++) {
            if (MAX(box[i].size.width, box.size.height) > MIN(box[i].size.width, box.size.height) * 30)
                continue;

            ellipse(cimg, box[i], Scalar(0, 0, 255), 1, LINE_AA);
            ellipse(cimg, box[i].center, box[i].size * 0.5f, box[i].angle, 0, 360, Scalar(0, 255, 255), 1, LINE_AA);
            Point2f vtx[4];
            box.points(vtx);
            for (int j = 0; j < 4; j++)
                cv::line(cimg, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
        }
        
        */



        win.set_image(cimg);
        //cout << rect[0];
    }
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

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
    double i = fabs(contourArea(cv::Mat(contour1)));
    double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}

void detectEye(cv::Mat * image, cv::Rect * rect, RotatedRect * boxpointer){
    image_window win;
    int factor1 = 4;
    int factor2 = 2;
    cv::Rect scaledRect;
    image_window win1;
    while (true) {
    
        if ((*rect).width == 0)
            continue;
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
        std::vector<std::vector<Point> > contours;
        //Mat bimage = croppedImage >= sliderPos;
        /*
        for (int x = 4; x < croppedImage.cols-5; x++) {
            bool black = true;
            int white = 0;
            //cout << +croppedImage.channels() << endl;
            if (croppedImage.channels() != 0)
                for (int y = croppedImage.rows -5 ; y >=4 ; y--) {
                    
                    
                    //if (croppedImage.at<Vec3b>(y, x)[0] != 0)
                    //    croppedImage.at<Vec3b>(y, x)[0] = 255;
                    
                    
                    //if (croppedImage.at<Vec3b>(y, x)[0] != 255 && croppedImage.at<Vec3b>(y, x)[0] != 0)
                        //cout << +croppedImage.at<Vec3b>(y, x)[0] << endl;
                    if (black) {
                        if (croppedImage.at<Vec3b>(y, x)[0] == 255) {
                            black = false;
                            white = 1;
                        }
                    }
                    else if (white > 0) {
                        if (croppedImage.at<Vec3b>(y, x)[0] == 0) {
                            white--;

                        }
                    }
                    else
                        croppedImage.at<Vec3b>(y, x)[0] = 0;
                    
                }
        }
        */
        Mat bimage = croppedImage.clone();
        findContours(bimage, contours, RETR_TREE, CHAIN_APPROX_NONE);
        Mat cimage = Mat::zeros(bimage.size(), CV_8UC3);
        // cimage = croppedImage.clone();

        cvtColor(render, render, COLOR_GRAY2BGR);
        int i = contours.size() - 1;
        if (i ==  -1)
            continue;
        size_t count = contours[i].size();
        if (count < 6)
            continue;
        std::sort(contours.begin(), contours.end(), compareContourAreas);
        std::vector<cv::Point> biggestContour = contours[contours.size() - 1];


        Mat pointsf;
        Mat(contours[i]).convertTo(pointsf, CV_32F);


        //RotatedRect box = minAreaRect(Mat(contours[i]));
        



        RotatedRect box = fitEllipse(pointsf);
        *boxpointer = box;
        if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 30)
            continue;
        drawContours(render, contours, (int)i, Scalar::all(255), 1, 8);

        ellipse(render, box, Scalar(0, 0, 255), 1, LINE_AA);
        ellipse(render, box.center, box.size * 0.5f, box.angle, 0, 360, Scalar(0, 255, 255), 1, LINE_AA);
        Point2f vtx[4];
        box.points(vtx);
        for (int j = 0; j < 4; j++)
            cv::line(render, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);
        
        //
        //cv_image<bgr_pixel> cimg1(cimage);
        
        cv_image<bgr_pixel> cimg1(render);
        win1.set_image(cimg1);

    }
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




int main(){




    //ellipseDetector((char *)"media/circles.jpg");
    const int grid = 5;
    
    cv::Mat image;
    cv::Rect rect[2];
    RotatedRect box[2];
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

    

    std::thread Capture(capture,&image);

    sleep(2u);
    std::thread Detectface(detectface, &image ,rect,box);
    sleep(1u);

    std::thread detectEyeLeft(detectEye, &image, rect+0, box+0);
    sleep(1u);
    std::thread detectEyeRight(detectEye, &image, rect+1, box+1);
    sleep(1u);

    /*
    //Mat frame = imread("media/circles.png");
    //cv::namedWindow("frame", WINDOW_AUTOSIZE);

    cv::createTrackbar("threshold", "frame", &sliderPos, 255);
    cv::createTrackbar("alpha", "frame", &alpha, 100);
    cv::createTrackbar("beta", "frame", &betaa, 200);
     */
    while (true) {
        cv::waitKey(0);
    }
}

