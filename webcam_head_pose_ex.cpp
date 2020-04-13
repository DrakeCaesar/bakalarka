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


int capture(cv::Mat* image)
{
    const unsigned int targetFramerate = 30;
    const unsigned int second = 1000000;
    const unsigned int targetFrameTime = second / targetFramerate;
    const std::string videoStreamAddress = "http://192.168.1.17:8080/video";
    cv::VideoCapture cap;

    /*
        for (int i = 1; i < 255; i++) {
        std::stringstream buffer;
        buffer << "http://192.168.1." << +i << ":8080/video";
        std::string videoStreamAddress = buffer.str();
        if (cap.open(videoStreamAddress,0)) {
            std::cout << "Opened video stream at " << videoStreamAddress << std::endl;
            break;
        }
    }
    */
    
    /*
    if (!cap.open(videoStreamAddress)) {
        std::cout << "Error opening video stream or file" << std::endl;
        //return -1;
    }
    */
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    while (true) {
        cv::Mat temp;
        temp = imread("media/face.jpg",1);
        //cap >> temp;
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
int beta = 0;       /*< Simple brightness control Enter the beta value [0-100]: 50 */

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

void getQuadrangleSubPix_8u32f_CnR(const uchar* src, size_t src_step, Size src_size,
    float* dst, size_t dst_step, Size win_size,
    const double* matrix, int cn)
{
    int x, y, k;
    double A11 = matrix[0], A12 = matrix[1], A13 = matrix[2];
    double A21 = matrix[3], A22 = matrix[4], A23 = matrix[5];

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);

    for (y = 0; y < win_size.height; y++, dst += dst_step)
    {
        double xs = A12 * y + A13;
        double ys = A22 * y + A23;
        double xe = A11 * (win_size.width - 1) + A12 * y + A13;
        double ye = A21 * (win_size.width - 1) + A22 * y + A23;

        if ((unsigned)(cvFloor(xs) - 1) < (unsigned)(src_size.width - 3) &&
            (unsigned)(cvFloor(ys) - 1) < (unsigned)(src_size.height - 3) &&
            (unsigned)(cvFloor(xe) - 1) < (unsigned)(src_size.width - 3) &&
            (unsigned)(cvFloor(ye) - 1) < (unsigned)(src_size.height - 3))
        {
            for (x = 0; x < win_size.width; x++)
            {
                int ixs = cvFloor(xs);
                int iys = cvFloor(ys);
                const uchar* ptr = src + src_step * iys;
                float a = (float)(xs - ixs), b = (float)(ys - iys), a1 = 1.f - a, b1 = 1.f - b;
                float w00 = a1 * b1, w01 = a * b1, w10 = a1 * b, w11 = a * b;
                xs += A11;
                ys += A21;

                if (cn == 1)
                {
                    ptr += ixs;
                    dst[x] = ptr[0] * w00 + ptr[1] * w01 + ptr[src_step] * w10 + ptr[src_step + 1] * w11;
                }
                else if (cn == 3)
                {
                    ptr += ixs * 3;
                    float t0 = ptr[0] * w00 + ptr[3] * w01 + ptr[src_step] * w10 + ptr[src_step + 3] * w11;
                    float t1 = ptr[1] * w00 + ptr[4] * w01 + ptr[src_step + 1] * w10 + ptr[src_step + 4] * w11;
                    float t2 = ptr[2] * w00 + ptr[5] * w01 + ptr[src_step + 2] * w10 + ptr[src_step + 5] * w11;

                    dst[x * 3] = t0;
                    dst[x * 3 + 1] = t1;
                    dst[x * 3 + 2] = t2;
                }
                else
                {
                    ptr += ixs * cn;
                    for (k = 0; k < cn; k++)
                        dst[x * cn + k] = ptr[k] * w00 + ptr[k + cn] * w01 +
                        ptr[src_step + k] * w10 + ptr[src_step + k + cn] * w11;
                }
            }
        }
        else
        {
            for (x = 0; x < win_size.width; x++)
            {
                int ixs = cvFloor(xs), iys = cvFloor(ys);
                float a = (float)(xs - ixs), b = (float)(ys - iys), a1 = 1.f - a, b1 = 1.f - b;
                float w00 = a1 * b1, w01 = a * b1, w10 = a1 * b, w11 = a * b;
                const uchar* ptr0, * ptr1;
                xs += A11; ys += A21;

                if ((unsigned)iys < (unsigned)(src_size.height - 1))
                    ptr0 = src + src_step * iys, ptr1 = ptr0 + src_step;
                else
                    ptr0 = ptr1 = src + (iys < 0 ? 0 : src_size.height - 1) * src_step;

                if ((unsigned)ixs < (unsigned)(src_size.width - 1))
                {
                    ptr0 += ixs * cn; ptr1 += ixs * cn;
                    for (k = 0; k < cn; k++)
                        dst[x * cn + k] = ptr0[k] * w00 + ptr0[k + cn] * w01 + ptr1[k] * w10 + ptr1[k + cn] * w11;
                }
                else
                {
                    ixs = ixs < 0 ? 0 : src_size.width - 1;
                    ptr0 += ixs * cn; ptr1 += ixs * cn;
                    for (k = 0; k < cn; k++)
                        dst[x * cn + k] = ptr0[k] * b1 + ptr1[k] * b;
                }
            }
        }
    }
}
void myGetQuadrangleSubPix(const Mat& src, Mat& dst, Mat& m)
{
    CV_Assert(src.channels() == dst.channels());

    cv::Size win_size = dst.size();
    double matrix[6];
    cv::Mat M(2, 3, CV_64F, matrix);
    m.convertTo(M, CV_64F);
    double dx = (win_size.width - 1) * 0.5;
    double dy = (win_size.height - 1) * 0.5;
    matrix[2] -= matrix[0] * dx + matrix[1] * dy;
    matrix[5] -= matrix[3] * dx + matrix[4] * dy;

    if (src.depth() == CV_8U && dst.depth() == CV_32F)
        getQuadrangleSubPix_8u32f_CnR(src.data, src.step, src.size(),
            (float*)dst.data, dst.step, dst.size(),
            matrix, src.channels());
    else
    {
        CV_Assert(src.depth() == dst.depth());
        cv::warpAffine(src, dst, M, dst.size(),
            cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
            cv::BORDER_REPLICATE);
    }
}
void getRotRectImg(cv::RotatedRect rr, Mat& img, Mat& dst)
{
    Mat m(2, 3, CV_64FC1);
    float ang = rr.angle * CV_PI / 180.0;
    m.at<double>(0, 0) = cos(ang);
    m.at<double>(1, 0) = sin(ang);
    m.at<double>(0, 1) = -sin(ang);
    m.at<double>(1, 1) = cos(ang);
    m.at<double>(0, 2) = rr.center.x;
    m.at<double>(1, 2) = rr.center.y;
    myGetQuadrangleSubPix(img, dst, m);
}


int main()
{   
    
    cv::Mat image;
    cv::Rect rect[2];
    RotatedRect box[2];

    
    image = imread("media/circles.png");
    namedWindow("out", WINDOW_AUTOSIZE);

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
    int rows = imagec.rows;
    int cols = imagec.cols;
    //cout << "cols: " << cols << endl;
    //cout << "rows: " << rows << endl;
    double ellipseX = imagec.cols / 2;
    double ellipseY = imagec.rows / 2;
    double ellipseW = 95;
    double ellipseH = 145;
    int ellipseD = 30;
    while (true) {
        cvtColor(image, imagec, COLOR_GRAY2BGR);
        ellipseD++;
        double ellipseA = 2 * pi * (1.f * ellipseD / 360);
        //RotatedRect rRect = RotatedRect(Point2f(100+random(20), 100 + random(20)), Size2f(100, 50), 0);
        
        RotatedRect rRect = RotatedRect(Point2f(ellipseX, ellipseY), Size2f(ellipseW, ellipseH), ellipseD);

        Mat rotatedImage(rRect.size, CV_32FC3);
        getRotRectImg(rRect, imagec, rotatedImage);
        
        //Rect brect = rRect.boundingRect();
        //cv::rectangle(imagec, brect, Scalar(255, 0, 0), 2);
        //ellipse(imagec, rRect.center, rRect.size * 0.5f, rRect.angle, 0, 360, Scalar(0, 255, 255), 1, LINE_AA);
        int blacks = 0;
        int whites = 0;
        int greys = 0;
        /*
        for (int j = 0; j < imagec.rows; j++) {
            for (int i = 0; i < imagec.cols; i++) {
                double point = pow((cos(ellipseA) * (i - ellipseX) + sin(ellipseA) * (j - ellipseY)) / (ellipseW / 2), 2) +
                    pow((sin(ellipseA) * (i - ellipseX) - cos(ellipseA) * (j - ellipseY)) / (ellipseH / 2), 2);
                if (point <= 1 && point > 0.95) {
                    Vec3b color = imagec.at<Vec3b>(Point(i, j));
                    Vec3b black = Vec3b(0, 0, 0);
                    Vec3b white = Vec3b(255, 255, 255);

                    if (color == black) {
                        imagec.at<Vec3b>(Point(i, j)) = Vec3b(0, 242, 221);
                        blacks++;
                    }
                    else if (color == white) {
                        imagec.at<Vec3b>(Point(i, j)) = Vec3b(0, 200, 50);
                        whites++;
                    }
                    else{
                        imagec.at<Vec3b>(Point(i, j)) = Vec3b(255, 0, 0);
                        greys++;
                    }

                }
            }
        }
        cout << "whites: " << whites << ", blacks: " << blacks << ", greys: " << greys << endl;
        */
        
        Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            cv::line(imagec, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);

        ellipse(imagec, rRect, Scalar(0, 0, 255), 1, LINE_AA);
        
        imshow("out", imagec);
        waitKey(1);
        

        
        double ellipseX2 = rotatedImage.cols / 2;
        double ellipseY2 = rotatedImage.rows / 2;
        double ellipseW2 = rotatedImage.cols;
        double ellipseH2 = rotatedImage.rows;
        int ellipseD2 = 0;
        double ellipseA2 = 2 * pi * (1.f * ellipseD2 / 360);
        for (int j = 0; j < rotatedImage.rows; j++) {
            for (int i = 0; i < rotatedImage.cols; i++) {
                double point = pow((cos(ellipseA2) * (i - ellipseX2) + sin(ellipseA2) * (j - ellipseY2)) / (ellipseW2 / 2), 2) +
                    pow((sin(ellipseA2) * (i - ellipseX2) - cos(ellipseA2) * (j - ellipseY2)) / (ellipseH2 / 2), 2);
                if (point <= 1 && point > 0) {
                    //cout << rotatedImage.at<Vec3f>(Point(i, j));
                    Vec3f color = rotatedImage.at<Vec3f>(Point(i, j));
                    Vec3f black = Vec3f(0, 0, 0);
                    Vec3f white = Vec3f(1, 1, 1);

                    if (color == black) {
                        rotatedImage.at<Vec3f>(Point(i, j)) = Vec3f(0, 1, 1);
                        blacks++;
                    }
                    else if (color == white) {
                        rotatedImage.at<Vec3f>(Point(i, j)) = Vec3f(0, 1, 0);
                        whites++;
                    }
                    else {
                        rotatedImage.at<Vec3f>(Point(i, j)) = Vec3f(1, 0, 0);
                        greys++;
                    }

                }
            }
        }
        //cout << "whites: " << whites << ", blacks: " << blacks << ", greys: " << greys << endl;
        
        imshow("rotImg", rotatedImage);


    }
    while (true) {
        cv::waitKey(0);
    }
    
    

    std::thread capture(capture,&image);
    Sleep(2000);
    std::thread detectface(detectface, &image ,rect,box);
    Sleep(1000);
    std::thread detectEyeLeft(detectEye, &image, rect+0, box+0);
    Sleep(1000);
    std::thread detectEyeRight(detectEye, &image, rect+1, box+1);
    Sleep(1000);


    Mat frame = imread("media/circles.png");
    namedWindow("frame", WINDOW_AUTOSIZE);

    createTrackbar("threshold", "frame", &sliderPos, 255);
    createTrackbar("alpha", "frame", &alpha, 100);
    createTrackbar("beta", "frame", &beta, 200);
    while (true) {
        cv::waitKey(0);
    }
}

