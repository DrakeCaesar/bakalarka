
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

#define frames 1238 //1238
using namespace dlib;
using namespace cv;
using namespace std;

std::mutex mtxCam;
std::atomic<bool> grabOn; //this is lock free
std::queue<Mat> buffer;

frontal_face_detector detector = get_frontal_face_detector();
shape_predictor pose_model, eyes_model;
image_window win;
image_window eyeL;
image_window eyeR;
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
    //(*image) = imread("face.jpeg");
    bool debug = true;
    cv::Mat resize, flip, grey, test, transpose;
    cv::cvtColor(* image, grey, COLOR_BGR2GRAY);
    cv::resize(grey, resize, cv::Size(grey.cols/8,grey.rows/8));
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
                //cv::rectangle(test, rect[j], cv::Scalar(0, 255, 0));
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
void ProcessFrame(Mat *src,int *bufSize)
{




    //*src = imread("face.jpeg");
    //imshow("Image main", *src);
    Rect eyes[2];
    //src = imread("face.jpeg");
    if(*bufSize > 8 ) return;
    //Mat newMat;
    //cv::transpose(*src,newMat);
    //if((*src).empty()) return;
    int faces = detectface(const_cast<const decltype(src)>(src),eyes);
    //if (faces) {
    //    detectEye(src, eyes + 1, &eyeL);
    //}

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

void processWithDlib(Mat * temp, std::vector<dlib::rectangle> * faces, int index){
    //
    //cv::resize(*temp,*temp,cv::Size((*temp).cols,(*temp).rows));
    cv_image<bgr_pixel> cimg(*temp);
    *faces = detector(cimg);

}

int main() {
    ofstream result ("compareResult.txt");

    //Draw axis
    Mat Eye_Waveform = Mat::zeros(900, 900, CV_8UC3); // Waveform image used to record blinks
    Point p1 = Point(10, 0);
    Point p2 = Point(10, 900);
    Point p3 = Point(0, 890);
    Point p4 = Point(900, 890);
    Scalar line_color = Scalar(255, 255, 255);
    cv::line(Eye_Waveform, p1, p2, line_color, 1, LINE_AA);
    cv::line(Eye_Waveform, p3, p4, line_color, 1, LINE_AA);



    //Store the coordinates of the last point of the eye
    int eye_previous_x = 10; //The abscissa of the origin
    int eye_previous_y = 890; //The ordinate of the origin
    int eye_now_x = 1;
    int eye_now_y = 1;


    //Store the number of blinks
    unsigned int count_blink = 0; //Number of blinks


    //Each blink EAR has to go through the process from greater than 0.2-less than 0.2-greater than 0.2
    float blink_EAR_before =0.0; // before blinking
    float blink_EAR_now =0.2; //In blinking
    float blink_EAR_after = 0.0; //After blinking


    try {

        //const std::string videoStreamAddress = "VID_20201119_195914.mp4";
        const std::string videoStreamAddress = "VID_20201207_235401.mp4";
        VideoCapture cap(videoStreamAddress);
        //VideoCapture cap(0);
        if (!cap.isOpened()) {//Open the camera
            printf("Unable to connect a camera");
            return 1;
        }
        frontal_face_detector detector = get_frontal_face_detector();

        shape_predictor pos_model;

        deserialize("shape_predictor_68_face_landmarks.dat") >> pos_model;
        Mat data[frames];
        for  (int i=0; i < frames; i++){
            //char filename[32];
            //snprintf(filename, sizeof(filename), "wink/img-%d.png",i+1);
            //data[i] = imread(filename);
            cap >> data[i];
            //cout << "loaded image " << string(filename)  << endl;
            if (i%100 == 0){
                cout << "loaded image " << i << endl;
            }
        }
        std::vector<dlib::rectangle> faces[frames];
        /*
        for (int j=0; j < frames; j = j+8) {
            std::thread t[8];
            for (int i = 0; i < 8; i++) {
                t[i] = std::thread(processWithDlib, &(data[j+i]), &(faces[j+i]));
                //if (j % 100 == 0) {
                    cout << "launched thread " << j+i << endl;
                //}
            }

            for (int i = 0; i < 8; ++i) {
                t[i].detach();
                //if (j % 100 == 0) {
                    cout << "joined thread " << j+i << endl;
                //}
            }
        }
         for (int j=0; j < frames; j++) {

            cout << "processed " << j << endl;
        }
        */


        bool faceDetected = false;
        for  (int k=0; k < frames; k++){
            if (waitKey(30) == 27) {
                break;
            }

            Mat temp = data[k];
            //cap >> temp;
            //cv::resize(temp,temp,cv::Size(temp.cols/2,temp.rows/2));
            //cv::transpose(temp, temp);
            //cv::imshow("video",temp);
            //cv::waitKey(0);
            //Convert the image into the form of BGR in dlib
            cv_image<bgr_pixel> cimg(temp);
            //win1.set_image(cimg);
            //std::vector<dlib::rectangle> faces = detector(cimg);
            bool load = false;
            std::vector<full_object_detection> shapes;
            if (load == true){
                char fileName[64];
                stringstream fileNameStringStream;
                fileNameStringStream << "facialData/" << k << ".0.txt";
                ifstream myfile (fileNameStringStream.str());
                string line;

                for (int i = 0; i < 68; i++){
                    std::getline(myfile, line);
                    std::stringstream buffer(line);
                    int tempx, tempy;
                    buffer >> tempx >> tempy;
                    full_object_detection tempShape;
                    tempShape.part(i).x() = tempx;
                    tempShape.part(i).y() = tempy;
                    shapes[0].part(i).x() = tempx;
                    shapes[0].part(i).y() = tempy;

                }


            }
            if (load == false) {
                processWithDlib(&(temp), &(faces[k]), k);
                unsigned int faceNumber = faces[k].size(); //Get the number of vectors in the container, that is, the number of faces
                for (unsigned int i = 0; i < faceNumber; i++) {
                    shapes.push_back(pos_model(cimg, faces[k][i]));
                }
            }
            if (!shapes.empty()) {
                if (faceDetected == false) {
                    cout << "Face detected at: " << k << endl;
                    faceDetected = true;
                }
            }
            else{
                if (faceDetected == true) {
                    cout << "Face not detected at: " << k << endl;
                    faceDetected = false;
                }
            }


            if (!shapes.empty()) {

                if (faceDetected == false){
                    cout << "Face detected at: " << k << endl;
                    faceDetected = true;
                }
                //if (false){

                int faceNumber = shapes.size();
                for (int j = 0; j < faceNumber; j++)
                {
                    ofstream myfile;
                    if (load == false) {
                        char fileName[64];
                        stringstream fileNameStringStream;
                        fileNameStringStream << "facialData/" << k << "." << j << ".txt";
                        //cout << fileNameStringStream.str() << endl;
                        myfile.open(fileNameStringStream.str(), ios::out);
                    }

                    for (int i = 0; i < 68; i++)
                    {
                        //Points used to draw eigenvalues
                        if (load == false) {
                            cv::circle(temp, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 1,
                                       cv::Scalar(0, 0, 255), -1);
                            if (load == false) {
                                myfile << shapes[j].part(i).x() << " " << shapes[j].part(i).y() << endl;
                            }
                        }
                        //Parameter description Image center line width color line type
                        //Display number
                        //cv::putText(temp, to_string(i), cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));

                    }
                    if (load == false) {
                        myfile.close();
                    }
                }

                //Left eye

                //The coordinates of point 36
                unsigned int x_36 = shapes[0].part(36).x();
                unsigned int y_36 = shapes[0].part(36).y();

                //The coordinates of point 37
                unsigned int x_37 = shapes[0].part(37).x();
                unsigned int y_37 = shapes[0].part(37).y();

                //The coordinates of point 38
                unsigned int x_38 = shapes[0].part(38).x();
                unsigned int y_38 = shapes[0].part(38).y();

                //The coordinates of point 39
                unsigned int x_39 = shapes[0].part(39).x();
                unsigned int y_39 = shapes[0].part(39).y();

                //The coordinates of point 40
                unsigned int x_40 = shapes[0].part(40).x();
                unsigned int y_40 = shapes[0].part(40).y();

                //The coordinates of point 41
                unsigned int x_41 = shapes[0].part(41).x();
                unsigned int y_41 = shapes[0].part(41).y();



                int height_left_eye1 = y_41-y_37; //Longitudinal distance from 37 to 41
                //cout << "Left Eye Height 1\t" << height_left_eye1 << endl;
                int height_left_eye2 = y_40-y_38; //Longitudinal distance from 38 to 40
                //cout << "Left Eye Height 2\t" << height_left_eye2 << endl;
                float height_left_eye = (height_left_eye1 + height_left_eye2) / 2; //up and down distance of eyes
                //cout << "Left Eye Height\t" << height_left_eye << endl;
                int length_left_eye = x_39 - x_36;
                //cout << "Left Eye Length\t" << length_left_eye << endl;
                if (height_left_eye == 0) //When the eyes are closed, the distance may be detected as 0 and the aspect ratio is wrong
                    height_left_eye = 1;

                float EAR_left_eye; //eye aspect ratio
                EAR_left_eye = height_left_eye / length_left_eye;

                //Right eye

                //The coordinates of point 42
                unsigned int x_42 = shapes[0].part(42).x();
                unsigned int y_42 = shapes[0].part(42).y();

                //The coordinates of point 37
                unsigned int x_43 = shapes[0].part(43).x();
                unsigned int y_43 = shapes[0].part(43).y();

                //The coordinates of point 38
                unsigned int x_44 = shapes[0].part(44).x();
                unsigned int y_44 = shapes[0].part(44).y();

                //The coordinates of point 39
                unsigned int x_45 = shapes[0].part(45).x();
                unsigned int y_45 = shapes[0].part(45).y();

                //The coordinates of point 40
                unsigned int x_46 = shapes[0].part(46).x();
                unsigned int y_46 = shapes[0].part(46).y();

                //The coordinates of point 41
                unsigned int x_47 = shapes[0].part(47).x();
                unsigned int y_47 = shapes[0].part(47).y();

                unsigned int height_right_eye1 = y_47-y_43; //Longitudinal distance from 37 to 41
                unsigned int height_right_eye2 = y_46-y_44; //Longitudinal distance from 38 to 40
                float height_right_eye = (height_right_eye1 + height_right_eye2) / 2; //up and down distance of eyes
                if (height_right_eye == 0) //When the eyes are closed, the distance may be detected as 0 and the aspect ratio is wrong
                    height_right_eye = 1;

                unsigned int length_right_eye = x_45 - x_42;

                float EAR_right_eye; //Aspect ratio of eyes
                EAR_right_eye = height_right_eye / length_right_eye;

                //Take the average aspect ratio of the two eyes as the aspect ratio of the eyes
                float EAR_eyes = (EAR_left_eye + EAR_right_eye) / 2;

                //cout << "The aspect ratio of the eyes is" << EAR_eyes << endl;

                //Draw the waveform of the eye
                eye_now_x = eye_now_x + 1; // abscissa (one point for every 10 images)
                eye_now_y = 900-(EAR_eyes * 900 ); //Vertical coordinate
                Point poi1 = Point(eye_previous_x, eye_previous_y); //previous point
                Point poi2 = Point(eye_now_x, eye_now_y); //Current point
                Scalar eyes_color = Scalar(0, 255, 0);
                cv::line(Eye_Waveform, poi1, poi2, eyes_color,1, LINE_AA); //Draw a line
                eye_previous_x = eye_now_x;
                eye_previous_y = eye_now_y;
                namedWindow("Blink waveform figure", WINDOW_AUTOSIZE);

                //Count the number of blinks
                if (blink_EAR_before < EAR_eyes) {
                    blink_EAR_before = EAR_eyes;
                }
                if (blink_EAR_now > EAR_eyes) {
                    blink_EAR_now = EAR_eyes;
                }
                if (blink_EAR_after < EAR_eyes) {
                    blink_EAR_after = EAR_eyes;
                }
                if (blink_EAR_before > 0.2 && blink_EAR_now <= 0.2 && blink_EAR_after > 0.2) {
                    count_blink = count_blink + 1;
                    cout << " blink at " << k << endl;
                    string line;
                    std::stringstream lineSteam;
                    ifstream myfile ("compareValues.txt");
                    if (myfile.is_open())
                    {
                        int closestMiddle = 10000;
                        int closestLeft = 0;
                        int closestRight = 0;
                        while ( getline (myfile,line) )
                        {

                            int left, right;
                            lineSteam.str(line);
                            lineSteam >> left >> right;
                            //cout << left << " " << right << endl;
                            int middle = (left + right) / 2;
                            if ( abs(k - middle) <  abs(k - closestMiddle)){
                                closestMiddle = middle;
                                closestLeft = left;
                                closestRight = right;
                            }
                        }
                        result << "Blink at: " << k << " Closest marked blink: " << (closestLeft) << " " << (closestRight) << endl;
                        cout << "Blink at: " << k << " Closest marked blink: " << (closestLeft) << " " << (closestRight) << endl;

                        myfile.close();
                    }

                    else cout << "Unable to open file";

                    blink_EAR_before = 0.0;
                    blink_EAR_now = 0.2;
                    blink_EAR_after = 0.0;
                }

                //Display height_left_eye, length_left_eye and ERA_left_eye

                //Convert hight_left_eye from float type to string type
                char count_blink_text[30];

                _gcvt_s(count_blink_text, count_blink, 10); //Convert hight_left_eye from float type to string type

                putText(temp, count_blink_text, Point(10, 100), FONT_HERSHEY_COMPLEX, 1.0, Scalar(0, 0, 255), 1, LINE_AA);


            }


            //Display it all on the screen display pictures of each frame
            cv::imshow("Dlib tag", temp);
            cv::imshow("Blink waveform figure", Eye_Waveform);


            //Time for one minute (60 seconds)
            clock_t start = clock();
            clock_t finish = clock();
            }

    }
    catch (serialization_error& e) {
        cout << "You need dlib‘s default face landmarking file to run this example." << endl;
        cout << endl << e.what() << endl;
    }
    catch (exception& e) {
        cout << e.what() << endl;

    }

}

int main3(){
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
    deserialize("eye_predictor.dat") >> eyes_model;
    Rect eyes[2];
    float wink[4];
    Mat src;
    VideoCapture cap;
    const std::string videoStreamAddress = "VID_20201023_163951.mp4";
    cap.open(videoStreamAddress);
    if (!cap.isOpened()) //check if we succeeded
        return -1;

    string faceDetectionStrings[2] = {"nebola","bola"};
    stringstream output;

    bool faceDetected = false;
    for (int i = 0; ; i++){
        cap >> src;
        if(src.empty()){
            ofstream myfile;
            myfile.open ("output.txt");
            myfile << output.str();
            myfile.close();
            break;
        }
        cv::transpose(src, src);
        //src = imread("face.jpeg");
        //imshow("Image main", src);
        //waitKey(0);
        //continue;
        //src = imread("face.jpeg");


        int faces = detectface(const_cast<const decltype(&src)>(&src),eyes);
        if (faceDetected != bool(faces)){
            faceDetected = bool(faces);
            output << "v snímke " << i << " " << faceDetectionStrings[faces] << "detegovana tvar" << endl;
        }

        if (faces) {
            detectEye(&src, eyes, &eyeR);
            detectEye(&src, eyes + 1, &eyeL);
        }

    }
}

int main1() {
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
    deserialize("eye_predictor.dat") >> eyes_model;

    Mat frame;
    VideoCapture cap;

    const std::string videoStreamAddress = "http://192.168.0.50:8080/video";
    //const std::string videoStreamAddress = "VID_20201023_163951.mp4";
    cap.open(videoStreamAddress);
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
            //std::thread processFrame(ProcessFrame, &frame, &bufSize);
            //sleep(100);
            ProcessFrame(&frame, &bufSize);    //process it
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
                ProcessFrame(&frame, &bufSize);
                buffer.pop();
            }
            cout << "done"<<endl;
            break; //exit from process loop
        }
    }
    cout << endl << "Press Enter to terminate"; cin.get();
    return 0;
}

