
#include "../src/ellipseDetector.h"
#include "../src/lib/CTPL/ctpl.h"
#include <atomic>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
#include <iostream>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

#include "../src/render_eye_detections.h"

#define frames 1238 // 1238

dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
dlib::shape_predictor pose_model;
dlib::image_window win;

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

int detectface(int index) {
    /*
        if (index % 1 == 0) {
            char console[64];
            snprintf(console, sizeof(console), "%d\n", index);
            std::cout << console;
        }
    */
    cv::Rect rect[2];

    char filename[64];
    snprintf(filename, sizeof(filename),
             "C:/Users/domin/dev/bakalarka/data/wink/img-%d.png", index + 1);

    cv::Mat image = cv::imread(filename);

    bool debug = false;
    cv::Mat resize, flip, grey, test;
    cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    cv::resize(grey, resize, cv::Size(grey.cols / 4, grey.rows / 4));
    dlib::array2d<unsigned char> cimg, cimg1;
    dlib::assign_image(cimg, dlib::cv_image<unsigned char>(resize));
    if (debug) {
        cv::cvtColor(resize, test, cv::COLOR_GRAY2BGR);
    }
    std::vector<dlib::rectangle> faces = detector(cimg);
    std::vector<dlib::full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i) {
        shapes.push_back(pose_model(cimg, faces[i]));
        for (int j = 0; j < 2; j++) {
            int x1 = INT_MAX, y1 = INT_MAX;
            int x2 = 0, y2 = 0;
            for (int k = 0; k < 6; k++) {
                x1 = min(x1, shapes[i].part(36 + j * 6 + k).x());
                x2 = max(x2, shapes[i].part(36 + j * 6 + k).x());
                y1 = min(y1, shapes[i].part(36 + j * 6 + k).y());
                y2 = max(y2, shapes[i].part(36 + j * 6 + k).y());
            }
            rect[j] = cv::Rect(x1, y1 - .5 * (y2 - y1), x2 - x1, 2 * (y2 - y1));
            if (debug) {
                // cv::rectangle(test, rect[j], cv::Scalar(0, 255, 0));
            }
        }
        dlib::full_object_detection remapped_image;
        if (debug) {
            win.clear_overlay();
            win.add_overlay(render_face_detections(shapes));
        }
    }
    if (debug) {
        dlib::cv_image<dlib::bgr_pixel> cimg1(test);
        win.set_image(cimg1);
    }
    return faces.size();
}

int main() {
    dlib::deserialize("C:/Users/domin/dev/bakalarka/data/"
                      "shape_predictor_68_face_landmarks.dat") >>
        pose_model;

    std::cout << "start" << std::endl;

    const int numThreads = 7;

    ctpl::thread_pool p(numThreads);

    using clock = std::chrono::system_clock;
    using ms = std::chrono::duration<double, std::milli>;

    while (p.n_idle() != numThreads)
        ;
    const auto before = clock::now();

    for (int k = 0; k < frames; k++) {

        // std::cout << p.n_idle() << std::endl;
        int current = p.n_idle();
        // std::cout << current << std::endl;
        if (current == 0) {
            --k;
            continue;
        }
        p.push([k](int) { detectface(k); });
    }

    while (p.n_idle() != numThreads)
        ;
    const ms duration = clock::now() - before;

    std::cout << "It took " << duration.count() << "ms" << std::endl;
    std::cout << "end" << std::endl;
}