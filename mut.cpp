#include "async_opencv_video_capture.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <mutex>
#include <iostream>
#include <chrono>
#include <thread>
int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "must enter url of media\n";
        return -1;
    }

    std::mutex emutex;
    //create the functor to handle the exception when cv::VideoCapture fail
    //to capture the frame and wait 30 msec between each frame
    long long constexpr wait_msec = 30;
    cl([&](std::exception const& ex)
    {
        //cerr of c++ is not a thread safe class, so we need to lock the mutex
        std::lock_guard<std::mutex> lock(emutex);
        std::cerr << "camera exception:" << ex.what() << std::endl;

        return true;
    }, wait_msec);
    cl.open_url(argv[1]);

    //add listener to process captured frame
    //the listener could process the task in another thread too,
    //to make things easier to explain, I prefer to process it in
    //the same thread of videoCapture
    cv::Mat img;
    cl.add_listener([&](cv::Mat input)
    {
        std::lock_guard<std::mutex> lock(emutex);
        img = input;
    }, &emutex);

    //execute the task(s)
    cl.run();

    //We must display the captured image at main thread but not
    //in the listener, because every manipulation related to gui
    //must perform in the main thread(it also called gui thread)
    for (int finished = false; finished != 'q';) {
        finished = std::tolower(cv::waitKey(30));
        std::lock_guard<std::mutex> lock(emutex);
        if (!img.empty()) {
            cv::imshow("frame", img);
        }
    }
}
void run()
{
    if (thread_) {
        //before we start the thread,
        //we need to stop it
        set_stop(true);
        //call join before task(s)
        //of the thread done
        thread_->join();
        set_stop(false);
    }

    //create a new thread
    create_thread();
}
void create_thread()
{
    thread_ = std::make_unique<std::thread>([this]()
    {
        //read the frames in infinite for loop
        for (cv::Mat frame;;) {
            std::lock_guard<Mutex> lock(mutex_);
            if (!stop_ && !listeners_.empty()) {
                try {
                    cap_ >> frame;
                }
                catch (std::exception const& ex) {
                    //reopen the camera if exception thrown ,this may happen frequently when you
                    //receive frames from network
                    cap_.open(url_);
                    cam_exception_listener_(ex);
                }

                if (!frame.empty()) {
                    for (auto& val : listeners_) {
                        val.second(frame);
                    }
                }
                else {
                    if (replay_) {
                        cap_.open(url_);
                    }
                    else {
                        break;
                    }
                }
                std::this_thread::sleep_for(wait_for_);
            }
            else {
                break;
            }
        }
    });
}
void set_stop(bool val)
{
    std::lock_guard<Mutex> lock(mutex_);
    stop_ = val;
}

void stop()
{
    set_stop(true);
}

template<typename Mutex>
~async_opencv_video_capture()
{
    stop();
    thread_->join();
}