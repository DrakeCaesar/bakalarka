cmake_minimum_required(VERSION 3.16)
project(bakalarka)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories(.)
add_subdirectory(../dlib dlib_build)
add_executable( bakalarka
                render_face.hpp
                webcam_head_pose_ex.cpp)
target_link_libraries( bakalarka dlib::dlib ${OpenCV_LIBS} )