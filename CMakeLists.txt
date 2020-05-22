cmake_minimum_required(VERSION 3.16)
project(bakalarka)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories(.)
add_subdirectory(../dlib dlib_build)
#set(USE_SSE2_INSTRUCTIONS ON CACHE BOOL "Use SSE2 instructions")
#set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Use SSE4 instructions")
#set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")
add_executable( bakalarka
                render_face.hpp
                webcam_head_pose_ex.cpp
                common.cpp
                common.h
                EllipseDetectorYaed.cpp
                EllipseDetectorYaed.h
        ellipseDetector.cpp ellipseDetector.h)
target_link_libraries( bakalarka dlib::dlib ${OpenCV_LIBS} )