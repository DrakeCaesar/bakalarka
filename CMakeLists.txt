cmake_minimum_required(VERSION 3.16)
project(bakalarka)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV CONFIG REQUIRED )
find_package( dlib CONFIG REQUIRED )

set(USE_SSE2_INSTRUCTIONS ON CACHE BOOL "Use SSE2 instructions")
set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Use SSE4 instructions")
set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")
set(WITH_FFMPEG "ON")
#add_definitions( "-O3 -mavx -I" )
add_definitions( "-I" )
add_executable(
    bakalarka
    src/webcam_head_pose_ex.cpp
    src/common.cpp
    src/ellipseDetector.cpp
    src/EllipseDetectorYaed.cpp
    src/render_eye_detections.cpp
)

target_link_libraries(bakalarka ${OpenCV_LIBS} dlib::dlib)




