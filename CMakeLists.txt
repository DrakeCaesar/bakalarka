cmake_minimum_required(VERSION 3.16)
project(bakalarka)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV CONFIG REQUIRED )
find_package( dlib CONFIG REQUIRED )


include_directories(.)
#set(USE_SSE2_INSTRUCTIONS ON CACHE BOOL "Use SSE2 instructions")
#set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Use SSE4 instructions")
#set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")
#add_definitions( "-O3 -mavx -I" )

add_definitions( "-I" )



add_executable(
    bakalarka
    webcam_head_pose_ex.cpp

)

#target_link_libraries( bakalarka dlib ${OpenCV_LIBS} )
#target_link_libraries(bakalarka PRIVATE dlib::dlib)
target_link_libraries(bakalarka PRIVATE dlib::dlib ${OpenCV_LIBS})


