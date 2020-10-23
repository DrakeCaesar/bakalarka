#pragma once
#include "dlib/image_processing/full_object_detection.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_processing/render_face_detections_abstract.h"
#include <vector>

namespace dlib
{
    std::vector<image_window::overlay_line> render_eye_detections (
        const std::vector<full_object_detection>& dets,
        const rgb_pixel color = rgb_pixel(0,255,0)
    );


    inline std::vector<image_window::overlay_line> render_eye_detections (
        const full_object_detection& det,
        const rgb_pixel color = rgb_pixel(0,255,0)
    );

}


