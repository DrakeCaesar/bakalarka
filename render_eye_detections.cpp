// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "render_eye_detections.h"

namespace dlib {
std::vector<image_window::overlay_line>
render_eye_detections(const std::vector<full_object_detection> &dets,
                      const rgb_pixel color) {
    std::vector<image_window::overlay_line> lines;
    for (unsigned long i = 0; i < dets.size(); ++i) {
        DLIB_CASSERT(
            dets[i].num_parts() == 12,
            "\t std::vector<image_window::overlay_line> render_eye_detections()"
                << "\n\t You have to give either a 5 point or 68 point face "
                   "landmarking output to this function. "
                << "\n\t dets[" << i
                << "].num_parts():  " << dets[i].num_parts());

        const full_object_detection &d = dets[i];

        // Left eye
        for (unsigned long i = 1; i <= 5; ++i)
            lines.push_back(
                image_window::overlay_line(d.part(i), d.part(i - 1), color));
        lines.push_back(
            image_window::overlay_line(d.part(0), d.part(5), color));

        // Right eye
        for (unsigned long i = 7; i <= 11; ++i)
            lines.push_back(
                image_window::overlay_line(d.part(i), d.part(i - 1), color));
        lines.push_back(
            image_window::overlay_line(d.part(6), d.part(11), color));
    }
    return lines;
}

// ----------------------------------------------------------------------------------------

inline std::vector<image_window::overlay_line>
render_eye_detections(const full_object_detection &det, const rgb_pixel color) {
    std::vector<full_object_detection> dets;
    dets.push_back(det);
    return render_eye_detections(dets, color);
}

// ----------------------------------------------------------------------------------------

} // namespace dlib
