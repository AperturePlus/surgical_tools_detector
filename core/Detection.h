#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace sgt {

/// Axis-aligned bounding box in pixel coordinates (top-left origin).
struct BBox {
    float x;  ///< left
    float y;  ///< top
    float w;  ///< width
    float h;  ///< height

    cv::Rect toRect() const {
        return cv::Rect(
            static_cast<int>(x), static_cast<int>(y),
            static_cast<int>(w), static_cast<int>(h));
    }
    cv::Rect2f toRect2f() const {
        return cv::Rect2f(x, y, w, h);
    }
};

/// A single detection result.
struct Detection {
    BBox        bbox;     ///< Bounding box in original frame coordinates
    int         classId;  ///< Class index (0-based)
    float       score;    ///< Confidence score [0, 1]
    std::string label;    ///< Display label (resolved by LabelProvider)
};

} // namespace sgt
