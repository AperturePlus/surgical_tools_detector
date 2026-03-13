#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace sgt {

/// Abstract interface for text rendering.
/// Default implementation uses cv::putText (ASCII-only, Hershey fonts).
/// Replace with FreeTypeFontRenderer (WITH_FREETYPE=ON) for CJK / Unicode.
class FontRenderer {
public:
    virtual ~FontRenderer() = default;

    /// Draw text at the given bottom-left origin.
    virtual void putText(cv::Mat&           img,
                         const std::string& text,
                         cv::Point          origin,
                         double             fontScale,
                         cv::Scalar         color,
                         int                thickness = 1) = 0;

    /// Estimate the rendered text bounding size; baseline is written via pointer.
    virtual cv::Size getTextSize(const std::string& text,
                                 double             fontScale,
                                 int                thickness,
                                 int*               baseline) = 0;
};

} // namespace sgt
