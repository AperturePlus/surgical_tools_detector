#pragma once

#include "core/FontRenderer.h"
#include <opencv2/imgproc.hpp>

namespace sgt {

/// FontRenderer implementation using cv::putText with Hershey fonts.
/// ASCII-only. Enable WITH_FREETYPE=ON and implement FreeTypeFontRenderer
/// to gain full CJK / Unicode support without changing any other code.
class OpenCVFontRenderer : public FontRenderer {
public:
    explicit OpenCVFontRenderer(int fontFace = cv::FONT_HERSHEY_SIMPLEX)
        : fontFace_(fontFace) {}

    void putText(cv::Mat&           img,
                 const std::string& text,
                 cv::Point          origin,
                 double             fontScale,
                 cv::Scalar         color,
                 int                thickness = 1) override;

    cv::Size getTextSize(const std::string& text,
                         double             fontScale,
                         int                thickness,
                         int*               baseline) override;

private:
    int fontFace_;
};

} // namespace sgt
