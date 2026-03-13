#include "render/OpenCVFontRenderer.h"

namespace sgt {

void OpenCVFontRenderer::putText(cv::Mat&           img,
                                 const std::string& text,
                                 cv::Point          origin,
                                 double             fontScale,
                                 cv::Scalar         color,
                                 int                thickness)
{
    cv::putText(img, text, origin, fontFace_,
                fontScale, color, thickness, cv::LINE_AA);
}

cv::Size OpenCVFontRenderer::getTextSize(const std::string& text,
                                         double             fontScale,
                                         int                thickness,
                                         int*               baseline)
{
    return cv::getTextSize(text, fontFace_, fontScale, thickness, baseline);
}

} // namespace sgt
