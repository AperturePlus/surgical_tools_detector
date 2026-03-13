#include "render/OpenCVRenderer.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace sgt {

// ─────────────────────────────────────────────────────────────────────────────
OpenCVRenderer::OpenCVRenderer(std::unique_ptr<FontRenderer> font,
                               const std::string& windowName)
    : font_(std::move(font))
    , windowName_(windowName)
{
    cv::namedWindow(windowName_, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(windowName_, 1280, 720);
}

OpenCVRenderer::~OpenCVRenderer()
{
    cv::destroyWindow(windowName_);
}

// ─────────────────────────────────────────────────────────────────────────────
// Golden-angle HSV color distribution — visually distinct across 38 classes
// ─────────────────────────────────────────────────────────────────────────────
cv::Scalar OpenCVRenderer::classColor(int classId)
{
    float hue = std::fmod(static_cast<float>(classId) * 137.508f, 360.0f);
    cv::Mat hsv(1, 1, CV_8UC3,
                cv::Scalar(static_cast<uchar>(hue / 2.0f), 200, 220));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto* p = bgr.ptr<uchar>(0);
    return cv::Scalar(p[0], p[1], p[2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Draw a filled badge with white text; auto-flips below the box when clipped
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVRenderer::drawLabelBadge(cv::Mat&           frame,
                                    const std::string& text,
                                    cv::Point          anchor,
                                    cv::Scalar         bgColor)
{
    double fontScale = config.fontScale100 / 100.0;
    int baseline = 0;
    cv::Size sz = font_->getTextSize(text, fontScale, 1, &baseline);

    int padX = 4, padY = 3;
    int bx = anchor.x;
    int by = anchor.y - sz.height - baseline - padY * 2;

    // Flip below the top edge if label goes out of frame
    if (by < 0) by = anchor.y + padY;

    // Prevent right-edge overflow
    if (bx + sz.width + padX * 2 > frame.cols)
        bx = frame.cols - sz.width - padX * 2;

    cv::Rect bg(bx, by, sz.width + padX * 2, sz.height + baseline + padY * 2);
    bg &= cv::Rect(0, 0, frame.cols, frame.rows); // clamp to frame

    cv::rectangle(frame, bg, bgColor, cv::FILLED);
    font_->putText(frame, text,
                   cv::Point(bx + padX, by + sz.height + padY),
                   fontScale,
                   cv::Scalar(255, 255, 255),
                   1);
}

// ─────────────────────────────────────────────────────────────────────────────
void OpenCVRenderer::drawDetections(cv::Mat&                      frame,
                                    const std::vector<Detection>& dets)
{
    for (const auto& d : dets) {
        cv::Scalar color = classColor(d.classId);
        cv::Rect   rect  = d.bbox.toRect();

        cv::rectangle(frame, rect, color, config.boxThickness);

        std::ostringstream label;
        label << (d.label.empty() ? "cls" + std::to_string(d.classId) : d.label);
        if (config.showConfScore) {
            label << " " << std::fixed << std::setprecision(0)
                  << (d.score * 100.0f) << "%";
        }
        drawLabelBadge(frame, label.str(),
                       cv::Point(rect.x, rect.y), color);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpenCVRenderer::drawHUD(cv::Mat& frame, const HUDData& hud)
{
    const double fs    = 0.58;
    const int    thick = 1;

    // ── FPS counter (top-right) ────────────────────────────────────────────
    if (config.showFPS) {
        std::ostringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(1) << hud.fps;
        int baseline = 0;
        cv::Size sz = font_->getTextSize(ss.str(), fs, thick, &baseline);
        cv::Point pos(frame.cols - sz.width - 10, sz.height + 10);

        // Drop-shadow for readability
        font_->putText(frame, ss.str(), {pos.x + 1, pos.y + 1},
                       fs, cv::Scalar(0, 0, 0), thick + 1);
        font_->putText(frame, ss.str(), pos,
                       fs, cv::Scalar(0, 230, 80), thick);
    }

    // ── Status bar (top-left): conf threshold + det count + key hints ─────
    {
        std::ostringstream ss;
        ss << "Conf:" << std::fixed << std::setprecision(2) << hud.confThresh
           << "  Det:" << hud.detections
           << "  [+/-]conf  [s]shot  [q]quit";

        font_->putText(frame, ss.str(), {9, 23},
                       fs, cv::Scalar(0, 0, 0), thick + 1);
        font_->putText(frame, ss.str(), {8, 22},
                       fs, cv::Scalar(220, 220, 220), thick);
    }

    // ── Screenshot toast (center) ──────────────────────────────────────────
    if (screenshotFramesLeft_ > 0) {
        --screenshotFramesLeft_;
        const double tfs = 0.75;
        int baseline = 0;
        cv::Size sz = font_->getTextSize(screenshotMsg_, tfs, 2, &baseline);
        cv::Point pos((frame.cols - sz.width) / 2,
                      frame.rows - sz.height - 20);

        font_->putText(frame, screenshotMsg_, {pos.x + 2, pos.y + 2},
                       tfs, cv::Scalar(0, 0, 0), 2);
        font_->putText(frame, screenshotMsg_, pos,
                       tfs, cv::Scalar(0, 200, 255), 2);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
int OpenCVRenderer::showFrame(const cv::Mat& frame)
{
    cv::imshow(windowName_, frame);
    return cv::waitKey(1);
}

// ─────────────────────────────────────────────────────────────────────────────
void OpenCVRenderer::onScreenshot(const std::string& path)
{
    // Show short base filename in toast
    auto sep = path.find_last_of("/\\");
    screenshotMsg_ = "Saved: " + (sep == std::string::npos ? path : path.substr(sep + 1));
    screenshotFramesLeft_ = 75; // ~2.5 s @ 30 fps
}

} // namespace sgt
