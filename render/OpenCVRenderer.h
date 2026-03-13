#pragma once

#include <memory>
#include <string>
#include "core/Renderer.h"
#include "core/FontRenderer.h"

namespace sgt {

/// Renderer implementation using OpenCV HighGUI (imshow / waitKey).
/// To switch to Dear ImGui or Qt:
///   implement a new class that derives from Renderer and swap in main.cpp.
class OpenCVRenderer : public Renderer {
public:
    explicit OpenCVRenderer(std::unique_ptr<FontRenderer> font,
                            const std::string& windowName = "SGTDetector");
    ~OpenCVRenderer() override;

    void drawDetections(cv::Mat&                      frame,
                        const std::vector<Detection>& dets) override;

    void drawHUD(cv::Mat& frame, const HUDData& hud) override;

    /// Displays the frame with imshow, calls waitKey(1).
    /// Returns the waitKey result (key code & 0xFF; negative = no key).
    int showFrame(const cv::Mat& frame) override;

    void onScreenshot(const std::string& path) override;

private:
    std::unique_ptr<FontRenderer> font_;
    std::string                   windowName_;

    // On-screen "Saved: ..." notification
    std::string screenshotMsg_;
    int         screenshotFramesLeft_ = 0;

    /// Deterministic per-class color using golden-angle HSV distribution.
    static cv::Scalar classColor(int classId);

    /// Draw a filled label badge above (or below) a point.
    void drawLabelBadge(cv::Mat&           frame,
                        const std::string& text,
                        cv::Point          anchor,
                        cv::Scalar         bgColor);
};

} // namespace sgt
