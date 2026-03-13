#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "Detection.h"

namespace sgt {

/// Data passed to Renderer::drawHUD() each frame.
/// Adding new fields here does not require changing any Renderer interface contract.
struct HUDData {
    float fps          = 0.0f;
    float confThresh   = 0.25f;
    int   detections   = 0;         ///< Number of detections this frame
};

/// Configuration knobs for Renderer implementations.
struct RendererConfig {
    int  boxThickness  = 2;
    int  fontScale100  = 52;    ///< Font scale * 100 (52 → 0.52)
    bool showConfScore = true;
    bool showFPS       = true;
};

/// Abstract interface for rendering detection results and displaying frames.
/// Replace implementation to switch from OpenCV HighGUI to ImGui, Qt, etc.
class Renderer {
public:
    virtual ~Renderer() = default;

    /// Draw bounding boxes + labels onto the frame in-place.
    virtual void drawDetections(cv::Mat&                      frame,
                                const std::vector<Detection>& dets) = 0;

    /// Overlay HUD information (FPS, threshold, hints) onto the frame in-place.
    virtual void drawHUD(cv::Mat& frame, const HUDData& hud) = 0;

    /// Display the frame and poll for input.
    /// @return cv::waitKey() result  (masked to 0xFF); negative = timeout.
    virtual int showFrame(const cv::Mat& frame) = 0;

    /// Called after a screenshot is saved so the renderer can show a flash/toast.
    virtual void onScreenshot(const std::string& /*path*/) {}

    RendererConfig config;
};

} // namespace sgt
