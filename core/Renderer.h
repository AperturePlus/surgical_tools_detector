#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "DefectResult.h"
#include "Detection.h"

namespace sgt {

/// Bitmask for active detection modes.
enum ModeMask : uint8_t {
    MODE_TOOL   = 0x01,
    MODE_GRASP  = 0x02,
    MODE_DEFECT = 0x04,
};

/// Data passed to Renderer::drawHUD() each frame.
struct HUDData {
    uint8_t activeModes    = MODE_TOOL;
    float fps              = 0.0f;
    float toolConfThresh   = 0.65f;
    float graspConfThresh  = 0.25f;
    float defectThresh     = 0.50f;
    int   toolDetections   = 0;
    int   graspDetections  = 0;
    int   defects          = 0;
};

/// Configuration knobs for Renderer implementations.
struct RendererConfig {
    int  boxThickness  = 2;
    int  fontScale100  = 52;    ///< Font scale * 100 (52 → 0.52)
    bool showConfScore = true;
    bool showFPS       = true;
};

/// Abstract interface for rendering detection results and displaying frames.
class Renderer {
public:
    virtual ~Renderer() = default;

    virtual void drawDetections(cv::Mat&                      frame,
                                const std::vector<Detection>& dets) = 0;

    virtual void drawDefects(cv::Mat&                         frame,
                             const std::vector<DefectResult>& defects) {}

    virtual void drawHUD(cv::Mat& frame, const HUDData& hud) = 0;

    /// @return cv::waitKey() result (masked to 0xFF); negative = timeout.
    virtual int showFrame(const cv::Mat& frame) = 0;

    virtual void onScreenshot(const std::string& /*path*/) {}

    RendererConfig config;
};

} // namespace sgt
