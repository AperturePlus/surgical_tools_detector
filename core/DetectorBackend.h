#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "Detection.h"

namespace sgt {

/// Abstract interface for object detection backends.
/// To add a new backend (TensorRT, OpenVINO, etc.):
///   1. Derive from this class
///   2. Implement detect(), getConfThresh(), setConfThresh(), getNmsThresh()
///   3. Swap in main.cpp via dependency injection
class DetectorBackend {
public:
    virtual ~DetectorBackend() = default;

    /// Run inference on a BGR frame; return detections in frame coordinates.
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;

    /// Confidence threshold currently in use.
    virtual float getConfThresh() const = 0;

    /// Update the confidence threshold (clamped to [0, 1]).
    virtual void setConfThresh(float thresh) = 0;

    /// NMS IoU threshold currently in use.
    virtual float getNmsThresh() const = 0;
};

} // namespace sgt
