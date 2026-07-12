#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "core/Detection.h"
#include "core/DefectResult.h"
#include "core/Renderer.h"

namespace xcwj {

class DetectorBackend;
class LabelProvider;
class OnnxDefectClassifier;

struct ImageDetectionResult {
    std::vector<Detection>    toolDetections;
    std::vector<Detection>    graspDetections;
    std::vector<DefectResult> defectResults;
};

/// Standalone image detection API — no GUI or camera dependency.
class ImageInputAPI {
public:
    ImageInputAPI(const std::string& toolModelPath,
                  const std::string& graspModelPath,
                  const std::string& defectModelPath,
                  const std::string& dictPath);
    ~ImageInputAPI();

    /// Run detection on a single image with the given mode mask.
    ImageDetectionResult detectImage(const cv::Mat& image, uint8_t modeMask);

private:
    std::unique_ptr<LabelProvider>        toolLabels_, graspLabels_;
    std::unique_ptr<DetectorBackend>      toolDet_, graspDet_;
    std::unique_ptr<OnnxDefectClassifier> defectCls_;

    void ensureLoaded(uint8_t bit);
    std::string toolPath_, graspPath_, defectPath_, dictPath_;
};

} // namespace xcwj
