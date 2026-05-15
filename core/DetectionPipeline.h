#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/Detection.h"
#include "core/DetectionMetadata.h"
#include "core/DefectResult.h"
#include "core/PerfStats.h"
#include "core/Renderer.h"

namespace sgt {

class DetectorBackend;
class LabelProvider;
class OnnxDefectClassifier;

struct DetectionFrameResult {
    cv::Mat rawFrame;
    cv::Mat annotatedFrame;
    std::vector<Detection> toolDetections;
    std::vector<Detection> graspDetections;
    std::vector<DefectResult> defectResults;
    uint8_t activeModes = MODE_TOOL;
    DetectionThresholds thresholds;
    PerfStats perf;
    float fps = 0.0f;

    int defectCount() const;
};

class FrameAnnotator {
public:
    cv::Mat annotate(const cv::Mat& frame,
                     const std::vector<Detection>& toolDets,
                     const std::vector<Detection>& graspDets,
                     const std::vector<DefectResult>& defects) const;

private:
    static cv::Scalar classColor(int classId);
    static void drawLabelBadge(cv::Mat& frame,
                               const std::string& text,
                               cv::Point anchor,
                               cv::Scalar bgColor);
};

class DetectionEngine {
public:
    DetectionEngine(const std::string& toolModelPath,
                    const std::string& graspModelPath,
                    const std::string& defectModelPath,
                    const std::string& dictPath);
    ~DetectionEngine();

    DetectionFrameResult process(const cv::Mat& frame, uint8_t modeMask);
    bool ensureLoaded(uint8_t bit, std::string* error = nullptr);
    void setThresholds(const DetectionThresholds& thresholds);
    DetectionThresholds thresholds() const;
    ModelInfo models() const;

private:
    std::unique_ptr<LabelProvider>        toolLabels_;
    std::unique_ptr<LabelProvider>        graspLabels_;
    std::unique_ptr<DetectorBackend>      toolDet_;
    std::unique_ptr<DetectorBackend>      graspDet_;
    std::unique_ptr<OnnxDefectClassifier> defectCls_;

    std::string toolPath_;
    std::string graspPath_;
    std::string defectPath_;
    std::string dictPath_;
    DetectionThresholds pendingThresholds_;
    FrameAnnotator annotator_;
};

} // namespace sgt
