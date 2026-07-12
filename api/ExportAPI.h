#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "core/Detection.h"
#include "core/DetectionMetadata.h"
#include "core/DefectResult.h"

namespace xcwj {

struct ExportMetadata {
    int                 schemaVersion = 1;
    std::string         id;
    std::string         timestamp;
    uint8_t             modeMask = 0;
    DetectionThresholds thresholds;
    ModelInfo           models;
    std::string         rawImagePath;
    std::string         annotatedImagePath;
    std::string         jsonPath;
};

struct ExportData {
    cv::Mat     rawFrame;
    cv::Mat     annotatedFrame;
    std::string statsJson;
};

/// Build export data (frame + JSON stats) from detection results.
ExportData exportFrame(const cv::Mat& frame,
                       const std::vector<Detection>& toolDets,
                       const std::vector<Detection>& graspDets,
                       const std::vector<DefectResult>& defects);

/// Build export data with raw/annotated frames and capture metadata.
ExportData exportFrame(const cv::Mat& rawFrame,
                       const cv::Mat& annotatedFrame,
                       const std::vector<Detection>& toolDets,
                       const std::vector<Detection>& graspDets,
                       const std::vector<DefectResult>& defects,
                       const ExportMetadata& meta);

/// Save frame as JPEG and stats as JSON to outputDir. Returns true on success.
bool saveExport(const ExportData& data, const std::string& outputDir);

} // namespace xcwj
