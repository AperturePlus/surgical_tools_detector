#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "core/Detection.h"
#include "core/DefectResult.h"

namespace sgt {

struct ExportData {
    cv::Mat     frame;
    std::string statsJson;
};

/// Build export data (frame + JSON stats) from detection results.
ExportData exportFrame(const cv::Mat& frame,
                       const std::vector<Detection>& toolDets,
                       const std::vector<Detection>& graspDets,
                       const std::vector<DefectResult>& defects);

/// Save frame as JPEG and stats as JSON to outputDir. Returns true on success.
bool saveExport(const ExportData& data, const std::string& outputDir);

} // namespace sgt
