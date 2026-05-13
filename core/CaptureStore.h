#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "api/ExportAPI.h"
#include "core/DetectionPipeline.h"

namespace sgt {

struct CaptureRecord {
    std::string id;
    std::string timestamp;
    std::string rawImagePath;
    std::string annotatedImagePath;
    std::string jsonPath;
    uint8_t modeMask = 0;
    DetectionThresholds thresholds;
    int toolCount = 0;
    int graspCount = 0;
    int defectCount = 0;
};

class CaptureStore {
public:
    explicit CaptureStore(std::filesystem::path rootDir);

    CaptureRecord saveCapture(const DetectionFrameResult& result,
                              const ModelInfo& models);
    std::vector<CaptureRecord> records() const;
    std::filesystem::path rootDir() const { return rootDir_; }

private:
    std::filesystem::path rootDir_;
    std::vector<CaptureRecord> records_;

    void loadExisting();
    void writeIndex() const;
};

} // namespace sgt
