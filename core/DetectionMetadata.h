#pragma once

#include <string>

namespace sgt {

struct DetectionThresholds {
    float tool   = 0.25f;
    float grasp  = 0.25f;
    float defect = 0.50f;
};

struct ModelInfo {
    std::string toolModel;
    std::string graspModel;
    std::string defectModel;
};

} // namespace sgt
