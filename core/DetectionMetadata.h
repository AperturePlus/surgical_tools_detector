#pragma once

#include <string>

namespace xcwj {

struct DetectionThresholds {
    float tool   = 0.65f;
    float grasp  = 0.25f;
    float defect = 0.50f;
};

struct ModelInfo {
    std::string toolModel;
    std::string graspModel;
    std::string defectModel;
};

} // namespace xcwj
