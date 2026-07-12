#pragma once

#include "Detection.h"

namespace xcwj {

/// Per-tool defect classification result.
struct DefectResult {
    int   toolIndex   = -1;     ///< Index into the tool detection vector
    BBox  bbox;                 ///< Tool box used for the crop
    float normalScore = 0.0f;   ///< Softmax probability for class 0
    float defectScore = 0.0f;   ///< Softmax probability for class 1
    bool  defective   = false;  ///< True when defectScore crosses threshold
};

} // namespace xcwj
