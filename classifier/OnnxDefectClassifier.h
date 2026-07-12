#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "core/DefectResult.h"
#include "core/Detection.h"

namespace xcwj {

/// Fixed-batch ONNX classifier for per-tool defect detection.
class OnnxDefectClassifier {
public:
    OnnxDefectClassifier(const std::string& modelPath,
                         int                inputSize    = 512,
                         int                batchSize    = 4,
                         float              defectThresh = 0.50f);

    std::vector<DefectResult> classify(const cv::Mat&                 frame,
                                       const std::vector<Detection>&  tools);

    float getDefectThresh() const { return defectThresh_; }
    void  setDefectThresh(float thresh);

private:
    Ort::Env                      env_;
    Ort::SessionOptions           sessionOpts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    int   inputSize_;
    int   batchSize_;
    float defectThresh_;

    std::string          inputName_;
    std::string          outputName_;
    std::vector<int64_t> inputShape_;

    void fillInputTensor(const cv::Mat& crop, float* dst) const;
};

} // namespace xcwj
