#pragma once

#include <string>
#include <vector>
#include <memory>
#include <algorithm>     // std::clamp
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "core/DetectorBackend.h"
#include "core/LabelProvider.h"

namespace xcwj {

/// YOLOv8 object detector backed by ONNX Runtime.
///
/// Execution Provider priority (runtime, no recompile needed):
///   CUDA (WITH_CUDA=ON)  →  DirectML (WITH_DIRECTML=ON)  →  CPU fallback
///
/// To add TensorRT or OpenVINO support:
///   create a new class that derives from DetectorBackend and swap in main.cpp.
class YoloOnnxDetector : public DetectorBackend {
public:
    /// @param modelPath   Path to the .onnx file.
    /// @param inputSize   Square input size used at training time (e.g. 640).
    /// @param confThresh  Minimum confidence to keep a detection [0, 1].
    /// @param nmsThresh   NMS IoU threshold [0, 1].
    /// @param labels      Optional LabelProvider for class name resolution.
    YoloOnnxDetector(const std::string&   modelPath,
                     int                  inputSize  = 640,
                     float                confThresh = 0.65f,
                     float                nmsThresh  = 0.45f,
                     const LabelProvider* labels     = nullptr);

    ~YoloOnnxDetector() override = default;

    std::vector<Detection> detect(const cv::Mat& frame) override;
    PerfStats lastPerfStats() const override { return lastPerf_; }

    float getConfThresh() const override { return confThresh_; }
    void  setConfThresh(float t) override {
        confThresh_ = std::clamp(t, 0.0f, 1.0f);
    }
    float getNmsThresh() const override { return nmsThresh_; }

private:
    // ── ORT session ────────────────────────────────────────────────────────
    Ort::Env                         env_;
    Ort::SessionOptions              sessionOpts_;
    std::unique_ptr<Ort::Session>    session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // ── Model metadata ─────────────────────────────────────────────────────
    int   inputSize_;
    int   numClasses_;
    float confThresh_;
    float nmsThresh_;

    std::string          inputName_;
    std::string          outputName_;
    std::vector<int64_t> inputShape_;   // [1, 3, H, W]
    ONNXTensorElementDataType inputElementType_  = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType outputElementType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

    const LabelProvider* labels_;       // non-owning, may be null

    // ── Helpers ────────────────────────────────────────────────────────────
    /// Letterbox-resize to inputSize×inputSize.
    void preprocess(const cv::Mat& frame,
                    float&         outScale,
                    int&           outPadLeft,
                    int&           outPadTop);

    float* fillInputTensorFloat();
    Ort::Float16_t* fillInputTensorFloat16();

    /// Parse the raw output tensor [4+nc, anchors] into Detection objects.
    std::vector<Detection> postprocess(const float*    data,
                                       int             anchors,
                                       const cv::Size& origSize,
                                       float           scale,
                                       int             padLeft,
                                       int             padTop);
    std::vector<Detection> postprocess(const Ort::Float16_t* data,
                                       int                   anchors,
                                       const cv::Size&       origSize,
                                       float                 scale,
                                       int                   padLeft,
                                       int                   padTop);

    cv::Mat resized_;
    cv::Mat padded_;
    std::vector<float> inputFp32_;
    std::vector<Ort::Float16_t> inputFp16_;
    std::vector<cv::Rect> boxes_;
    std::vector<float> scores_;
    std::vector<int> classIds_;
    std::vector<int> nmsIndices_;
    PerfStats lastPerf_;
};

} // namespace xcwj
