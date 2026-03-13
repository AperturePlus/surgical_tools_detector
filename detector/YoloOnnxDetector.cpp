#include "detector/YoloOnnxDetector.h"

#include <algorithm>
#include <iostream>
#include <filesystem>
#include <thread>

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

namespace sgt {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
YoloOnnxDetector::YoloOnnxDetector(const std::string&   modelPath,
                                   int                  inputSize,
                                   float                confThresh,
                                   float                nmsThresh,
                                   const LabelProvider* labels)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SGTDetector")
    , inputSize_(inputSize)
    , numClasses_(0)
    , confThresh_(confThresh)
    , nmsThresh_(nmsThresh)
    , labels_(labels)
{
    // ── Session options ────────────────────────────────────────────────────
    int threads = static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency() / 2));
    sessionOpts_.SetIntraOpNumThreads(threads);
    sessionOpts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // ── Execution Provider selection ───────────────────────────────────────
    // When compiled with WITH_CUDA=ON, try CUDA first; fall back to CPU.
    bool sessionCreated = false;

#ifdef SGT_WITH_CUDA
    {
        Ort::SessionOptions cudaOpts;
        cudaOpts.SetIntraOpNumThreads(threads);
        cudaOpts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        try {
            OrtCUDAProviderOptions cudaProviderOpts{};
            cudaOpts.AppendExecutionProvider_CUDA(cudaProviderOpts);
            auto wpath = std::filesystem::path(modelPath).wstring();
            session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), cudaOpts);
            sessionOpts_ = std::move(cudaOpts);
            std::cout << "[SGTDetector] CUDA Execution Provider enabled.\n";
            sessionCreated = true;
        } catch (const Ort::Exception& e) {
            std::cerr << "[SGTDetector] CUDA EP unavailable: " << e.what()
                      << "\n  → Falling back to CPU.\n";
        }
    }
#endif

    if (!sessionCreated) {
        // Pure CPU session
#ifdef _WIN32
        auto wpath = std::filesystem::path(modelPath).wstring();
        session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), sessionOpts_);
#else
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOpts_);
#endif
        std::cout << "[SGTDetector] Using CPU Execution Provider ("
                  << threads << " threads).\n";
    }

    // ── Query input / output names ─────────────────────────────────────────
    {
        auto ptr = session_->GetInputNameAllocated(0, allocator_);
        inputName_ = ptr.get();
    }
    {
        auto ptr = session_->GetOutputNameAllocated(0, allocator_);
        outputName_ = ptr.get();
    }

    // ── Infer numClasses from output shape  [1, 4+nc, anchors] ────────────
    {
        auto outInfo  = session_->GetOutputTypeInfo(0);
        auto outShape = outInfo.GetTensorTypeAndShapeInfo().GetShape();
        // dim[1] may be -1 (dynamic) in some exports; fall back to labels count
        if (outShape.size() >= 2 && outShape[1] > 4) {
            numClasses_ = static_cast<int>(outShape[1]) - 4;
        } else if (labels_) {
            numClasses_ = labels_->numClasses();
        } else {
            numClasses_ = 38; // hard default matching m4.yaml
        }
    }

    inputShape_ = {1, 3, inputSize_, inputSize_};

    std::cout << "[SGTDetector] Model : " << modelPath << "\n"
              << "  Input  : " << inputName_
              << "  [1,3," << inputSize_ << "," << inputSize_ << "]\n"
              << "  Output : " << outputName_
              << "  [1," << (4 + numClasses_) << ",anchors]\n"
              << "  Classes: " << numClasses_ << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Letterbox pre-processing
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat YoloOnnxDetector::preprocess(const cv::Mat& frame,
                                     float&         outScale,
                                     int&           outPadLeft,
                                     int&           outPadTop) const
{
    int srcW = frame.cols, srcH = frame.rows;
    outScale = std::min(static_cast<float>(inputSize_) / srcW,
                        static_cast<float>(inputSize_) / srcH);

    int newW = static_cast<int>(std::round(srcW * outScale));
    int newH = static_cast<int>(std::round(srcH * outScale));
    outPadLeft = (inputSize_ - newW) / 2;
    outPadTop  = (inputSize_ - newH) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // Fill letterbox background (grey 114, matching YOLOv8 defaults)
    cv::Mat padded(inputSize_, inputSize_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(outPadLeft, outPadTop, newW, newH)));

    // BGR → float32 NCHW blob; swapRB=true handles BGR→RGB
    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob,
                           1.0 / 255.0,
                           cv::Size(),          // no additional resize
                           cv::Scalar(),        // no mean subtraction
                           true,                // swapRB: BGR → RGB
                           false,               // no crop
                           CV_32F);
    return blob; // shape [1, 3, inputSize_, inputSize_]
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-processing: decode [4+nc, anchors] → Detection list
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection>
YoloOnnxDetector::postprocess(const float*    data,
                              int             anchors,
                              const cv::Size& origSize,
                              float           scale,
                              int             padLeft,
                              int             padTop) const
{
    // data layout (row-major):  data[row * anchors + col]
    //   row 0: cx  (pixel coords in letterboxed space)
    //   row 1: cy
    //   row 2: w
    //   row 3: h
    //   row 4..4+nc-1: class scores (0-1, already activated by YOLOv8 export)

    std::vector<cv::Rect>  boxes;   // NMSBoxes requires integer Rect
    std::vector<float>     scores;
    std::vector<int>       classIds;
    boxes.reserve(256);
    scores.reserve(256);
    classIds.reserve(256);

    for (int a = 0; a < anchors; ++a) {
        // Find argmax class score
        float maxScore = -1.0f;
        int   bestCls  = 0;
        for (int c = 0; c < numClasses_; ++c) {
            float s = data[(4 + c) * anchors + a];
            if (s > maxScore) { maxScore = s; bestCls = c; }
        }
        if (maxScore < confThresh_) continue;

        // cx, cy, w, h in letterboxed 960×960 coordinate space
        float cx = data[0 * anchors + a];
        float cy = data[1 * anchors + a];
        float bw = data[2 * anchors + a];
        float bh = data[3 * anchors + a];

        // Remove padding and scale back to original frame coordinates
        float x1 = (cx - bw * 0.5f - padLeft) / scale;
        float y1 = (cy - bh * 0.5f - padTop)  / scale;
        float rw  = bw / scale;
        float rh  = bh / scale;

        // Clamp to frame boundaries
        x1 = std::clamp(x1, 0.0f, static_cast<float>(origSize.width));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(origSize.height));
        float x2 = std::clamp(x1 + rw, 0.0f, static_cast<float>(origSize.width));
        float y2 = std::clamp(y1 + rh, 0.0f, static_cast<float>(origSize.height));

        if (x2 <= x1 || y2 <= y1) continue; // degenerate box

        boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        scores.push_back(maxScore);
        classIds.push_back(bestCls);
    }

    // NMS (per-class via class ID vector)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confThresh_, nmsThresh_, indices);

    std::vector<Detection> result;
    result.reserve(indices.size());
    for (int idx : indices) {
        Detection d;
        d.bbox    = {static_cast<float>(boxes[idx].x),
                     static_cast<float>(boxes[idx].y),
                     static_cast<float>(boxes[idx].width),
                     static_cast<float>(boxes[idx].height)};
        d.classId = classIds[idx];
        d.score   = scores[idx];
        d.label   = labels_ ? labels_->getLabel(classIds[idx]) : std::string();
        result.push_back(std::move(d));
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// detect() — public entry point
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> YoloOnnxDetector::detect(const cv::Mat& frame)
{
    float scale;
    int   padLeft, padTop;
    cv::Mat blob = preprocess(frame, scale, padLeft, padTop);

    // Build input tensor (blob data ownership stays with cv::Mat)
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        reinterpret_cast<float*>(blob.data),
        static_cast<size_t>(1) * 3 * inputSize_ * inputSize_,
        inputShape_.data(),
        inputShape_.size());

    const char* inputNames[]  = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 inputNames,  &inputTensor, 1,
                                 outputNames, 1);

    // Output shape: [1, 4+nc, anchors]
    auto   shape   = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int    anchors = static_cast<int>(shape[2]);
    const float* data = outputs[0].GetTensorMutableData<float>();

    return postprocess(data, anchors, frame.size(), scale, padLeft, padTop);
}

} // namespace sgt
