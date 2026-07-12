#include "detector/YoloOnnxDetector.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <sstream>
#include <unordered_map>

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

namespace xcwj {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end = Clock::now())
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool equalsIgnoreCase(const std::string& a, const std::string& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

bool hasProvider(const std::vector<std::string>& providers,
                 const std::vector<std::string>& aliases)
{
    for (const auto& provider : providers) {
        for (const auto& alias : aliases) {
            if (equalsIgnoreCase(provider, alias)) return true;
        }
    }
    return false;
}

std::string joinProviders(const std::vector<std::string>& providers)
{
    if (providers.empty()) return "(none)";
    std::ostringstream oss;
    for (size_t i = 0; i < providers.size(); ++i) {
        if (i) oss << ", ";
        oss << providers[i];
    }
    return oss.str();
}

std::vector<Ort::ConstEpDevice> selectEpDevices(
    const std::vector<Ort::ConstEpDevice>& devices,
    const std::vector<std::string>& aliases)
{
    std::vector<Ort::ConstEpDevice> matched;
    for (const auto& d : devices) {
        for (const auto& alias : aliases) {
            if (equalsIgnoreCase(d.EpName(), alias)) {
                matched.push_back(d);
                break;
            }
        }
    }
    return matched;
}

std::string joinEpDeviceNames(const std::vector<Ort::ConstEpDevice>& devices)
{
    if (devices.empty()) return "(none)";
    std::ostringstream oss;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (i) oss << ", ";
        oss << devices[i].EpName();
    }
    return oss.str();
}

const char* tensorElementTypeName(ONNXTensorElementDataType type)
{
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "float16";
    default:
        return "unsupported";
    }
}

template <typename ReadValue>
std::vector<Detection> postprocessDetections(ReadValue read,
                                             int numClasses,
                                             float confThresh,
                                             float nmsThresh,
                                             const LabelProvider* labels,
                                             int anchors,
                                             const cv::Size& origSize,
                                             float scale,
                                             int padLeft,
                                             int padTop,
                                             std::vector<cv::Rect>& boxes,
                                             std::vector<float>& scores,
                                             std::vector<int>& classIds,
                                             std::vector<int>& nmsIndices)
{
    boxes.clear();
    scores.clear();
    classIds.clear();
    nmsIndices.clear();
    boxes.reserve(256);
    scores.reserve(256);
    classIds.reserve(256);

    for (int a = 0; a < anchors; ++a) {
        float maxScore = -1.0f;
        int bestCls = 0;
        for (int c = 0; c < numClasses; ++c) {
            float s = read(4 + c, a);
            if (s > maxScore) {
                maxScore = s;
                bestCls = c;
            }
        }
        if (maxScore < confThresh) continue;

        const float cx = read(0, a);
        const float cy = read(1, a);
        const float bw = read(2, a);
        const float bh = read(3, a);

        float x1 = (cx - bw * 0.5f - padLeft) / scale;
        float y1 = (cy - bh * 0.5f - padTop) / scale;
        const float rw = bw / scale;
        const float rh = bh / scale;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(origSize.width));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(origSize.height));
        const float x2 = std::clamp(x1 + rw, 0.0f, static_cast<float>(origSize.width));
        const float y2 = std::clamp(y1 + rh, 0.0f, static_cast<float>(origSize.height));

        if (x2 <= x1 || y2 <= y1) continue;

        boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        scores.push_back(maxScore);
        classIds.push_back(bestCls);
    }

    cv::dnn::NMSBoxes(boxes, scores, confThresh, nmsThresh, nmsIndices);

    std::vector<Detection> result;
    result.reserve(nmsIndices.size());
    for (int idx : nmsIndices) {
        Detection d;
        d.bbox = {static_cast<float>(boxes[idx].x),
                  static_cast<float>(boxes[idx].y),
                  static_cast<float>(boxes[idx].width),
                  static_cast<float>(boxes[idx].height)};
        d.classId = classIds[idx];
        d.score = scores[idx];
        d.label = labels ? labels->getLabel(classIds[idx]) : std::string();
        result.push_back(std::move(d));
    }
    return result;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
YoloOnnxDetector::YoloOnnxDetector(const std::string&   modelPath,
                                   int                  inputSize,
                                   float                confThresh,
                                   float                nmsThresh,
                                   const LabelProvider* labels)
    : env_(ORT_LOGGING_LEVEL_WARNING, "XunChaWeiJian")
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
    const auto availableProviders = Ort::GetAvailableProviders();
    const auto availableEpDevices = env_.GetEpDevices();
    std::cout << "[XunChaWeiJian] ONNX Runtime available EPs: "
              << joinProviders(availableProviders) << "\n";
    std::cout << "[XunChaWeiJian] ONNX Runtime available EP devices: "
              << joinEpDeviceNames(availableEpDevices) << "\n";

    bool sessionCreated = false;
    std::string activeEp = "CPU";

    struct EpCandidate {
        const char* label;
        std::vector<std::string> aliases;
    };
    std::vector<EpCandidate> candidates;

#ifdef XCWJ_WITH_CUDA
    candidates.push_back({"CUDA", {"CUDAExecutionProvider", "CUDA"}});
#endif
#ifdef XCWJ_WITH_DIRECTML
#ifdef _WIN32
    candidates.push_back({"DirectML", {"DmlExecutionProvider", "DMLExecutionProvider", "DML"}});
#endif
#endif

    for (const auto& candidate : candidates) {
        if (!hasProvider(availableProviders, candidate.aliases)) {
            continue;
        }

        auto epDevices = selectEpDevices(availableEpDevices, candidate.aliases);
        if (epDevices.empty()) {
            std::cerr << "[XunChaWeiJian] " << candidate.label
                      << " provider reported by ORT but no EP device is available.\n";
            continue;
        }

        Ort::SessionOptions epOpts;
        epOpts.SetIntraOpNumThreads(threads);
        epOpts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        try {
            epOpts.AppendExecutionProvider_V2(
                env_,
                epDevices,
                std::unordered_map<std::string, std::string>{});
            auto wpath = std::filesystem::path(modelPath).wstring();
            session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), epOpts);
            sessionOpts_ = std::move(epOpts);
            activeEp = candidate.label;
            std::cout << "[XunChaWeiJian] " << candidate.label
                      << " Execution Provider enabled.\n";
            sessionCreated = true;
            break;
        } catch (const Ort::Exception& e) {
            std::cerr << "[XunChaWeiJian] " << candidate.label
                      << " EP init failed: " << e.what() << "\n";
        }
    }

#ifndef XCWJ_WITH_CUDA
    if (hasProvider(availableProviders, {"CUDAExecutionProvider"})) {
        std::cerr << "[XunChaWeiJian] CUDA is available in ORT, but this binary was built without WITH_CUDA=ON.\n";
    }
#endif
#ifndef XCWJ_WITH_DIRECTML
    if (hasProvider(availableProviders, {"DmlExecutionProvider", "DMLExecutionProvider", "DML"})) {
        std::cerr << "[XunChaWeiJian] DirectML is available in ORT, but this binary was built without WITH_DIRECTML=ON.\n";
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
        std::cout << "[XunChaWeiJian] Using CPU Execution Provider (" << threads << " threads).\n";
    } else {
        std::cout << "[XunChaWeiJian] Active EP: " << activeEp << "\n";
    }

    // ── Query input / output names ─────────────────────────────────────────
    {
        auto ptr = session_->GetInputNameAllocated(0, allocator_);
        inputName_ = ptr.get();
        auto inputInfo = session_->GetInputTypeInfo(0);
        inputElementType_ = inputInfo.GetTensorTypeAndShapeInfo().GetElementType();
    }
    {
        auto ptr = session_->GetOutputNameAllocated(0, allocator_);
        outputName_ = ptr.get();
        auto outputInfo = session_->GetOutputTypeInfo(0);
        outputElementType_ = outputInfo.GetTensorTypeAndShapeInfo().GetElementType();
    }

    if (inputElementType_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        inputElementType_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        throw std::runtime_error(std::string("unsupported YOLO input tensor type: ") +
                                 tensorElementTypeName(inputElementType_));
    }
    if (outputElementType_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        outputElementType_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        throw std::runtime_error(std::string("unsupported YOLO output tensor type: ") +
                                 tensorElementTypeName(outputElementType_));
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

    std::cout << "[XunChaWeiJian] Model : " << modelPath << "\n"
              << "  Input  : " << inputName_
              << "  [1,3," << inputSize_ << "," << inputSize_ << "]\n"
              << "  Input type : " << tensorElementTypeName(inputElementType_) << "\n"
              << "  Output : " << outputName_
              << "  [1," << (4 + numClasses_) << ",anchors]\n"
              << "  Output type: " << tensorElementTypeName(outputElementType_) << "\n"
              << "  Classes: " << numClasses_ << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Letterbox pre-processing
// ─────────────────────────────────────────────────────────────────────────────
void YoloOnnxDetector::preprocess(const cv::Mat& frame,
                                  float&         outScale,
                                  int&           outPadLeft,
                                  int&           outPadTop)
{
    int srcW = frame.cols, srcH = frame.rows;
    outScale = std::min(static_cast<float>(inputSize_) / srcW,
                        static_cast<float>(inputSize_) / srcH);

    int newW = static_cast<int>(std::round(srcW * outScale));
    int newH = static_cast<int>(std::round(srcH * outScale));
    outPadLeft = (inputSize_ - newW) / 2;
    outPadTop  = (inputSize_ - newH) / 2;

    cv::resize(frame, resized_, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // Fill letterbox background (grey 114, matching YOLOv8 defaults)
    padded_.create(inputSize_, inputSize_, CV_8UC3);
    padded_.setTo(cv::Scalar(114, 114, 114));
    resized_.copyTo(padded_(cv::Rect(outPadLeft, outPadTop, newW, newH)));
}

float* YoloOnnxDetector::fillInputTensorFloat()
{
    const size_t area = static_cast<size_t>(inputSize_) * inputSize_;
    inputFp32_.resize(area * 3);
    float* rPlane = inputFp32_.data();
    float* gPlane = rPlane + area;
    float* bPlane = gPlane + area;
    constexpr float inv255 = 1.0f / 255.0f;

    for (int y = 0; y < inputSize_; ++y) {
        const auto* src = padded_.ptr<unsigned char>(y);
        const size_t rowOffset = static_cast<size_t>(y) * inputSize_;
        for (int x = 0; x < inputSize_; ++x) {
            const auto* px = src + x * 3;
            const size_t i = rowOffset + static_cast<size_t>(x);
            rPlane[i] = static_cast<float>(px[2]) * inv255;
            gPlane[i] = static_cast<float>(px[1]) * inv255;
            bPlane[i] = static_cast<float>(px[0]) * inv255;
        }
    }
    return inputFp32_.data();
}

Ort::Float16_t* YoloOnnxDetector::fillInputTensorFloat16()
{
    const size_t area = static_cast<size_t>(inputSize_) * inputSize_;
    inputFp16_.resize(area * 3);
    Ort::Float16_t* rPlane = inputFp16_.data();
    Ort::Float16_t* gPlane = rPlane + area;
    Ort::Float16_t* bPlane = gPlane + area;
    constexpr float inv255 = 1.0f / 255.0f;

    for (int y = 0; y < inputSize_; ++y) {
        const auto* src = padded_.ptr<unsigned char>(y);
        const size_t rowOffset = static_cast<size_t>(y) * inputSize_;
        for (int x = 0; x < inputSize_; ++x) {
            const auto* px = src + x * 3;
            const size_t i = rowOffset + static_cast<size_t>(x);
            rPlane[i] = Ort::Float16_t(static_cast<float>(px[2]) * inv255);
            gPlane[i] = Ort::Float16_t(static_cast<float>(px[1]) * inv255);
            bPlane[i] = Ort::Float16_t(static_cast<float>(px[0]) * inv255);
        }
    }
    return inputFp16_.data();
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
                              int             padTop)
{
    return postprocessDetections(
        [data, anchors](int row, int col) {
            return data[row * anchors + col];
        },
        numClasses_, confThresh_, nmsThresh_, labels_,
        anchors, origSize, scale, padLeft, padTop,
        boxes_, scores_, classIds_, nmsIndices_);
}

std::vector<Detection>
YoloOnnxDetector::postprocess(const Ort::Float16_t* data,
                              int                   anchors,
                              const cv::Size&       origSize,
                              float                 scale,
                              int                   padLeft,
                              int                   padTop)
{
    return postprocessDetections(
        [data, anchors](int row, int col) {
            return data[row * anchors + col].ToFloat();
        },
        numClasses_, confThresh_, nmsThresh_, labels_,
        anchors, origSize, scale, padLeft, padTop,
        boxes_, scores_, classIds_, nmsIndices_);
}

// ─────────────────────────────────────────────────────────────────────────────
// detect() — public entry point
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> YoloOnnxDetector::detect(const cv::Mat& frame)
{
    lastPerf_ = {};
    float scale;
    int   padLeft, padTop;
    auto stageStart = Clock::now();
    preprocess(frame, scale, padLeft, padTop);
    lastPerf_.preprocessMs = elapsedMs(stageStart);

    // Build input tensor (blob data ownership stays with cv::Mat)
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const size_t tensorSize = static_cast<size_t>(1) * 3 * inputSize_ * inputSize_;
    Ort::Value inputTensor{nullptr};
    stageStart = Clock::now();
    if (inputElementType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        auto* inputFp16 = fillInputTensorFloat16();
        lastPerf_.inputCastMs = elapsedMs(stageStart);
        inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            memInfo,
            inputFp16,
            tensorSize,
            inputShape_.data(),
            inputShape_.size());
    } else {
        auto* inputFp32 = fillInputTensorFloat();
        lastPerf_.inputCastMs = elapsedMs(stageStart);
        inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            inputFp32,
            tensorSize,
            inputShape_.data(),
            inputShape_.size());
    }

    const char* inputNames[]  = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};

    stageStart = Clock::now();
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 inputNames,  &inputTensor, 1,
                                 outputNames, 1);
    lastPerf_.ortRunMs = elapsedMs(stageStart);

    // Output shape: [1, 4+nc, anchors]
    auto   outputInfo = outputs[0].GetTensorTypeAndShapeInfo();
    auto   shape      = outputInfo.GetShape();
    int    anchors    = static_cast<int>(shape[2]);
    const auto outputElementType = outputInfo.GetElementType();
    stageStart = Clock::now();
    if (outputElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto* outputFp16 = outputs[0].GetTensorData<Ort::Float16_t>();
        lastPerf_.outputCastMs = 0.0;
        auto result = postprocess(outputFp16, anchors, frame.size(), scale, padLeft, padTop);
        lastPerf_.postprocessMs = elapsedMs(stageStart);
        return result;
    } else if (outputElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const auto* data = outputs[0].GetTensorData<float>();
        lastPerf_.outputCastMs = 0.0;
        auto result = postprocess(data, anchors, frame.size(), scale, padLeft, padTop);
        lastPerf_.postprocessMs = elapsedMs(stageStart);
        return result;
    } else {
        throw std::runtime_error(std::string("unsupported YOLO output tensor type: ") +
                                 tensorElementTypeName(outputElementType));
    }
}

} // namespace xcwj

