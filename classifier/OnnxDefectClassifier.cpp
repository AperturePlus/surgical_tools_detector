#include "classifier/OnnxDefectClassifier.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

namespace xcwj {

namespace {

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

cv::Rect clippedRect(const BBox& bbox, const cv::Size& frameSize)
{
    cv::Rect rect(static_cast<int>(std::round(bbox.x)),
                  static_cast<int>(std::round(bbox.y)),
                  static_cast<int>(std::round(bbox.w)),
                  static_cast<int>(std::round(bbox.h)));
    rect &= cv::Rect(0, 0, frameSize.width, frameSize.height);
    return rect;
}

std::pair<float, float> softmax2(float a, float b)
{
    const float m = std::max(a, b);
    const float e0 = std::exp(a - m);
    const float e1 = std::exp(b - m);
    const float sum = e0 + e1;
    return {e0 / sum, e1 / sum};
}

} // namespace

OnnxDefectClassifier::OnnxDefectClassifier(const std::string& modelPath,
                                           int                inputSize,
                                           int                batchSize,
                                           float              defectThresh)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SGTDefectClassifier")
    , inputSize_(inputSize)
    , batchSize_(batchSize)
    , defectThresh_(std::clamp(defectThresh, 0.0f, 1.0f))
{
    int threads = static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency() / 2));
    sessionOpts_.SetIntraOpNumThreads(threads);
    sessionOpts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    const auto availableProviders = Ort::GetAvailableProviders();
    const auto availableEpDevices = env_.GetEpDevices();
    std::cout << "[DefectClassifier] ONNX Runtime available EPs: "
              << joinProviders(availableProviders) << "\n";

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
        if (!hasProvider(availableProviders, candidate.aliases)) continue;

        auto epDevices = selectEpDevices(availableEpDevices, candidate.aliases);
        if (epDevices.empty()) continue;

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
            sessionCreated = true;
            break;
        } catch (const Ort::Exception& e) {
            std::cerr << "[DefectClassifier] " << candidate.label
                      << " EP init failed: " << e.what() << "\n";
        }
    }

    if (!sessionCreated) {
#ifdef _WIN32
        auto wpath = std::filesystem::path(modelPath).wstring();
        session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), sessionOpts_);
#else
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOpts_);
#endif
    }

    {
        auto ptr = session_->GetInputNameAllocated(0, allocator_);
        inputName_ = ptr.get();
    }
    {
        auto ptr = session_->GetOutputNameAllocated(0, allocator_);
        outputName_ = ptr.get();
    }

    auto inputInfo = session_->GetInputTypeInfo(0);
    auto shape = inputInfo.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 4) {
        if (shape[0] > 0) batchSize_ = static_cast<int>(shape[0]);
        if (shape[2] > 0) inputSize_ = static_cast<int>(shape[2]);
    }
    inputShape_ = {batchSize_, 3, inputSize_, inputSize_};

    auto outputInfo = session_->GetOutputTypeInfo(0);
    auto outputShape = outputInfo.GetTensorTypeAndShapeInfo().GetShape();
    if (outputShape.size() != 2 || outputShape[1] != 2) {
        throw std::runtime_error("defect classifier output must be [batch,2]");
    }

    std::cout << "[DefectClassifier] Model : " << modelPath << "\n"
              << "  Input  : " << inputName_
              << "  [" << batchSize_ << ",3," << inputSize_ << "," << inputSize_ << "]\n"
              << "  Output : " << outputName_
              << "  [" << batchSize_ << ",2]\n"
              << "  Active EP: " << activeEp << "\n";
}

void OnnxDefectClassifier::setDefectThresh(float thresh)
{
    defectThresh_ = std::clamp(thresh, 0.0f, 1.0f);
}

void OnnxDefectClassifier::fillInputTensor(const cv::Mat& crop, float* dst) const
{
    int srcW = crop.cols;
    int srcH = crop.rows;
    float scale = std::min(static_cast<float>(inputSize_) / srcW,
                           static_cast<float>(inputSize_) / srcH);
    int newW = std::max(1, static_cast<int>(std::round(srcW * scale)));
    int newH = std::max(1, static_cast<int>(std::round(srcH * scale)));
    int padLeft = (inputSize_ - newW) / 2;
    int padTop = (inputSize_ - newH) / 2;

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(inputSize_, inputSize_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(padLeft, padTop, newW, newH)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    static constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float stdv[3] = {0.229f, 0.224f, 0.225f};
    const int area = inputSize_ * inputSize_;

    for (int y = 0; y < inputSize_; ++y) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inputSize_; ++x) {
            const int idx = y * inputSize_ + x;
            for (int c = 0; c < 3; ++c) {
                float v = static_cast<float>(row[x][c]) / 255.0f;
                dst[c * area + idx] = (v - mean[c]) / stdv[c];
            }
        }
    }
}

std::vector<DefectResult>
OnnxDefectClassifier::classify(const cv::Mat&                frame,
                               const std::vector<Detection>& tools)
{
    std::vector<std::pair<int, cv::Rect>> candidates;
    candidates.reserve(tools.size());
    for (int i = 0; i < static_cast<int>(tools.size()); ++i) {
        cv::Rect rect = clippedRect(tools[i].bbox, frame.size());
        if (rect.width > 1 && rect.height > 1) {
            candidates.emplace_back(i, rect);
        }
    }
    if (candidates.empty()) return {};

    std::vector<DefectResult> results;
    results.reserve(candidates.size());

    const size_t imageFloats = static_cast<size_t>(3) * inputSize_ * inputSize_;
    const size_t batchFloats = static_cast<size_t>(batchSize_) * imageFloats;

    for (size_t offset = 0; offset < candidates.size(); offset += batchSize_) {
        std::vector<float> inputData(batchFloats, 0.0f);
        const int valid = static_cast<int>(
            std::min<size_t>(batchSize_, candidates.size() - offset));

        for (int b = 0; b < valid; ++b) {
            const cv::Rect rect = candidates[offset + b].second;
            cv::Mat crop = frame(rect).clone();
            fillInputTensor(crop, inputData.data() + static_cast<size_t>(b) * imageFloats);
        }

        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            inputData.data(),
            inputData.size(),
            inputShape_.data(),
            inputShape_.size());

        const char* inputNames[] = {inputName_.c_str()};
        const char* outputNames[] = {outputName_.c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     inputNames, &inputTensor, 1,
                                     outputNames, 1);

        const float* logits = outputs[0].GetTensorData<float>();
        for (int b = 0; b < valid; ++b) {
            auto scores = softmax2(logits[b * 2 + 0], logits[b * 2 + 1]);
            int toolIndex = candidates[offset + b].first;

            DefectResult r;
            r.toolIndex = toolIndex;
            r.bbox = tools[toolIndex].bbox;
            r.normalScore = scores.first;
            r.defectScore = scores.second;
            r.defective = r.defectScore >= defectThresh_;
            results.push_back(r);
        }
    }

    return results;
}

} // namespace xcwj
