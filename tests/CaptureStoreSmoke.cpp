#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>

#include "core/CaptureStore.h"
#include "core/Renderer.h"

namespace fs = std::filesystem;

static std::string readText(const fs::path& path)
{
    std::ifstream f(path);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

int main()
{
    fs::path root = fs::temp_directory_path() / "sgtdetector_capture_store_smoke";
    fs::remove_all(root);

    sgt::DetectionFrameResult result;
    result.rawFrame = cv::Mat(32, 48, CV_8UC3, cv::Scalar(20, 40, 60));
    result.annotatedFrame = cv::Mat(32, 48, CV_8UC3, cv::Scalar(60, 40, 20));
    result.activeModes = sgt::MODE_TOOL | sgt::MODE_DEFECT;
    result.thresholds = {0.25f, 0.30f, 0.55f};

    sgt::Detection det;
    det.classId = 1;
    det.label = "骨钳";
    det.score = 0.91f;
    det.bbox = {1, 2, 20, 12};
    result.toolDetections.push_back(det);

    sgt::DefectResult defect;
    defect.toolIndex = 0;
    defect.bbox = det.bbox;
    defect.normalScore = 0.1f;
    defect.defectScore = 0.9f;
    defect.defective = true;
    result.defectResults.push_back(defect);

    sgt::CaptureStore store(root);
    auto record = store.saveCapture(result, {"tool.onnx", "grasp.onnx", "defect.onnx"});

    bool ok = fs::exists(record.rawImagePath)
        && fs::exists(record.annotatedImagePath)
        && fs::exists(record.jsonPath)
        && fs::exists(root / "index.json");

    std::string json = readText(record.jsonPath);
    ok = ok
        && json.find("\"schemaVersion\":1") != std::string::npos
        && json.find("\"label\":\"骨钳\"") != std::string::npos
        && json.find("\"defectCount\":1") != std::string::npos;

    sgt::CaptureStore loaded(root);
    ok = ok && loaded.records().size() == 1;

    fs::remove_all(root);
    if (!ok) {
        std::cerr << "CaptureStore smoke test failed\n";
        return 1;
    }
    return 0;
}
