/// SGTDetector — YOLOv8 real-time dental instrument detection
///
/// Usage:
///   SGTDetector [camera_id]         (default: 0)
///   SGTDetector [camera_id] [model] (override model path)
///
/// Keyboard shortcuts:
///   +  / =    raise confidence threshold +0.05
///   -  / _    lower confidence threshold -0.05
///   s  / S    save screenshot (screenshot_YYYYMMDD_HHMMSS.jpg)
///   q  / ESC  quit

#include <iostream>
#include <memory>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdlib>   // std::atoi

#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/DetectorBackend.h"
#include "core/LabelProvider.h"
#include "core/Renderer.h"

#include "detector/YoloOnnxDetector.h"
#include "label/DictLabelProvider.h"
#include "render/OpenCVFontRenderer.h"
#include "render/OpenCVRenderer.h"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Class names — must match the order in m4.yaml "names" field exactly.
// ─────────────────────────────────────────────────────────────────────────────
static const std::vector<std::string> CLASS_NAMES = {
    "cefangkaikouqi",        //  0
    "guqian",                //  1
    "gujian",                //  2
    "gudao",                 //  3
    "xianjian",              //  4
    "yating",                //  5
    "bayaqian",              //  6
    "guchui",                //  7
    "shaozi",                //  8
    "7haodaobing",           //  9
    "3haodaobing",           // 10
    "dagumo",                // 11
    "gucuo",                 // 12
    "zhenchi",               // 13
    "zhizhixueqian",         // 14
    "wanzhixueqian",         // 15
    "paqian",                // 16
    "kekeqian",              // 17
    "xiaoduqian",            // 18
    "xichiqian",             // 19
    "zuzhinie",              // 20
    "dangou",                // 21
    "pingnie",               // 22
    "zuzhijian",             // 23
    "zhijiaoqian",           // 24
    "jiazhuangxianlagou",    // 25
    "huanqian",              // 26
    "xiaogumo",              // 27
    "sichilagou",            // 28
    "guachi",                // 29
    "gangsijian",            // 30
    "xiaolagou",             // 31
    "eliekaikouqi",          // 32
    "yasheban",              // 33
    "eliejian",              // 34
    "xueguanshenjingboliqi1",// 35
    "xueguanshenjingboliqi2",// 36
    "yingeboliqi",           // 37
};

// ─────────────────────────────────────────────────────────────────────────────
// Save screenshot to the working directory; return the file path.
// ─────────────────────────────────────────────────────────────────────────────
static std::string saveScreenshot(const cv::Mat& frame)
{
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << "screenshot_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".jpg";
    const std::string path = oss.str();
    cv::imwrite(path, frame);
    return path;
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolve an asset path: check argv[0] directory first, then CWD.
// ─────────────────────────────────────────────────────────────────────────────
static std::string resolveAsset(const fs::path& exeDir, const std::string& name)
{
    fs::path candidate = exeDir / name;
    if (fs::exists(candidate)) return candidate.string();
    if (fs::exists(fs::path(name))) return name;
    return name; // let downstream code emit the error
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── Parse CLI ──────────────────────────────────────────────────────────
    int cameraId = 0;
    if (argc > 1) cameraId = std::atoi(argv[1]);

    auto exeDir = fs::path(argv[0]).parent_path();
    std::string modelPath = resolveAsset(exeDir, "best.onnx");
    std::string dictPath  = resolveAsset(exeDir, "labels.dict");
    if (argc > 2) modelPath = argv[2];

    // ── Assemble modules via dependency injection ──────────────────────────
    std::unique_ptr<sgt::LabelProvider>   labels;
    std::unique_ptr<sgt::DetectorBackend> detector;
    std::unique_ptr<sgt::Renderer>        renderer;

    try {
        labels = std::make_unique<sgt::DictLabelProvider>(dictPath, CLASS_NAMES);

        detector = std::make_unique<sgt::YoloOnnxDetector>(
            modelPath,
            /*inputSize*/  960,
            /*confThresh*/ 0.25f,
            /*nmsThresh*/  0.45f,
            labels.get());

        // ── Swap these two lines to switch to a different renderer/font ────
        auto font = std::make_unique<sgt::OpenCVFontRenderer>();
        renderer  = std::make_unique<sgt::OpenCVRenderer>(std::move(font));
        // ─────────────────────────────────────────────────────────────────

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Initialization failed: " << e.what() << "\n";
        return 1;
    }

    // ── Open camera ────────────────────────────────────────────────────────
    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened()) {
        std::cerr << "[FATAL] Cannot open camera " << cameraId << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "[SGTDetector] Camera " << cameraId
              << " opened. Press q/ESC to quit.\n";

    // ── Main loop ──────────────────────────────────────────────────────────
    cv::Mat frame;
    auto    lastTime   = std::chrono::steady_clock::now();
    float   fps        = 0.0f;
    float   confThresh = detector->getConfThresh();

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[WARN] Empty frame, skipping...\n";
            continue;
        }

        // ── Inference ──────────────────────────────────────────────────────
        std::vector<sgt::Detection> dets;
        try {
            dets = detector->detect(frame);
        } catch (const std::exception& e) {
            std::cerr << "[WARN] Inference error: " << e.what() << "\n";
        }

        // ── FPS (exponential moving average, weight = 0.1) ─────────────────
        {
            auto   now     = std::chrono::steady_clock::now();
            float  elapsed = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            float  instant = (elapsed > 0.0f) ? (1.0f / elapsed) : 0.0f;
            fps = fps * 0.9f + instant * 0.1f;
        }

        // ── Render ─────────────────────────────────────────────────────────
        renderer->drawDetections(frame, dets);
        renderer->drawHUD(frame, {fps, confThresh, static_cast<int>(dets.size())});

        // ── Display + keyboard ─────────────────────────────────────────────
        int key = renderer->showFrame(frame) & 0xFF;

        if (key == 27 || key == 'q' || key == 'Q') break;

        if (key == '+' || key == '=') {
            confThresh = std::min(confThresh + 0.05f, 0.95f);
            detector->setConfThresh(confThresh);
            std::cout << "[SGTDetector] Conf threshold -> " << confThresh << "\n";
        }
        if (key == '-' || key == '_') {
            confThresh = std::max(confThresh - 0.05f, 0.05f);
            detector->setConfThresh(confThresh);
            std::cout << "[SGTDetector] Conf threshold -> " << confThresh << "\n";
        }
        if (key == 's' || key == 'S') {
            std::string path = saveScreenshot(frame);
            std::cout << "[SGTDetector] Screenshot saved: " << path << "\n";
            renderer->onScreenshot(path);
        }
    }

    cap.release();
    std::cout << "[SGTDetector] Exited.\n";
    return 0;
}
