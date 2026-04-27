/// SGTDetector - real-time surgical tool, grasp, and defect detection.
///
/// Usage:
///   SGTDetector [camera_id] [--mode tool|grasp|defect] [--tool-model p] [--grasp-model p] [--defect-model p]
///
/// Keyboard shortcuts:
///   1/2/3     toggle mode (tool/grasp/defect)
///   +/-       raise/lower active threshold
///   c/C       capture frame with full detection
///   s/S       save screenshot
///   q/ESC     quit

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "classifier/OnnxDefectClassifier.h"
#include "core/DetectorBackend.h"
#include "core/LabelProvider.h"
#include "core/Renderer.h"
#include "detector/YoloOnnxDetector.h"
#include "label/DictLabelProvider.h"
#include "render/OpenCVFontRenderer.h"
#include "render/OpenCVRenderer.h"

namespace fs = std::filesystem;

// ── Constants ────────────────────────────────────────────────────────────────

static constexpr const char* DEFAULT_TOOL_MODEL  = "surgical_tool_detector_yolov8_640.onnx";
static constexpr const char* DEFAULT_GRASP_MODEL = "grasp_state_detector_yolov8_640.onnx";
static constexpr const char* DEFAULT_DEFECT_MODEL = "tool_defect_classifier_resnet_512_b4.onnx";
static constexpr const char* WINDOW_NAME = "SGTDetector";

static const std::vector<std::string> CLASS_NAMES = {
    "cefangkaikouqi","guqian","gujian","gudao","xianjian","yating",
    "bayaqian","guchui","shaozi","7haodaobing","3haodaobing","dagumo",
    "gucuo","zhenchi","zhizhixueqian","wanzhixueqian","paqian","kekeqian",
    "xiaoduqian","xichiqian","zuzhinie","dangou","pingnie","zuzhijian",
    "zhijiaoqian","jiazhuangxianlagou","huanqian","xiaogumo","sichilagou",
    "guachi","gangsijian","xiaolagou","eliekaikouqi","yasheban","eliejian",
    "xueguanshenjingboliqi1","xueguanshenjingboliqi2","yingeboliqi",
};

static const std::vector<std::string> GRASP_CLASS_NAMES = {"work", "grasp"};

// ── CLI parsing ──────────────────────────────────────────────────────────────

struct AppOptions {
    int         cameraId    = 0;
    uint8_t     modeMask    = sgt::MODE_TOOL;
    std::string toolModel   = DEFAULT_TOOL_MODEL;
    std::string graspModel  = DEFAULT_GRASP_MODEL;
    std::string defectModel = DEFAULT_DEFECT_MODEL;
};

static uint8_t parseMode(const std::string& raw) {
    std::string m = raw;
    std::transform(m.begin(), m.end(), m.begin(), ::tolower);
    if (m == "tool")   return sgt::MODE_TOOL;
    if (m == "grasp")  return sgt::MODE_GRASP;
    if (m == "defect") return sgt::MODE_DEFECT;
    throw std::runtime_error("invalid mode: " + raw);
}

static std::string requireVal(int& i, int argc, char* argv[], const std::string& opt) {
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + opt);
    return argv[++i];
}

static AppOptions parseArgs(int argc, char* argv[]) {
    AppOptions opts;
    bool camSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: SGTDetector [camera_id] [--mode tool|grasp|defect] ...\n";
            std::exit(0);
        } else if (arg == "--mode")        { opts.modeMask = parseMode(requireVal(i, argc, argv, arg)); }
        else if (arg == "--tool-model")    { opts.toolModel = requireVal(i, argc, argv, arg); }
        else if (arg == "--grasp-model")   { opts.graspModel = requireVal(i, argc, argv, arg); }
        else if (arg == "--defect-model")  { opts.defectModel = requireVal(i, argc, argv, arg); }
        else if (arg.rfind("--", 0) == 0)  { throw std::runtime_error("unknown option: " + arg); }
        else if (!camSet) { opts.cameraId = std::atoi(arg.c_str()); camSet = true; }
        else { throw std::runtime_error("unexpected argument: " + arg); }
    }
    return opts;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string resolveAsset(const fs::path& exeDir, const std::string& name) {
    if (fs::path(name).is_absolute() && fs::exists(name)) return name;
    for (auto& p : {exeDir / name, exeDir / "models" / name,
                    fs::path(name), fs::path("assets/models") / name})
        if (fs::exists(p)) return p.string();
    return name;
}

static std::string timestampFilename(const char* prefix, const char* ext) {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{}; 
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << prefix << std::put_time(&tm, "%Y%m%d_%H%M%S") << ext;
    return oss.str();
}

static int countDefects(const std::vector<sgt::DefectResult>& v) {
    return static_cast<int>(std::count_if(v.begin(), v.end(),
        [](const sgt::DefectResult& r) { return r.defective; }));
}

// ── Mode button overlay ─────────────────────────────────────────────────────

struct ModeButtonState {
    struct Hit { uint8_t bit; cv::Rect rect; };
    std::vector<Hit> hits;
    uint8_t pendingToggle = 0;
};

static void onModeMouse(int event, int x, int y, int, void* ud) {
    if (event != cv::EVENT_LBUTTONUP || !ud) return;
    auto* s = static_cast<ModeButtonState*>(ud);
    for (auto& h : s->hits)
        if (h.rect.contains(cv::Point(x, y))) { s->pendingToggle = h.bit; break; }
}

static void drawModeButtons(cv::Mat& frame, uint8_t mask, ModeButtonState& st) {
    struct Spec { uint8_t bit; const char* label; };
    static constexpr std::array<Spec, 3> BTNS = {{{sgt::MODE_TOOL,"Tool"},{sgt::MODE_GRASP,"Grasp"},{sgt::MODE_DEFECT,"Defect"}}};
    constexpr int W = 88, H = 30, gap = 8, margin = 12;
    int startX = frame.cols - margin - 3 * W - 2 * gap;
    st.hits.clear();
    for (size_t i = 0; i < BTNS.size(); ++i) {
        cv::Rect r(startX + int(i) * (W + gap), margin, W, H);
        st.hits.push_back({BTNS[i].bit, r});
        bool on = mask & BTNS[i].bit;
        cv::rectangle(frame, r, on ? cv::Scalar(50,160,70) : cv::Scalar(55,55,55), cv::FILLED);
        cv::rectangle(frame, r, on ? cv::Scalar(120,255,150) : cv::Scalar(180,180,180), 1);
        int bl = 0;
        auto sz = cv::getTextSize(BTNS[i].label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &bl);
        cv::putText(frame, BTNS[i].label,
            {r.x + (r.width - sz.width)/2, r.y + (r.height + sz.height)/2 - 2},
            cv::FONT_HERSHEY_SIMPLEX, 0.5, {240,240,240}, 1, cv::LINE_AA);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    AppOptions opts;
    try { opts = parseArgs(argc, argv); }
    catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n"; return 1;
    }

    auto exeDir = fs::path(argv[0]).parent_path();
    std::string toolPath   = resolveAsset(exeDir, opts.toolModel);
    std::string graspPath  = resolveAsset(exeDir, opts.graspModel);
    std::string defectPath = resolveAsset(exeDir, opts.defectModel);
    std::string dictPath   = resolveAsset(exeDir, "labels.dict");

    // Lazy-loaded models
    std::unique_ptr<sgt::LabelProvider>        toolLabels, graspLabels;
    std::unique_ptr<sgt::DetectorBackend>      toolDet, graspDet;
    std::unique_ptr<sgt::OnnxDefectClassifier> defectCls;
    uint8_t activeMask = opts.modeMask;

    auto ensureLoaded = [&](uint8_t bit) -> bool {
        try {
            if (bit == sgt::MODE_TOOL && !toolDet) {
                if (!toolLabels) toolLabels = std::make_unique<sgt::DictLabelProvider>(dictPath, CLASS_NAMES);
                toolDet = std::make_unique<sgt::YoloOnnxDetector>(toolPath, 640, 0.25f, 0.45f, toolLabels.get());
            }
            if (bit == sgt::MODE_GRASP && !graspDet) {
                if (!graspLabels) graspLabels = std::make_unique<sgt::DictLabelProvider>("", GRASP_CLASS_NAMES);
                graspDet = std::make_unique<sgt::YoloOnnxDetector>(graspPath, 640, 0.25f, 0.45f, graspLabels.get());
            }
            if (bit == sgt::MODE_DEFECT && !defectCls)
                defectCls = std::make_unique<sgt::OnnxDefectClassifier>(defectPath, 512, 4, 0.50f);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[FATAL] Load failed: " << e.what() << "\n"; return false;
        }
    };

    // Init renderer
    std::unique_ptr<sgt::Renderer> renderer;
    try {
        renderer = std::make_unique<sgt::OpenCVRenderer>(
            std::make_unique<sgt::OpenCVFontRenderer>(), WINDOW_NAME);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n"; return 1;
    }

    // Load initial modes
    for (uint8_t b : {sgt::MODE_TOOL, sgt::MODE_GRASP, sgt::MODE_DEFECT})
        if ((activeMask & b) && !ensureLoaded(b)) return 1;

    cv::VideoCapture cap(opts.cameraId);
    if (!cap.isOpened()) { std::cerr << "[FATAL] Cannot open camera\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "[SGTDetector] Camera " << opts.cameraId << " opened.\n";

    ModeButtonState btnState;
    cv::setMouseCallback(WINDOW_NAME, onModeMouse, &btnState);

    auto toggleMode = [&](uint8_t bit) {
        uint8_t next = activeMask ^ bit;
        if (next == 0) return; // keep at least one
        if ((next & bit) && !ensureLoaded(bit)) return;
        activeMask = next;
    };

    cv::Mat frame;
    auto lastTime = std::chrono::steady_clock::now();
    float fps = 0.0f;

    while (true) {
        if (!cap.read(frame) || frame.empty()) continue;

        // ── Inference ────────────────────────────────────────────────────
        std::vector<sgt::Detection>    toolDets, graspDets;
        std::vector<sgt::DefectResult> defectRes;

        try {
            if ((activeMask & sgt::MODE_TOOL) && toolDet)
                toolDets = toolDet->detect(frame);
            if ((activeMask & sgt::MODE_GRASP) && graspDet)
                graspDets = graspDet->detect(frame);
            if ((activeMask & sgt::MODE_DEFECT) && defectCls) {
                sgt::Detection full;
                full.bbox = {0, 0, float(frame.cols), float(frame.rows)};
                defectRes = defectCls->classify(frame, {full});
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARN] Inference: " << e.what() << "\n";
        }

        // ── FPS ──────────────────────────────────────────────────────────
        {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            fps = fps * 0.9f + (dt > 0 ? 1.0f / dt : 0) * 0.1f;
        }

        // ── Render ───────────────────────────────────────────────────────
        if (!toolDets.empty())  renderer->drawDetections(frame, toolDets);
        if (!graspDets.empty()) renderer->drawDetections(frame, graspDets);
        if (!defectRes.empty()) renderer->drawDefects(frame, defectRes);

        sgt::HUDData hud;
        hud.activeModes    = activeMask;
        hud.fps            = fps;
        hud.toolConfThresh = toolDet ? toolDet->getConfThresh() : 0.25f;
        hud.graspConfThresh= graspDet ? graspDet->getConfThresh() : 0.25f;
        hud.defectThresh   = defectCls ? defectCls->getDefectThresh() : 0.50f;
        hud.toolDetections = int(toolDets.size());
        hud.graspDetections= int(graspDets.size());
        hud.defects        = countDefects(defectRes);
        renderer->drawHUD(frame, hud);
        drawModeButtons(frame, activeMask, btnState);

        int key = renderer->showFrame(frame) & 0xFF;

        // ── Button click ─────────────────────────────────────────────────
        if (btnState.pendingToggle) {
            toggleMode(btnState.pendingToggle);
            btnState.pendingToggle = 0;
        }

        if (key == 27 || key == 'q' || key == 'Q') break;
        if (key == '1') toggleMode(sgt::MODE_TOOL);
        if (key == '2') toggleMode(sgt::MODE_GRASP);
        if (key == '3') toggleMode(sgt::MODE_DEFECT);

        // ── Threshold adjust (applies to first active detector) ──────────
        if (key == '+' || key == '=' || key == '-' || key == '_') {
            float delta = (key == '+' || key == '=') ? 0.05f : -0.05f;
            if ((activeMask & sgt::MODE_TOOL) && toolDet)
                toolDet->setConfThresh(std::clamp(toolDet->getConfThresh() + delta, 0.05f, 0.95f));
            else if ((activeMask & sgt::MODE_GRASP) && graspDet)
                graspDet->setConfThresh(std::clamp(graspDet->getConfThresh() + delta, 0.05f, 0.95f));
            else if ((activeMask & sgt::MODE_DEFECT) && defectCls)
                defectCls->setDefectThresh(std::clamp(defectCls->getDefectThresh() + delta, 0.05f, 0.95f));
        }

        // ── Screenshot ───────────────────────────────────────────────────
        if (key == 's' || key == 'S') {
            auto path = timestampFilename("screenshot_", ".jpg");
            cv::imwrite(path, frame);
            std::cout << "[SGTDetector] Screenshot: " << path << "\n";
            renderer->onScreenshot(path);
        }

        // ── Capture mode (full annotated frame) ──────────────────────────
        if (key == 'c' || key == 'C') {
            auto path = timestampFilename("capture_", ".jpg");
            cv::imwrite(path, frame);
            std::cout << "[SGTDetector] Capture: " << path
                      << " | T:" << toolDets.size()
                      << " G:" << graspDets.size()
                      << " D:" << countDefects(defectRes) << "\n";
            renderer->onScreenshot(path);
        }
    }

    cap.release();
    std::cout << "[SGTDetector] Exited.\n";
    return 0;
}
