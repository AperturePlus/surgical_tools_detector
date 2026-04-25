/// SGTDetector - real-time surgical tool, grasp, and defect detection.
///
/// Usage:
///   SGTDetector [camera_id] [tool_model]
///   SGTDetector [camera_id] --mode <tool|grasp|defect> [--tool-model <path>] [--grasp-model <path>] [--defect-model <path>]
///
/// Keyboard shortcuts:
///   1/2/3     switch mode (tool/grasp/defect)
///   +  / =    raise active threshold +0.05
///   -  / _    lower active threshold -0.05
///   mouse     click top-right mode buttons
///   s  / S    save screenshot (screenshot_YYYYMMDD_HHMMSS.jpg)
///   q  / ESC  quit

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

static constexpr const char* DEFAULT_TOOL_MODEL =
    "surgical_tool_detector_yolov8_640.onnx";
static constexpr const char* DEFAULT_GRASP_MODEL =
    "grasp_state_detector_yolov8_640.onnx";
static constexpr const char* DEFAULT_DEFECT_MODEL =
    "tool_defect_classifier_resnet_512_b4.onnx";
static constexpr const char* WINDOW_NAME = "SGTDetector";

static const std::vector<std::string> CLASS_NAMES = {
    "cefangkaikouqi",
    "guqian",
    "gujian",
    "gudao",
    "xianjian",
    "yating",
    "bayaqian",
    "guchui",
    "shaozi",
    "7haodaobing",
    "3haodaobing",
    "dagumo",
    "gucuo",
    "zhenchi",
    "zhizhixueqian",
    "wanzhixueqian",
    "paqian",
    "kekeqian",
    "xiaoduqian",
    "xichiqian",
    "zuzhinie",
    "dangou",
    "pingnie",
    "zuzhijian",
    "zhijiaoqian",
    "jiazhuangxianlagou",
    "huanqian",
    "xiaogumo",
    "sichilagou",
    "guachi",
    "gangsijian",
    "xiaolagou",
    "eliekaikouqi",
    "yasheban",
    "eliejian",
    "xueguanshenjingboliqi1",
    "xueguanshenjingboliqi2",
    "yingeboliqi",
};

static const std::vector<std::string> GRASP_CLASS_NAMES = {
    "work",
    "grasp",
};

enum class InferenceMode {
    Tool,
    Grasp,
    Defect,
};

static std::string toLowerAscii(std::string text)
{
    std::transform(
        text.begin(), text.end(), text.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

static const char* modeToString(InferenceMode mode)
{
    switch (mode) {
    case InferenceMode::Tool:
        return "tool";
    case InferenceMode::Grasp:
        return "grasp";
    case InferenceMode::Defect:
        return "defect";
    }
    return "tool";
}

static InferenceMode parseMode(const std::string& rawMode)
{
    const std::string mode = toLowerAscii(rawMode);
    if (mode == "tool") return InferenceMode::Tool;
    if (mode == "grasp") return InferenceMode::Grasp;
    if (mode == "defect") return InferenceMode::Defect;
    throw std::runtime_error(
        "invalid mode: " + rawMode + " (expected: tool|grasp|defect)");
}

struct ModeButtonOverlayState {
    std::vector<std::pair<InferenceMode, cv::Rect>> hitRegions;
    InferenceMode pendingMode = InferenceMode::Tool;
    bool hasPendingMode = false;
};

static void onModeOverlayMouse(int event, int x, int y, int, void* userdata)
{
    if (event != cv::EVENT_LBUTTONUP || userdata == nullptr) {
        return;
    }

    auto* state = static_cast<ModeButtonOverlayState*>(userdata);
    const cv::Point clickPoint(x, y);
    for (const auto& [mode, rect] : state->hitRegions) {
        if (rect.contains(clickPoint)) {
            state->pendingMode = mode;
            state->hasPendingMode = true;
            break;
        }
    }
}

static void drawModeButtons(cv::Mat& frame,
                            InferenceMode currentMode,
                            ModeButtonOverlayState& state)
{
    struct ModeButtonSpec {
        InferenceMode mode;
        const char* label;
    };

    static constexpr std::array<ModeButtonSpec, 3> MODE_BUTTONS = {{
        {InferenceMode::Tool, "Tool"},
        {InferenceMode::Grasp, "Grasp"},
        {InferenceMode::Defect, "Defect"},
    }};

    static constexpr int margin = 12;
    static constexpr int buttonWidth = 88;
    static constexpr int buttonHeight = 30;
    static constexpr int gap = 8;

    state.hitRegions.clear();
    const int totalWidth = static_cast<int>(MODE_BUTTONS.size()) * buttonWidth
        + static_cast<int>(MODE_BUTTONS.size() - 1) * gap;
    const int startX = std::max(margin, frame.cols - margin - totalWidth);
    const int startY = margin;

    for (size_t i = 0; i < MODE_BUTTONS.size(); ++i) {
        const auto& button = MODE_BUTTONS[i];
        const cv::Rect rect(
            startX + static_cast<int>(i) * (buttonWidth + gap),
            startY,
            buttonWidth,
            buttonHeight);

        state.hitRegions.emplace_back(button.mode, rect);

        const bool active = (button.mode == currentMode);
        const cv::Scalar fillColor = active
            ? cv::Scalar(50, 160, 70)
            : cv::Scalar(55, 55, 55);
        const cv::Scalar borderColor = active
            ? cv::Scalar(120, 255, 150)
            : cv::Scalar(180, 180, 180);
        cv::rectangle(frame, rect, fillColor, cv::FILLED);
        cv::rectangle(frame, rect, borderColor, 1);

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            button.label,
            cv::FONT_HERSHEY_SIMPLEX,
            0.50,
            1,
            &baseline);
        const cv::Point textPos(
            rect.x + (rect.width - textSize.width) / 2,
            rect.y + (rect.height + textSize.height) / 2 - 2);
        cv::putText(
            frame,
            button.label,
            textPos,
            cv::FONT_HERSHEY_SIMPLEX,
            0.50,
            cv::Scalar(240, 240, 240),
            1,
            cv::LINE_AA);
    }
}

struct AppOptions {
    int         cameraId      = 0;
    InferenceMode mode        = InferenceMode::Tool;
    std::string toolModel     = DEFAULT_TOOL_MODEL;
    std::string graspModel    = DEFAULT_GRASP_MODEL;
    std::string defectModel   = DEFAULT_DEFECT_MODEL;
};

static void printUsage(const char* exe)
{
    std::cout
        << "Usage:\n"
        << "  " << exe << " [camera_id] [tool_model]\n"
        << "  " << exe << " [camera_id] [--mode tool|grasp|defect] [--tool-model path] [--grasp-model path] [--defect-model path]\n"
        << "Options:\n"
        << "  --mode <tool|grasp|defect>  Inference mode (default: tool)\n"
        << "  --tool-model <path>      Surgical tool detector ONNX\n"
        << "  --grasp-model <path>     Grasp/work detector ONNX\n"
        << "  --defect-model <path>    Tool defect classifier ONNX\n"
        << "  --help                   Show this help\n";
}

static std::string requireValue(int& i, int argc, char* argv[], const std::string& opt)
{
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + opt);
    }
    return argv[++i];
}

static AppOptions parseArgs(int argc, char* argv[])
{
    AppOptions opts;
    bool cameraSet = false;
    bool legacyToolModelSet = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--mode") {
            opts.mode = parseMode(requireValue(i, argc, argv, arg));
        } else if (arg == "--tool-model") {
            opts.toolModel = requireValue(i, argc, argv, arg);
        } else if (arg == "--grasp-model") {
            opts.graspModel = requireValue(i, argc, argv, arg);
        } else if (arg == "--defect-model") {
            opts.defectModel = requireValue(i, argc, argv, arg);
        } else if (arg == "--disable-grasp" || arg == "--disable-defect") {
            throw std::runtime_error(
                arg + " is no longer supported; use --mode tool|grasp|defect");
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown option: " + arg);
        } else if (!cameraSet) {
            opts.cameraId = std::atoi(arg.c_str());
            cameraSet = true;
        } else if (!legacyToolModelSet) {
            opts.toolModel = arg;
            legacyToolModelSet = true;
        } else {
            throw std::runtime_error("unexpected positional argument: " + arg);
        }
    }

    // Keep legacy positional model argument usable across new modes.
    if (legacyToolModelSet) {
        if (opts.mode == InferenceMode::Grasp) {
            opts.graspModel = opts.toolModel;
        } else if (opts.mode == InferenceMode::Defect) {
            opts.defectModel = opts.toolModel;
        }
    }

    return opts;
}

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

static std::string resolveAsset(const fs::path& exeDir, const std::string& name)
{
    fs::path raw(name);
    std::vector<fs::path> candidates;

    if (raw.is_absolute()) {
        candidates.push_back(raw);
    } else {
        candidates.push_back(exeDir / raw);
        candidates.push_back(exeDir / "models" / raw);
        candidates.push_back(raw);
        candidates.push_back(fs::path("assets") / "models" / raw);
        candidates.push_back(fs::path("assets") / raw);
    }

    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) return candidate.string();
    }
    return raw.string();
}

static int countDefects(const std::vector<sgt::DefectResult>& defects)
{
    return static_cast<int>(std::count_if(
        defects.begin(), defects.end(),
        [](const sgt::DefectResult& r) { return r.defective; }));
}

int main(int argc, char* argv[])
{
    AppOptions opts;
    try {
        opts = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }

    auto exeDir = fs::path(argv[0]).parent_path();
    std::string toolModelPath = resolveAsset(exeDir, opts.toolModel);
    std::string graspModelPath = resolveAsset(exeDir, opts.graspModel);
    std::string defectModelPath = resolveAsset(exeDir, opts.defectModel);
    std::string dictPath = resolveAsset(exeDir, "labels.dict");

    std::unique_ptr<sgt::LabelProvider>        toolLabels;
    std::unique_ptr<sgt::LabelProvider>        graspLabels;
    std::unique_ptr<sgt::DetectorBackend>      toolDetector;
    std::unique_ptr<sgt::DetectorBackend>      graspDetector;
    std::unique_ptr<sgt::OnnxDefectClassifier> defectClassifier;
    std::unique_ptr<sgt::Renderer>             renderer;
    InferenceMode                              currentMode = opts.mode;
    std::string                                activeModelPath;
    float                                      detectorConfThresh = 0.0f;

    auto ensureModeLoaded = [&](InferenceMode mode) -> bool {
        try {
            if (mode == InferenceMode::Tool) {
                if (!toolDetector) {
                    if (!toolLabels) {
                        toolLabels = std::make_unique<sgt::DictLabelProvider>(
                            dictPath,
                            CLASS_NAMES);
                    }
                    toolDetector = std::make_unique<sgt::YoloOnnxDetector>(
                        toolModelPath,
                        640,
                        0.25f,
                        0.45f,
                        toolLabels.get());
                }
                activeModelPath = toolModelPath;
                detectorConfThresh = toolDetector->getConfThresh();
                return true;
            }

            if (mode == InferenceMode::Grasp) {
                if (!graspDetector) {
                    if (!graspLabels) {
                        graspLabels = std::make_unique<sgt::DictLabelProvider>(
                            "",
                            GRASP_CLASS_NAMES);
                    }
                    graspDetector = std::make_unique<sgt::YoloOnnxDetector>(
                        graspModelPath,
                        640,
                        0.25f,
                        0.45f,
                        graspLabels.get());
                }
                activeModelPath = graspModelPath;
                detectorConfThresh = graspDetector->getConfThresh();
                return true;
            }

            if (!defectClassifier) {
                defectClassifier = std::make_unique<sgt::OnnxDefectClassifier>(
                    defectModelPath,
                    512,
                    4,
                    0.50f);
            }
            activeModelPath = defectModelPath;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[FATAL] Failed to load mode " << modeToString(mode)
                      << ": " << e.what() << "\n";
            return false;
        }
    };

    try {
        auto font = std::make_unique<sgt::OpenCVFontRenderer>();
        renderer = std::make_unique<sgt::OpenCVRenderer>(
            std::move(font),
            WINDOW_NAME);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Initialization failed: " << e.what() << "\n";
        return 1;
    }

    if (!ensureModeLoaded(currentMode)) {
        return 1;
    }

    cv::VideoCapture cap(opts.cameraId);
    if (!cap.isOpened()) {
        std::cerr << "[FATAL] Cannot open camera " << opts.cameraId << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "[SGTDetector] Camera " << opts.cameraId
              << " opened. Press q/ESC to quit.\n";
    std::cout << "[SGTDetector] Mode         : " << modeToString(currentMode) << "\n";
    std::cout << "[SGTDetector] Active model : " << activeModelPath << "\n";

    ModeButtonOverlayState modeButtonState;
    cv::setMouseCallback(WINDOW_NAME, onModeOverlayMouse, &modeButtonState);

    auto switchMode = [&](InferenceMode requestedMode, const char* source) {
        if (requestedMode == currentMode) return;

        if (!ensureModeLoaded(requestedMode)) {
            return;
        }

        currentMode = requestedMode;
        std::cout << "[SGTDetector] Mode switched (" << source << "): "
                  << modeToString(currentMode)
                  << " | model: " << activeModelPath << "\n";
    };

    cv::Mat frame;
    auto  lastTime          = std::chrono::steady_clock::now();
    float fps               = 0.0f;

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[WARN] Empty frame, skipping...\n";
            continue;
        }

        sgt::DetectorBackend* activeDetector = nullptr;
        sgt::OnnxDefectClassifier* activeDefectClassifier = nullptr;
        if (currentMode == InferenceMode::Tool) {
            activeDetector = toolDetector.get();
        } else if (currentMode == InferenceMode::Grasp) {
            activeDetector = graspDetector.get();
        } else {
            activeDefectClassifier = defectClassifier.get();
        }

        std::vector<sgt::Detection>    detections;
        std::vector<sgt::DefectResult> defectResults;

        try {
            if (activeDetector) {
                detections = activeDetector->detect(frame);
            }
            if (activeDefectClassifier) {
                sgt::Detection frameRegion;
                frameRegion.bbox = {
                    0.0f,
                    0.0f,
                    static_cast<float>(frame.cols),
                    static_cast<float>(frame.rows),
                };
                frameRegion.label = "frame";
                defectResults = activeDefectClassifier->classify(
                    frame,
                    std::vector<sgt::Detection>{frameRegion});
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARN] Inference error (" << modeToString(currentMode)
                      << "): " << e.what() << "\n";
        }

        {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            float instant = (elapsed > 0.0f) ? (1.0f / elapsed) : 0.0f;
            fps = fps * 0.9f + instant * 0.1f;
        }

        if (activeDetector) {
            renderer->drawDetections(frame, detections);
        }
        if (activeDefectClassifier) {
            renderer->drawDefects(frame, defectResults);
        }

        sgt::HUDData hud;
        hud.fps = fps;
        hud.modeName = modeToString(currentMode);
        hud.confThresh = detectorConfThresh;
        hud.graspConfThresh = 0.0f;
        hud.defectThresh = activeDefectClassifier
            ? activeDefectClassifier->getDefectThresh()
            : 0.0f;
        hud.detections = static_cast<int>(detections.size());
        hud.graspDetections = 0;
        hud.defects = countDefects(defectResults);
        hud.graspEnabled = false;
        hud.defectEnabled = (activeDefectClassifier != nullptr);
        renderer->drawHUD(frame, hud);
        drawModeButtons(frame, currentMode, modeButtonState);

        int key = renderer->showFrame(frame) & 0xFF;

        if (modeButtonState.hasPendingMode) {
            switchMode(modeButtonState.pendingMode, "GUI Button");
            modeButtonState.hasPendingMode = false;
        }

        if (key == 27 || key == 'q' || key == 'Q') break;

        if (key == '1') switchMode(InferenceMode::Tool, "Hotkey");
        if (key == '2') switchMode(InferenceMode::Grasp, "Hotkey");
        if (key == '3') switchMode(InferenceMode::Defect, "Hotkey");

        if (key == '+' || key == '=') {
            if (activeDetector) {
                detectorConfThresh = std::min(detectorConfThresh + 0.05f, 0.95f);
                activeDetector->setConfThresh(detectorConfThresh);
                std::cout << "[SGTDetector] " << modeToString(currentMode)
                          << " conf -> " << detectorConfThresh << "\n";
            } else if (activeDefectClassifier) {
                float defectThresh =
                    std::min(activeDefectClassifier->getDefectThresh() + 0.05f, 0.95f);
                activeDefectClassifier->setDefectThresh(defectThresh);
                std::cout << "[SGTDetector] defect threshold -> "
                          << defectThresh << "\n";
            }
        }
        if (key == '-' || key == '_') {
            if (activeDetector) {
                detectorConfThresh = std::max(detectorConfThresh - 0.05f, 0.05f);
                activeDetector->setConfThresh(detectorConfThresh);
                std::cout << "[SGTDetector] " << modeToString(currentMode)
                          << " conf -> " << detectorConfThresh << "\n";
            } else if (activeDefectClassifier) {
                float defectThresh =
                    std::max(activeDefectClassifier->getDefectThresh() - 0.05f, 0.05f);
                activeDefectClassifier->setDefectThresh(defectThresh);
                std::cout << "[SGTDetector] defect threshold -> "
                          << defectThresh << "\n";
            }
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
