#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <QApplication>

#include "core/AppSettings.h"
#include "core/CaptureStore.h"
#include "core/DetectionPipeline.h"
#include "ui/AppShell.h"
#include "ui/ThemeManager.h"

namespace fs = std::filesystem;

static constexpr const char* DEFAULT_TOOL_MODEL = "inventory_detection.onnx";
static constexpr const char* DEFAULT_GRASP_MODEL = "instrument_endpoint_detection.onnx";
static constexpr const char* DEFAULT_DEFECT_MODEL = "tool_defect_classifier_resnet_512_b4.onnx";

static uint8_t parseMode(const std::string& raw)
{
    std::string m = raw;
    std::transform(m.begin(), m.end(), m.begin(), ::tolower);
    if (m == "tool") return xcwj::MODE_TOOL;
    if (m == "grasp") return xcwj::MODE_GRASP;
    if (m == "defect") return xcwj::MODE_DEFECT;
    throw std::runtime_error("invalid mode: " + raw);
}

static std::string requireVal(int& i, int argc, char* argv[], const std::string& opt)
{
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + opt);
    return argv[++i];
}

static xcwj::ui::AppOptions parseArgs(int argc, char* argv[])
{
    xcwj::ui::AppOptions opts;
    opts.modeMask = xcwj::MODE_TOOL;
    opts.toolModel = DEFAULT_TOOL_MODEL;
    opts.graspModel = DEFAULT_GRASP_MODEL;
    opts.defectModel = DEFAULT_DEFECT_MODEL;

    bool camSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: XunChaWeiJian [camera_id] [--mode tool|grasp|defect] "
                      << "[--tool-model p] [--grasp-model p] [--defect-model p]\n";
            std::exit(0);
        } else if (arg == "--mode") {
            opts.modeMask = parseMode(requireVal(i, argc, argv, arg));
        } else if (arg == "--tool-model") {
            opts.toolModel = requireVal(i, argc, argv, arg);
        } else if (arg == "--grasp-model") {
            opts.graspModel = requireVal(i, argc, argv, arg);
        } else if (arg == "--defect-model") {
            opts.defectModel = requireVal(i, argc, argv, arg);
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown option: " + arg);
        } else if (!camSet) {
            opts.cameraId = std::atoi(arg.c_str());
            opts.cameraIdFromCli = true;
            camSet = true;
        } else {
            throw std::runtime_error("unexpected argument: " + arg);
        }
    }
    return opts;
}

static std::string resolveAsset(const fs::path& exeDir, const std::string& name)
{
    if (fs::path(name).is_absolute() && fs::exists(name)) return name;
    for (auto& p : {exeDir / name, exeDir / "models" / name,
                    fs::path(name), fs::path("assets/models") / name}) {
        if (fs::exists(p)) return p.string();
    }
    return name;
}

int main(int argc, char* argv[])
{
    xcwj::ui::AppOptions opts;
    try {
        opts = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }

    QApplication app(argc, argv);
    xcwj::ui::ThemeManager::instance().apply(&app);

    xcwj::AppSettings settings;
    if (!opts.cameraIdFromCli) {
        opts.cameraId = settings.cameraId();
    }
    opts.modeMask = settings.modeMask();
    opts.thresholds = settings.defaultThresholds();

    fs::path exeDir = fs::path(argv[0]).parent_path();
    std::string toolPath = resolveAsset(exeDir, opts.toolModel);
    std::string graspPath = resolveAsset(exeDir, opts.graspModel);
    std::string defectPath = resolveAsset(exeDir, opts.defectModel);
    std::string dictPath = resolveAsset(exeDir, "labels.dict");

    auto engine = std::make_unique<xcwj::DetectionEngine>(toolPath, graspPath, defectPath, dictPath);

    QString captureDirPref = settings.captureDir();
    fs::path captureDir = captureDirPref.isEmpty()
        ? (exeDir / "captures")
        : fs::path(captureDirPref.toStdString());
    auto store = std::make_unique<xcwj::CaptureStore>(captureDir);

    xcwj::ui::AppShell window(opts, std::move(engine), std::move(store));
    window.show();
    return app.exec();
}
