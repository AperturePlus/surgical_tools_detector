#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <QDateTime>
#include <QMainWindow>
#include <QTimer>

#include <opencv2/videoio.hpp>

#include "core/DetectionPipeline.h"
#include "core/Renderer.h"

class QStackedWidget;

namespace xcwj {
class CaptureStore;
class DetectionEngine;
}

namespace xcwj::ui {

struct AppOptions {
    int cameraId = 0;
    bool cameraIdFromCli = false;       // CLI numeric arg sets this true
    uint8_t modeMask = MODE_TOOL;
    DetectionThresholds thresholds;     // seeded from AppSettings::defaultThresholds()
    std::string toolModel;
    std::string graspModel;
    std::string defectModel;
};

class GalleryPage;
class LivePage;
class SettingsPage;
class Sidebar;

class AppShell final : public QMainWindow {
    Q_OBJECT

public:
    AppShell(AppOptions opts,
             std::unique_ptr<DetectionEngine> engine,
             std::unique_ptr<CaptureStore> store,
             QWidget* parent = nullptr);
    ~AppShell() override;

private:
    AppOptions opts_;
    std::unique_ptr<DetectionEngine> engine_;
    std::unique_ptr<CaptureStore> store_;
    uint8_t activeMask_;
    DetectionThresholds thresholds_;
    DetectionFrameResult lastResult_;
    cv::VideoCapture cap_;
    QTimer timer_;
    QDateTime lastFrameTime_;
    float fps_ = 0.0f;
    bool profileEnabled_ = false;
    int profileFrameCount_ = 0;
    PerfStats profileTotals_;

    Sidebar* sidebar_ = nullptr;
    QStackedWidget* stack_ = nullptr;
    LivePage* livePage_ = nullptr;
    GalleryPage* galleryPage_ = nullptr;
    SettingsPage* settingsPage_ = nullptr;

    void buildUi();
    void wireEvents();
    void startCamera();
    void processFrame();
    void updateFps();
    void recordProfile(const PerfStats& perf);
    void captureCurrent();
};

} // namespace xcwj::ui
