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

namespace sgt {
class CaptureStore;
class DetectionEngine;
}

namespace sgt::ui {

struct AppOptions {
    int cameraId = 0;
    uint8_t modeMask = MODE_TOOL;
    std::string toolModel;
    std::string graspModel;
    std::string defectModel;
};

class GalleryPage;
class LivePage;
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

    Sidebar* sidebar_ = nullptr;
    QStackedWidget* stack_ = nullptr;
    LivePage* livePage_ = nullptr;
    GalleryPage* galleryPage_ = nullptr;

    void buildUi();
    void wireEvents();
    void startCamera();
    void processFrame();
    void updateFps();
    void captureCurrent();
};

} // namespace sgt::ui
