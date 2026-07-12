#pragma once

#include <cstdint>

#include <QWidget>

namespace xcwj {
struct DetectionFrameResult;
}

namespace xcwj::ui {

class CaptureFAB;
class ModePillBar;
class StatusChip;

class HudOverlay final : public QWidget {
    Q_OBJECT

public:
    explicit HudOverlay(uint8_t initialModeMask, QWidget* parent = nullptr);

    void setCameraStatus(const QString& text);
    void setFps(float fps);
    void setModeMask(uint8_t mask);
    void setFrameSummary(const DetectionFrameResult& result);

signals:
    void toggleMode(uint8_t bit);
    void captureRequested();

private:
    StatusChip* cameraChip_ = nullptr;
    StatusChip* fpsChip_ = nullptr;
    ModePillBar* modePillBar_ = nullptr;
    CaptureFAB* captureButton_ = nullptr;
};

} // namespace xcwj::ui
