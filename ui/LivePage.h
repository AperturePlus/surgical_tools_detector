#pragma once

#include <cstdint>

#include <QWidget>

#include "core/DetectionMetadata.h"
#include "core/DetectionPipeline.h"

namespace xcwj::ui {

class ControlPanel;
class HudOverlay;
class LivePreviewWidget;

class LivePage final : public QWidget {
    Q_OBJECT

public:
    explicit LivePage(uint8_t initialModeMask,
                      const DetectionThresholds& initialThresholds,
                      QWidget* parent = nullptr);

    PerfStats setFrameResult(const DetectionFrameResult& result, uint8_t activeMask);
    void setCameraStatus(const QString& text);
    void setFps(float fps);
    void setModels(const ModelInfo& models);
    void setModeMask(uint8_t modeMask);
    void toggleMode(uint8_t bit);

signals:
    void modeMaskChanged(uint8_t modeMask);
    void thresholdsChanged(const DetectionThresholds& thresholds);
    void captureRequested();

private:
    LivePreviewWidget* preview_ = nullptr;
    HudOverlay* hud_ = nullptr;
    ControlPanel* controlPanel_ = nullptr;
};

} // namespace xcwj::ui
