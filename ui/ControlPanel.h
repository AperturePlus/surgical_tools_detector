#pragma once

#include <cstdint>

#include <QFrame>

#include "core/DetectionMetadata.h"
#include "core/DetectionPipeline.h"

class QLabel;
class QSlider;
class QTableWidget;

namespace sgt::ui {

class ControlPanel final : public QFrame {
    Q_OBJECT

public:
    explicit ControlPanel(uint8_t initialModeMask,
                          const DetectionThresholds& initialThresholds,
                          QWidget* parent = nullptr);

    uint8_t modeMask() const;
    DetectionThresholds thresholds() const { return thresholds_; }
    void setFrameResult(const DetectionFrameResult& result);
    void setModels(const ModelInfo& models);
    void setModeMask(uint8_t modeMask);
    void toggleMode(uint8_t bit);

public slots:
    void toggleToolMode();
    void toggleGraspMode();
    void toggleDefectMode();

signals:
    void modeMaskChanged(uint8_t modeMask);
    void thresholdsChanged(const DetectionThresholds& thresholds);

private:
    uint8_t currentModeMask_ = 0;
    DetectionThresholds thresholds_;
    QSlider* toolSlider_ = nullptr;
    QSlider* graspSlider_ = nullptr;
    QSlider* defectSlider_ = nullptr;
    QLabel* toolThresholdLabel_ = nullptr;
    QLabel* graspThresholdLabel_ = nullptr;
    QLabel* defectThresholdLabel_ = nullptr;
    QLabel* toolModelLabel_ = nullptr;
    QLabel* graspModelLabel_ = nullptr;
    QLabel* defectModelLabel_ = nullptr;
    QTableWidget* resultTable_ = nullptr;

    QSlider* makeSlider(float value, QLabel*& label, const QString& name);
    QWidget* makeSliderRow(QSlider* slider, QLabel* label);
    void updateThresholdLabel(QLabel* label, const QString& name, int value);
    void emitModeIfChanged(uint8_t next);
};

} // namespace sgt::ui
