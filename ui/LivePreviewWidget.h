#pragma once

#include <cstdint>

#include <QFrame>
#include <QPixmap>

#include "core/DetectionPipeline.h"
#include "core/PerfStats.h"

class QLabel;
class QResizeEvent;

namespace sgt::ui {

class LivePreviewWidget final : public QFrame {
    Q_OBJECT

public:
    explicit LivePreviewWidget(QWidget* parent = nullptr);

    PerfStats setResult(const DetectionFrameResult& result, uint8_t activeMask);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    QLabel* videoLabel_ = nullptr;
    QPixmap currentPixmap_;

    double updatePixmap();
};

} // namespace sgt::ui
