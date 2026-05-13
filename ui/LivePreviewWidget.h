#pragma once

#include <cstdint>

#include <QFrame>
#include <QPixmap>

#include "core/DetectionPipeline.h"

class QLabel;
class QResizeEvent;

namespace sgt::ui {

class LivePreviewWidget final : public QFrame {
    Q_OBJECT

public:
    explicit LivePreviewWidget(QWidget* parent = nullptr);

    void setResult(const DetectionFrameResult& result, uint8_t activeMask);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    QLabel* videoLabel_ = nullptr;
    QPixmap currentPixmap_;

    void updatePixmap();
};

} // namespace sgt::ui
