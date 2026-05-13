#include "ui/LivePreviewWidget.h"

#include <QLabel>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QVBoxLayout>

#include "ui/QtImageUtils.h"

namespace sgt::ui {

LivePreviewWidget::LivePreviewWidget(QWidget* parent)
    : QFrame(parent)
{
    setObjectName("LivePreview");
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    videoLabel_ = new QLabel("Waiting for camera frame");
    videoLabel_->setObjectName("VideoSurface");
    videoLabel_->setAlignment(Qt::AlignCenter);
    videoLabel_->setMinimumSize(720, 420);
    videoLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    layout->addWidget(videoLabel_, 1);
}

void LivePreviewWidget::setResult(const DetectionFrameResult& result, uint8_t activeMask)
{
    Q_UNUSED(activeMask);
    currentPixmap_ = matToPixmap(result.annotatedFrame);
    updatePixmap();
}

void LivePreviewWidget::resizeEvent(QResizeEvent* event)
{
    QFrame::resizeEvent(event);
    updatePixmap();
}

void LivePreviewWidget::updatePixmap()
{
    if (currentPixmap_.isNull()) return;
    videoLabel_->setPixmap(currentPixmap_.scaled(videoLabel_->size(),
        Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

} // namespace sgt::ui
