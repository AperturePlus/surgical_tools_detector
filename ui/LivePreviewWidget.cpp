#include "ui/LivePreviewWidget.h"

#include <chrono>

#include <QLabel>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QVBoxLayout>

#include "ui/QtImageUtils.h"

namespace sgt::ui {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end = Clock::now())
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

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

PerfStats LivePreviewWidget::setResult(const DetectionFrameResult& result, uint8_t activeMask)
{
    Q_UNUSED(activeMask);
    PerfStats perf;
    auto start = Clock::now();
    currentPixmap_ = matToPixmap(result.annotatedFrame);
    perf.qtImageMs = elapsedMs(start);
    perf.qtScaleDisplayMs = updatePixmap();
    return perf;
}

void LivePreviewWidget::resizeEvent(QResizeEvent* event)
{
    QFrame::resizeEvent(event);
    (void)updatePixmap();
}

double LivePreviewWidget::updatePixmap()
{
    if (currentPixmap_.isNull()) return 0.0;
    auto start = Clock::now();
    videoLabel_->setPixmap(currentPixmap_.scaled(videoLabel_->size(),
        Qt::KeepAspectRatio, Qt::FastTransformation));
    return elapsedMs(start);
}

} // namespace sgt::ui
