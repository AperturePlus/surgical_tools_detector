#include "ui/LivePage.h"

#include <QFrame>
#include <QHBoxLayout>
#include <QSplitter>
#include <QStackedLayout>
#include <QVBoxLayout>

#include "ui/ControlPanel.h"
#include "ui/HudOverlay.h"
#include "ui/LivePreviewWidget.h"

namespace sgt::ui {

LivePage::LivePage(uint8_t initialModeMask,
                   const DetectionThresholds& initialThresholds,
                   QWidget* parent)
    : QWidget(parent)
{
    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(20, 18, 20, 18);
    root->setSpacing(0);

    auto* splitter = new QSplitter(Qt::Horizontal);
    splitter->setChildrenCollapsible(false);
    splitter->setHandleWidth(12);

    auto* videoStage = new QFrame();
    videoStage->setObjectName("VideoStage");
    auto* stack = new QStackedLayout(videoStage);
    stack->setContentsMargins(12, 12, 12, 12);
    stack->setStackingMode(QStackedLayout::StackAll);
    preview_ = new LivePreviewWidget();
    hud_ = new HudOverlay(initialModeMask);
    stack->addWidget(preview_);
    stack->addWidget(hud_);
    stack->setCurrentWidget(hud_);

    controlPanel_ = new ControlPanel(initialModeMask, initialThresholds);
    splitter->addWidget(videoStage);
    splitter->addWidget(controlPanel_);
    splitter->setStretchFactor(0, 7);
    splitter->setStretchFactor(1, 3);
    root->addWidget(splitter);

    connect(hud_, &HudOverlay::toggleMode, this, [this](uint8_t bit) {
        controlPanel_->toggleMode(bit);
    });
    connect(hud_, &HudOverlay::captureRequested, this, &LivePage::captureRequested);
    connect(controlPanel_, &ControlPanel::modeMaskChanged, this, [this](uint8_t mask) {
        hud_->setModeMask(mask);
        emit modeMaskChanged(mask);
    });
    connect(controlPanel_, &ControlPanel::thresholdsChanged, this, &LivePage::thresholdsChanged);
}

PerfStats LivePage::setFrameResult(const DetectionFrameResult& result, uint8_t activeMask)
{
    PerfStats perf = preview_->setResult(result, activeMask);
    controlPanel_->setFrameResult(result);
    hud_->setModeMask(activeMask);
    hud_->setFrameSummary(result);
    return perf;
}

void LivePage::setCameraStatus(const QString& text)
{
    hud_->setCameraStatus(text);
}

void LivePage::setFps(float fps)
{
    hud_->setFps(fps);
}

void LivePage::setModels(const ModelInfo& models)
{
    controlPanel_->setModels(models);
}

void LivePage::setModeMask(uint8_t modeMask)
{
    controlPanel_->setModeMask(modeMask);
    hud_->setModeMask(controlPanel_->modeMask());
}

void LivePage::toggleMode(uint8_t bit)
{
    controlPanel_->toggleMode(bit);
}

} // namespace sgt::ui
