#include "ui/HudOverlay.h"

#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>

#include "core/DetectionPipeline.h"
#include "ui/CaptureFAB.h"
#include "ui/ModePillBar.h"
#include "ui/StatusChip.h"
#include "ui/ThemeManager.h"

namespace xcwj::ui {

HudOverlay::HudOverlay(uint8_t initialModeMask, QWidget* parent)
    : QWidget(parent)
{
    setObjectName("HudOverlay");
    setAttribute(Qt::WA_StyledBackground, false);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(0);

    auto* top = new QHBoxLayout();
    top->setContentsMargins(0, 0, 0, 0);
    top->setSpacing(8);
    cameraChip_ = new StatusChip("Camera starting");
    fpsChip_ = new StatusChip("FPS 0.0");
    fpsChip_->setDotColor(ThemeManager::instance().tokens().info);
    modePillBar_ = new ModePillBar(initialModeMask);
    top->addWidget(cameraChip_);
    top->addWidget(fpsChip_);
    top->addStretch();
    top->addWidget(modePillBar_);
    root->addLayout(top);
    root->addStretch();

    auto* bottom = new QHBoxLayout();
    bottom->setContentsMargins(0, 0, 0, 0);
    bottom->addStretch();
    captureButton_ = new CaptureFAB();
    bottom->addWidget(captureButton_);
    root->addLayout(bottom);

    connect(modePillBar_, &ModePillBar::toggleMode, this, &HudOverlay::toggleMode);
    connect(captureButton_, &QPushButton::clicked, this, &HudOverlay::captureRequested);

    connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, [this]() {
        const auto& t = ThemeManager::instance().tokens();
        fpsChip_->setDotColor(t.info);
        const QString camColor = lastCameraStatus_.contains("unavailable", Qt::CaseInsensitive)
            ? t.danger : t.accent;
        cameraChip_->setDotColor(camColor);
    });
}

void HudOverlay::setCameraStatus(const QString& text)
{
    lastCameraStatus_ = text;
    cameraChip_->setText(text);
    const auto& tokens = ThemeManager::instance().tokens();
    const QString color = text.contains("unavailable", Qt::CaseInsensitive)
        ? tokens.danger
        : tokens.accent;
    cameraChip_->setDotColor(color);
}

void HudOverlay::setFps(float fps)
{
    fpsChip_->setText(QString("FPS %1").arg(fps, 0, 'f', 1));
}

void HudOverlay::setModeMask(uint8_t mask)
{
    modePillBar_->setModeMask(mask);
}

void HudOverlay::setFrameSummary(const DetectionFrameResult& result)
{
    Q_UNUSED(result);
}

} // namespace xcwj::ui
