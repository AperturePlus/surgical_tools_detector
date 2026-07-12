#include "ui/ModePillBar.h"

#include <QHBoxLayout>
#include <QPushButton>
#include <QSignalBlocker>

#include "core/Renderer.h"

namespace xcwj::ui {

ModePillBar::ModePillBar(uint8_t initialMask, QWidget* parent)
    : QFrame(parent)
    , modeMask_(initialMask ? initialMask : MODE_TOOL)
{
    setObjectName("ModePillBar");
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    toolButton_ = makeButton("Tool", MODE_TOOL);
    graspButton_ = makeButton("Grasp", MODE_GRASP);
    defectButton_ = makeButton("Defect", MODE_DEFECT);
    layout->addWidget(toolButton_);
    layout->addWidget(graspButton_);
    layout->addWidget(defectButton_);
    syncButtons();
}

void ModePillBar::setModeMask(uint8_t mask)
{
    modeMask_ = mask ? mask : MODE_TOOL;
    syncButtons();
}

QPushButton* ModePillBar::makeButton(const QString& text, uint8_t bit)
{
    auto* button = new QPushButton(text);
    button->setObjectName("ModePill");
    button->setCheckable(true);
    button->setCursor(Qt::PointingHandCursor);
    connect(button, &QPushButton::clicked, this, [this, bit]() {
        emit toggleMode(bit);
        syncButtons();
    });
    return button;
}

void ModePillBar::syncButtons()
{
    const QSignalBlocker blockTool(toolButton_);
    const QSignalBlocker blockGrasp(graspButton_);
    const QSignalBlocker blockDefect(defectButton_);
    toolButton_->setChecked(modeMask_ & MODE_TOOL);
    graspButton_->setChecked(modeMask_ & MODE_GRASP);
    defectButton_->setChecked(modeMask_ & MODE_DEFECT);
}

} // namespace xcwj::ui
