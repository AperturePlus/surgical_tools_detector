#include "ui/StatusChip.h"

#include <QHBoxLayout>
#include <QLabel>

namespace sgt::ui {

StatusChip::StatusChip(const QString& text, QWidget* parent)
    : QFrame(parent)
{
    setObjectName("StatusChip");
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(10, 4, 10, 4);
    layout->setSpacing(6);

    auto* dot = new QLabel("o");
    dot->setObjectName("StatusChipDot");
    dot->setFixedWidth(10);
    layout->addWidget(dot);

    textLabel_ = new QLabel(text);
    textLabel_->setObjectName("StatusChipText");
    layout->addWidget(textLabel_);
}

void StatusChip::setText(const QString& text)
{
    textLabel_->setText(text);
}

} // namespace sgt::ui
