#include "ui/StatusChip.h"

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>

namespace sgt::ui {

StatusChip::StatusChip(const QString& text, QWidget* parent)
    : QFrame(parent)
{
    setObjectName("StatusChip");
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(10, 4, 12, 4);
    layout->setSpacing(8);

    auto* dot = new QFrame();
    dot->setObjectName("StatusChipDot");
    dot->setFixedSize(8, 8);
    layout->addWidget(dot, 0, Qt::AlignVCenter);

    textLabel_ = new QLabel(text);
    textLabel_->setObjectName("StatusChipText");
    layout->addWidget(textLabel_);
}

void StatusChip::setText(const QString& text)
{
    textLabel_->setText(text);
}

} // namespace sgt::ui
