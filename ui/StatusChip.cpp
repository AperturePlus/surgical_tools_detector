#include "ui/StatusChip.h"

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>

namespace xcwj::ui {

StatusChip::StatusChip(const QString& text, QWidget* parent)
    : QFrame(parent)
{
    setObjectName("StatusChip");
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(10, 4, 12, 4);
    layout->setSpacing(8);

    dot_ = new QFrame();
    dot_->setObjectName("StatusChipDot");
    dot_->setFixedSize(8, 8);
    layout->addWidget(dot_, 0, Qt::AlignVCenter);

    textLabel_ = new QLabel(text);
    textLabel_->setObjectName("StatusChipText");
    layout->addWidget(textLabel_);
}

void StatusChip::setText(const QString& text)
{
    textLabel_->setText(text);
}

void StatusChip::setDotColor(const QString& color)
{
    dot_->setStyleSheet(QString("background: %1; border: none; border-radius: 4px;").arg(color));
}

} // namespace xcwj::ui
