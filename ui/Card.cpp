#include "ui/Card.h"

#include <QLabel>
#include <QVBoxLayout>

namespace xcwj::ui {

Card::Card(const QString& title, QWidget* parent)
    : Card(title, Variant::Standard, parent) {}

Card::Card(const QString& title, Variant variant, QWidget* parent)
    : QFrame(parent)
{
    init(title, variant);
}

void Card::init(const QString& title, Variant variant)
{
    setObjectName(variant == Variant::Flat ? "CardFlat" : "Card");
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(14, 12, 14, 14);
    layout->setSpacing(10);

    auto* titleLabel = new QLabel(title);
    titleLabel->setObjectName("CardTitle");
    layout->addWidget(titleLabel);

    bodyLayout_ = new QVBoxLayout();
    bodyLayout_->setContentsMargins(0, 0, 0, 0);
    bodyLayout_->setSpacing(8);
    layout->addLayout(bodyLayout_);
}

} // namespace xcwj::ui
