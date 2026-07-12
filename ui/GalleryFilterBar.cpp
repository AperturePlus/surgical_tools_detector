#include "ui/GalleryFilterBar.h"

#include <QButtonGroup>
#include <QDateEdit>
#include <QHBoxLayout>
#include <QPushButton>

namespace xcwj::ui {

GalleryFilterBar::GalleryFilterBar(QWidget* parent)
    : QFrame(parent)
{
    setObjectName("GalleryFilterBar");
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    group_ = new QButtonGroup(this);
    group_->setExclusive(true);

    allBtn_    = makeChip("All",     PresetAll);
    todayBtn_  = makeChip("Today",   PresetToday);
    last7Btn_  = makeChip("Last 7",  PresetLast7);
    last30Btn_ = makeChip("Last 30", PresetLast30);
    customBtn_ = makeChip("Custom",  PresetCustom);

    layout->addWidget(allBtn_);
    layout->addWidget(todayBtn_);
    layout->addWidget(last7Btn_);
    layout->addWidget(last30Btn_);
    layout->addWidget(customBtn_);

    fromEdit_ = new QDateEdit(QDate::currentDate().addDays(-30));
    toEdit_   = new QDateEdit(QDate::currentDate());
    for (auto* edit : { fromEdit_, toEdit_ }) {
        edit->setDisplayFormat("yyyy-MM-dd");
        edit->setCalendarPopup(true);
        edit->setVisible(false);
    }
    layout->addSpacing(8);
    layout->addWidget(fromEdit_);
    layout->addWidget(toEdit_);
    layout->addStretch();

    connect(group_, &QButtonGroup::idClicked, this,
            [this](int id) { onPresetChanged(id); });
    connect(fromEdit_, &QDateEdit::dateChanged, this,
            [this]() { onCustomDateChanged(); });
    connect(toEdit_, &QDateEdit::dateChanged, this,
            [this]() { onCustomDateChanged(); });

    allBtn_->setChecked(true);
    onPresetChanged(PresetAll);
}

QPushButton* GalleryFilterBar::makeChip(const QString& text, Preset preset)
{
    auto* button = new QPushButton(text);
    button->setObjectName("FilterChip");
    button->setCheckable(true);
    button->setCursor(Qt::PointingHandCursor);
    group_->addButton(button, static_cast<int>(preset));
    return button;
}

void GalleryFilterBar::onPresetChanged(int presetId)
{
    const QDate today = QDate::currentDate();
    const bool custom = (presetId == PresetCustom);
    fromEdit_->setVisible(custom);
    toEdit_->setVisible(custom);

    QDate from;
    QDate to;
    switch (presetId) {
        case PresetAll:
            break;
        case PresetToday:
            from = today;
            to = today;
            break;
        case PresetLast7:
            from = today.addDays(-6);
            to = today;
            break;
        case PresetLast30:
            from = today.addDays(-29);
            to = today;
            break;
        case PresetCustom:
            from = fromEdit_->date();
            to = toEdit_->date();
            break;
    }
    emitIfChanged(from, to);
}

void GalleryFilterBar::onCustomDateChanged()
{
    if (!customBtn_->isChecked()) return;
    emitIfChanged(fromEdit_->date(), toEdit_->date());
}

void GalleryFilterBar::emitIfChanged(QDate from, QDate to)
{
    if (from == from_ && to == to_) return;
    from_ = from;
    to_ = to;
    emit rangeChanged(from_, to_);
}

} // namespace xcwj::ui
