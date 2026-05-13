#pragma once

#include <QDate>
#include <QFrame>

class QButtonGroup;
class QDateEdit;
class QPushButton;

namespace sgt::ui {

class GalleryFilterBar final : public QFrame {
    Q_OBJECT

public:
    explicit GalleryFilterBar(QWidget* parent = nullptr);

    QDate from() const { return from_; }
    QDate to()   const { return to_; }

signals:
    void rangeChanged(QDate from, QDate to);

private:
    enum Preset { PresetAll = 0, PresetToday, PresetLast7, PresetLast30, PresetCustom };

    QButtonGroup* group_ = nullptr;
    QPushButton* allBtn_ = nullptr;
    QPushButton* todayBtn_ = nullptr;
    QPushButton* last7Btn_ = nullptr;
    QPushButton* last30Btn_ = nullptr;
    QPushButton* customBtn_ = nullptr;

    QDateEdit* fromEdit_ = nullptr;
    QDateEdit* toEdit_ = nullptr;

    QDate from_;
    QDate to_;

    QPushButton* makeChip(const QString& text, Preset preset);
    void onPresetChanged(int presetId);
    void onCustomDateChanged();
    void emitIfChanged(QDate from, QDate to);
};

} // namespace sgt::ui
