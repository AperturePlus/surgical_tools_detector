#pragma once

#include <vector>

#include <QDate>
#include <QWidget>

#include "core/CaptureStore.h"

class QLabel;
class QLineEdit;
class QVBoxLayout;

namespace xcwj::ui {

class GalleryFilterBar;

class GalleryPage final : public QWidget {
    Q_OBJECT

public:
    explicit GalleryPage(QWidget* parent = nullptr);

    void setRecords(const std::vector<CaptureRecord>& records);
    void addRecord(const CaptureRecord& record);

private:
    std::vector<CaptureRecord> records_;
    QLabel* titleLabel_ = nullptr;
    QLineEdit* searchEdit_ = nullptr;
    GalleryFilterBar* filterBar_ = nullptr;
    QWidget* content_ = nullptr;
    QVBoxLayout* contentLayout_ = nullptr;
    QDate filterFrom_;
    QDate filterTo_;

    void rebuild();
    void openDetail(const QString& id);
    bool matchesFilter(const CaptureRecord& record) const;
    bool matchesSearch(const CaptureRecord& record) const;
    bool matchesDateRange(const CaptureRecord& record) const;
    QString dateHeading(const QString& timestamp) const;
};

} // namespace xcwj::ui
