#pragma once

#include <vector>

#include <QWidget>

#include "core/CaptureStore.h"

class QLabel;
class QLineEdit;
class QVBoxLayout;

namespace sgt::ui {

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
    QWidget* content_ = nullptr;
    QVBoxLayout* contentLayout_ = nullptr;

    void rebuild();
    void openDetail(const QString& id);
    bool matchesFilter(const CaptureRecord& record) const;
    QString dateHeading(const QString& timestamp) const;
};

} // namespace sgt::ui
