#pragma once

#include <QFrame>
#include <QPixmap>

#include "core/CaptureStore.h"

class QLabel;
class QEnterEvent;
class QMouseEvent;

namespace xcwj::ui {

class ThumbCard final : public QFrame {
    Q_OBJECT

public:
    explicit ThumbCard(const CaptureRecord& record, QWidget* parent = nullptr);

signals:
    void activated(const QString& id);

protected:
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    CaptureRecord record_;
    QLabel* imageLabel_ = nullptr;
    QPixmap thumbnail_;

    void loadThumbnail();
    void setThumbnail(const QPixmap& pixmap);
    QString displayTime() const;
};

} // namespace xcwj::ui
