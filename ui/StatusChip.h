#pragma once

#include <QFrame>
#include <QString>

class QLabel;

namespace xcwj::ui {

class StatusChip final : public QFrame {
    Q_OBJECT

public:
    explicit StatusChip(const QString& text, QWidget* parent = nullptr);

    void setText(const QString& text);
    void setDotColor(const QString& color);

private:
    QFrame* dot_ = nullptr;
    QLabel* textLabel_ = nullptr;
};

} // namespace xcwj::ui
