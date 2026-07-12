#pragma once

#include <QFrame>

class QLabel;

namespace xcwj::ui {

class StatusChip final : public QFrame {
    Q_OBJECT

public:
    explicit StatusChip(const QString& text, QWidget* parent = nullptr);

    void setText(const QString& text);

private:
    QLabel* textLabel_ = nullptr;
};

} // namespace xcwj::ui
