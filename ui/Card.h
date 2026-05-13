#pragma once

#include <QFrame>

class QVBoxLayout;

namespace sgt::ui {

class Card final : public QFrame {
    Q_OBJECT

public:
    explicit Card(const QString& title, QWidget* parent = nullptr);

    QVBoxLayout* bodyLayout() const { return bodyLayout_; }

private:
    QVBoxLayout* bodyLayout_ = nullptr;
};

} // namespace sgt::ui
