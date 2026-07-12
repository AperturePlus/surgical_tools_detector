#pragma once

#include <QFrame>

class QVBoxLayout;

namespace xcwj::ui {

class Card final : public QFrame {
    Q_OBJECT

public:
    enum class Variant { Standard, Flat };

    explicit Card(const QString& title, QWidget* parent = nullptr);
    Card(const QString& title, Variant variant, QWidget* parent = nullptr);

    QVBoxLayout* bodyLayout() const { return bodyLayout_; }

private:
    void init(const QString& title, Variant variant);

    QVBoxLayout* bodyLayout_ = nullptr;
};

} // namespace xcwj::ui
