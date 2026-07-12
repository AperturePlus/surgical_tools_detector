#pragma once

#include <cstdint>

#include <QFrame>

class QPushButton;

namespace xcwj::ui {

class ModePillBar final : public QFrame {
    Q_OBJECT

public:
    explicit ModePillBar(uint8_t initialMask, QWidget* parent = nullptr);

    void setModeMask(uint8_t mask);
    uint8_t modeMask() const { return modeMask_; }

signals:
    void toggleMode(uint8_t bit);

private:
    uint8_t modeMask_ = 0;
    QPushButton* toolButton_ = nullptr;
    QPushButton* graspButton_ = nullptr;
    QPushButton* defectButton_ = nullptr;

    QPushButton* makeButton(const QString& text, uint8_t bit);
    void syncButtons();
};

} // namespace xcwj::ui
