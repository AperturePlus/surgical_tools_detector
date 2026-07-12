#pragma once

#include <QPushButton>

namespace xcwj::ui {

class CaptureFAB final : public QPushButton {
    Q_OBJECT

public:
    explicit CaptureFAB(QWidget* parent = nullptr);
};

} // namespace xcwj::ui
