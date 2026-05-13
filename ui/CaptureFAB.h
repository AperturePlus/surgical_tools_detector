#pragma once

#include <QPushButton>

namespace sgt::ui {

class CaptureFAB final : public QPushButton {
    Q_OBJECT

public:
    explicit CaptureFAB(QWidget* parent = nullptr);
};

} // namespace sgt::ui
