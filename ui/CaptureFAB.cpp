#include "ui/CaptureFAB.h"

namespace sgt::ui {

CaptureFAB::CaptureFAB(QWidget* parent)
    : QPushButton("Capture", parent)
{
    setObjectName("CaptureFAB");
    setMinimumSize(116, 48);
    setCursor(Qt::PointingHandCursor);
}

} // namespace sgt::ui
