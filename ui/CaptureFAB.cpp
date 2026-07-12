#include "ui/CaptureFAB.h"

namespace xcwj::ui {

CaptureFAB::CaptureFAB(QWidget* parent)
    : QPushButton("Capture", parent)
{
    setObjectName("CaptureFAB");
    setMinimumSize(116, 48);
    setCursor(Qt::PointingHandCursor);
}

} // namespace xcwj::ui
