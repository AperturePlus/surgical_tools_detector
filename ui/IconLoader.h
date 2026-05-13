#pragma once

#include <QIcon>
#include <QSize>
#include <QString>

namespace sgt::ui {

// Loads ":/icons/<alias>.svg", substitutes the {{stroke}} placeholder with
// `strokeColor`, and rasterizes to a device-pixel-aware QIcon at `size`.
// Returns a null icon if the resource is missing or fails to render.
class IconLoader {
public:
    static QIcon load(const QString& alias,
                      const QString& strokeColor,
                      QSize size = QSize(22, 22));
};

} // namespace sgt::ui
