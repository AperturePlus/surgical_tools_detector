#pragma once

#include <QString>

namespace sgt::ui {

struct ThemeTokens {
    QString name;
    QString bg;
    QString surface;
    QString elevated;
    QString border;
    QString textPrimary;
    QString textSecondary;
    QString accent;
    QString accentHover;
    QString info;
    QString warn;
    QString danger;
    QString shadow;
};

namespace Theme {
    ThemeTokens dark();
    ThemeTokens light();
    QString renderQss(const ThemeTokens& tokens);
}

} // namespace sgt::ui
