#pragma once

#include <QString>

namespace xcwj::ui {

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
    QString chipBg;
    QString chipBorder;
    QString focusRing;
    QString scrollBar;
    QString scrollBarHover;
};

namespace Theme {
    ThemeTokens dark();
    ThemeTokens light();
    QString renderQss(const ThemeTokens& tokens);
}

} // namespace xcwj::ui
