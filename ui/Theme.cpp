#include "ui/Theme.h"

#include <QFile>
#include <QIODevice>

namespace xcwj::ui::Theme {

ThemeTokens dark()
{
    return {
        "dark",
        "#0E141B", "#161D26", "#1E2733", "#243140",
        "#E6EDF3", "#8B98A8",
        "#14B8A6", "#2DD4BF",
        "#38BDF8", "#F59E0B", "#EF4444",
        "rgba(0,0,0,0.35)",
        "rgba(14,20,27,0.72)", "#243140", "#38BDF8", "#2A3441", "#3A4654"
    };
}

ThemeTokens light()
{
    return {
        "light",
        "#F4F6F9", "#FFFFFF", "#FFFFFF", "#E2E8F0",
        "#0F172A", "#64748B",
        "#0F766E", "#0D9488",
        "#2563EB", "#D97706", "#DC2626",
        "rgba(15,23,42,0.12)",
        "rgba(255,255,255,0.82)", "#E2E8F0", "#2563EB", "#CBD5E1", "#94A3B8"
    };
}

QString renderQss(const ThemeTokens& t)
{
    QFile f(":/qss/base.qss");
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    QString s = QString::fromUtf8(f.readAll());
    s.replace("{{bg}}", t.bg);
    s.replace("{{surface}}", t.surface);
    s.replace("{{elevated}}", t.elevated);
    s.replace("{{border}}", t.border);
    s.replace("{{textPrimary}}", t.textPrimary);
    s.replace("{{textSecondary}}", t.textSecondary);
    s.replace("{{accent}}", t.accent);
    s.replace("{{accentHover}}", t.accentHover);
    s.replace("{{info}}", t.info);
    s.replace("{{warn}}", t.warn);
    s.replace("{{danger}}", t.danger);
    s.replace("{{shadow}}", t.shadow);
    s.replace("{{chipBg}}", t.chipBg);
    s.replace("{{chipBorder}}", t.chipBorder);
    s.replace("{{focusRing}}", t.focusRing);
    s.replace("{{scrollBar}}", t.scrollBar);
    s.replace("{{scrollBarHover}}", t.scrollBarHover);
    return s;
}

} // namespace xcwj::ui::Theme
