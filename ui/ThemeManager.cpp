#include "ui/ThemeManager.h"

#include <QApplication>

#include "core/AppSettings.h"

namespace xcwj::ui {

ThemeManager& ThemeManager::instance()
{
    static ThemeManager s;
    return s;
}

ThemeManager::ThemeManager()
{
    AppSettings settings;
    const QString mode = settings.themeMode();
    tokens_ = (mode == "light") ? Theme::light() : Theme::dark();
}

void ThemeManager::apply(QApplication* app)
{
    app_ = app;
    applyToApp();
}

void ThemeManager::setMode(const QString& mode)
{
    const ThemeTokens next = (mode == "light") ? Theme::light() : Theme::dark();
    if (next.name == tokens_.name) return;
    tokens_ = next;
    AppSettings settings;
    settings.setThemeMode(tokens_.name);
    applyToApp();
    emit themeChanged(tokens_);
}

void ThemeManager::toggle()
{
    setMode(tokens_.name == "dark" ? "light" : "dark");
}

void ThemeManager::applyToApp()
{
    if (!app_) return;
    app_->setStyleSheet(Theme::renderQss(tokens_));
}

} // namespace xcwj::ui
