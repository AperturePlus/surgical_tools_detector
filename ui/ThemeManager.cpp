#include "ui/ThemeManager.h"

#include <QApplication>
#include <QSettings>

namespace sgt::ui {

ThemeManager& ThemeManager::instance()
{
    static ThemeManager s;
    return s;
}

ThemeManager::ThemeManager()
{
    QSettings settings("SGT", "Detector");
    const QString mode = settings.value("ui/theme", "dark").toString();
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
    QSettings("SGT", "Detector").setValue("ui/theme", tokens_.name);
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

} // namespace sgt::ui
