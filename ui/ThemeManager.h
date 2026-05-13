#pragma once

#include <QObject>

#include "ui/Theme.h"

class QApplication;

namespace sgt::ui {

class ThemeManager final : public QObject {
    Q_OBJECT

public:
    static ThemeManager& instance();

    void apply(QApplication* app);
    void setMode(const QString& mode);
    QString mode() const { return tokens_.name; }
    const ThemeTokens& tokens() const { return tokens_; }

public slots:
    void toggle();

signals:
    void themeChanged(const ThemeTokens& tokens);

private:
    ThemeManager();
    ThemeTokens tokens_;
    QApplication* app_ = nullptr;

    void applyToApp();
};

} // namespace sgt::ui
