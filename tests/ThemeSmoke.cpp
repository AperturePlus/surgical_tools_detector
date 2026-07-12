#include <iostream>

#include <QCoreApplication>

#include "ui/Theme.h"

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);

    const auto dark = xcwj::ui::Theme::dark();
    const auto light = xcwj::ui::Theme::light();

    if (dark.name != "dark" || light.name != "light") {
        std::cerr << "name mismatch\n";
        return 1;
    }
    if (dark.bg == light.bg) {
        std::cerr << "dark and light share bg\n";
        return 1;
    }

    const QString darkQss = xcwj::ui::Theme::renderQss(dark);
    const QString lightQss = xcwj::ui::Theme::renderQss(light);

    if (darkQss.isEmpty() || lightQss.isEmpty()) {
        std::cerr << "qss empty (resource not registered?)\n";
        return 1;
    }
    if (darkQss.contains("{{")) {
        std::cerr << "dark qss has unsubstituted placeholders\n";
        return 1;
    }
    if (!darkQss.contains(dark.accent)) {
        std::cerr << "dark accent token missing in rendered qss\n";
        return 1;
    }
    if (!lightQss.contains(light.accent)) {
        std::cerr << "light accent token missing in rendered qss\n";
        return 1;
    }
    return 0;
}
