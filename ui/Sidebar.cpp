#include "ui/Sidebar.h"

#include <QButtonGroup>
#include <QLabel>
#include <QToolButton>
#include <QVBoxLayout>

#include "ui/IconLoader.h"
#include "ui/ThemeManager.h"

namespace sgt::ui {

namespace {

constexpr QSize kNavIconSize{22, 22};
constexpr QSize kThemeIconSize{20, 20};

} // namespace

Sidebar::Sidebar(QWidget* parent)
    : QFrame(parent)
{
    setObjectName("Sidebar");
    setFixedWidth(64);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 14, 8, 14);
    layout->setSpacing(10);

    auto* brand = new QLabel("SGT");
    brand->setObjectName("BrandMark");
    brand->setAlignment(Qt::AlignCenter);
    brand->setFixedHeight(40);
    layout->addWidget(brand);
    layout->addSpacing(12);

    navGroup_ = new QButtonGroup(this);
    navGroup_->setExclusive(true);

    auto* live = makeNavButton("Live", "nav-live", 0);
    auto* gallery = makeNavButton("Gallery", "nav-gallery", 1);
    auto* settings = makeNavButton("Settings", "nav-settings", 2);

    layout->addWidget(live);
    layout->addWidget(gallery);
    layout->addWidget(settings);
    layout->addStretch();

    themeButton_ = new QToolButton();
    themeButton_->setObjectName("NavButton");
    themeButton_->setIconSize(kThemeIconSize);
    themeButton_->setFixedSize(48, 44);
    themeButton_->setCursor(Qt::PointingHandCursor);
    connect(themeButton_, &QToolButton::clicked,
            &ThemeManager::instance(), &ThemeManager::toggle);
    layout->addWidget(themeButton_);

    connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, [this]() {
        refreshNavIcons();
        refreshThemeIcon();
    });
    connect(navGroup_, &QButtonGroup::idToggled, this, [this](int, bool) {
        refreshNavIcons();
    });

    refreshNavIcons();
    refreshThemeIcon();
    setCurrentPage(0);
}

void Sidebar::setCurrentPage(int index)
{
    if (auto* button = navGroup_->button(index)) {
        button->setChecked(true);
    }
}

QToolButton* Sidebar::makeNavButton(const QString& tooltip,
                                    const QString& iconAlias,
                                    int pageIndex)
{
    auto* button = new QToolButton();
    button->setObjectName("NavButton");
    button->setIconSize(kNavIconSize);
    button->setCheckable(true);
    button->setToolTip(tooltip);
    button->setCursor(Qt::PointingHandCursor);
    button->setFixedSize(48, 44);
    navGroup_->addButton(button, pageIndex);
    navButtons_.insert(pageIndex, button);
    navIconAliases_.insert(pageIndex, iconAlias);
    connect(button, &QToolButton::clicked, this, [this, pageIndex]() {
        emit pageRequested(pageIndex);
    });
    return button;
}

void Sidebar::refreshNavIcons()
{
    const auto& tokens = ThemeManager::instance().tokens();
    for (auto it = navButtons_.constBegin(); it != navButtons_.constEnd(); ++it) {
        QToolButton* button = it.value();
        const QString alias = navIconAliases_.value(it.key());
        const bool active = button->isChecked() && button->isEnabled();
        const QString color = active ? tokens.accent : tokens.textSecondary;
        button->setIcon(IconLoader::load(alias, color, kNavIconSize));
    }
}

void Sidebar::refreshThemeIcon()
{
    if (!themeButton_) return;
    const bool dark = ThemeManager::instance().mode() == "dark";
    const auto& tokens = ThemeManager::instance().tokens();
    themeButton_->setToolTip(dark ? "Switch to light theme" : "Switch to dark theme");
    themeButton_->setIcon(IconLoader::load(dark ? "theme-sun" : "theme-moon",
                                           tokens.textSecondary,
                                           kThemeIconSize));
}

} // namespace sgt::ui
