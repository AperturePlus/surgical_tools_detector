#include "ui/Sidebar.h"

#include <QButtonGroup>
#include <QLabel>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>

#include "ui/ThemeManager.h"

namespace sgt::ui {

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

    auto* live = makeNavButton("Live", QStyle::SP_ComputerIcon, 0);
    auto* gallery = makeNavButton("Gallery", QStyle::SP_DirIcon, 1);
    auto* settings = makeNavButton("Settings", QStyle::SP_FileDialogDetailedView, 2);
    settings->setEnabled(false);

    layout->addWidget(live);
    layout->addWidget(gallery);
    layout->addWidget(settings);
    layout->addStretch();

    themeButton_ = new QToolButton();
    themeButton_->setObjectName("NavButton");
    themeButton_->setIconSize({22, 22});
    themeButton_->setCursor(Qt::PointingHandCursor);
    connect(themeButton_, &QToolButton::clicked, &ThemeManager::instance(), &ThemeManager::toggle);
    connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, [this]() { updateThemeButton(); });
    layout->addWidget(themeButton_);
    updateThemeButton();

    setCurrentPage(0);
}

void Sidebar::setCurrentPage(int index)
{
    if (auto* button = navGroup_->button(index)) {
        button->setChecked(true);
    }
}

QToolButton* Sidebar::makeNavButton(const QString& tooltip,
                                    QStyle::StandardPixmap icon,
                                    int pageIndex)
{
    auto* button = new QToolButton();
    button->setObjectName("NavButton");
    button->setIcon(style()->standardIcon(icon));
    button->setIconSize({22, 22});
    button->setCheckable(true);
    button->setToolTip(tooltip);
    button->setCursor(Qt::PointingHandCursor);
    button->setFixedSize(48, 44);
    navGroup_->addButton(button, pageIndex);
    connect(button, &QToolButton::clicked, this, [this, pageIndex]() {
        emit pageRequested(pageIndex);
    });
    return button;
}

void Sidebar::updateThemeButton()
{
    const bool dark = ThemeManager::instance().mode() == "dark";
    themeButton_->setToolTip(dark ? "Switch to light theme" : "Switch to dark theme");
    themeButton_->setIcon(style()->standardIcon(dark ? QStyle::SP_DialogYesButton
                                                     : QStyle::SP_DialogNoButton));
}

} // namespace sgt::ui
