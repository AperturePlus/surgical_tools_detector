#pragma once

#include <QFrame>
#include <QStyle>

class QButtonGroup;
class QToolButton;

namespace sgt::ui {

class Sidebar final : public QFrame {
    Q_OBJECT

public:
    explicit Sidebar(QWidget* parent = nullptr);

    void setCurrentPage(int index);

signals:
    void pageRequested(int index);

private:
    QButtonGroup* navGroup_ = nullptr;
    QToolButton* themeButton_ = nullptr;

    QToolButton* makeNavButton(const QString& tooltip, QStyle::StandardPixmap icon, int pageIndex);
    void updateThemeButton();
};

} // namespace sgt::ui
