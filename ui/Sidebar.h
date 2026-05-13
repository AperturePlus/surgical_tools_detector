#pragma once

#include <QFrame>
#include <QHash>

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
    QHash<int, QToolButton*> navButtons_;
    QHash<int, QString> navIconAliases_;

    QToolButton* makeNavButton(const QString& tooltip,
                               const QString& iconAlias,
                               int pageIndex);
    void refreshNavIcons();
    void refreshThemeIcon();
};

} // namespace sgt::ui
