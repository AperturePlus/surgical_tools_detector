#pragma once

#include <vector>

#include <QDialog>
#include <QPixmap>

#include "core/CaptureStore.h"

class QLabel;
class QKeyEvent;
class QPlainTextEdit;
class QResizeEvent;
class QScrollArea;

namespace xcwj::ui {

class CaptureDetailDialog final : public QDialog {
    Q_OBJECT

public:
    CaptureDetailDialog(std::vector<CaptureRecord> records,
                        int selectedIndex,
                        QWidget* parent = nullptr);

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    std::vector<CaptureRecord> records_;
    int currentIndex_ = 0;

    QLabel* imageLabel_ = nullptr;
    QLabel* titleLabel_ = nullptr;
    QLabel* metaLabel_ = nullptr;
    QPlainTextEdit* jsonText_ = nullptr;
    QWidget* filmstripContent_ = nullptr;
    QPixmap fullPixmap_;

    void showRecord(int index);
    void updateImage();
    void rebuildFilmstrip();
    void navigate(int delta);
};

} // namespace xcwj::ui
