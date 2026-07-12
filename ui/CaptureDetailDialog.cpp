#include "ui/CaptureDetailDialog.h"

#include <algorithm>

#include <QDesktopServices>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QIcon>
#include <QKeyEvent>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QScrollArea>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

#include <opencv2/imgcodecs.hpp>

#include "ui/QtImageUtils.h"
#include "ui/ThumbnailCache.h"

namespace xcwj::ui {

namespace {

QString thresholdText(const DetectionThresholds& t)
{
    return QString("Tool %1%  Grasp %2%  Defect %3%")
        .arg(t.tool * 100.0f, 0, 'f', 0)
        .arg(t.grasp * 100.0f, 0, 'f', 0)
        .arg(t.defect * 100.0f, 0, 'f', 0);
}

} // namespace

CaptureDetailDialog::CaptureDetailDialog(std::vector<CaptureRecord> records,
                                         int selectedIndex,
                                         QWidget* parent)
    : QDialog(parent)
    , records_(std::move(records))
{
    if (!records_.empty()) {
        currentIndex_ = std::clamp(selectedIndex, 0, static_cast<int>(records_.size()) - 1);
    }
    setWindowTitle("Capture detail");
    resize(1150, 720);

    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(18, 18, 18, 18);
    root->setSpacing(16);

    auto* left = new QVBoxLayout();
    left->setSpacing(10);
    imageLabel_ = new QLabel("Image unavailable");
    imageLabel_->setObjectName("VideoSurface");
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setMinimumSize(640, 420);
    left->addWidget(imageLabel_, 1);

    auto* stripArea = new QScrollArea();
    stripArea->setWidgetResizable(true);
    stripArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    stripArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    stripArea->setFixedHeight(82);
    filmstripContent_ = new QWidget();
    stripArea->setWidget(filmstripContent_);
    left->addWidget(stripArea);
    root->addLayout(left, 3);

    auto* right = new QVBoxLayout();
    right->setSpacing(12);
    titleLabel_ = new QLabel();
    titleLabel_->setObjectName("PanelTitle");
    right->addWidget(titleLabel_);

    metaLabel_ = new QLabel();
    metaLabel_->setObjectName("SubtleText");
    metaLabel_->setWordWrap(true);
    right->addWidget(metaLabel_);

    jsonText_ = new QPlainTextEdit();
    jsonText_->setReadOnly(true);
    jsonText_->setMinimumWidth(340);
    right->addWidget(jsonText_, 1);

    auto* buttons = new QHBoxLayout();
    auto* folderButton = new QPushButton("Open folder");
    auto* exportButton = new QPushButton("Export...");
    exportButton->setEnabled(false);
    exportButton->setToolTip("Export is planned for a later phase.");
    buttons->addWidget(folderButton);
    buttons->addWidget(exportButton);
    right->addLayout(buttons);
    root->addLayout(right, 1);

    connect(folderButton, &QPushButton::clicked, this, [this]() {
        if (records_.empty()) return;
        const QFileInfo info(QString::fromStdString(records_[currentIndex_].annotatedImagePath));
        QDesktopServices::openUrl(QUrl::fromLocalFile(info.absolutePath()));
    });

    showRecord(currentIndex_);
}

void CaptureDetailDialog::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Left) {
        navigate(-1);
        return;
    }
    if (event->key() == Qt::Key_Right) {
        navigate(1);
        return;
    }
    QDialog::keyPressEvent(event);
}

void CaptureDetailDialog::resizeEvent(QResizeEvent* event)
{
    QDialog::resizeEvent(event);
    updateImage();
}

void CaptureDetailDialog::showRecord(int index)
{
    if (records_.empty()) return;
    currentIndex_ = std::clamp(index, 0, static_cast<int>(records_.size()) - 1);
    const auto& record = records_[currentIndex_];

    titleLabel_->setText(QString::fromStdString(record.id));
    metaLabel_->setText(QString("Timestamp: %1\nModes: %2\nThresholds: %3\nCounts: T%4  G%5  D%6")
        .arg(QString::fromStdString(record.timestamp))
        .arg(modeText(record.modeMask))
        .arg(thresholdText(record.thresholds))
        .arg(record.toolCount)
        .arg(record.graspCount)
        .arg(record.defectCount));
    jsonText_->setPlainText(readTextFile(QString::fromStdString(record.jsonPath)));

    const cv::Mat image = cv::imread(record.annotatedImagePath, cv::IMREAD_COLOR);
    fullPixmap_ = image.empty() ? QPixmap{} : matToPixmap(image);
    updateImage();
    rebuildFilmstrip();
}

void CaptureDetailDialog::updateImage()
{
    if (fullPixmap_.isNull()) {
        imageLabel_->setText("Image unavailable");
        imageLabel_->setPixmap({});
        return;
    }
    imageLabel_->setText({});
    imageLabel_->setPixmap(fullPixmap_.scaled(imageLabel_->size(),
        Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void CaptureDetailDialog::rebuildFilmstrip()
{
    if (auto* oldLayout = filmstripContent_->layout()) {
        QLayoutItem* item = nullptr;
        while ((item = oldLayout->takeAt(0))) {
            if (auto* widget = item->widget()) {
                delete widget;
            }
            delete item;
        }
        delete oldLayout;
    }
    auto* layout = new QHBoxLayout(filmstripContent_);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(8);

    for (int i = 0; i < static_cast<int>(records_.size()); ++i) {
        auto* button = new QToolButton();
        button->setCheckable(true);
        button->setChecked(i == currentIndex_);
        button->setFixedSize(76, 54);
        button->setIconSize({72, 48});
        button->setToolTip(QString::fromStdString(records_[i].id));
        layout->addWidget(button);

        const int index = i;
        connect(button, &QToolButton::clicked, this, [this, index]() { showRecord(index); });
        QPointer<QToolButton> guard(button);
        ThumbnailCache::instance().requestThumbnail(
            QString::fromStdString(records_[i].id),
            QString::fromStdString(records_[i].annotatedImagePath),
            [guard](QPixmap pixmap) {
                if (!guard || pixmap.isNull()) return;
                guard->setIcon(QIcon(pixmap));
            },
            {72, 48});
    }
    layout->addStretch();
}

void CaptureDetailDialog::navigate(int delta)
{
    if (records_.empty()) return;
    showRecord(std::clamp(currentIndex_ + delta, 0, static_cast<int>(records_.size()) - 1));
}

} // namespace xcwj::ui
