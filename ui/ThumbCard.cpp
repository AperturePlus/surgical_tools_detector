#include "ui/ThumbCard.h"

#include <QDateTime>
#include <QColor>
#include <QEnterEvent>
#include <QGraphicsDropShadowEffect>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPointer>
#include <QVBoxLayout>

#include "ui/ThemeManager.h"
#include "ui/ThumbnailCache.h"

namespace xcwj::ui {

ThumbCard::ThumbCard(const CaptureRecord& record, QWidget* parent)
    : QFrame(parent)
    , record_(record)
{
    setObjectName("ThumbCard");
    setFixedSize(192, 158);
    setCursor(Qt::PointingHandCursor);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    imageLabel_ = new QLabel("Loading");
    imageLabel_->setObjectName("ThumbImage");
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setFixedSize(192, 120);
    layout->addWidget(imageLabel_);

    auto* meta = new QWidget();
    auto* metaLayout = new QHBoxLayout(meta);
    metaLayout->setContentsMargins(10, 6, 10, 6);
    metaLayout->setSpacing(6);
    auto* timeLabel = new QLabel(displayTime());
    timeLabel->setObjectName("ThumbMeta");
    metaLayout->addWidget(timeLabel);
    metaLayout->addStretch();

    auto addBadge = [metaLayout](const QString& text, bool warn) {
        auto* label = new QLabel(text);
        label->setObjectName(warn ? "BadgeWarn" : "BadgeOk");
        metaLayout->addWidget(label);
    };
    addBadge(QString("T%1").arg(record_.toolCount), false);
    addBadge(QString("G%1").arg(record_.graspCount), false);
    addBadge(QString("D%1").arg(record_.defectCount), record_.defectCount > 0);
    layout->addWidget(meta);

    connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, [this]() {
        if (thumbnail_.isNull()) {
            imageLabel_->setText("Loading");
        }
        loadThumbnail();
    });
    loadThumbnail();
}

void ThumbCard::enterEvent(QEnterEvent* event)
{
    QFrame::enterEvent(event);
    auto* effect = new QGraphicsDropShadowEffect(this);
    effect->setBlurRadius(18);
    effect->setOffset(0, 6);
    effect->setColor(QColor(0, 0, 0, 90));
    setGraphicsEffect(effect);
}

void ThumbCard::leaveEvent(QEvent* event)
{
    QFrame::leaveEvent(event);
    setGraphicsEffect(nullptr);
}

void ThumbCard::mousePressEvent(QMouseEvent* event)
{
    QFrame::mousePressEvent(event);
    if (event->button() == Qt::LeftButton) {
        emit activated(QString::fromStdString(record_.id));
    }
}

void ThumbCard::mouseDoubleClickEvent(QMouseEvent* event)
{
    QFrame::mouseDoubleClickEvent(event);
    if (event->button() == Qt::LeftButton) {
        emit activated(QString::fromStdString(record_.id));
    }
}

void ThumbCard::loadThumbnail()
{
    QPointer<ThumbCard> guard(this);
    ThumbnailCache::instance().requestThumbnail(
        QString::fromStdString(record_.id),
        QString::fromStdString(record_.annotatedImagePath),
        [guard](QPixmap pixmap) {
            if (!guard) return;
            guard->setThumbnail(pixmap);
        });
}

void ThumbCard::setThumbnail(const QPixmap& pixmap)
{
    thumbnail_ = pixmap;
    if (thumbnail_.isNull()) {
        imageLabel_->setText("Image unavailable");
        imageLabel_->setPixmap({});
        return;
    }
    imageLabel_->setText({});
    imageLabel_->setPixmap(thumbnail_);
}

QString ThumbCard::displayTime() const
{
    const QString ts = QString::fromStdString(record_.timestamp);
    const QDateTime dt = QDateTime::fromString(ts, Qt::ISODate);
    if (dt.isValid()) return dt.time().toString("HH:mm:ss");
    return ts.size() >= 19 ? ts.mid(11, 8) : ts;
}

} // namespace xcwj::ui
