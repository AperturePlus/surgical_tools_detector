#include "ui/ThumbnailCache.h"

#include <algorithm>

#include <QImage>
#include <QMetaObject>
#include <QRunnable>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ui/QtImageUtils.h"

namespace xcwj::ui {

namespace {

QImage decodeThumbnail(const QString& path, const QSize& size)
{
    cv::Mat src = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
    if (src.empty() || size.width() <= 0 || size.height() <= 0) {
        return {};
    }

    const double scale = std::min(size.width() / static_cast<double>(src.cols),
                                  size.height() / static_cast<double>(src.rows));
    const int scaledW = std::max(1, static_cast<int>(src.cols * scale));
    const int scaledH = std::max(1, static_cast<int>(src.rows * scale));
    cv::Mat scaled;
    cv::resize(src, scaled, {scaledW, scaledH}, 0, 0, cv::INTER_AREA);

    cv::Mat canvas(size.height(), size.width(), CV_8UC3, cv::Scalar(18, 24, 32));
    const int x = (size.width() - scaledW) / 2;
    const int y = (size.height() - scaledH) / 2;
    scaled.copyTo(canvas(cv::Rect(x, y, scaledW, scaledH)));
    return matToImage(canvas);
}

} // namespace

ThumbnailCache& ThumbnailCache::instance()
{
    static ThumbnailCache cache;
    return cache;
}

ThumbnailCache::ThumbnailCache(QObject* parent)
    : QObject(parent)
{
    pool_.setMaxThreadCount(1);
}

void ThumbnailCache::requestThumbnail(const QString& id,
                                      const QString& path,
                                      Callback onReady,
                                      QSize size)
{
    const QString key = keyFor(id, path, size);
    if (cache_.contains(key)) {
        lru_.removeAll(key);
        lru_.prepend(key);
        onReady(cache_.value(key));
        return;
    }

    pending_[key].append(std::move(onReady));
    if (pending_[key].size() > 1) {
        return;
    }

    pool_.start(QRunnable::create([this, key, path, size]() {
        const QImage image = decodeThumbnail(path, size);
        QMetaObject::invokeMethod(this, [this, key, image]() {
            QPixmap pixmap = image.isNull() ? QPixmap{} : QPixmap::fromImage(image);
            remember(key, pixmap);
            const auto callbacks = pending_.take(key);
            for (const auto& callback : callbacks) {
                callback(pixmap);
            }
        }, Qt::QueuedConnection);
    }));
}

QString ThumbnailCache::keyFor(const QString& id, const QString& path, const QSize& size) const
{
    const QString base = id.isEmpty() ? path : id;
    return QString("%1:%2x%3").arg(base).arg(size.width()).arg(size.height());
}

void ThumbnailCache::remember(const QString& key, const QPixmap& pixmap)
{
    cache_.insert(key, pixmap);
    lru_.removeAll(key);
    lru_.prepend(key);
    while (lru_.size() > maxEntries_) {
        cache_.remove(lru_.takeLast());
    }
}

} // namespace xcwj::ui
