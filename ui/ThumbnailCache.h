#pragma once

#include <functional>

#include <QHash>
#include <QList>
#include <QObject>
#include <QPixmap>
#include <QSize>
#include <QThreadPool>

namespace sgt::ui {

class ThumbnailCache final : public QObject {
    Q_OBJECT

public:
    using Callback = std::function<void(QPixmap)>;

    static ThumbnailCache& instance();

    void requestThumbnail(const QString& id,
                          const QString& path,
                          Callback onReady,
                          QSize size = QSize(192, 120));

private:
    explicit ThumbnailCache(QObject* parent = nullptr);

    QThreadPool pool_;
    QHash<QString, QPixmap> cache_;
    QHash<QString, QList<Callback>> pending_;
    QList<QString> lru_;
    int maxEntries_ = 256;

    QString keyFor(const QString& id, const QString& path, const QSize& size) const;
    void remember(const QString& key, const QPixmap& pixmap);
};

} // namespace sgt::ui
