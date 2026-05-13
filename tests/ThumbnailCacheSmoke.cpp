#include <filesystem>
#include <iostream>

#include <QApplication>
#include <QEventLoop>
#include <QTimer>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ui/ThumbnailCache.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    const std::filesystem::path dir = std::filesystem::current_path() / "thumbnail_cache_smoke";
    std::filesystem::create_directories(dir);
    const std::filesystem::path imagePath = dir / "sample.jpg";

    cv::Mat sample(80, 120, CV_8UC3, cv::Scalar(20, 160, 220));
    if (!cv::imwrite(imagePath.string(), sample)) {
        std::cerr << "failed to write sample image\n";
        return 1;
    }

    QPixmap result;
    bool done = false;
    QEventLoop loop;
    QTimer::singleShot(5000, &loop, &QEventLoop::quit);
    sgt::ui::ThumbnailCache::instance().requestThumbnail(
        "sample",
        QString::fromStdString(imagePath.string()),
        [&](QPixmap pixmap) {
            result = pixmap;
            done = true;
            loop.quit();
        },
        {64, 40});
    if (!done) {
        loop.exec();
    }

    if (!done) {
        std::cerr << "thumbnail callback timed out\n";
        return 1;
    }
    if (result.isNull()) {
        std::cerr << "thumbnail pixmap is null\n";
        return 1;
    }
    if (result.size() != QSize(64, 40)) {
        std::cerr << "thumbnail size mismatch\n";
        return 1;
    }
    return 0;
}
