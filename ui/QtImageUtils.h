#pragma once

#include <cstdint>

#include <QImage>
#include <QPixmap>
#include <QString>

#include <opencv2/core.hpp>

namespace sgt::ui {

QImage matToImage(const cv::Mat& mat);
QPixmap matToPixmap(const cv::Mat& mat);
QString modeText(uint8_t mask);
QString readTextFile(const QString& path);

} // namespace sgt::ui
