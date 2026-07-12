#include "ui/QtImageUtils.h"

#include <fstream>
#include <sstream>

#include <QStringList>
#include <opencv2/imgproc.hpp>

#include "core/Renderer.h"

namespace xcwj::ui {

QImage matToImage(const cv::Mat& mat)
{
    if (mat.empty()) return {};
    cv::Mat rgb;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGBA);
    } else {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    }
    QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                 QImage::Format_RGB888);
    return image.copy();
}

QPixmap matToPixmap(const cv::Mat& mat)
{
    QImage image = matToImage(mat);
    return image.isNull() ? QPixmap{} : QPixmap::fromImage(image);
}

QString modeText(uint8_t mask)
{
    QStringList modes;
    if (mask & MODE_TOOL) modes << "Tool";
    if (mask & MODE_GRASP) modes << "Grasp";
    if (mask & MODE_DEFECT) modes << "Defect";
    return modes.join("+");
}

QString readTextFile(const QString& path)
{
    std::ifstream f(path.toStdString());
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return QString::fromStdString(ss.str());
}

} // namespace xcwj::ui
