#include "ui/IconLoader.h"

#include <QColor>
#include <QGuiApplication>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>
#include <QPointF>
#include <QRectF>
#include <QScreen>

namespace xcwj::ui {

namespace {

constexpr qreal kCanvas = 24.0;
constexpr qreal kStroke = 1.6;

qreal devicePixelRatioForActiveScreen()
{
    if (auto* screen = QGuiApplication::primaryScreen()) {
        return screen->devicePixelRatio();
    }
    return 1.0;
}

void preparePainter(QPainter& p, const QColor& color, QSize size)
{
    p.setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    // Scale a 24x24 logical canvas to the requested icon size.
    p.scale(size.width() / kCanvas, size.height() / kCanvas);
    QPen pen(color);
    pen.setWidthF(kStroke);
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush);
}

void paintLive(QPainter& p, const QColor& color)
{
    QPainterPath body;
    body.moveTo(3.0, 8.5);
    body.lineTo(6.4, 8.5);
    body.lineTo(7.9, 6.2);
    body.quadTo(8.4, 5.5, 9.3, 5.5);
    body.lineTo(14.7, 5.5);
    body.quadTo(15.6, 5.5, 16.1, 6.2);
    body.lineTo(17.6, 8.5);
    body.lineTo(21.0, 8.5);
    body.lineTo(21.0, 17.6);
    body.quadTo(21.0, 19.3, 19.3, 19.3);
    body.lineTo(4.7, 19.3);
    body.quadTo(3.0, 19.3, 3.0, 17.6);
    body.closeSubpath();
    p.drawPath(body);
    p.drawEllipse(QPointF(12.0, 13.4), 3.4, 3.4);

    // Solid indicator dot.
    p.setPen(Qt::NoPen);
    p.setBrush(color);
    p.drawEllipse(QPointF(17.8, 10.7), 0.7, 0.7);
}

void paintGallery(QPainter& p, const QColor& /*color*/)
{
    QRectF frame(3.2, 4.6, 17.6, 14.8);
    p.drawRoundedRect(frame, 2.2, 2.2);
    p.drawEllipse(QPointF(8.0, 9.0), 1.3, 1.3);

    QPainterPath horizon;
    horizon.moveTo(3.6, 17.4);
    horizon.lineTo(9.2, 11.8);
    horizon.lineTo(13.0, 15.6);
    horizon.lineTo(15.7, 13.0);
    horizon.lineTo(20.4, 17.6);
    p.drawPath(horizon);
}

void paintSettings(QPainter& p, const QColor& /*color*/)
{
    const QPointF center(12.0, 12.0);
    // Center wheel
    p.drawEllipse(center, 2.7, 2.7);

    // Eight short teeth ringing the wheel.
    for (int i = 0; i < 8; ++i) {
        const qreal angle = i * 45.0;
        p.save();
        p.translate(center);
        p.rotate(angle);
        p.drawLine(QPointF(0.0, -4.6), QPointF(0.0, -7.0));
        p.restore();
    }
}

void paintSun(QPainter& p, const QColor& /*color*/)
{
    const QPointF center(12.0, 12.0);
    p.drawEllipse(center, 3.6, 3.6);

    // Cardinal rays
    p.drawLine(QPointF(12.0, 2.4), QPointF(12.0, 4.8));
    p.drawLine(QPointF(12.0, 19.2), QPointF(12.0, 21.6));
    p.drawLine(QPointF(2.4, 12.0), QPointF(4.8, 12.0));
    p.drawLine(QPointF(19.2, 12.0), QPointF(21.6, 12.0));

    // Diagonal rays
    const qreal a = 4.6;
    const qreal b = 6.4;
    p.drawLine(QPointF(12.0 - b * 0.7071, 12.0 - b * 0.7071),
               QPointF(12.0 - a * 0.7071, 12.0 - a * 0.7071));
    p.drawLine(QPointF(12.0 + b * 0.7071, 12.0 - b * 0.7071),
               QPointF(12.0 + a * 0.7071, 12.0 - a * 0.7071));
    p.drawLine(QPointF(12.0 - b * 0.7071, 12.0 + b * 0.7071),
               QPointF(12.0 - a * 0.7071, 12.0 + a * 0.7071));
    p.drawLine(QPointF(12.0 + b * 0.7071, 12.0 + b * 0.7071),
               QPointF(12.0 + a * 0.7071, 12.0 + a * 0.7071));
}

void paintMoon(QPainter& p, const QColor& /*color*/)
{
    // Outer disc, then mask with an offset disc to leave a crescent.
    QPainterPath outer;
    outer.addEllipse(QPointF(12.0, 12.0), 7.2, 7.2);
    QPainterPath bite;
    bite.addEllipse(QPointF(14.5, 10.3), 6.6, 6.6);
    p.drawPath(outer.subtracted(bite));
}

bool dispatch(const QString& alias, QPainter& p, const QColor& color)
{
    if (alias == "nav-live")     { paintLive(p, color);     return true; }
    if (alias == "nav-gallery")  { paintGallery(p, color);  return true; }
    if (alias == "nav-settings") { paintSettings(p, color); return true; }
    if (alias == "theme-sun")    { paintSun(p, color);      return true; }
    if (alias == "theme-moon")   { paintMoon(p, color);     return true; }
    return false;
}

} // namespace

QIcon IconLoader::load(const QString& alias, const QString& strokeColor, QSize size)
{
    const QColor color(strokeColor);
    if (!color.isValid()) return {};

    const qreal dpr = devicePixelRatioForActiveScreen();
    QPixmap pixmap(size * dpr);
    pixmap.setDevicePixelRatio(dpr);
    pixmap.fill(Qt::transparent);

    {
        QPainter painter(&pixmap);
        preparePainter(painter, color, size);
        if (!dispatch(alias, painter, color)) {
            return {};
        }
    }

    QIcon icon;
    icon.addPixmap(pixmap, QIcon::Normal, QIcon::Off);
    icon.addPixmap(pixmap, QIcon::Normal, QIcon::On);
    icon.addPixmap(pixmap, QIcon::Active, QIcon::Off);
    icon.addPixmap(pixmap, QIcon::Active, QIcon::On);
    return icon;
}

} // namespace xcwj::ui
