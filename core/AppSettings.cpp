#include "core/AppSettings.h"

#include <utility>

#include "core/Renderer.h"

namespace xcwj {

namespace {

constexpr const char* kCameraId       = "app/cameraId";
constexpr const char* kCaptureDir     = "app/captureDir";
constexpr const char* kModeMask       = "defaults/modeMask";
constexpr const char* kThresholdTool  = "defaults/threshold/tool";
constexpr const char* kThresholdGrasp = "defaults/threshold/grasp";
constexpr const char* kThresholdDef   = "defaults/threshold/defect";
constexpr const char* kThemeMode      = "ui/theme";

} // namespace

AppSettings::AppSettings()
    : settings_(std::make_unique<QSettings>(QStringLiteral("XunChaWeiJian"),
                                            QStringLiteral("XunChaWeiJian")))
    , legacySettings_(std::make_unique<QSettings>(QStringLiteral("SGT"),
                                                  QStringLiteral("Detector")))
{
    migrateLegacySettings();
}

AppSettings::AppSettings(QSettings::Format format, const QString& path)
    : settings_(std::make_unique<QSettings>(path, format))
{}

AppSettings::AppSettings(std::unique_ptr<QSettings> settings,
                         std::unique_ptr<QSettings> legacySettings)
    : settings_(std::move(settings))
    , legacySettings_(std::move(legacySettings))
{
    migrateLegacySettings();
}

void AppSettings::migrateLegacySettings()
{
    if (!legacySettings_) return;

    const char* const keys[] = {
        kCameraId,
        kCaptureDir,
        kModeMask,
        kThresholdTool,
        kThresholdGrasp,
        kThresholdDef,
        kThemeMode
    };

    bool migrated = false;
    for (const char* key : keys) {
        if (settings_->contains(key) || !legacySettings_->contains(key)) continue;
        settings_->setValue(key, legacySettings_->value(key));
        migrated = true;
    }
    if (migrated) settings_->sync();
}

int AppSettings::cameraId() const
{
    return settings_->value(kCameraId, 0).toInt();
}

void AppSettings::setCameraId(int id)
{
    settings_->setValue(kCameraId, id);
}

uint8_t AppSettings::modeMask() const
{
    const int raw = settings_->value(kModeMask,
                                     static_cast<int>(MODE_TOOL)).toInt();
    const uint8_t mask = static_cast<uint8_t>(raw)
        & (MODE_TOOL | MODE_GRASP | MODE_DEFECT);
    return mask ? mask : static_cast<uint8_t>(MODE_TOOL);
}

void AppSettings::setModeMask(uint8_t mask)
{
    const uint8_t clean = mask & (MODE_TOOL | MODE_GRASP | MODE_DEFECT);
    settings_->setValue(kModeMask,
                        static_cast<int>(clean ? clean : MODE_TOOL));
}

DetectionThresholds AppSettings::defaultThresholds() const
{
    DetectionThresholds factory;  // 0.65 / 0.25 / 0.50
    DetectionThresholds out;
    out.tool   = static_cast<float>(settings_->value(kThresholdTool,  factory.tool ).toDouble());
    out.grasp  = static_cast<float>(settings_->value(kThresholdGrasp, factory.grasp).toDouble());
    out.defect = static_cast<float>(settings_->value(kThresholdDef,   factory.defect).toDouble());
    return out;
}

void AppSettings::setDefaultThresholds(const DetectionThresholds& t)
{
    settings_->setValue(kThresholdTool,  static_cast<double>(t.tool));
    settings_->setValue(kThresholdGrasp, static_cast<double>(t.grasp));
    settings_->setValue(kThresholdDef,   static_cast<double>(t.defect));
}

QString AppSettings::captureDir() const
{
    return settings_->value(kCaptureDir, QString{}).toString();
}

void AppSettings::setCaptureDir(const QString& dir)
{
    if (dir.isEmpty()) {
        settings_->remove(kCaptureDir);
    } else {
        settings_->setValue(kCaptureDir, dir);
    }
}

QString AppSettings::themeMode() const
{
    return settings_->value(kThemeMode, QStringLiteral("dark")).toString();
}

void AppSettings::setThemeMode(const QString& mode)
{
    settings_->setValue(kThemeMode, mode);
}

void AppSettings::resetDefaults()
{
    settings_->remove(kModeMask);
    settings_->remove(kThresholdTool);
    settings_->remove(kThresholdGrasp);
    settings_->remove(kThresholdDef);
}

} // namespace xcwj
