#pragma once

#include <memory>

#include <QSettings>
#include <QString>

#include "core/DetectionMetadata.h"

namespace sgt {

/// Strongly-typed facade over QSettings("SGT","Detector").
/// Centralises the key strings so callers don't sprinkle QSettings
/// access across the codebase.
class AppSettings {
public:
    AppSettings();
    AppSettings(QSettings::Format format, const QString& path);  // for tests

    int cameraId() const;
    void setCameraId(int id);

    uint8_t modeMask() const;            // never returns 0 (clamped to MODE_TOOL)
    void setModeMask(uint8_t mask);

    DetectionThresholds defaultThresholds() const;
    void setDefaultThresholds(const DetectionThresholds& t);

    QString captureDir() const;          // empty if unset (caller picks fallback)
    void setCaptureDir(const QString& dir);

    /// Removes only the keys under "defaults/" (thresholds, modeMask).
    /// Camera id and capture dir are preserved.
    void resetDefaults();

private:
    std::unique_ptr<QSettings> settings_;
};

} // namespace sgt
