#include <cmath>
#include <iostream>
#include <memory>

#include <QCoreApplication>
#include <QFile>
#include <QSettings>
#include <QString>
#include <QTemporaryDir>

#include "core/AppSettings.h"
#include "core/DetectionMetadata.h"
#include "core/Renderer.h"

namespace {

bool nearly(float a, float b)
{
    return std::fabs(a - b) < 1e-4f;
}

#define CHECK(cond, msg) do {                                            \
    if (!(cond)) {                                                       \
        std::cerr << "FAIL: " << (msg) << "  at " << __LINE__ << "\n";   \
        return 1;                                                        \
    }                                                                    \
} while (0)

} // namespace

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);

    QTemporaryDir tmp;
    if (!tmp.isValid()) { std::cerr << "tmp dir failed\n"; return 1; }
    const QString path = tmp.filePath("settings.ini");

    // Round-trip: write everything, reopen, read back.
    {
        xcwj::AppSettings s(QSettings::IniFormat, path);
        s.setCameraId(2);
        s.setModeMask(static_cast<uint8_t>(xcwj::MODE_TOOL | xcwj::MODE_DEFECT));
        xcwj::DetectionThresholds t;
        t.tool = 0.42f; t.grasp = 0.33f; t.defect = 0.81f;
        s.setDefaultThresholds(t);
        s.setCaptureDir("D:/captures");
    }
    {
        xcwj::AppSettings s(QSettings::IniFormat, path);
        CHECK(s.cameraId() == 2, "cameraId round-trip");
        CHECK(s.modeMask() == (xcwj::MODE_TOOL | xcwj::MODE_DEFECT), "modeMask round-trip");
        const auto t = s.defaultThresholds();
        CHECK(nearly(t.tool,   0.42f), "tool threshold round-trip");
        CHECK(nearly(t.grasp,  0.33f), "grasp threshold round-trip");
        CHECK(nearly(t.defect, 0.81f), "defect threshold round-trip");
        CHECK(s.captureDir() == "D:/captures", "captureDir round-trip");
    }

    // resetDefaults clears defaults/* but preserves app/*.
    {
        xcwj::AppSettings s(QSettings::IniFormat, path);
        s.resetDefaults();
        CHECK(s.cameraId() == 2, "cameraId survives reset");
        CHECK(s.captureDir() == "D:/captures", "captureDir survives reset");
        CHECK(s.modeMask() == xcwj::MODE_TOOL, "modeMask resets to TOOL");
        const auto t = s.defaultThresholds();
        CHECK(nearly(t.tool,   0.65f), "tool threshold resets to factory");
        CHECK(nearly(t.grasp,  0.25f), "grasp threshold resets to factory");
        CHECK(nearly(t.defect, 0.50f), "defect threshold resets to factory");
    }

    // setCaptureDir("") removes the key.
    {
        xcwj::AppSettings s(QSettings::IniFormat, path);
        s.setCaptureDir("");
        CHECK(s.captureDir().isEmpty(), "empty captureDir round-trips as empty");
    }

    // Mask sanitization: setting MODE_TOOL|MODE_GRASP|0x80 strips the spurious bit.
    {
        xcwj::AppSettings s(QSettings::IniFormat, path);
        s.setModeMask(static_cast<uint8_t>(xcwj::MODE_TOOL | xcwj::MODE_GRASP | 0x80));
        CHECK(s.modeMask() == (xcwj::MODE_TOOL | xcwj::MODE_GRASP), "mask is sanitised");
    }

    // Legacy settings migration: values from old store are copied into new store once.
    {
        const QString legacyPath = tmp.filePath("legacy.ini");
        const QString newPath = tmp.filePath("new.ini");

        QSettings legacy(legacyPath, QSettings::IniFormat);
        legacy.setValue("app/cameraId", 5);
        legacy.setValue("app/captureDir", "D:/legacy_captures");
        legacy.setValue("defaults/modeMask", static_cast<int>(xcwj::MODE_GRASP));
        legacy.setValue("defaults/threshold/tool", 0.11);
        legacy.setValue("defaults/threshold/grasp", 0.22);
        legacy.setValue("defaults/threshold/defect", 0.33);
        legacy.setValue("ui/theme", "light");
        legacy.sync();

        xcwj::AppSettings migrated(
            std::make_unique<QSettings>(newPath, QSettings::IniFormat),
            std::make_unique<QSettings>(legacyPath, QSettings::IniFormat));
        CHECK(migrated.cameraId() == 5, "legacy cameraId migrated");
        CHECK(migrated.captureDir() == "D:/legacy_captures", "legacy captureDir migrated");
        CHECK(migrated.modeMask() == xcwj::MODE_GRASP, "legacy modeMask migrated");
        CHECK(nearly(migrated.defaultThresholds().tool, 0.11f), "legacy tool threshold migrated");
        CHECK(nearly(migrated.defaultThresholds().grasp, 0.22f), "legacy grasp threshold migrated");
        CHECK(nearly(migrated.defaultThresholds().defect, 0.33f), "legacy defect threshold migrated");
        CHECK(migrated.themeMode() == "light", "legacy theme migrated");

        QSettings fresh(newPath, QSettings::IniFormat);
        CHECK(fresh.value("app/cameraId").toInt() == 5, "new store received migrated cameraId");
        CHECK(fresh.value("ui/theme").toString() == "light", "new store received migrated theme");
    }

    QFile::remove(path);
    return 0;
}
