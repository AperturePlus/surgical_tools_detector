#include <cmath>
#include <iostream>

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
        sgt::AppSettings s(QSettings::IniFormat, path);
        s.setCameraId(2);
        s.setModeMask(static_cast<uint8_t>(sgt::MODE_TOOL | sgt::MODE_DEFECT));
        sgt::DetectionThresholds t;
        t.tool = 0.42f; t.grasp = 0.33f; t.defect = 0.81f;
        s.setDefaultThresholds(t);
        s.setCaptureDir("D:/captures");
    }
    {
        sgt::AppSettings s(QSettings::IniFormat, path);
        CHECK(s.cameraId() == 2, "cameraId round-trip");
        CHECK(s.modeMask() == (sgt::MODE_TOOL | sgt::MODE_DEFECT), "modeMask round-trip");
        const auto t = s.defaultThresholds();
        CHECK(nearly(t.tool,   0.42f), "tool threshold round-trip");
        CHECK(nearly(t.grasp,  0.33f), "grasp threshold round-trip");
        CHECK(nearly(t.defect, 0.81f), "defect threshold round-trip");
        CHECK(s.captureDir() == "D:/captures", "captureDir round-trip");
    }

    // resetDefaults clears defaults/* but preserves app/*.
    {
        sgt::AppSettings s(QSettings::IniFormat, path);
        s.resetDefaults();
        CHECK(s.cameraId() == 2, "cameraId survives reset");
        CHECK(s.captureDir() == "D:/captures", "captureDir survives reset");
        CHECK(s.modeMask() == sgt::MODE_TOOL, "modeMask resets to TOOL");
        const auto t = s.defaultThresholds();
        CHECK(nearly(t.tool,   0.65f), "tool threshold resets to factory");
        CHECK(nearly(t.grasp,  0.25f), "grasp threshold resets to factory");
        CHECK(nearly(t.defect, 0.50f), "defect threshold resets to factory");
    }

    // setCaptureDir("") removes the key.
    {
        sgt::AppSettings s(QSettings::IniFormat, path);
        s.setCaptureDir("");
        CHECK(s.captureDir().isEmpty(), "empty captureDir round-trips as empty");
    }

    // Mask sanitization: setting MODE_TOOL|MODE_GRASP|0x80 strips the spurious bit.
    {
        sgt::AppSettings s(QSettings::IniFormat, path);
        s.setModeMask(static_cast<uint8_t>(sgt::MODE_TOOL | sgt::MODE_GRASP | 0x80));
        CHECK(s.modeMask() == (sgt::MODE_TOOL | sgt::MODE_GRASP), "mask is sanitised");
    }

    QFile::remove(path);
    return 0;
}
