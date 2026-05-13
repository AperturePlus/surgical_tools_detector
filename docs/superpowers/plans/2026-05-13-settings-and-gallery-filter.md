# Settings Page + Gallery Date Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the disabled Sidebar Settings entry with a working Settings page that persists user preferences (camera id, default thresholds, default mode mask, capture output directory) via `QSettings`, and add a date-range filter bar to the Gallery (All / Today / Last 7 / Last 30 / Custom).

**Architecture:** A new `core/AppSettings` class wraps `QSettings("SGT","Detector")` with strongly typed accessors. `main.cpp` reads it on startup into `AppOptions` (CLI args still win for camera id), and `AppShell` seeds its `thresholds_` from the persisted defaults. `ui/SettingsPage` is the page widget added to the existing `QStackedWidget`; the Sidebar's Settings nav button is enabled. `ui/GalleryFilterBar` is a chip bar inserted into `GalleryPage`, emitting a `(QDate from, QDate to)` range that composes with the existing search filter.

**Tech Stack:** Qt 6 Widgets, C++17, OpenCV (`core`, `imgproc`, `imgcodecs`, `videoio`, `dnn`), ONNX Runtime 1.24, CMake + Ninja under MSYS2 MinGW64, CTest.

**Spec:** `docs/superpowers/specs/2026-05-13-settings-and-gallery-filter-design.md`

---

## Conventions

- Branch: continue on `codex/enhance`.
- Commits: one per task. Use Conventional Commits (`feat(ui): …`, `feat(core): …`, `refactor(ui): …`, `test: …`, `build: …`).
- New non-UI code goes in `namespace sgt`; new UI widgets in `namespace sgt::ui`.
- Compile/test loop: `cmake --build build` then `ctest --test-dir build --output-on-failure`. UI behavior verified by build + manual smoke launch (`./build/SGTDetector.exe`).
- The build directory already exists as `build/`; if not, run `cmake -B build -G Ninja` once. Post-build hooks copy ONNX Runtime, models, labels, and Qt platform plugins.
- `MODE_TOOL` / `MODE_GRASP` / `MODE_DEFECT` live in `core/Renderer.h` (`sgt::ModeMask`); they are visible without qualification inside `namespace sgt` and `namespace sgt::ui`.
- `DetectionThresholds` lives in `core/DetectionMetadata.h` with defaults `tool=0.25, grasp=0.25, defect=0.50`.

---

## Task 1: AppSettings facade + smoke test

**Files:**
- Create: `core/AppSettings.h`
- Create: `core/AppSettings.cpp`
- Create: `tests/AppSettingsSmoke.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1.1: Write `core/AppSettings.h`**

```cpp
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
```

- [ ] **Step 1.2: Write `core/AppSettings.cpp`**

```cpp
#include "core/AppSettings.h"

#include "core/Renderer.h"

namespace sgt {

namespace {

constexpr const char* kCameraId       = "app/cameraId";
constexpr const char* kCaptureDir     = "app/captureDir";
constexpr const char* kModeMask       = "defaults/modeMask";
constexpr const char* kThresholdTool  = "defaults/threshold/tool";
constexpr const char* kThresholdGrasp = "defaults/threshold/grasp";
constexpr const char* kThresholdDef   = "defaults/threshold/defect";

} // namespace

AppSettings::AppSettings()
    : settings_(std::make_unique<QSettings>(QStringLiteral("SGT"),
                                            QStringLiteral("Detector")))
{}

AppSettings::AppSettings(QSettings::Format format, const QString& path)
    : settings_(std::make_unique<QSettings>(path, format))
{}

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
    DetectionThresholds factory;  // 0.25 / 0.25 / 0.50
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

void AppSettings::resetDefaults()
{
    settings_->remove(kModeMask);
    settings_->remove(kThresholdTool);
    settings_->remove(kThresholdGrasp);
    settings_->remove(kThresholdDef);
}

} // namespace sgt
```

- [ ] **Step 1.3: Write `tests/AppSettingsSmoke.cpp`**

```cpp
#include <cmath>
#include <cstdio>
#include <iostream>

#include <QCoreApplication>
#include <QFile>
#include <QSettings>
#include <QString>
#include <QTemporaryFile>

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

    QTemporaryFile tmp;
    if (!tmp.open()) { std::cerr << "tmp file failed\n"; return 1; }
    const QString path = tmp.fileName();
    tmp.close();

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
        CHECK(nearly(t.tool,   0.25f), "tool threshold resets to factory");
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
```

- [ ] **Step 1.4: Update `CMakeLists.txt`**

In the `set(SGT_SOURCES ...)` block, append after `core/CaptureStore.cpp`:

```cmake
    core/AppSettings.cpp
```

In the `set(SGT_HEADERS ...)` block, append after `core/CaptureStore.h`:

```cmake
    core/AppSettings.h
```

After the existing `add_executable(ThumbnailCacheSmoke ...)` block, add:

```cmake
add_executable(AppSettingsSmoke
    tests/AppSettingsSmoke.cpp
    core/AppSettings.cpp
)
target_include_directories(AppSettingsSmoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AppSettingsSmoke PRIVATE Qt6::Core)
```

After the existing `add_test(NAME ThumbnailCacheSmoke ...)` line, add:

```cmake
add_test(NAME AppSettingsSmoke COMMAND AppSettingsSmoke)
```

- [ ] **Step 1.5: Build**

Run:
```bash
cmake --build build
```
Expected: clean build, both `SGTDetector.exe` and `AppSettingsSmoke.exe` produced.

- [ ] **Step 1.6: Run the smoke test**

Run:
```bash
ctest --test-dir build -R AppSettingsSmoke --output-on-failure
```
Expected: `1/1 Test #N: AppSettingsSmoke ......... Passed`. Failures of "round-trip" indicate a key spelling drift; failures of "sanitised" indicate a missing `& (MODE_TOOL|MODE_GRASP|MODE_DEFECT)` in `setModeMask`.

- [ ] **Step 1.7: Commit**

```bash
git add core/AppSettings.h core/AppSettings.cpp \
        tests/AppSettingsSmoke.cpp CMakeLists.txt
git commit -m "feat(core): add QSettings-backed AppSettings facade"
```

---

## Task 2: AppOptions extension + main.cpp wiring + AppShell threshold seeding

This task is purely plumbing — no new UI yet. After it lands, `AppSettings` values written by hand into the INI/registry are picked up at startup, and CLI args still override the camera id.

**Files:**
- Modify: `ui/AppShell.h`
- Modify: `ui/AppShell.cpp`
- Modify: `main.cpp`

- [ ] **Step 2.1: Extend `AppOptions` in `ui/AppShell.h`**

Replace the existing struct (lines 25-31):

```cpp
struct AppOptions {
    int cameraId = 0;
    bool cameraIdFromCli = false;       // CLI numeric arg sets this true
    uint8_t modeMask = MODE_TOOL;
    DetectionThresholds thresholds;     // seeded from AppSettings::defaultThresholds()
    std::string toolModel;
    std::string graspModel;
    std::string defectModel;
};
```

- [ ] **Step 2.2: Seed `thresholds_` in `ui/AppShell.cpp`**

Modify the constructor body. After the existing initializer list (line 27, `, activeMask_(...)`), add an initializer for `thresholds_`:

Change:
```cpp
    , activeMask_(opts_.modeMask ? opts_.modeMask : MODE_TOOL)
{
```

to:
```cpp
    , activeMask_(opts_.modeMask ? opts_.modeMask : MODE_TOOL)
    , thresholds_(opts_.thresholds)
{
    engine_->setThresholds(thresholds_);
```

The `setThresholds` call ensures the engine uses the persisted defaults from frame 1 (otherwise it picks up the engine-internal defaults until the user touches a slider).

- [ ] **Step 2.3: Mark CLI camera id in `main.cpp::parseArgs`**

In the `else if (!camSet)` branch (currently lines 64-66), add `opts.cameraIdFromCli = true;` next to `camSet = true;`:

```cpp
        } else if (!camSet) {
            opts.cameraId = std::atoi(arg.c_str());
            opts.cameraIdFromCli = true;
            camSet = true;
        }
```

- [ ] **Step 2.4: Read AppSettings in `main.cpp::main`**

Add the include near the other core includes (currently around line 12):

```cpp
#include "core/AppSettings.h"
```

Then, in `main()` between `parseArgs` (currently line 88) and `QApplication app(...)` (line 94), insert AppSettings overrides. Replace the block from "QApplication app(argc, argv);" through the `auto store = ...` line (currently lines 94-104) with:

```cpp
    QApplication app(argc, argv);
    sgt::ui::ThemeManager::instance().apply(&app);

    sgt::AppSettings settings;
    if (!opts.cameraIdFromCli) {
        opts.cameraId = settings.cameraId();
    }
    opts.modeMask = settings.modeMask();
    opts.thresholds = settings.defaultThresholds();

    fs::path exeDir = fs::path(argv[0]).parent_path();
    std::string toolPath = resolveAsset(exeDir, opts.toolModel);
    std::string graspPath = resolveAsset(exeDir, opts.graspModel);
    std::string defectPath = resolveAsset(exeDir, opts.defectModel);
    std::string dictPath = resolveAsset(exeDir, "labels.dict");

    auto engine = std::make_unique<sgt::DetectionEngine>(toolPath, graspPath, defectPath, dictPath);

    QString captureDirPref = settings.captureDir();
    fs::path captureDir = captureDirPref.isEmpty()
        ? (exeDir / "captures")
        : fs::path(captureDirPref.toStdString());
    auto store = std::make_unique<sgt::CaptureStore>(captureDir);
```

Note: CLI arg `--mode` is parsed before `parseArgs` returns, so its value is in `opts.modeMask` when we reach this point — we then overwrite with the persisted `modeMask`. That is intentional for this task: the persisted setting is the user's "default mode" preference. CLI `--mode` users can clear the QSettings key from Settings → Reset to factory; if there is demand, a future `modeMaskFromCli` flag can mirror the camera-id pattern. (Capture this as a known limitation; not a regression because nothing on `main` previously persisted the mode.)

- [ ] **Step 2.5: Build**

Run:
```bash
cmake --build build
```
Expected: clean build. If you see "no member named `cameraIdFromCli`", the AppOptions edit (Step 2.1) is missing or in the wrong namespace.

- [ ] **Step 2.6: Manual smoke**

Run:
```bash
./build/SGTDetector.exe
```
Expected: app launches identically to before (camera, gallery, theme toggle still functional). If your `QSettings` registry has stale keys from earlier sessions, you may see a different startup mode — that's the new behavior, not a regression.

- [ ] **Step 2.7: Commit**

```bash
git add ui/AppShell.h ui/AppShell.cpp main.cpp
git commit -m "feat(app): seed AppOptions from AppSettings; CLI camera id wins"
```

---

## Task 3: SettingsPage UI + Sidebar enable + add to stack

**Files:**
- Create: `ui/SettingsPage.h`
- Create: `ui/SettingsPage.cpp`
- Modify: `ui/Sidebar.cpp`
- Modify: `ui/AppShell.h`
- Modify: `ui/AppShell.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 3.1: Write `ui/SettingsPage.h`**

```cpp
#pragma once

#include <QWidget>

#include "core/AppSettings.h"

class QCheckBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSlider;
class QSpinBox;

namespace sgt::ui {

class Card;

class SettingsPage final : public QWidget {
    Q_OBJECT

public:
    explicit SettingsPage(QWidget* parent = nullptr);

private:
    AppSettings settings_;

    QSpinBox* cameraSpin_ = nullptr;

    QSlider* toolSlider_ = nullptr;
    QSlider* graspSlider_ = nullptr;
    QSlider* defectSlider_ = nullptr;
    QLabel* toolValueLabel_ = nullptr;
    QLabel* graspValueLabel_ = nullptr;
    QLabel* defectValueLabel_ = nullptr;
    QCheckBox* toolModeBox_ = nullptr;
    QCheckBox* graspModeBox_ = nullptr;
    QCheckBox* defectModeBox_ = nullptr;
    QPushButton* resetButton_ = nullptr;

    QLineEdit* captureDirEdit_ = nullptr;
    QPushButton* browseButton_ = nullptr;

    Card* buildSourceCard();
    Card* buildDefaultsCard();
    Card* buildStorageCard();

    QSlider* makeThresholdSlider(float initial, QLabel* valueLabel);
    void writeModeMask();
    void populateFromSettings();
    static QString thresholdLabelText(int percent);
};

} // namespace sgt::ui
```

- [ ] **Step 3.2: Write `ui/SettingsPage.cpp`**

```cpp
#include "ui/SettingsPage.h"

#include <QCheckBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

#include "core/Renderer.h"
#include "ui/Card.h"

namespace sgt::ui {

namespace {

constexpr int kSliderMin = 5;
constexpr int kSliderMax = 95;

QLabel* makeHintLabel()
{
    auto* hint = new QLabel("Applies on next launch.");
    hint->setObjectName("SubtleText");
    return hint;
}

} // namespace

SettingsPage::SettingsPage(QWidget* parent)
    : QWidget(parent)
{
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(22, 18, 22, 18);
    root->setSpacing(14);

    auto* title = new QLabel("Settings");
    title->setObjectName("AppTitle");
    auto* subtitle = new QLabel("Preferences are saved automatically. Some changes apply on next launch.");
    subtitle->setObjectName("SubtleText");
    root->addWidget(title);
    root->addWidget(subtitle);

    root->addWidget(buildSourceCard());
    root->addWidget(buildDefaultsCard());
    root->addWidget(buildStorageCard());
    root->addStretch();

    populateFromSettings();
}

Card* SettingsPage::buildSourceCard()
{
    auto* card = new Card("Capture Source");
    auto* row = new QHBoxLayout();
    row->setSpacing(10);
    auto* label = new QLabel("Camera index");
    cameraSpin_ = new QSpinBox();
    cameraSpin_->setRange(0, 15);
    cameraSpin_->setFixedWidth(96);
    row->addWidget(label);
    row->addStretch();
    row->addWidget(cameraSpin_);
    card->bodyLayout()->addLayout(row);
    card->bodyLayout()->addWidget(makeHintLabel());

    connect(cameraSpin_, qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int value) { settings_.setCameraId(value); });
    return card;
}

Card* SettingsPage::buildDefaultsCard()
{
    auto* card = new Card("Detection Defaults");
    auto* body = card->bodyLayout();

    auto addSliderRow = [&](const QString& name,
                            QSlider*& slider, QLabel*& valueLabel) {
        auto* row = new QHBoxLayout();
        auto* nameLabel = new QLabel(name);
        valueLabel = new QLabel("--");
        valueLabel->setObjectName("SubtleText");
        valueLabel->setMinimumWidth(48);
        valueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        row->addWidget(nameLabel);
        row->addStretch();
        row->addWidget(valueLabel);
        body->addLayout(row);
        slider = makeThresholdSlider(0.5f, valueLabel);
        body->addWidget(slider);
    };

    addSliderRow("Tool threshold",   toolSlider_,   toolValueLabel_);
    addSliderRow("Grasp threshold",  graspSlider_,  graspValueLabel_);
    addSliderRow("Defect threshold", defectSlider_, defectValueLabel_);

    auto* modesRow = new QHBoxLayout();
    auto* modesLabel = new QLabel("Active modes");
    toolModeBox_   = new QCheckBox("Tool");
    graspModeBox_  = new QCheckBox("Grasp");
    defectModeBox_ = new QCheckBox("Defect");
    modesRow->addWidget(modesLabel);
    modesRow->addStretch();
    modesRow->addWidget(toolModeBox_);
    modesRow->addWidget(graspModeBox_);
    modesRow->addWidget(defectModeBox_);
    body->addLayout(modesRow);

    auto* footer = new QHBoxLayout();
    footer->addWidget(makeHintLabel());
    footer->addStretch();
    resetButton_ = new QPushButton("Reset to factory");
    resetButton_->setCursor(Qt::PointingHandCursor);
    footer->addWidget(resetButton_);
    body->addLayout(footer);

    auto wireSlider = [this](QSlider* slider, QLabel* label, float DetectionThresholds::* member) {
        connect(slider, &QSlider::valueChanged, this,
                [this, label, member](int value) {
                    label->setText(thresholdLabelText(value));
                    DetectionThresholds t = settings_.defaultThresholds();
                    t.*member = value / 100.0f;
                    settings_.setDefaultThresholds(t);
                });
    };
    wireSlider(toolSlider_,   toolValueLabel_,   &DetectionThresholds::tool);
    wireSlider(graspSlider_,  graspValueLabel_,  &DetectionThresholds::grasp);
    wireSlider(defectSlider_, defectValueLabel_, &DetectionThresholds::defect);

    auto wireModeBox = [this](QCheckBox* box) {
        connect(box, &QCheckBox::toggled, this, [this]() { writeModeMask(); });
    };
    wireModeBox(toolModeBox_);
    wireModeBox(graspModeBox_);
    wireModeBox(defectModeBox_);

    connect(resetButton_, &QPushButton::clicked, this, [this]() {
        settings_.resetDefaults();
        populateFromSettings();
    });

    return card;
}

Card* SettingsPage::buildStorageCard()
{
    auto* card = new Card("Capture Storage");
    auto* row = new QHBoxLayout();
    row->setSpacing(8);
    auto* label = new QLabel("Output folder");
    captureDirEdit_ = new QLineEdit();
    captureDirEdit_->setReadOnly(true);
    captureDirEdit_->setPlaceholderText("Default: <exe-dir>/captures");
    browseButton_ = new QPushButton("Browse...");
    browseButton_->setCursor(Qt::PointingHandCursor);
    row->addWidget(label);
    row->addWidget(captureDirEdit_, 1);
    row->addWidget(browseButton_);
    card->bodyLayout()->addLayout(row);
    card->bodyLayout()->addWidget(makeHintLabel());

    connect(browseButton_, &QPushButton::clicked, this, [this]() {
        const QString chosen = QFileDialog::getExistingDirectory(
            this, "Choose capture output folder", captureDirEdit_->text());
        if (chosen.isEmpty()) return;
        captureDirEdit_->setText(chosen);
        settings_.setCaptureDir(chosen);
    });
    return card;
}

QSlider* SettingsPage::makeThresholdSlider(float initial, QLabel* /*valueLabel*/)
{
    auto* slider = new QSlider(Qt::Horizontal);
    slider->setRange(kSliderMin, kSliderMax);
    slider->setValue(static_cast<int>(initial * 100.0f));
    slider->setMinimumHeight(30);
    return slider;
}

void SettingsPage::writeModeMask()
{
    uint8_t mask = 0;
    if (toolModeBox_->isChecked())   mask |= MODE_TOOL;
    if (graspModeBox_->isChecked())  mask |= MODE_GRASP;
    if (defectModeBox_->isChecked()) mask |= MODE_DEFECT;
    if (!mask) {
        // Refuse the empty mask; force Tool back on without re-emitting.
        QSignalBlocker block(toolModeBox_);
        toolModeBox_->setChecked(true);
        mask = MODE_TOOL;
    }
    settings_.setModeMask(mask);
}

void SettingsPage::populateFromSettings()
{
    {
        QSignalBlocker b(cameraSpin_);
        cameraSpin_->setValue(settings_.cameraId());
    }

    const auto t = settings_.defaultThresholds();
    auto setSlider = [](QSlider* s, QLabel* l, float v) {
        QSignalBlocker b(s);
        const int percent = static_cast<int>(v * 100.0f);
        s->setValue(percent);
        l->setText(thresholdLabelText(percent));
    };
    setSlider(toolSlider_,   toolValueLabel_,   t.tool);
    setSlider(graspSlider_,  graspValueLabel_,  t.grasp);
    setSlider(defectSlider_, defectValueLabel_, t.defect);

    const uint8_t mask = settings_.modeMask();
    auto setBox = [](QCheckBox* box, bool checked) {
        QSignalBlocker b(box);
        box->setChecked(checked);
    };
    setBox(toolModeBox_,   mask & MODE_TOOL);
    setBox(graspModeBox_,  mask & MODE_GRASP);
    setBox(defectModeBox_, mask & MODE_DEFECT);

    captureDirEdit_->setText(settings_.captureDir());
}

QString SettingsPage::thresholdLabelText(int percent)
{
    return QStringLiteral("%1%").arg(percent);
}

} // namespace sgt::ui
```

- [ ] **Step 3.3: Enable the Settings nav button**

In `ui/Sidebar.cpp`, remove these two lines (currently around line 36-37):

```cpp
    auto* settings = makeNavButton("Settings", "nav-settings", 2);
    settings->setEnabled(false);
```

Replace with:

```cpp
    auto* settings = makeNavButton("Settings", "nav-settings", 2);
```

(Just delete the `setEnabled(false)` line; nothing else changes.)

- [ ] **Step 3.4: Add SettingsPage to AppShell stack**

In `ui/AppShell.h`, add a forward declaration:

```cpp
class SettingsPage;
```

next to the existing `class GalleryPage;`. And add a member next to the existing pages:

```cpp
    SettingsPage* settingsPage_ = nullptr;
```

In `ui/AppShell.cpp`:

Add the include:
```cpp
#include "ui/SettingsPage.h"
```

In `buildUi()`, after the existing `galleryPage_ = new GalleryPage();` (line 54), insert:

```cpp
    settingsPage_ = new SettingsPage();
```

After the existing `stack_->addWidget(galleryPage_);` (line 59), insert:

```cpp
    stack_->addWidget(settingsPage_);
```

The Sidebar already emits `pageRequested(2)` for the Settings button; `wireEvents` already routes that to `stack_->setCurrentIndex(index)` with bounds-check, so no further wiring is needed.

- [ ] **Step 3.5: Update `CMakeLists.txt`**

Append to `SGT_SOURCES` (after `ui/Sidebar.cpp`):

```cmake
    ui/SettingsPage.cpp
```

Append to `SGT_HEADERS` (after `ui/Sidebar.h`):

```cmake
    ui/SettingsPage.h
```

- [ ] **Step 3.6: Build**

Run:
```bash
cmake --build build
```
Expected: clean build. Any error about `Card::bodyLayout()` being missing means the include path is wrong — `ui/Card.h` already exposes `bodyLayout()`; verify the include in `SettingsPage.cpp`.

- [ ] **Step 3.7: Manual smoke**

Run:
```bash
./build/SGTDetector.exe
```

Verify:
- Sidebar Settings icon is now clickable; clicking it shows the Settings page.
- Capture Source SpinBox shows `0` (or your saved value).
- Detection Defaults shows three sliders with Tool 25%, Grasp 25%, Defect 50% on first launch.
- All three Active modes checkboxes default to Tool only checked (or whatever was persisted).
- Move Tool slider to 60%, then close and relaunch the app → Settings page now opens with Tool slider at 60% and Live page's Tool threshold also starts at 60%.
- `Reset to factory` puts the three sliders back to 25/25/50 and resets modes to Tool-only without touching Camera index or Output folder.
- Browse opens a folder picker; selecting a folder writes the path into the text field.

- [ ] **Step 3.8: Commit**

```bash
git add ui/SettingsPage.h ui/SettingsPage.cpp \
        ui/Sidebar.cpp ui/AppShell.h ui/AppShell.cpp \
        CMakeLists.txt
git commit -m "feat(ui): add Settings page and enable Sidebar entry"
```

---

## Task 4: GalleryFilterBar widget + QSS

**Files:**
- Create: `ui/GalleryFilterBar.h`
- Create: `ui/GalleryFilterBar.cpp`
- Modify: `assets/qss/base.qss`
- Modify: `CMakeLists.txt`

- [ ] **Step 4.1: Write `ui/GalleryFilterBar.h`**

```cpp
#pragma once

#include <QDate>
#include <QFrame>

class QButtonGroup;
class QDateEdit;
class QPushButton;

namespace sgt::ui {

class GalleryFilterBar final : public QFrame {
    Q_OBJECT

public:
    explicit GalleryFilterBar(QWidget* parent = nullptr);

    QDate from() const { return from_; }
    QDate to()   const { return to_; }

signals:
    /// Emitted whenever the active range changes.
    /// `from` or `to` may be invalid -> open on that side. Both invalid -> no filter.
    void rangeChanged(QDate from, QDate to);

private:
    enum Preset { PresetAll = 0, PresetToday, PresetLast7, PresetLast30, PresetCustom };

    QButtonGroup* group_ = nullptr;
    QPushButton* allBtn_ = nullptr;
    QPushButton* todayBtn_ = nullptr;
    QPushButton* last7Btn_ = nullptr;
    QPushButton* last30Btn_ = nullptr;
    QPushButton* customBtn_ = nullptr;

    QDateEdit* fromEdit_ = nullptr;
    QDateEdit* toEdit_ = nullptr;

    QDate from_;
    QDate to_;

    QPushButton* makeChip(const QString& text, Preset preset);
    void onPresetChanged(int presetId);
    void onCustomDateChanged();
    void emitIfChanged(QDate from, QDate to);
};

} // namespace sgt::ui
```

- [ ] **Step 4.2: Write `ui/GalleryFilterBar.cpp`**

```cpp
#include "ui/GalleryFilterBar.h"

#include <QButtonGroup>
#include <QDateEdit>
#include <QHBoxLayout>
#include <QPushButton>

namespace sgt::ui {

GalleryFilterBar::GalleryFilterBar(QWidget* parent)
    : QFrame(parent)
{
    setObjectName("GalleryFilterBar");
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    group_ = new QButtonGroup(this);
    group_->setExclusive(true);

    allBtn_    = makeChip("All",     PresetAll);
    todayBtn_  = makeChip("Today",   PresetToday);
    last7Btn_  = makeChip("Last 7",  PresetLast7);
    last30Btn_ = makeChip("Last 30", PresetLast30);
    customBtn_ = makeChip("Custom",  PresetCustom);

    layout->addWidget(allBtn_);
    layout->addWidget(todayBtn_);
    layout->addWidget(last7Btn_);
    layout->addWidget(last30Btn_);
    layout->addWidget(customBtn_);

    fromEdit_ = new QDateEdit(QDate::currentDate().addDays(-30));
    toEdit_   = new QDateEdit(QDate::currentDate());
    for (auto* edit : { fromEdit_, toEdit_ }) {
        edit->setDisplayFormat("yyyy-MM-dd");
        edit->setCalendarPopup(true);
        edit->setVisible(false);
    }
    layout->addSpacing(8);
    layout->addWidget(fromEdit_);
    layout->addWidget(toEdit_);
    layout->addStretch();

    connect(group_, &QButtonGroup::idClicked, this,
            [this](int id) { onPresetChanged(id); });
    connect(fromEdit_, &QDateEdit::dateChanged, this,
            [this]() { onCustomDateChanged(); });
    connect(toEdit_, &QDateEdit::dateChanged, this,
            [this]() { onCustomDateChanged(); });

    allBtn_->setChecked(true);
    onPresetChanged(PresetAll);
}

QPushButton* GalleryFilterBar::makeChip(const QString& text, Preset preset)
{
    auto* button = new QPushButton(text);
    button->setObjectName("FilterChip");
    button->setCheckable(true);
    button->setCursor(Qt::PointingHandCursor);
    group_->addButton(button, static_cast<int>(preset));
    return button;
}

void GalleryFilterBar::onPresetChanged(int presetId)
{
    const QDate today = QDate::currentDate();
    const bool custom = (presetId == PresetCustom);
    fromEdit_->setVisible(custom);
    toEdit_->setVisible(custom);

    QDate from;
    QDate to;
    switch (presetId) {
        case PresetAll:
            // both invalid -> no filter
            break;
        case PresetToday:
            from = today;
            to = today;
            break;
        case PresetLast7:
            from = today.addDays(-6);
            to = today;
            break;
        case PresetLast30:
            from = today.addDays(-29);
            to = today;
            break;
        case PresetCustom:
            from = fromEdit_->date();
            to = toEdit_->date();
            break;
    }
    emitIfChanged(from, to);
}

void GalleryFilterBar::onCustomDateChanged()
{
    if (!customBtn_->isChecked()) return;
    emitIfChanged(fromEdit_->date(), toEdit_->date());
}

void GalleryFilterBar::emitIfChanged(QDate from, QDate to)
{
    if (from == from_ && to == to_) return;
    from_ = from;
    to_ = to;
    emit rangeChanged(from_, to_);
}

} // namespace sgt::ui
```

- [ ] **Step 4.3: Add QSS rules in `assets/qss/base.qss`**

Append the following block at the end of the file (after the existing `QToolButton:checked { ... }` block):

```css
QFrame#GalleryFilterBar { background: transparent; }

QPushButton#FilterChip {
    background: {{surface}};
    color: {{textSecondary}};
    border: 1px solid {{border}};
    border-radius: 14px;
    padding: 6px 14px;
}
QPushButton#FilterChip:hover    { color: {{textPrimary}}; }
QPushButton#FilterChip:checked  {
    background: {{accent}};
    color: #FFFFFF;
    border-color: {{accent}};
}

QDateEdit {
    background: {{surface}};
    color: {{textPrimary}};
    border: 1px solid {{border}};
    border-radius: 6px;
    padding: 4px 8px;
}
QDateEdit::drop-down { width: 18px; border: none; }
```

- [ ] **Step 4.4: Update `CMakeLists.txt`**

Append to `SGT_SOURCES` (after `ui/GalleryPage.cpp`):

```cmake
    ui/GalleryFilterBar.cpp
```

Append to `SGT_HEADERS` (after `ui/GalleryPage.h`):

```cmake
    ui/GalleryFilterBar.h
```

- [ ] **Step 4.5: Build**

Run:
```bash
cmake --build build
```
Expected: clean build. The widget is unused at this point (next task wires it into `GalleryPage`).

- [ ] **Step 4.6: Commit**

```bash
git add ui/GalleryFilterBar.h ui/GalleryFilterBar.cpp \
        assets/qss/base.qss CMakeLists.txt
git commit -m "feat(ui): add GalleryFilterBar widget with date presets"
```

---

## Task 5: GalleryPage integrates filter bar

**Files:**
- Modify: `ui/GalleryPage.h`
- Modify: `ui/GalleryPage.cpp`

- [ ] **Step 5.1: Rewrite `ui/GalleryPage.h`**

Replace the file with:

```cpp
#pragma once

#include <vector>

#include <QDate>
#include <QWidget>

#include "core/CaptureStore.h"

class QLabel;
class QLineEdit;
class QVBoxLayout;

namespace sgt::ui {

class GalleryFilterBar;

class GalleryPage final : public QWidget {
    Q_OBJECT

public:
    explicit GalleryPage(QWidget* parent = nullptr);

    void setRecords(const std::vector<CaptureRecord>& records);
    void addRecord(const CaptureRecord& record);

private:
    std::vector<CaptureRecord> records_;
    QLabel* titleLabel_ = nullptr;
    QLineEdit* searchEdit_ = nullptr;
    GalleryFilterBar* filterBar_ = nullptr;
    QWidget* content_ = nullptr;
    QVBoxLayout* contentLayout_ = nullptr;
    QDate filterFrom_;
    QDate filterTo_;

    void rebuild();
    void openDetail(const QString& id);
    bool matchesFilter(const CaptureRecord& record) const;
    bool matchesSearch(const CaptureRecord& record) const;
    bool matchesDateRange(const CaptureRecord& record) const;
    QString dateHeading(const QString& timestamp) const;
};

} // namespace sgt::ui
```

- [ ] **Step 5.2: Modify `ui/GalleryPage.cpp`**

Add the new include:
```cpp
#include "ui/GalleryFilterBar.h"
```

In the constructor, after the existing `root->addLayout(header);` (currently around line 52), insert:

```cpp
    filterBar_ = new GalleryFilterBar();
    root->addWidget(filterBar_);
    connect(filterBar_, &GalleryFilterBar::rangeChanged, this,
            [this](QDate from, QDate to) {
                filterFrom_ = from;
                filterTo_ = to;
                rebuild();
            });
```

Replace the existing `matchesFilter` method body (currently lines 148-156) with:

```cpp
bool GalleryPage::matchesFilter(const CaptureRecord& record) const
{
    return matchesSearch(record) && matchesDateRange(record);
}

bool GalleryPage::matchesSearch(const CaptureRecord& record) const
{
    const QString needle = searchEdit_->text().trimmed();
    if (needle.isEmpty()) return true;
    const QString id = QString::fromStdString(record.id);
    const QString timestamp = QString::fromStdString(record.timestamp);
    return id.contains(needle, Qt::CaseInsensitive)
        || timestamp.contains(needle, Qt::CaseInsensitive);
}

bool GalleryPage::matchesDateRange(const CaptureRecord& record) const
{
    if (!filterFrom_.isValid() && !filterTo_.isValid()) return true;
    const QDate captureDay = QDate::fromString(
        QString::fromStdString(record.timestamp).left(10), Qt::ISODate);
    if (!captureDay.isValid()) return false;
    if (filterFrom_.isValid() && captureDay < filterFrom_) return false;
    if (filterTo_.isValid()   && captureDay > filterTo_)   return false;
    return true;
}
```

- [ ] **Step 5.3: Build**

Run:
```bash
cmake --build build
```
Expected: clean build.

- [ ] **Step 5.4: Manual smoke**

Run:
```bash
./build/SGTDetector.exe
```

Verify on the Gallery page:
- A row of 5 chips appears below the header: `All` selected by default, then `Today` `Last 7` `Last 30` `Custom`.
- Clicking `Today` hides every card whose timestamp is not from today.
- Clicking `Last 7` shows the past 7 days inclusive (today plus 6 prior).
- Clicking `Custom` reveals two date pickers next to it; setting `From` to a past day and `To` to today filters accordingly.
- Typing into the search box while a date filter is active narrows the result further (search AND date both apply).
- Clicking `All` returns to the unfiltered view; the date pickers hide again.
- Switching pages and coming back to Gallery preserves the active chip (since the bar lives in the page).

- [ ] **Step 5.5: Commit**

```bash
git add ui/GalleryPage.h ui/GalleryPage.cpp
git commit -m "feat(ui): integrate date-range filter into Gallery"
```

---

## Task 6: Final acceptance pass

This task has no code changes — it walks the spec's §7 acceptance list end-to-end and adds a single wrap-up commit if any plan-level docs need updating (none expected).

**Files:**
- (none modified by default)

- [ ] **Step 6.1: Run the full test suite**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: 4/4 tests pass — `CaptureStoreSmoke`, `ThemeSmoke`, `ThumbnailCacheSmoke`, `AppSettingsSmoke`.

- [ ] **Step 6.2: Settings persistence acceptance**

Launch:
```bash
./build/SGTDetector.exe
```

Walk through:
1. Sidebar → Settings is reachable; Capture Source / Detection Defaults / Capture Storage cards render.
2. Slide Tool threshold to 60%, grasp to 40%, defect to 70%; check Tool only; set output folder to a writable path; close the app.
3. Relaunch with no CLI args. Settings page reflects 60/40/70 and Tool-only. Live page Tool/Grasp/Defect sliders start at 60/40/70.
4. Relaunch with `./build/SGTDetector.exe 2`. Camera index in Settings still shows 0 (CLI override, persisted untouched), but the running camera is index 2.
5. Settings → `Reset to factory` puts thresholds back to 25/25/50 and modes to Tool-only without resetting camera id or output folder.

- [ ] **Step 6.3: Gallery filter acceptance**

Still in the running app:
1. Capture three frames in Live (`C` key); switch to Gallery → 3 cards appear under "Today".
2. `All` shows them and any prior cards. `Today` shows only the 3 just-captured. `Last 7` includes both. `Custom` lets you pick an explicit range.
3. Search `2026-05` while `Last 7` is active narrows further; clearing search restores the date-only filter.
4. Clicking a card still opens `CaptureDetailDialog`; ←/→ navigation, Esc close, "Open folder" button still work.

- [ ] **Step 6.4: Theme + plumbing regression check**

1. Theme toggle still flips light/dark across all three pages (Live, Gallery, Settings).
2. Switching pages does not stop the camera (FPS continues to update on Live).
3. `./build/SGTDetector.exe --help` still prints usage; unknown args still error out.

- [ ] **Step 6.5: Commit (only if anything had to be touched)**

If you discovered something the spec missed (e.g., a missing QSS class for `QSpinBox` darkening), apply the fix in the smallest possible diff and commit:
```bash
git commit -m "fix(ui): <one-line>"
```
If everything passed cleanly, no commit is needed for this task.

---

## Verification matrix (spec → task)

| Spec section                                | Implemented in |
|---------------------------------------------|----------------|
| §3.2 `core/AppSettings`                     | Task 1         |
| §3.3 `ui/SettingsPage` (4 cards, autosave)  | Task 3         |
| §3.4 `ui/GalleryFilterBar`                  | Task 4         |
| §3.5 GalleryPage filter integration         | Task 5         |
| §3.6 Sidebar Settings enable                | Task 3 step 3.3|
| §3.7 AppOptions extension + main.cpp wiring | Task 2         |
| §3.8 QSS additions                          | Task 4 step 4.3|
| §7 Automated test (`AppSettingsSmoke`)      | Task 1         |
| §7 Manual acceptance                        | Task 6         |
