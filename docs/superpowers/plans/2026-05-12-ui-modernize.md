# SGTDetector UI Modernization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the Qt6 Widgets UI into a sidebar-navigated, multi-page shell with a dedicated gallery page that renders real thumbnails, fix the JPEG-plugin-related thumbnail bug, and add a token-based dark/light theme system.

**Architecture:** Replace `ui/MainWindow` with `AppShell` (`QMainWindow` owning camera, timer, detection engine and capture store) hosting a left `Sidebar` and a `QStackedWidget` of pages (`LivePage`, `GalleryPage`). Migrate the existing live widgets into `LivePage`, add an HUD overlay on top of the video, and rebuild the gallery as a flow of `ThumbCard`s backed by an OpenCV-decoded `ThumbnailCache`. Style flows from `ThemeManager` (`QSettings`-persisted) through a tokenized `base.qss` template applied to `qApp`.

**Tech Stack:** Qt 6 Widgets, C++17, OpenCV (`core`, `imgproc`, `imgcodecs`, `videoio`, `dnn`), ONNX Runtime 1.24, CMake + Ninja under MSYS2 MinGW64, CTest.

**Spec:** `docs/superpowers/specs/2026-05-12-ui-modernize-design.md`

---

## Conventions

- Branch: continue on `codex/enhance`.
- Commits: one per task unless a task explicitly splits commits. Use Conventional Commits (`feat(ui): …`, `fix(ui): …`, `refactor(ui): …`, `test: …`, `build: …`).
- Headers go in `ui/`, sources in `ui/`. Namespace is `sgt::ui` for new widgets, `sgt` for non-UI helpers.
- Compile/test loop: `cmake --build build` then `ctest --test-dir build --output-on-failure` for unit tests; run `./build/SGTDetector.exe` for manual smoke (every GUI task ends with one).
- GUI widgets are verified by build + visible behaviour. Where logic is pure (`Theme::renderQss`, `ThumbnailCache` decode), use real TDD with a smoke binary registered in CTest.
- The build directory may already exist as `build/`. If not, run `cmake -B build -G Ninja` once. The post-build hooks already copy ORT, models, labels, and Qt platform plugins.

---

## Task 1: Theme tokens, QSS template, and ThemeManager (TDD)

**Files:**
- Create: `assets/qss/base.qss`
- Create: `assets/icons.qrc`
- Create: `ui/Theme.h`, `ui/Theme.cpp`
- Create: `ui/ThemeManager.h`, `ui/ThemeManager.cpp`
- Create: `tests/ThemeSmoke.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1.1: Write `ui/Theme.h`**

```cpp
#pragma once

#include <QString>

namespace sgt::ui {

struct ThemeTokens {
    QString name;            // "dark" / "light"
    QString bg;
    QString surface;
    QString elevated;
    QString border;
    QString textPrimary;
    QString textSecondary;
    QString accent;
    QString accentHover;
    QString info;
    QString warn;
    QString danger;
    QString shadow;          // rgba string e.g. "rgba(0,0,0,0.35)"
};

namespace Theme {
    ThemeTokens dark();
    ThemeTokens light();
    QString renderQss(const ThemeTokens& tokens);
}

} // namespace sgt::ui
```

- [ ] **Step 1.2: Write `assets/qss/base.qss`** — token-placeholder QSS used as the entire app stylesheet.

```css
* { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; font-size: 14px; }

QWidget {
    background: {{bg}};
    color: {{textPrimary}};
}

QMainWindow, QDialog { background: {{bg}}; }

QFrame#Sidebar {
    background: {{elevated}};
    border-right: 1px solid {{border}};
}

QToolButton#NavButton {
    background: transparent;
    border: none;
    color: {{textSecondary}};
    padding: 10px 0;
    border-radius: 8px;
}
QToolButton#NavButton:hover { background: {{surface}}; color: {{textPrimary}}; }
QToolButton#NavButton:checked {
    background: {{surface}};
    color: {{accent}};
    border-left: 3px solid {{accent}};
}

QFrame#Card {
    background: {{surface}};
    border: 1px solid {{border}};
    border-radius: 10px;
}
QLabel#CardTitle { font-size: 13px; font-weight: 700; color: {{textSecondary}}; padding: 4px 2px 8px 2px; }

QFrame#VideoStage { background: {{elevated}}; border-radius: 10px; }
QLabel#VideoSurface { background: #000000; color: {{textSecondary}}; border-radius: 8px; font-size: 16px; }

QFrame#StatusChip {
    background: rgba(14,20,27,0.72);
    color: {{textPrimary}};
    border: 1px solid {{border}};
    border-radius: 14px;
    padding: 4px 12px;
}
QLabel#StatusChipText { background: transparent; color: {{textPrimary}}; }
QLabel#StatusChipDot { background: transparent; color: {{accent}}; }

QPushButton#ModePill {
    background: rgba(14,20,27,0.72);
    color: {{textSecondary}};
    border: 1px solid {{border}};
    border-radius: 14px;
    padding: 6px 14px;
}
QPushButton#ModePill:hover { color: {{textPrimary}}; }
QPushButton#ModePill:checked {
    background: {{accent}};
    color: #FFFFFF;
    border-color: {{accent}};
}

QPushButton#CaptureFAB {
    background: {{accent}};
    color: #FFFFFF;
    border: none;
    border-radius: 28px;
    font-weight: 700;
    font-size: 14px;
}
QPushButton#CaptureFAB:hover { background: {{accentHover}}; }

QPushButton {
    background: {{surface}};
    color: {{textPrimary}};
    border: 1px solid {{border}};
    border-radius: 6px;
    padding: 8px 14px;
}
QPushButton:hover { background: {{elevated}}; }
QPushButton:disabled { color: {{textSecondary}}; }

QLineEdit, QPlainTextEdit, QTableWidget, QListWidget, QScrollArea {
    background: {{surface}};
    color: {{textPrimary}};
    border: 1px solid {{border}};
    border-radius: 6px;
    selection-background-color: {{accent}};
}

QHeaderView::section {
    background: {{surface}};
    color: {{textSecondary}};
    border: none;
    border-bottom: 1px solid {{border}};
    padding: 6px 8px;
    font-weight: 700;
}

QSlider::groove:horizontal { height: 6px; background: {{border}}; border-radius: 3px; }
QSlider::handle:horizontal { background: {{accent}}; width: 18px; margin: -7px 0; border-radius: 9px; }
QSlider::handle:horizontal:hover { background: {{accentHover}}; }

QFrame#ThumbCard {
    background: {{surface}};
    border: 1px solid {{border}};
    border-radius: 10px;
}
QFrame#ThumbCard:hover { border-color: {{accent}}; }
QLabel#ThumbImage { background: {{elevated}}; border-top-left-radius: 9px; border-top-right-radius: 9px; }
QLabel#ThumbMeta  { color: {{textSecondary}}; padding: 6px 10px; }
QLabel#BadgeWarn  { color: {{warn}}; font-weight: 700; }
QLabel#BadgeOk    { color: {{textSecondary}}; }

QLabel#AppTitle { font-size: 22px; font-weight: 700; color: {{textPrimary}}; }
QLabel#SubtleText { color: {{textSecondary}}; }
QLabel#PanelTitle { font-size: 15px; font-weight: 700; color: {{textPrimary}}; }
QLabel#DateHeading { font-size: 13px; font-weight: 700; color: {{textSecondary}}; padding: 12px 4px 6px 4px; text-transform: uppercase; }
```

- [ ] **Step 1.3: Write `assets/icons.qrc`** — placeholder; SVG files are added later as needed but the resource manifest must exist now so CMake stays consistent.

```xml
<RCC>
    <qresource prefix="/">
        <file alias="qss/base.qss">qss/base.qss</file>
    </qresource>
</RCC>
```

- [ ] **Step 1.4: Write `ui/Theme.cpp`**

```cpp
#include "ui/Theme.h"

#include <QFile>
#include <QIODevice>

namespace sgt::ui::Theme {

ThemeTokens dark()
{
    return {
        "dark",
        "#0E141B", "#161D26", "#1E2733", "#243140",
        "#E6EDF3", "#8B98A8",
        "#14B8A6", "#2DD4BF",
        "#38BDF8", "#F59E0B", "#EF4444",
        "rgba(0,0,0,0.35)"
    };
}

ThemeTokens light()
{
    return {
        "light",
        "#F4F6F9", "#FFFFFF", "#FFFFFF", "#E2E8F0",
        "#0F172A", "#64748B",
        "#0F766E", "#0D9488",
        "#2563EB", "#D97706", "#DC2626",
        "rgba(15,23,42,0.12)"
    };
}

QString renderQss(const ThemeTokens& t)
{
    QFile f(":/qss/base.qss");
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    QString s = QString::fromUtf8(f.readAll());
    s.replace("{{bg}}", t.bg);
    s.replace("{{surface}}", t.surface);
    s.replace("{{elevated}}", t.elevated);
    s.replace("{{border}}", t.border);
    s.replace("{{textPrimary}}", t.textPrimary);
    s.replace("{{textSecondary}}", t.textSecondary);
    s.replace("{{accent}}", t.accent);
    s.replace("{{accentHover}}", t.accentHover);
    s.replace("{{info}}", t.info);
    s.replace("{{warn}}", t.warn);
    s.replace("{{danger}}", t.danger);
    s.replace("{{shadow}}", t.shadow);
    return s;
}

} // namespace sgt::ui::Theme
```

- [ ] **Step 1.5: Write `ui/ThemeManager.h`**

```cpp
#pragma once

#include <QObject>

#include "ui/Theme.h"

class QApplication;

namespace sgt::ui {

class ThemeManager final : public QObject {
    Q_OBJECT

public:
    static ThemeManager& instance();

    void apply(QApplication* app);
    void setMode(const QString& mode);   // "dark" or "light"
    QString mode() const { return tokens_.name; }
    const ThemeTokens& tokens() const { return tokens_; }
    void toggle();

signals:
    void themeChanged(const ThemeTokens& tokens);

private:
    ThemeManager();
    ThemeTokens tokens_;
    QApplication* app_ = nullptr;

    void applyToApp();
};

} // namespace sgt::ui
```

- [ ] **Step 1.6: Write `ui/ThemeManager.cpp`**

```cpp
#include "ui/ThemeManager.h"

#include <QApplication>
#include <QSettings>

namespace sgt::ui {

ThemeManager& ThemeManager::instance()
{
    static ThemeManager s;
    return s;
}

ThemeManager::ThemeManager()
{
    QSettings settings("SGT", "Detector");
    const QString mode = settings.value("ui/theme", "dark").toString();
    tokens_ = (mode == "light") ? Theme::light() : Theme::dark();
}

void ThemeManager::apply(QApplication* app)
{
    app_ = app;
    applyToApp();
}

void ThemeManager::setMode(const QString& mode)
{
    const ThemeTokens next = (mode == "light") ? Theme::light() : Theme::dark();
    if (next.name == tokens_.name) return;
    tokens_ = next;
    QSettings("SGT", "Detector").setValue("ui/theme", tokens_.name);
    applyToApp();
    emit themeChanged(tokens_);
}

void ThemeManager::toggle()
{
    setMode(tokens_.name == "dark" ? "light" : "dark");
}

void ThemeManager::applyToApp()
{
    if (!app_) return;
    app_->setStyleSheet(Theme::renderQss(tokens_));
}

} // namespace sgt::ui
```

- [ ] **Step 1.7: Write `tests/ThemeSmoke.cpp`** — pure logic test, no GUI.

```cpp
#include <cstdlib>
#include <iostream>

#include <QCoreApplication>
#include <QResource>

#include "ui/Theme.h"

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);

    const auto dark = sgt::ui::Theme::dark();
    const auto light = sgt::ui::Theme::light();

    if (dark.name != "dark" || light.name != "light") {
        std::cerr << "name mismatch\n";
        return 1;
    }
    if (dark.bg == light.bg) {
        std::cerr << "dark and light share bg\n";
        return 1;
    }

    const QString darkQss = sgt::ui::Theme::renderQss(dark);
    const QString lightQss = sgt::ui::Theme::renderQss(light);

    if (darkQss.isEmpty() || lightQss.isEmpty()) {
        std::cerr << "qss empty (resource not registered?)\n";
        return 1;
    }
    if (darkQss.contains("{{")) {
        std::cerr << "dark qss has unsubstituted placeholders\n";
        return 1;
    }
    if (!darkQss.contains(dark.accent)) {
        std::cerr << "dark accent token missing in rendered qss\n";
        return 1;
    }
    if (!lightQss.contains(light.accent)) {
        std::cerr << "light accent token missing in rendered qss\n";
        return 1;
    }
    return 0;
}
```

- [ ] **Step 1.8: Update `CMakeLists.txt`** — declare the resource, add new sources, register the smoke test.

In the `find_package` block, replace the OpenCV line to include `imgcodecs`:

```cmake
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
```

Add Qt resource just after `set(CMAKE_AUTOUIC ON)`:

```cmake
qt_add_resources(SGT_QRC_FILES assets/icons.qrc)
```

Append to `SGT_SOURCES`:

```cmake
    ui/Theme.cpp
    ui/ThemeManager.cpp
```

Append to `SGT_HEADERS`:

```cmake
    ui/Theme.h
    ui/ThemeManager.h
```

Pass the generated `.cpp` into the executable target — change:

```cmake
add_executable(SGTDetector ${SGT_SOURCES} ${SGT_HEADERS})
```

to:

```cmake
add_executable(SGTDetector ${SGT_SOURCES} ${SGT_HEADERS} ${SGT_QRC_FILES})
```

Add a new test executable just before `enable_testing()`:

```cmake
add_executable(ThemeSmoke
    tests/ThemeSmoke.cpp
    ui/Theme.cpp
    ${SGT_QRC_FILES}
)
target_include_directories(ThemeSmoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ThemeSmoke PRIVATE Qt6::Core)
add_test(NAME ThemeSmoke COMMAND ThemeSmoke)
```

- [ ] **Step 1.9: Build + run the smoke test**

Run:
```bash
cmake --build build
ctest --test-dir build -R ThemeSmoke --output-on-failure
```
Expected: `1/1 Test #N: ThemeSmoke ......... Passed`. If it fails with "qss empty", the resource didn't compile — confirm `${SGT_QRC_FILES}` is passed to the test target.

- [ ] **Step 1.10: Commit**

```bash
git add CMakeLists.txt assets/icons.qrc assets/qss/base.qss \
        ui/Theme.h ui/Theme.cpp ui/ThemeManager.h ui/ThemeManager.cpp \
        tests/ThemeSmoke.cpp
git commit -m "feat(ui): add tokenized dark/light theme infrastructure"
```

---

## Task 2: Wire ThemeManager into the running app, delete legacy AppStyle

**Files:**
- Modify: `main.cpp:93-94`
- Modify: `ui/MainWindow.cpp:36`
- Delete: `ui/AppStyle.h`, `ui/AppStyle.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 2.1: Apply theme in `main.cpp`** — insert after `QApplication app(argc, argv);` (currently line 93):

```cpp
    sgt::ui::ThemeManager::instance().apply(&app);
```

Add the include with the other UI includes near the top:

```cpp
#include "ui/ThemeManager.h"
```

- [ ] **Step 2.2: Drop `setStyleSheet(appStyleSheet())` from `ui/MainWindow.cpp`** — delete line 36 entirely. Also delete `#include "ui/AppStyle.h"` (line 17).

- [ ] **Step 2.3: Delete legacy files**

```bash
rm ui/AppStyle.h ui/AppStyle.cpp
```

- [ ] **Step 2.4: Remove them from `CMakeLists.txt`** — delete the two lines `ui/AppStyle.cpp` and `ui/AppStyle.h` from `SGT_SOURCES` / `SGT_HEADERS`.

- [ ] **Step 2.5: Build and smoke**

Run:
```bash
cmake --build build
./build/SGTDetector.exe
```
Expected: the existing UI launches with the new dark theme applied (Sidebar / Cards / ThumbCard QSS classes are present but no widgets reference them yet — the existing widgets simply pick up generic styles).

- [ ] **Step 2.6: Commit**

```bash
git add main.cpp ui/MainWindow.cpp CMakeLists.txt
git rm ui/AppStyle.h ui/AppStyle.cpp
git commit -m "refactor(ui): replace AppStyle with ThemeManager"
```

---

## Task 3: Sidebar widget

**Files:**
- Create: `ui/Sidebar.h`, `ui/Sidebar.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 3.1: Write `ui/Sidebar.h`**

```cpp
#pragma once

#include <QFrame>

class QButtonGroup;
class QToolButton;

namespace sgt::ui {

class Sidebar final : public QFrame {
    Q_OBJECT

public:
    enum class Page { Live = 0, Gallery = 1, Settings = 2 };

    explicit Sidebar(QWidget* parent = nullptr);

    void setCurrentPage(Page page);

signals:
    void pageRequested(int index);
    void themeToggleRequested();

private:
    QButtonGroup* group_ = nullptr;
    QToolButton* liveBtn_ = nullptr;
    QToolButton* galleryBtn_ = nullptr;
    QToolButton* settingsBtn_ = nullptr;
    QToolButton* themeBtn_ = nullptr;

    QToolButton* makeNavButton(const QString& glyph, const QString& tooltip);
};

} // namespace sgt::ui
```

- [ ] **Step 3.2: Write `ui/Sidebar.cpp`**

```cpp
#include "ui/Sidebar.h"

#include <QButtonGroup>
#include <QToolButton>
#include <QVBoxLayout>

#include "ui/ThemeManager.h"

namespace sgt::ui {

Sidebar::Sidebar(QWidget* parent)
    : QFrame(parent)
{
    setObjectName("Sidebar");
    setFixedWidth(64);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 14, 8, 14);
    layout->setSpacing(6);

    group_ = new QButtonGroup(this);
    group_->setExclusive(true);

    liveBtn_     = makeNavButton(QStringLiteral("◉"), tr("Live detection"));
    galleryBtn_  = makeNavButton(QStringLiteral("▦"), tr("Captures"));
    settingsBtn_ = makeNavButton(QStringLiteral("⚙"), tr("Settings"));
    settingsBtn_->setEnabled(false);
    settingsBtn_->setToolTip(tr("Settings (coming soon)"));

    group_->addButton(liveBtn_,     static_cast<int>(Page::Live));
    group_->addButton(galleryBtn_,  static_cast<int>(Page::Gallery));
    group_->addButton(settingsBtn_, static_cast<int>(Page::Settings));

    layout->addWidget(liveBtn_);
    layout->addWidget(galleryBtn_);
    layout->addWidget(settingsBtn_);
    layout->addStretch();

    themeBtn_ = makeNavButton(QStringLiteral("☾"), tr("Toggle theme"));
    themeBtn_->setCheckable(false);
    layout->addWidget(themeBtn_);

    auto syncThemeGlyph = [this](const ThemeTokens& t) {
        themeBtn_->setText(t.name == "dark" ? QStringLiteral("☀") : QStringLiteral("☾"));
    };
    syncThemeGlyph(ThemeManager::instance().tokens());
    connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, syncThemeGlyph);

    connect(group_, &QButtonGroup::idClicked, this, &Sidebar::pageRequested);
    connect(themeBtn_, &QToolButton::clicked, this, &Sidebar::themeToggleRequested);

    liveBtn_->setChecked(true);
}

void Sidebar::setCurrentPage(Page page)
{
    auto* btn = qobject_cast<QToolButton*>(group_->button(static_cast<int>(page)));
    if (btn) btn->setChecked(true);
}

QToolButton* Sidebar::makeNavButton(const QString& glyph, const QString& tooltip)
{
    auto* btn = new QToolButton(this);
    btn->setObjectName("NavButton");
    btn->setText(glyph);
    btn->setToolTip(tooltip);
    btn->setCheckable(true);
    btn->setAutoRaise(true);
    btn->setFixedSize(48, 48);
    QFont f = btn->font();
    f.setPointSize(18);
    btn->setFont(f);
    return btn;
}

} // namespace sgt::ui
```

- [ ] **Step 3.3: Add to `CMakeLists.txt`** — append `ui/Sidebar.cpp` to `SGT_SOURCES`, `ui/Sidebar.h` to `SGT_HEADERS`.

- [ ] **Step 3.4: Build**

Run:
```bash
cmake --build build
```
Expected: clean build, no link errors. `Sidebar` is unreferenced yet but compiles.

- [ ] **Step 3.5: Commit**

```bash
git add ui/Sidebar.h ui/Sidebar.cpp CMakeLists.txt
git commit -m "feat(ui): add Sidebar nav widget with theme toggle"
```

---

## Task 4: AppShell skeleton + LivePage adopting current widgets

**Files:**
- Create: `ui/AppShell.h`, `ui/AppShell.cpp`
- Create: `ui/LivePage.h`, `ui/LivePage.cpp`
- Create: `ui/GalleryPage.h`, `ui/GalleryPage.cpp` (placeholder body for now)
- Modify: `main.cpp`
- Modify: `CMakeLists.txt`
- Delete: `ui/MainWindow.h`, `ui/MainWindow.cpp` (after AppShell takes over)

This task migrates the existing camera/timer/engine ownership from `MainWindow` into `AppShell`, parks the existing `LivePreviewWidget` / `ControlPanel` inside `LivePage`, and keeps `GalleryPanel` running on a new `GalleryPage` until Task 5 replaces it. Because `MainWindow` and `GalleryPanel` are deleted together with their replacements in later tasks, we keep them coexistent now only via includes — there is no need to leave a compatibility shim.

- [ ] **Step 4.1: Write `ui/LivePage.h`**

```cpp
#pragma once

#include <cstdint>

#include <QWidget>

#include "core/DetectionPipeline.h"

namespace sgt::ui {

class LivePreviewWidget;
class ControlPanel;

class LivePage final : public QWidget {
    Q_OBJECT

public:
    LivePage(uint8_t initialModeMask,
             const DetectionThresholds& initialThresholds,
             QWidget* parent = nullptr);

    ControlPanel* controlPanel() const { return controlPanel_; }
    LivePreviewWidget* preview() const { return preview_; }

public slots:
    void applyFrame(const DetectionFrameResult& result, uint8_t activeMask);
    void setCameraStatus(const QString& text);
    void setFps(float fps);

signals:
    void captureRequested();
    void modeMaskChanged(uint8_t mask);
    void thresholdsChanged(const DetectionThresholds& t);

private:
    LivePreviewWidget* preview_ = nullptr;
    ControlPanel* controlPanel_ = nullptr;
    class QLabel* cameraStatus_ = nullptr;
    class QLabel* fpsLabel_ = nullptr;
    class QPushButton* captureButton_ = nullptr;
};

} // namespace sgt::ui
```

- [ ] **Step 4.2: Write `ui/LivePage.cpp`**

```cpp
#include "ui/LivePage.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSplitter>
#include <QVBoxLayout>

#include "ui/ControlPanel.h"
#include "ui/LivePreviewWidget.h"

namespace sgt::ui {

LivePage::LivePage(uint8_t initialModeMask,
                   const DetectionThresholds& initialThresholds,
                   QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(20, 18, 20, 18);
    layout->setSpacing(14);

    auto* header = new QHBoxLayout();
    auto* title = new QLabel("Live Detection", this);
    title->setObjectName("AppTitle");
    auto* sub = new QLabel("Surgical tool detection workstation", this);
    sub->setObjectName("SubtleText");
    auto* titles = new QVBoxLayout();
    titles->addWidget(title);
    titles->addWidget(sub);
    header->addLayout(titles);
    header->addStretch();

    cameraStatus_ = new QLabel("Camera starting", this);
    cameraStatus_->setObjectName("StatusChipText");
    fpsLabel_ = new QLabel("FPS 0.0", this);
    fpsLabel_->setObjectName("StatusChipText");
    captureButton_ = new QPushButton("Capture", this);
    captureButton_->setMinimumHeight(44);
    captureButton_->setMinimumWidth(120);
    captureButton_->setObjectName("CaptureFAB");
    header->addWidget(cameraStatus_);
    header->addSpacing(8);
    header->addWidget(fpsLabel_);
    header->addSpacing(12);
    header->addWidget(captureButton_);
    layout->addLayout(header);

    auto* splitter = new QSplitter(Qt::Horizontal, this);
    splitter->setChildrenCollapsible(false);
    preview_ = new LivePreviewWidget(splitter);
    controlPanel_ = new ControlPanel(initialModeMask, initialThresholds, splitter);
    splitter->addWidget(preview_);
    splitter->addWidget(controlPanel_);
    splitter->setStretchFactor(0, 7);
    splitter->setStretchFactor(1, 3);
    layout->addWidget(splitter, 1);

    connect(captureButton_, &QPushButton::clicked, this, &LivePage::captureRequested);
    connect(controlPanel_, &ControlPanel::modeMaskChanged, this, &LivePage::modeMaskChanged);
    connect(controlPanel_, &ControlPanel::thresholdsChanged, this, &LivePage::thresholdsChanged);
}

void LivePage::applyFrame(const DetectionFrameResult& result, uint8_t activeMask)
{
    preview_->setResult(result, activeMask);
}

void LivePage::setCameraStatus(const QString& text)
{
    cameraStatus_->setText(text);
}

void LivePage::setFps(float fps)
{
    fpsLabel_->setText(QString("FPS %1").arg(fps, 0, 'f', 1));
}

} // namespace sgt::ui
```

- [ ] **Step 4.3: Write `ui/GalleryPage.h` (placeholder)**

```cpp
#pragma once

#include <QWidget>
#include <vector>

#include "core/CaptureStore.h"

namespace sgt::ui {

class GalleryPanel;

class GalleryPage final : public QWidget {
    Q_OBJECT

public:
    explicit GalleryPage(QWidget* parent = nullptr);

    void setRecords(const std::vector<sgt::CaptureRecord>& records);

private:
    GalleryPanel* panel_ = nullptr;
};

} // namespace sgt::ui
```

- [ ] **Step 4.4: Write `ui/GalleryPage.cpp` (placeholder wrapping legacy `GalleryPanel`)**

```cpp
#include "ui/GalleryPage.h"

#include <QVBoxLayout>

#include "ui/GalleryPanel.h"

namespace sgt::ui {

GalleryPage::GalleryPage(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(20, 18, 20, 18);
    panel_ = new GalleryPanel(this);
    layout->addWidget(panel_);
}

void GalleryPage::setRecords(const std::vector<sgt::CaptureRecord>& records)
{
    panel_->setRecords(records);
}

} // namespace sgt::ui
```

- [ ] **Step 4.5: Write `ui/AppShell.h`**

```cpp
#pragma once

#include <cstdint>
#include <memory>

#include <QDateTime>
#include <QMainWindow>
#include <QTimer>

#include <opencv2/videoio.hpp>

#include "core/DetectionPipeline.h"
#include "core/Renderer.h"

namespace sgt {
class CaptureStore;
class DetectionEngine;
}

namespace sgt::ui {

struct AppOptions {
    int cameraId = 0;
    uint8_t modeMask = MODE_TOOL;
    std::string toolModel;
    std::string graspModel;
    std::string defectModel;
};

class Sidebar;
class LivePage;
class GalleryPage;
class QStackedWidget;

class AppShell final : public QMainWindow {
    Q_OBJECT

public:
    AppShell(AppOptions opts,
             std::unique_ptr<DetectionEngine> engine,
             std::unique_ptr<CaptureStore> store,
             QWidget* parent = nullptr);
    ~AppShell() override;

private:
    AppOptions opts_;
    std::unique_ptr<DetectionEngine> engine_;
    std::unique_ptr<CaptureStore> store_;
    uint8_t activeMask_;
    DetectionThresholds thresholds_;
    DetectionFrameResult lastResult_;
    cv::VideoCapture cap_;
    QTimer timer_;
    QDateTime lastFrameTime_;
    float fps_ = 0.0f;

    Sidebar* sidebar_ = nullptr;
    QStackedWidget* stack_ = nullptr;
    LivePage* livePage_ = nullptr;
    GalleryPage* galleryPage_ = nullptr;

    void buildUi();
    void wireEvents();
    void processFrame();
    void updateFps();
    void captureCurrent();
    void refreshGallery(bool selectLatest = false);
};

} // namespace sgt::ui
```

`AppOptions` is duplicated here intentionally — Task 4.10 deletes `MainWindow.h` (the original home of `AppOptions`), so callers henceforth get it from `AppShell.h`.

- [ ] **Step 4.6: Write `ui/AppShell.cpp`**

```cpp
#include "ui/AppShell.h"

#include <QHBoxLayout>
#include <QMessageBox>
#include <QShortcut>
#include <QStackedWidget>
#include <QStatusBar>
#include <QWidget>

#include "core/CaptureStore.h"
#include "ui/ControlPanel.h"
#include "ui/GalleryPage.h"
#include "ui/LivePage.h"
#include "ui/Sidebar.h"
#include "ui/ThemeManager.h"

namespace sgt::ui {

AppShell::AppShell(AppOptions opts,
                   std::unique_ptr<DetectionEngine> engine,
                   std::unique_ptr<CaptureStore> store,
                   QWidget* parent)
    : QMainWindow(parent)
    , opts_(std::move(opts))
    , engine_(std::move(engine))
    , store_(std::move(store))
    , activeMask_(opts_.modeMask ? opts_.modeMask : MODE_TOOL)
{
    buildUi();
    wireEvents();
    refreshGallery();

    cap_.open(opts_.cameraId);
    if (!cap_.isOpened()) {
        livePage_->setCameraStatus("Camera unavailable");
        statusBar()->showMessage("Cannot open camera " + QString::number(opts_.cameraId));
    } else {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        livePage_->setCameraStatus("Camera " + QString::number(opts_.cameraId) + " online");
        timer_.start(1);
    }
}

AppShell::~AppShell()
{
    timer_.stop();
    cap_.release();
}

void AppShell::buildUi()
{
    setWindowTitle("SGTDetector");
    resize(1440, 900);

    auto* root = new QWidget(this);
    auto* layout = new QHBoxLayout(root);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    sidebar_ = new Sidebar(root);
    layout->addWidget(sidebar_);

    stack_ = new QStackedWidget(root);
    livePage_    = new LivePage(activeMask_, thresholds_, stack_);
    galleryPage_ = new GalleryPage(stack_);
    stack_->addWidget(livePage_);
    stack_->addWidget(galleryPage_);
    layout->addWidget(stack_, 1);

    setCentralWidget(root);
    statusBar()->showMessage("Ready");
}

void AppShell::wireEvents()
{
    connect(&timer_, &QTimer::timeout, this, [this]() { processFrame(); });
    connect(livePage_, &LivePage::captureRequested, this, [this]() { captureCurrent(); });
    connect(livePage_, &LivePage::modeMaskChanged, this,
            [this](uint8_t mask) { activeMask_ = mask; });
    connect(livePage_, &LivePage::thresholdsChanged, this,
            [this](const DetectionThresholds& t) {
                thresholds_ = t;
                engine_->setThresholds(thresholds_);
            });
    connect(sidebar_, &Sidebar::pageRequested,
            stack_, &QStackedWidget::setCurrentIndex);
    connect(sidebar_, &Sidebar::themeToggleRequested,
            &ThemeManager::instance(), &ThemeManager::toggle);

    new QShortcut(QKeySequence(Qt::Key_1), this,
                  [this]() { livePage_->controlPanel()->toggleToolMode(); });
    new QShortcut(QKeySequence(Qt::Key_2), this,
                  [this]() { livePage_->controlPanel()->toggleGraspMode(); });
    new QShortcut(QKeySequence(Qt::Key_3), this,
                  [this]() { livePage_->controlPanel()->toggleDefectMode(); });
    new QShortcut(QKeySequence(Qt::Key_C), this, [this]() { captureCurrent(); });
    new QShortcut(QKeySequence(Qt::Key_Q), this, [this]() { close(); });
    new QShortcut(QKeySequence(Qt::Key_Escape), this, [this]() { close(); });
}

void AppShell::processFrame()
{
    if (!cap_.isOpened()) return;

    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) return;

    try {
        lastResult_ = engine_->process(frame, activeMask_);
        updateFps();
        lastResult_.fps = fps_;
        livePage_->applyFrame(lastResult_, activeMask_);
    } catch (const std::exception& e) {
        statusBar()->showMessage("Inference error: " + QString::fromLocal8Bit(e.what()), 3000);
    }
}

void AppShell::updateFps()
{
    QDateTime now = QDateTime::currentDateTimeUtc();
    if (lastFrameTime_.isValid()) {
        qint64 ms = lastFrameTime_.msecsTo(now);
        if (ms > 0) {
            float instant = 1000.0f / static_cast<float>(ms);
            fps_ = fps_ * 0.9f + instant * 0.1f;
        }
    }
    lastFrameTime_ = now;
    livePage_->setFps(fps_);
}

void AppShell::captureCurrent()
{
    if (lastResult_.rawFrame.empty()) {
        statusBar()->showMessage("No frame to capture", 2500);
        return;
    }
    try {
        auto record = store_->saveCapture(lastResult_, engine_->models());
        statusBar()->showMessage("Capture saved: " + QString::fromStdString(record.id), 3500);
        refreshGallery(true);
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Capture failed", QString::fromLocal8Bit(e.what()));
    }
}

void AppShell::refreshGallery(bool /*selectLatest*/)
{
    galleryPage_->setRecords(store_->records());
}

} // namespace sgt::ui
```

- [ ] **Step 4.7: Update `main.cpp`** — replace `ui/MainWindow.h` include with `ui/AppShell.h`, and substitute `MainWindow` for `AppShell`:

Replace lines 14 and 104 from:
```cpp
#include "ui/MainWindow.h"
...
    sgt::ui::MainWindow window(opts, std::move(engine), std::move(store));
```
to:
```cpp
#include "ui/AppShell.h"
...
    sgt::ui::AppShell window(opts, std::move(engine), std::move(store));
```

- [ ] **Step 4.8: Update `CMakeLists.txt`** — replace `ui/MainWindow.cpp/.h` in `SGT_SOURCES` / `SGT_HEADERS` with the new files:

Remove:
```cmake
    ui/MainWindow.cpp
```
and:
```cmake
    ui/MainWindow.h
```

Add to `SGT_SOURCES`:
```cmake
    ui/AppShell.cpp
    ui/LivePage.cpp
    ui/GalleryPage.cpp
```

Add to `SGT_HEADERS`:
```cmake
    ui/AppShell.h
    ui/LivePage.h
    ui/GalleryPage.h
```

- [ ] **Step 4.9: Build + smoke**

Run:
```bash
cmake --build build
./build/SGTDetector.exe
```
Expected: window shows sidebar on the left with two enabled icons (Live, Gallery) and a theme toggle at the bottom. Live page shows the camera + thresholds + (legacy text-only) live data. Clicking the gallery icon switches to the (still old-style) gallery panel. Theme toggle swaps light/dark and the choice persists across launches.

- [ ] **Step 4.10: Delete the obsolete `MainWindow`**

```bash
rm ui/MainWindow.h ui/MainWindow.cpp
```

Rebuild:
```bash
cmake --build build
```
Expected: clean build (no references to `MainWindow` remain).

- [ ] **Step 4.11: Commit**

```bash
git add ui/AppShell.h ui/AppShell.cpp ui/LivePage.h ui/LivePage.cpp \
        ui/GalleryPage.h ui/GalleryPage.cpp \
        main.cpp CMakeLists.txt
git rm ui/MainWindow.h ui/MainWindow.cpp
git commit -m "refactor(ui): split MainWindow into AppShell + LivePage + GalleryPage"
```

---

## Task 5: FlowLayout + ThumbCard + GalleryPage proper (synchronous decode)

This task replaces the legacy `GalleryPanel` with a real flow-layout page. JPEG decoding goes through OpenCV (synchronous in this task; async cache lands in Task 6) so the bug is already fixed by the end of this commit.

**Files:**
- Create: `ui/FlowLayout.h`, `ui/FlowLayout.cpp`
- Create: `ui/ThumbCard.h`, `ui/ThumbCard.cpp`
- Modify: `ui/GalleryPage.h`, `ui/GalleryPage.cpp`
- Modify: `CMakeLists.txt`
- Delete: `ui/GalleryPanel.h`, `ui/GalleryPanel.cpp`

- [ ] **Step 5.1: Write `ui/FlowLayout.h`** — verbatim Qt FlowLayout example header.

```cpp
#pragma once

#include <QLayout>
#include <QList>
#include <QRect>
#include <QStyle>

namespace sgt::ui {

class FlowLayout final : public QLayout {
    Q_OBJECT

public:
    explicit FlowLayout(QWidget* parent, int margin = -1, int hSpacing = -1, int vSpacing = -1);
    explicit FlowLayout(int margin = -1, int hSpacing = -1, int vSpacing = -1);
    ~FlowLayout() override;

    void addItem(QLayoutItem* item) override;
    int horizontalSpacing() const;
    int verticalSpacing() const;
    Qt::Orientations expandingDirections() const override;
    bool hasHeightForWidth() const override;
    int heightForWidth(int width) const override;
    int count() const override;
    QLayoutItem* itemAt(int index) const override;
    QSize minimumSize() const override;
    void setGeometry(const QRect& rect) override;
    QSize sizeHint() const override;
    QLayoutItem* takeAt(int index) override;

private:
    int doLayout(const QRect& rect, bool testOnly) const;
    int smartSpacing(QStyle::PixelMetric pm) const;

    QList<QLayoutItem*> items_;
    int hSpace_;
    int vSpace_;
};

} // namespace sgt::ui
```

- [ ] **Step 5.2: Write `ui/FlowLayout.cpp`**

```cpp
#include "ui/FlowLayout.h"

#include <QWidget>

namespace sgt::ui {

FlowLayout::FlowLayout(QWidget* parent, int margin, int hSpacing, int vSpacing)
    : QLayout(parent), hSpace_(hSpacing), vSpace_(vSpacing)
{
    setContentsMargins(margin, margin, margin, margin);
}

FlowLayout::FlowLayout(int margin, int hSpacing, int vSpacing)
    : hSpace_(hSpacing), vSpace_(vSpacing)
{
    setContentsMargins(margin, margin, margin, margin);
}

FlowLayout::~FlowLayout()
{
    while (auto* item = takeAt(0)) delete item;
}

void FlowLayout::addItem(QLayoutItem* item) { items_.append(item); }
int FlowLayout::horizontalSpacing() const { return hSpace_ >= 0 ? hSpace_ : smartSpacing(QStyle::PM_LayoutHorizontalSpacing); }
int FlowLayout::verticalSpacing() const   { return vSpace_ >= 0 ? vSpace_ : smartSpacing(QStyle::PM_LayoutVerticalSpacing); }
int FlowLayout::count() const             { return items_.size(); }
QLayoutItem* FlowLayout::itemAt(int i) const { return items_.value(i); }
QLayoutItem* FlowLayout::takeAt(int i)    { return (i >= 0 && i < items_.size()) ? items_.takeAt(i) : nullptr; }
Qt::Orientations FlowLayout::expandingDirections() const { return {}; }
bool FlowLayout::hasHeightForWidth() const { return true; }
int FlowLayout::heightForWidth(int width) const { return doLayout(QRect(0, 0, width, 0), true); }
QSize FlowLayout::sizeHint() const        { return minimumSize(); }

QSize FlowLayout::minimumSize() const
{
    QSize size;
    for (auto* it : items_) size = size.expandedTo(it->minimumSize());
    const auto m = contentsMargins();
    size += QSize(m.left() + m.right(), m.top() + m.bottom());
    return size;
}

void FlowLayout::setGeometry(const QRect& rect)
{
    QLayout::setGeometry(rect);
    doLayout(rect, false);
}

int FlowLayout::doLayout(const QRect& rect, bool testOnly) const
{
    const auto m = contentsMargins();
    QRect eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom());
    int x = eff.x();
    int y = eff.y();
    int lineHeight = 0;

    for (auto* item : items_) {
        const int spaceX = horizontalSpacing() >= 0 ? horizontalSpacing() : item->widget()->style()->layoutSpacing(QSizePolicy::PushButton, QSizePolicy::PushButton, Qt::Horizontal);
        const int spaceY = verticalSpacing()   >= 0 ? verticalSpacing()   : item->widget()->style()->layoutSpacing(QSizePolicy::PushButton, QSizePolicy::PushButton, Qt::Vertical);
        int next = x + item->sizeHint().width() + spaceX;
        if (next - spaceX > eff.right() && lineHeight > 0) {
            x = eff.x();
            y = y + lineHeight + spaceY;
            next = x + item->sizeHint().width() + spaceX;
            lineHeight = 0;
        }
        if (!testOnly) item->setGeometry(QRect(QPoint(x, y), item->sizeHint()));
        x = next;
        lineHeight = qMax(lineHeight, item->sizeHint().height());
    }
    return y + lineHeight - rect.y() + m.bottom();
}

int FlowLayout::smartSpacing(QStyle::PixelMetric pm) const
{
    QObject* p = parent();
    if (!p) return -1;
    if (p->isWidgetType()) {
        auto* w = static_cast<QWidget*>(p);
        return w->style()->pixelMetric(pm, nullptr, w);
    }
    return static_cast<QLayout*>(p)->spacing();
}

} // namespace sgt::ui
```

- [ ] **Step 5.3: Write `ui/ThumbCard.h`**

```cpp
#pragma once

#include <QFrame>
#include <QString>

#include "core/CaptureStore.h"

class QLabel;

namespace sgt::ui {

class ThumbCard final : public QFrame {
    Q_OBJECT

public:
    explicit ThumbCard(const sgt::CaptureRecord& record, QWidget* parent = nullptr);

    QString id() const { return id_; }

signals:
    void activated(const QString& id);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    QString id_;
    QString imagePath_;
    QLabel* image_ = nullptr;

    void loadThumbnailSync();
};

} // namespace sgt::ui
```

- [ ] **Step 5.4: Write `ui/ThumbCard.cpp`** — synchronous OpenCV decode (Task 6 swaps it for the async cache).

```cpp
#include "ui/ThumbCard.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QVBoxLayout>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ui/QtImageUtils.h"

namespace sgt::ui {

namespace {

QString hhmmss(const std::string& isoTimestamp)
{
    // isoTimestamp looks like "2026-05-12T13:45:09"
    if (isoTimestamp.size() < 19) return QString::fromStdString(isoTimestamp);
    return QString::fromStdString(isoTimestamp.substr(11, 8));
}

} // namespace

ThumbCard::ThumbCard(const sgt::CaptureRecord& record, QWidget* parent)
    : QFrame(parent)
    , id_(QString::fromStdString(record.id))
    , imagePath_(QString::fromStdString(record.annotatedImagePath))
{
    setObjectName("ThumbCard");
    setFixedSize(208, 174);
    setCursor(Qt::PointingHandCursor);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    image_ = new QLabel(this);
    image_->setObjectName("ThumbImage");
    image_->setFixedSize(208, 130);
    image_->setAlignment(Qt::AlignCenter);
    image_->setText("…");
    layout->addWidget(image_);

    auto* meta = new QHBoxLayout();
    meta->setContentsMargins(10, 6, 10, 8);
    meta->setSpacing(8);
    auto* time = new QLabel(hhmmss(record.timestamp), this);
    time->setObjectName("ThumbMeta");
    meta->addWidget(time);
    meta->addStretch();

    auto makeBadge = [this](const QString& letter, int count) {
        auto* b = new QLabel(QString("%1 %2").arg(letter).arg(count), this);
        b->setObjectName(count > 0 ? "BadgeOk" : "BadgeOk");
        return b;
    };
    meta->addWidget(makeBadge("T", record.toolCount));
    meta->addWidget(makeBadge("G", record.graspCount));
    auto* d = new QLabel(QString("D %1").arg(record.defectCount), this);
    d->setObjectName(record.defectCount > 0 ? "BadgeWarn" : "BadgeOk");
    meta->addWidget(d);
    layout->addLayout(meta);

    loadThumbnailSync();
}

void ThumbCard::loadThumbnailSync()
{
    cv::Mat mat = cv::imread(imagePath_.toStdString(), cv::IMREAD_COLOR);
    if (mat.empty()) {
        image_->setText("(no image)");
        return;
    }
    cv::Mat resized;
    cv::resize(mat, resized, cv::Size(208, 130), 0, 0, cv::INTER_AREA);
    image_->setPixmap(matToPixmap(resized));
}

void ThumbCard::mousePressEvent(QMouseEvent* e)
{
    if (e->button() == Qt::LeftButton) emit activated(id_);
    QFrame::mousePressEvent(e);
}

void ThumbCard::mouseDoubleClickEvent(QMouseEvent* e)
{
    if (e->button() == Qt::LeftButton) emit activated(id_);
    QFrame::mouseDoubleClickEvent(e);
}

} // namespace sgt::ui
```

- [ ] **Step 5.5: Rewrite `ui/GalleryPage.h`**

```cpp
#pragma once

#include <vector>

#include <QString>
#include <QWidget>

#include "core/CaptureStore.h"

class QLabel;
class QLineEdit;
class QScrollArea;
class QVBoxLayout;

namespace sgt::ui {

class GalleryPage final : public QWidget {
    Q_OBJECT

public:
    explicit GalleryPage(QWidget* parent = nullptr);

    void setRecords(const std::vector<sgt::CaptureRecord>& records);

signals:
    void captureOpened(const QString& id);

private:
    std::vector<sgt::CaptureRecord> records_;
    QLabel* titleLabel_ = nullptr;
    QLineEdit* searchEdit_ = nullptr;
    QScrollArea* scroll_ = nullptr;
    QVBoxLayout* contentLayout_ = nullptr;

    void rebuildContent();
    QString currentFilter() const;
};

} // namespace sgt::ui
```

- [ ] **Step 5.6: Write `ui/GalleryPage.cpp`**

```cpp
#include "ui/GalleryPage.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>

#include <map>

#include "ui/FlowLayout.h"
#include "ui/ThumbCard.h"

namespace sgt::ui {

GalleryPage::GalleryPage(QWidget* parent)
    : QWidget(parent)
{
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(20, 18, 20, 18);
    root->setSpacing(12);

    auto* header = new QHBoxLayout();
    titleLabel_ = new QLabel("Captures · 0", this);
    titleLabel_->setObjectName("AppTitle");
    searchEdit_ = new QLineEdit(this);
    searchEdit_->setPlaceholderText("Search by id or date");
    searchEdit_->setMaximumWidth(280);
    auto* exportBtn = new QPushButton("Export all…", this);
    exportBtn->setEnabled(false);
    exportBtn->setToolTip("Export pipeline coming soon");

    header->addWidget(titleLabel_);
    header->addStretch();
    header->addWidget(searchEdit_);
    header->addWidget(exportBtn);
    root->addLayout(header);

    scroll_ = new QScrollArea(this);
    scroll_->setWidgetResizable(true);
    auto* content = new QWidget(scroll_);
    contentLayout_ = new QVBoxLayout(content);
    contentLayout_->setContentsMargins(0, 0, 0, 0);
    contentLayout_->setSpacing(4);
    contentLayout_->addStretch();
    scroll_->setWidget(content);
    root->addWidget(scroll_, 1);

    connect(searchEdit_, &QLineEdit::textChanged, this, [this]() { rebuildContent(); });
}

void GalleryPage::setRecords(const std::vector<sgt::CaptureRecord>& records)
{
    records_ = records;
    titleLabel_->setText(QString("Captures · %1").arg(records_.size()));
    rebuildContent();
}

QString GalleryPage::currentFilter() const
{
    return searchEdit_->text().trimmed().toLower();
}

void GalleryPage::rebuildContent()
{
    // Drop existing day-group widgets (everything except the trailing stretch).
    while (contentLayout_->count() > 1) {
        auto* item = contentLayout_->takeAt(0);
        if (item->widget()) item->widget()->deleteLater();
        delete item;
    }

    const QString filter = currentFilter();

    // Group by date (records_ is already sorted timestamp-desc by CaptureStore).
    std::map<std::string, std::vector<const sgt::CaptureRecord*>, std::greater<>> byDay;
    for (const auto& r : records_) {
        if (!filter.isEmpty()) {
            const QString id = QString::fromStdString(r.id).toLower();
            const QString ts = QString::fromStdString(r.timestamp).toLower();
            if (!id.contains(filter) && !ts.contains(filter)) continue;
        }
        const std::string day = r.timestamp.size() >= 10 ? r.timestamp.substr(0, 10) : "unknown";
        byDay[day].push_back(&r);
    }

    if (byDay.empty()) {
        auto* empty = new QLabel("No captures match.", scroll_->widget());
        empty->setObjectName("SubtleText");
        empty->setAlignment(Qt::AlignCenter);
        contentLayout_->insertWidget(contentLayout_->count() - 1, empty);
        return;
    }

    for (const auto& [day, items] : byDay) {
        auto* heading = new QLabel(QString::fromStdString(day), scroll_->widget());
        heading->setObjectName("DateHeading");
        contentLayout_->insertWidget(contentLayout_->count() - 1, heading);

        auto* flowHost = new QWidget(scroll_->widget());
        auto* flow = new FlowLayout(flowHost, 0, 12, 12);
        for (const auto* r : items) {
            auto* card = new ThumbCard(*r, flowHost);
            connect(card, &ThumbCard::activated, this, &GalleryPage::captureOpened);
            flow->addWidget(card);
        }
        contentLayout_->insertWidget(contentLayout_->count() - 1, flowHost);
    }
}

} // namespace sgt::ui
```

- [ ] **Step 5.7: Update `CMakeLists.txt`**

Append to `SGT_SOURCES`:
```cmake
    ui/FlowLayout.cpp
    ui/ThumbCard.cpp
```
Append to `SGT_HEADERS`:
```cmake
    ui/FlowLayout.h
    ui/ThumbCard.h
```
Remove from both lists: `ui/GalleryPanel.cpp` and `ui/GalleryPanel.h`.

- [ ] **Step 5.8: Delete legacy gallery panel**

```bash
rm ui/GalleryPanel.h ui/GalleryPanel.cpp
```

- [ ] **Step 5.9: Build + smoke**

Run:
```bash
cmake --build build
./build/SGTDetector.exe
```
Expected: switching to the Gallery icon shows a grid of `ThumbCard`s with **actual thumbnail images** (the original bug fix), grouped by date, with a search field that filters in real time. Press `C` on Live to capture; switch back to Gallery to see the new card appear after the next refresh.

- [ ] **Step 5.10: Commit**

```bash
git add ui/FlowLayout.h ui/FlowLayout.cpp ui/ThumbCard.h ui/ThumbCard.cpp \
        ui/GalleryPage.h ui/GalleryPage.cpp CMakeLists.txt
git rm ui/GalleryPanel.h ui/GalleryPanel.cpp
git commit -m "feat(ui): replace GalleryPanel with flow-layout GalleryPage + ThumbCard"
```

---

## Task 6: ThumbnailCache (async OpenCV decode, TDD)

**Files:**
- Create: `ui/ThumbnailCache.h`, `ui/ThumbnailCache.cpp`
- Create: `tests/ThumbnailCacheSmoke.cpp`
- Modify: `ui/ThumbCard.h`, `ui/ThumbCard.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 6.1: Write `ui/ThumbnailCache.h`**

```cpp
#pragma once

#include <QHash>
#include <QObject>
#include <QPixmap>
#include <QSize>
#include <QString>
#include <functional>

class QThreadPool;

namespace sgt::ui {

class ThumbnailCache final : public QObject {
    Q_OBJECT

public:
    static ThumbnailCache& instance();

    using Callback = std::function<void(QPixmap)>;

    // If the thumbnail is cached, callback fires synchronously on this thread.
    // Otherwise it is decoded on a worker thread, then callback fires on the
    // caller's thread via QueuedConnection.
    void request(const QString& id,
                 const QString& path,
                 const QSize& size,
                 Callback cb);

    // Synchronous decode helper, primarily for tests. Returns null on failure.
    static QPixmap decodeBlocking(const QString& path, const QSize& size);

private:
    explicit ThumbnailCache(QObject* parent = nullptr);

    struct Key {
        QString id;
        int w;
        int h;
        bool operator==(const Key& other) const { return id == other.id && w == other.w && h == other.h; }
    };
    friend size_t qHash(const Key& k, size_t seed) noexcept;

    QHash<Key, QPixmap> cache_;
    QThreadPool* pool_;
};

} // namespace sgt::ui
```

- [ ] **Step 6.2: Write `ui/ThumbnailCache.cpp`**

```cpp
#include "ui/ThumbnailCache.h"

#include <QMetaObject>
#include <QPointer>
#include <QRunnable>
#include <QThreadPool>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ui/QtImageUtils.h"

namespace sgt::ui {

size_t qHash(const ThumbnailCache::Key& k, size_t seed) noexcept
{
    return qHashMulti(seed, k.id, k.w, k.h);
}

ThumbnailCache& ThumbnailCache::instance()
{
    static ThumbnailCache c;
    return c;
}

ThumbnailCache::ThumbnailCache(QObject* parent)
    : QObject(parent)
    , pool_(new QThreadPool(this))
{
    pool_->setMaxThreadCount(2);
}

QPixmap ThumbnailCache::decodeBlocking(const QString& path, const QSize& size)
{
    cv::Mat mat = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
    if (mat.empty()) return {};
    cv::Mat resized;
    cv::resize(mat, resized, cv::Size(size.width(), size.height()), 0, 0, cv::INTER_AREA);
    return matToPixmap(resized);
}

namespace {

class DecodeJob final : public QRunnable {
public:
    DecodeJob(QPointer<ThumbnailCache> cache,
              QString path, QSize size,
              ThumbnailCache::Callback cb)
        : cache_(std::move(cache))
        , path_(std::move(path))
        , size_(size)
        , cb_(std::move(cb)) {}

    void run() override {
        QPixmap pix = ThumbnailCache::decodeBlocking(path_, size_);
        if (!cache_) return;
        auto cb = cb_;
        // Dispatch to the cache's thread (created on the UI thread, so this
        // marshals back to the UI thread).
        QMetaObject::invokeMethod(cache_.data(), [cb, pix]() {
            if (cb) cb(pix);
        }, Qt::QueuedConnection);
    }

private:
    QPointer<ThumbnailCache> cache_;
    QString path_;
    QSize size_;
    ThumbnailCache::Callback cb_;
};

} // namespace

void ThumbnailCache::request(const QString& id, const QString& path, const QSize& size, Callback cb)
{
    const Key key{id, size.width(), size.height()};
    auto it = cache_.constFind(key);
    if (it != cache_.cend() && !it.value().isNull()) {
        if (cb) cb(it.value());
        return;
    }

    // Reserve a placeholder so a flurry of identical requests doesn't all queue.
    cache_.insert(key, QPixmap());

    QPointer<ThumbnailCache> self(this);
    auto* job = new DecodeJob(self, path, size,
        [self, key, cb](QPixmap pix) {
            if (self) self->cache_.insert(key, pix);
            if (cb) cb(pix);
        });
    job->setAutoDelete(true);
    pool_->start(job);
}

} // namespace sgt::ui
```

- [ ] **Step 6.3: Write `tests/ThumbnailCacheSmoke.cpp`**

```cpp
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <QCoreApplication>
#include <QImage>
#include <QPixmap>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "ui/ThumbnailCache.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);

    fs::path tmp = fs::temp_directory_path() / "sgt_thumb_smoke.jpg";
    cv::Mat src(120, 200, CV_8UC3, cv::Scalar(80, 160, 240));
    if (!cv::imwrite(tmp.string(), src)) {
        std::cerr << "imwrite failed\n";
        return 1;
    }

    QPixmap blocking = sgt::ui::ThumbnailCache::decodeBlocking(
        QString::fromStdString(tmp.string()), QSize(64, 40));
    fs::remove(tmp);

    if (blocking.isNull()) {
        std::cerr << "decodeBlocking returned null\n";
        return 1;
    }
    if (blocking.width() != 64 || blocking.height() != 40) {
        std::cerr << "wrong size: " << blocking.width() << "x" << blocking.height() << "\n";
        return 1;
    }
    return 0;
}
```

- [ ] **Step 6.4: Register the test in `CMakeLists.txt`**

Add new sources to `SGT_SOURCES`/`SGT_HEADERS`:
```cmake
    ui/ThumbnailCache.cpp
```
```cmake
    ui/ThumbnailCache.h
```

Add the smoke test executable just below `add_test(NAME ThemeSmoke ...)`:

```cmake
add_executable(ThumbnailCacheSmoke
    tests/ThumbnailCacheSmoke.cpp
    ui/ThumbnailCache.cpp
    ui/QtImageUtils.cpp
)
target_include_directories(ThumbnailCacheSmoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ThumbnailCacheSmoke PRIVATE
    Qt6::Core
    Qt6::Gui
    ${OpenCV_LIBS}
)
add_test(NAME ThumbnailCacheSmoke COMMAND ThumbnailCacheSmoke)
```

- [ ] **Step 6.5: Run the test**

```bash
cmake --build build
ctest --test-dir build -R ThumbnailCacheSmoke --output-on-failure
```
Expected: passes. If you see `null pixmap`, OpenCV `imgcodecs` likely failed to write JPG — confirm Task 1 added `imgcodecs` to the OpenCV `COMPONENTS`.

- [ ] **Step 6.6: Switch `ThumbCard` to the async cache**

Change `ui/ThumbCard.h`'s `loadThumbnailSync()` to `requestThumbnail()`, and rewrite `ui/ThumbCard.cpp::loadThumbnailSync` (and its constructor call) as below.

`ui/ThumbCard.h` change:
```cpp
    void requestThumbnail();
```

`ui/ThumbCard.cpp` — replace the `loadThumbnailSync` body and its call:

```cpp
#include "ui/ThumbnailCache.h"
...
ThumbCard::ThumbCard(const sgt::CaptureRecord& record, QWidget* parent)
    : QFrame(parent)
    , id_(QString::fromStdString(record.id))
    , imagePath_(QString::fromStdString(record.annotatedImagePath))
{
    // ... (unchanged widget construction) ...
    requestThumbnail();
}

void ThumbCard::requestThumbnail()
{
    QPointer<QLabel> safeLabel(image_);
    ThumbnailCache::instance().request(id_, imagePath_, QSize(208, 130),
        [safeLabel](QPixmap pix) {
            if (!safeLabel) return;
            if (pix.isNull()) { safeLabel->setText("(no image)"); return; }
            safeLabel->setPixmap(pix);
        });
}
```

Remove the now-dead `<opencv2/...>` includes and the `cv::imread` block from `ThumbCard.cpp`. Add `#include <QPointer>`.

- [ ] **Step 6.7: Build + smoke**

```bash
cmake --build build
./build/SGTDetector.exe
```
Expected: gallery thumbnails appear, possibly with a brief blank placeholder on first paint (the async load), then snap into place. Scrolling the gallery doesn't stall the UI thread.

- [ ] **Step 6.8: Commit**

```bash
git add ui/ThumbnailCache.h ui/ThumbnailCache.cpp \
        ui/ThumbCard.h ui/ThumbCard.cpp \
        tests/ThumbnailCacheSmoke.cpp CMakeLists.txt
git commit -m "feat(ui): async OpenCV-backed ThumbnailCache"
```

---

## Task 7: CaptureDetailDialog

**Files:**
- Create: `ui/CaptureDetailDialog.h`, `ui/CaptureDetailDialog.cpp`
- Modify: `ui/AppShell.h`, `ui/AppShell.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 7.1: Write `ui/CaptureDetailDialog.h`**

```cpp
#pragma once

#include <vector>

#include <QDialog>
#include <QString>

#include "core/CaptureStore.h"

class QKeyEvent;
class QLabel;
class QPlainTextEdit;

namespace sgt::ui {

class CaptureDetailDialog final : public QDialog {
    Q_OBJECT

public:
    CaptureDetailDialog(std::vector<sgt::CaptureRecord> records,
                        int initialIndex,
                        QWidget* parent = nullptr);

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    std::vector<sgt::CaptureRecord> records_;
    int index_;
    QLabel* image_ = nullptr;
    QLabel* metaTitle_ = nullptr;
    QLabel* metaSummary_ = nullptr;
    QPlainTextEdit* json_ = nullptr;

    void showIndex(int index);
    void step(int delta);
};

} // namespace sgt::ui
```

- [ ] **Step 7.2: Write `ui/CaptureDetailDialog.cpp`**

```cpp
#include "ui/CaptureDetailDialog.h"

#include <QDesktopServices>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QPushButton>
#include <QUrl>
#include <QVBoxLayout>

#include "ui/QtImageUtils.h"

namespace sgt::ui {

CaptureDetailDialog::CaptureDetailDialog(std::vector<sgt::CaptureRecord> records,
                                         int initialIndex,
                                         QWidget* parent)
    : QDialog(parent)
    , records_(std::move(records))
    , index_(initialIndex)
{
    setWindowTitle("Capture");
    setModal(true);
    resize(1200, 740);

    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(20, 20, 20, 20);
    root->setSpacing(16);

    image_ = new QLabel(this);
    image_->setAlignment(Qt::AlignCenter);
    image_->setMinimumSize(700, 480);
    image_->setObjectName("VideoSurface");
    root->addWidget(image_, 7);

    auto* side = new QVBoxLayout();
    side->setSpacing(10);

    metaTitle_ = new QLabel(this);
    metaTitle_->setObjectName("PanelTitle");
    side->addWidget(metaTitle_);

    metaSummary_ = new QLabel(this);
    metaSummary_->setWordWrap(true);
    metaSummary_->setObjectName("SubtleText");
    side->addWidget(metaSummary_);

    json_ = new QPlainTextEdit(this);
    json_->setReadOnly(true);
    side->addWidget(json_, 1);

    auto* row = new QHBoxLayout();
    auto* prev = new QPushButton("‹ Prev", this);
    auto* next = new QPushButton("Next ›", this);
    auto* openFolder = new QPushButton("Open folder", this);
    auto* exportBtn = new QPushButton("Export…", this);
    exportBtn->setEnabled(false);
    exportBtn->setToolTip("Export pipeline coming soon");
    row->addWidget(prev);
    row->addWidget(next);
    row->addStretch();
    row->addWidget(openFolder);
    row->addWidget(exportBtn);
    side->addLayout(row);

    root->addLayout(side, 3);

    connect(prev, &QPushButton::clicked, this, [this]() { step(-1); });
    connect(next, &QPushButton::clicked, this, [this]() { step(+1); });
    connect(openFolder, &QPushButton::clicked, this, [this]() {
        if (index_ < 0 || index_ >= static_cast<int>(records_.size())) return;
        const QString path = QString::fromStdString(records_[index_].annotatedImagePath);
        QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(path).absolutePath()));
    });

    showIndex(index_);
}

void CaptureDetailDialog::showIndex(int index)
{
    if (records_.empty()) return;
    index_ = qBound(0, index, static_cast<int>(records_.size()) - 1);
    const auto& r = records_[index_];

    QPixmap pix(QString::fromStdString(r.annotatedImagePath));
    if (!pix.isNull()) {
        image_->setPixmap(pix.scaled(image_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else {
        image_->setText("Image unavailable");
        image_->setPixmap({});
    }

    metaTitle_->setText(QString::fromStdString(r.id));
    metaSummary_->setText(QString("Captured %1 · Tools %2 · Grasp %3 · Defects %4")
        .arg(QString::fromStdString(r.timestamp))
        .arg(r.toolCount).arg(r.graspCount).arg(r.defectCount));
    json_->setPlainText(readTextFile(QString::fromStdString(r.jsonPath)));
}

void CaptureDetailDialog::step(int delta)
{
    showIndex(index_ + delta);
}

void CaptureDetailDialog::keyPressEvent(QKeyEvent* event)
{
    switch (event->key()) {
    case Qt::Key_Left:  step(-1); return;
    case Qt::Key_Right: step(+1); return;
    case Qt::Key_Escape: reject(); return;
    default: break;
    }
    QDialog::keyPressEvent(event);
}

} // namespace sgt::ui
```

- [ ] **Step 7.3: Wire it from `AppShell`** — handle `GalleryPage::captureOpened`.

In `ui/AppShell.cpp::wireEvents`, after the sidebar/theme-toggle connections, add:

```cpp
    connect(galleryPage_, &GalleryPage::captureOpened, this, [this](const QString& id) {
        auto records = store_->records();
        int idx = 0;
        for (size_t i = 0; i < records.size(); ++i) {
            if (QString::fromStdString(records[i].id) == id) { idx = static_cast<int>(i); break; }
        }
        CaptureDetailDialog dlg(records, idx, this);
        dlg.exec();
    });
```

Add at the top of `ui/AppShell.cpp`:
```cpp
#include "ui/CaptureDetailDialog.h"
```

- [ ] **Step 7.4: Update `CMakeLists.txt`** — append:
```cmake
    ui/CaptureDetailDialog.cpp
```
```cmake
    ui/CaptureDetailDialog.h
```

- [ ] **Step 7.5: Build + smoke**

```bash
cmake --build build
./build/SGTDetector.exe
```
Expected: clicking a `ThumbCard` opens a modal dialog showing the annotated frame, JSON metadata, ←/→ navigation between captures, `Open folder` opens the capture's day directory in Windows Explorer, Esc closes the dialog.

- [ ] **Step 7.6: Commit**

```bash
git add ui/CaptureDetailDialog.h ui/CaptureDetailDialog.cpp \
        ui/AppShell.cpp CMakeLists.txt
git commit -m "feat(ui): add modal CaptureDetailDialog with ←/→ navigation"
```

---

## Task 8: HUD overlay on Live page

**Files:**
- Create: `ui/StatusChip.h`, `ui/StatusChip.cpp`
- Create: `ui/ModePillBar.h`, `ui/ModePillBar.cpp`
- Create: `ui/HudOverlay.h`, `ui/HudOverlay.cpp`
- Modify: `ui/LivePage.h`, `ui/LivePage.cpp`
- Modify: `ui/LivePreviewWidget.cpp` (no longer prints FPS/status text)
- Modify: `CMakeLists.txt`

- [ ] **Step 8.1: Write `ui/StatusChip.h`**

```cpp
#pragma once

#include <QFrame>

class QLabel;

namespace sgt::ui {

class StatusChip final : public QFrame {
    Q_OBJECT

public:
    enum class Tone { Neutral, Ok, Warn, Danger };

    StatusChip(const QString& text, Tone tone, QWidget* parent = nullptr);

    void setText(const QString& text);
    void setTone(Tone tone);

private:
    QLabel* dot_ = nullptr;
    QLabel* label_ = nullptr;
    Tone tone_;

    void applyTone();
};

} // namespace sgt::ui
```

- [ ] **Step 8.2: Write `ui/StatusChip.cpp`**

```cpp
#include "ui/StatusChip.h"

#include <QHBoxLayout>
#include <QLabel>

#include "ui/ThemeManager.h"

namespace sgt::ui {

StatusChip::StatusChip(const QString& text, Tone tone, QWidget* parent)
    : QFrame(parent), tone_(tone)
{
    setObjectName("StatusChip");
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(10, 4, 12, 4);
    layout->setSpacing(8);

    dot_ = new QLabel(QStringLiteral("●"), this);
    dot_->setObjectName("StatusChipDot");
    label_ = new QLabel(text, this);
    label_->setObjectName("StatusChipText");
    layout->addWidget(dot_);
    layout->addWidget(label_);

    applyTone();
    connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this,
            [this](const ThemeTokens&) { applyTone(); });
}

void StatusChip::setText(const QString& text) { label_->setText(text); }

void StatusChip::setTone(Tone tone)
{
    if (tone_ == tone) return;
    tone_ = tone;
    applyTone();
}

void StatusChip::applyTone()
{
    const auto& t = ThemeManager::instance().tokens();
    QString color;
    switch (tone_) {
    case Tone::Neutral: color = t.textSecondary; break;
    case Tone::Ok:      color = t.accent;         break;
    case Tone::Warn:    color = t.warn;           break;
    case Tone::Danger:  color = t.danger;         break;
    }
    dot_->setStyleSheet(QString("color: %1; background: transparent;").arg(color));
}

} // namespace sgt::ui
```

- [ ] **Step 8.3: Write `ui/ModePillBar.h`**

```cpp
#pragma once

#include <cstdint>

#include <QFrame>

#include "core/Renderer.h"   // for MODE_TOOL / MODE_GRASP / MODE_DEFECT

class QPushButton;

namespace sgt::ui {

class ModePillBar final : public QFrame {
    Q_OBJECT

public:
    explicit ModePillBar(uint8_t initialMask, QWidget* parent = nullptr);

    void setMask(uint8_t mask);
    uint8_t mask() const;

signals:
    void maskChanged(uint8_t mask);

private:
    QPushButton* tool_ = nullptr;
    QPushButton* grasp_ = nullptr;
    QPushButton* defect_ = nullptr;

    QPushButton* makePill(const QString& text);
    void emitIfChanged(uint8_t before);
};

} // namespace sgt::ui
```

- [ ] **Step 8.4: Write `ui/ModePillBar.cpp`**

```cpp
#include "ui/ModePillBar.h"

#include <QHBoxLayout>
#include <QPushButton>

namespace sgt::ui {

ModePillBar::ModePillBar(uint8_t initialMask, QWidget* parent)
    : QFrame(parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    tool_   = makePill("Tool");
    grasp_  = makePill("Grasp");
    defect_ = makePill("Defect");
    tool_  ->setChecked(initialMask & MODE_TOOL);
    grasp_ ->setChecked(initialMask & MODE_GRASP);
    defect_->setChecked(initialMask & MODE_DEFECT);

    layout->addWidget(tool_);
    layout->addWidget(grasp_);
    layout->addWidget(defect_);

    auto handle = [this]() {
        uint8_t before = 0; // computed in emitIfChanged
        emitIfChanged(before);
    };
    connect(tool_,   &QPushButton::toggled, this, handle);
    connect(grasp_,  &QPushButton::toggled, this, handle);
    connect(defect_, &QPushButton::toggled, this, handle);
}

QPushButton* ModePillBar::makePill(const QString& text)
{
    auto* b = new QPushButton(text, this);
    b->setObjectName("ModePill");
    b->setCheckable(true);
    b->setCursor(Qt::PointingHandCursor);
    return b;
}

void ModePillBar::setMask(uint8_t m)
{
    QSignalBlocker b1(tool_), b2(grasp_), b3(defect_);
    tool_  ->setChecked(m & MODE_TOOL);
    grasp_ ->setChecked(m & MODE_GRASP);
    defect_->setChecked(m & MODE_DEFECT);
}

uint8_t ModePillBar::mask() const
{
    uint8_t m = 0;
    if (tool_  ->isChecked()) m |= MODE_TOOL;
    if (grasp_ ->isChecked()) m |= MODE_GRASP;
    if (defect_->isChecked()) m |= MODE_DEFECT;
    return m;
}

void ModePillBar::emitIfChanged(uint8_t)
{
    uint8_t m = mask();
    if (m == 0) {
        // Refuse the "none" state — restore Tool as a safe default.
        QSignalBlocker b(tool_);
        tool_->setChecked(true);
        m = MODE_TOOL;
    }
    emit maskChanged(m);
}

} // namespace sgt::ui
```

- [ ] **Step 8.5: Write `ui/HudOverlay.h`**

```cpp
#pragma once

#include <cstdint>

#include <QWidget>

namespace sgt::ui {

class StatusChip;
class ModePillBar;
class QPushButton;

class HudOverlay final : public QWidget {
    Q_OBJECT

public:
    HudOverlay(uint8_t initialMask, QWidget* host);

    void setCameraStatus(const QString& text);
    void setFps(float fps);
    void setMask(uint8_t mask);

signals:
    void captureRequested();
    void maskChanged(uint8_t mask);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    QWidget* host_;
    StatusChip* cameraChip_ = nullptr;
    StatusChip* fpsChip_ = nullptr;
    ModePillBar* pills_ = nullptr;
    QPushButton* fab_ = nullptr;

    void relayout();
};

} // namespace sgt::ui
```

- [ ] **Step 8.6: Write `ui/HudOverlay.cpp`**

```cpp
#include "ui/HudOverlay.h"

#include <QEvent>
#include <QPushButton>
#include <QResizeEvent>

#include "ui/ModePillBar.h"
#include "ui/StatusChip.h"

namespace sgt::ui {

HudOverlay::HudOverlay(uint8_t initialMask, QWidget* host)
    : QWidget(host)
    , host_(host)
{
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
    setAttribute(Qt::WA_NoSystemBackground, true);
    setAttribute(Qt::WA_TranslucentBackground, true);

    cameraChip_ = new StatusChip("Camera starting", StatusChip::Tone::Warn, this);
    fpsChip_    = new StatusChip("FPS 0.0",         StatusChip::Tone::Neutral, this);
    pills_      = new ModePillBar(initialMask, this);
    fab_        = new QPushButton("◉ Capture", this);
    fab_->setObjectName("CaptureFAB");
    fab_->setCursor(Qt::PointingHandCursor);
    fab_->setFixedSize(140, 48);

    connect(pills_, &ModePillBar::maskChanged, this, &HudOverlay::maskChanged);
    connect(fab_,   &QPushButton::clicked,     this, &HudOverlay::captureRequested);

    host_->installEventFilter(this);
    relayout();
}

void HudOverlay::setCameraStatus(const QString& text)
{
    cameraChip_->setText(text);
    cameraChip_->setTone(text.contains("online") ? StatusChip::Tone::Ok
                                                 : StatusChip::Tone::Warn);
}

void HudOverlay::setFps(float fps)
{
    fpsChip_->setText(QString("FPS %1").arg(fps, 0, 'f', 1));
}

void HudOverlay::setMask(uint8_t m) { pills_->setMask(m); }

bool HudOverlay::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == host_ && event->type() == QEvent::Resize) {
        setGeometry(host_->rect());
        relayout();
    }
    return QWidget::eventFilter(watched, event);
}

void HudOverlay::relayout()
{
    const int margin = 14;
    cameraChip_->move(margin, margin);
    fpsChip_->move(cameraChip_->x() + cameraChip_->sizeHint().width() + 8, margin);

    const QSize pillSize = pills_->sizeHint();
    pills_->resize(pillSize);
    pills_->move(width() - pillSize.width() - margin, margin);

    fab_->move(width() - fab_->width() - margin, height() - fab_->height() - margin);
}

} // namespace sgt::ui
```

- [ ] **Step 8.7: Update `ui/LivePage.h`** — remove the in-page header chips/button and own a `HudOverlay` instead. Replace the full file with:

```cpp
#pragma once

#include <cstdint>

#include <QWidget>

#include "core/DetectionPipeline.h"

namespace sgt::ui {

class LivePreviewWidget;
class ControlPanel;
class HudOverlay;
class QFrame;

class LivePage final : public QWidget {
    Q_OBJECT

public:
    LivePage(uint8_t initialModeMask,
             const DetectionThresholds& initialThresholds,
             QWidget* parent = nullptr);

    ControlPanel* controlPanel() const { return controlPanel_; }
    LivePreviewWidget* preview() const { return preview_; }

public slots:
    void applyFrame(const DetectionFrameResult& result, uint8_t activeMask);
    void setCameraStatus(const QString& text);
    void setFps(float fps);
    void setMaskFromOutside(uint8_t mask);  // keeps HUD pills in sync

signals:
    void captureRequested();
    void modeMaskChanged(uint8_t mask);
    void thresholdsChanged(const DetectionThresholds& t);

private:
    QFrame* stage_ = nullptr;
    LivePreviewWidget* preview_ = nullptr;
    HudOverlay* hud_ = nullptr;
    ControlPanel* controlPanel_ = nullptr;
};

} // namespace sgt::ui
```

- [ ] **Step 8.8: Replace `ui/LivePage.cpp`**

```cpp
#include "ui/LivePage.h"

#include <QFrame>
#include <QHBoxLayout>
#include <QSplitter>
#include <QVBoxLayout>

#include "ui/ControlPanel.h"
#include "ui/HudOverlay.h"
#include "ui/LivePreviewWidget.h"

namespace sgt::ui {

LivePage::LivePage(uint8_t initialModeMask,
                   const DetectionThresholds& initialThresholds,
                   QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(20, 18, 20, 18);
    layout->setSpacing(14);

    auto* splitter = new QSplitter(Qt::Horizontal, this);
    splitter->setChildrenCollapsible(false);

    stage_ = new QFrame(splitter);
    stage_->setObjectName("VideoStage");
    auto* stageLayout = new QVBoxLayout(stage_);
    stageLayout->setContentsMargins(0, 0, 0, 0);
    preview_ = new LivePreviewWidget(stage_);
    stageLayout->addWidget(preview_);

    hud_ = new HudOverlay(initialModeMask, stage_);
    hud_->raise();

    controlPanel_ = new ControlPanel(initialModeMask, initialThresholds, splitter);

    splitter->addWidget(stage_);
    splitter->addWidget(controlPanel_);
    splitter->setStretchFactor(0, 7);
    splitter->setStretchFactor(1, 3);
    layout->addWidget(splitter, 1);

    connect(hud_, &HudOverlay::captureRequested, this, &LivePage::captureRequested);
    connect(hud_, &HudOverlay::maskChanged, this, [this](uint8_t m) {
        controlPanel_->setMaskFromOutside(m);
        emit modeMaskChanged(m);
    });
    connect(controlPanel_, &ControlPanel::modeMaskChanged, this, [this](uint8_t m) {
        hud_->setMask(m);
        emit modeMaskChanged(m);
    });
    connect(controlPanel_, &ControlPanel::thresholdsChanged, this, &LivePage::thresholdsChanged);
}

void LivePage::applyFrame(const DetectionFrameResult& result, uint8_t activeMask)
{
    preview_->setResult(result, activeMask);
}

void LivePage::setCameraStatus(const QString& text) { hud_->setCameraStatus(text); }
void LivePage::setFps(float fps)                    { hud_->setFps(fps); }
void LivePage::setMaskFromOutside(uint8_t mask)     { hud_->setMask(mask); }

} // namespace sgt::ui
```

> Note: `ControlPanel::setMaskFromOutside` is added in Task 9. For this task, temporarily implement it as a no-op stub at the bottom of `ui/ControlPanel.h` / `ControlPanel.cpp` so this code compiles. Task 9 replaces the stub with the real implementation.

- [ ] **Step 8.9: Add the temporary `setMaskFromOutside` stub to `ControlPanel`**

In `ui/ControlPanel.h`, add to the public slots block:
```cpp
    void setMaskFromOutside(uint8_t mask);
```
In `ui/ControlPanel.cpp`, append:
```cpp
void ControlPanel::setMaskFromOutside(uint8_t mask)
{
    (void)mask;  // implemented in Task 9
}
```

- [ ] **Step 8.10: Drop the duplicate summary line from `LivePreviewWidget`** — the HUD owns status. In `ui/LivePreviewWidget.cpp`:

- Delete the `summaryLabel_` widget construction (lines around 26-28) and the `summaryLabel_->setText(...)` call inside `setResult`.
- Delete the matching field from `ui/LivePreviewWidget.h`.

- [ ] **Step 8.11: CMake — add new sources**

Append to `SGT_SOURCES`:
```cmake
    ui/HudOverlay.cpp
    ui/ModePillBar.cpp
    ui/StatusChip.cpp
```
Append to `SGT_HEADERS`:
```cmake
    ui/HudOverlay.h
    ui/ModePillBar.h
    ui/StatusChip.h
```

- [ ] **Step 8.12: Build + smoke**

```bash
cmake --build build
./build/SGTDetector.exe
```
Expected: video fills the left pane; top-left two chips (camera + FPS); top-right Tool/Grasp/Defect pills; bottom-right round-cornered Capture button. Clicking a pill toggles modes and ControlPanel's checkboxes flip in step; toggling a ControlPanel checkbox flips the HUD pill.

- [ ] **Step 8.13: Commit**

```bash
git add ui/StatusChip.h ui/StatusChip.cpp \
        ui/ModePillBar.h ui/ModePillBar.cpp \
        ui/HudOverlay.h ui/HudOverlay.cpp \
        ui/LivePage.h ui/LivePage.cpp \
        ui/LivePreviewWidget.h ui/LivePreviewWidget.cpp \
        ui/ControlPanel.h ui/ControlPanel.cpp \
        CMakeLists.txt
git commit -m "feat(ui): floating HUD with status chips, mode pills, capture FAB"
```

---

## Task 9: ControlPanel cards + structured live-data table

**Files:**
- Modify: `ui/ControlPanel.h`, `ui/ControlPanel.cpp`
- Modify: `ui/AppShell.cpp` (replace `setResultText` with `applyResult`)

- [ ] **Step 9.1: Rewrite `ui/ControlPanel.h`**

```cpp
#pragma once

#include <cstdint>

#include <QFrame>

#include "core/DetectionMetadata.h"
#include "core/DetectionPipeline.h"

class QLabel;
class QSlider;
class QTableWidget;

namespace sgt::ui {

class ControlPanel final : public QFrame {
    Q_OBJECT

public:
    explicit ControlPanel(uint8_t initialModeMask,
                          const DetectionThresholds& initialThresholds,
                          QWidget* parent = nullptr);

    uint8_t modeMask() const { return currentModeMask_; }
    DetectionThresholds thresholds() const { return thresholds_; }

public slots:
    void toggleToolMode();
    void toggleGraspMode();
    void toggleDefectMode();
    void setMaskFromOutside(uint8_t mask);
    void applyResult(const DetectionFrameResult& result);

signals:
    void modeMaskChanged(uint8_t modeMask);
    void thresholdsChanged(const DetectionThresholds& thresholds);

private:
    uint8_t currentModeMask_ = 0;
    DetectionThresholds thresholds_;

    QSlider* toolSlider_ = nullptr;
    QSlider* graspSlider_ = nullptr;
    QSlider* defectSlider_ = nullptr;
    QLabel* toolValue_ = nullptr;
    QLabel* graspValue_ = nullptr;
    QLabel* defectValue_ = nullptr;
    QTableWidget* table_ = nullptr;

    QFrame* makeCard(const QString& title, QWidget* body) const;
    QSlider* makeSlider(float initialValue, QLabel*& valueLabel);
    void toggleBit(uint8_t bit);
    void emitMask(uint8_t mask);
};

} // namespace sgt::ui
```

- [ ] **Step 9.2: Rewrite `ui/ControlPanel.cpp`**

```cpp
#include "ui/ControlPanel.h"

#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QSignalBlocker>
#include <QSlider>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include "ui/ThemeManager.h"

namespace sgt::ui {

ControlPanel::ControlPanel(uint8_t initialModeMask,
                           const DetectionThresholds& initialThresholds,
                           QWidget* parent)
    : QFrame(parent)
    , currentModeMask_(initialModeMask ? initialModeMask : MODE_TOOL)
    , thresholds_(initialThresholds)
{
    setMinimumWidth(340);
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(12);

    // Thresholds card
    auto* tBody = new QWidget(this);
    auto* tLayout = new QVBoxLayout(tBody);
    tLayout->setContentsMargins(0, 0, 0, 0);
    tLayout->setSpacing(10);

    auto addSliderRow = [&](const QString& name, QSlider*& slider, QLabel*& value, float initial) {
        auto* row = new QHBoxLayout();
        auto* lab = new QLabel(name, tBody);
        value = new QLabel(QString::number(static_cast<int>(initial * 100)) + "%", tBody);
        value->setObjectName("SubtleText");
        row->addWidget(lab);
        row->addStretch();
        row->addWidget(value);
        tLayout->addLayout(row);
        slider = makeSlider(initial, value);
        slider->setProperty("rowName", name);
        tLayout->addWidget(slider);
    };
    addSliderRow("Tool",   toolSlider_,   toolValue_,   thresholds_.tool);
    addSliderRow("Grasp",  graspSlider_,  graspValue_,  thresholds_.grasp);
    addSliderRow("Defect", defectSlider_, defectValue_, thresholds_.defect);

    layout->addWidget(makeCard("Thresholds", tBody));

    // Live data card
    table_ = new QTableWidget(0, 3, this);
    table_->setHorizontalHeaderLabels({"Mode", "Label", "Score"});
    table_->verticalHeader()->setVisible(false);
    table_->horizontalHeader()->setStretchLastSection(true);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->setSelectionMode(QAbstractItemView::NoSelection);
    table_->setFocusPolicy(Qt::NoFocus);
    table_->setMinimumHeight(220);
    layout->addWidget(makeCard("Live data", table_), 1);

    auto wireSlider = [this](QSlider* s, QLabel* lab, float* dst) {
        connect(s, &QSlider::valueChanged, this, [this, lab, dst](int v) {
            *dst = v / 100.0f;
            lab->setText(QString::number(v) + "%");
            emit thresholdsChanged(thresholds_);
        });
    };
    wireSlider(toolSlider_,   toolValue_,   &thresholds_.tool);
    wireSlider(graspSlider_,  graspValue_,  &thresholds_.grasp);
    wireSlider(defectSlider_, defectValue_, &thresholds_.defect);
}

QFrame* ControlPanel::makeCard(const QString& title, QWidget* body) const
{
    auto* card = new QFrame;
    card->setObjectName("Card");
    auto* lay = new QVBoxLayout(card);
    lay->setContentsMargins(14, 12, 14, 14);
    lay->setSpacing(8);
    auto* t = new QLabel(title.toUpper(), card);
    t->setObjectName("CardTitle");
    lay->addWidget(t);
    lay->addWidget(body, 1);
    return card;
}

QSlider* ControlPanel::makeSlider(float value, QLabel*& /*valueLabel*/)
{
    auto* s = new QSlider(Qt::Horizontal, this);
    s->setRange(5, 95);
    s->setValue(static_cast<int>(value * 100.0f));
    return s;
}

void ControlPanel::toggleToolMode()   { toggleBit(MODE_TOOL); }
void ControlPanel::toggleGraspMode()  { toggleBit(MODE_GRASP); }
void ControlPanel::toggleDefectMode() { toggleBit(MODE_DEFECT); }

void ControlPanel::toggleBit(uint8_t bit)
{
    uint8_t next = currentModeMask_ ^ bit;
    if (next == 0) next = bit;  // refuse "none"
    emitMask(next);
}

void ControlPanel::setMaskFromOutside(uint8_t mask)
{
    if (mask == 0) mask = MODE_TOOL;
    if (mask == currentModeMask_) return;
    currentModeMask_ = mask;
    // No internal UI to sync now that mode toggles live in HUD.
}

void ControlPanel::emitMask(uint8_t mask)
{
    if (mask == currentModeMask_) return;
    currentModeMask_ = mask;
    emit modeMaskChanged(currentModeMask_);
}

void ControlPanel::applyResult(const DetectionFrameResult& result)
{
    table_->setRowCount(0);

    auto addRow = [&](const QString& mode, const QString& label, float score, bool defective = false) {
        const int row = table_->rowCount();
        table_->insertRow(row);
        auto* m = new QTableWidgetItem(mode);
        auto* l = new QTableWidgetItem(label);
        auto* s = new QTableWidgetItem(QString::number(score * 100.0f, 'f', 1) + "%");
        if (defective) {
            const QColor warn(ThemeManager::instance().tokens().warn);
            for (auto* it : {m, l, s}) it->setForeground(warn);
        }
        table_->setItem(row, 0, m);
        table_->setItem(row, 1, l);
        table_->setItem(row, 2, s);
    };

    for (const auto& d : result.toolDetections)  addRow("Tool",   QString::fromStdString(d.label), d.score);
    for (const auto& d : result.graspDetections) addRow("Grasp",  QString::fromStdString(d.label), d.score);
    for (const auto& d : result.defectResults) {
        addRow("Defect", d.defective ? "DEFECT" : "normal", d.defectScore, d.defective);
    }
}

} // namespace sgt::ui
```

- [ ] **Step 9.3: Wire `applyResult` from `AppShell`** — in `ui/AppShell.cpp::processFrame`, after `livePage_->applyFrame(...)`, add:

```cpp
        livePage_->controlPanel()->applyResult(lastResult_);
```

Remove any leftover call to `controlPanel_->setResultText(...)` if present.

- [ ] **Step 9.4: Drop the now-unused `setResultText` references**

Search:
```bash
grep -rn "setResultText" ui/ main.cpp
```
Expected: no matches outside this comment.

- [ ] **Step 9.5: Build + smoke**

```bash
cmake --build build
./build/SGTDetector.exe
```
Expected: right rail has two cards — "THRESHOLDS" with three labeled sliders showing live percentages, and "LIVE DATA" with a 3-column table that updates each frame; defective rows render in the warn colour.

- [ ] **Step 9.6: Commit**

```bash
git add ui/ControlPanel.h ui/ControlPanel.cpp ui/AppShell.cpp
git commit -m "feat(ui): card-based ControlPanel with structured live-data table"
```

---

## Task 10: Deployment, plugins, finalization

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 10.1: Flip `SGT_RUN_WINDEPLOYQT` default to `ON`**

In `CMakeLists.txt`, change:
```cmake
option(SGT_RUN_WINDEPLOYQT "Run windeployqt6 after build"                  OFF)
```
to:
```cmake
option(SGT_RUN_WINDEPLOYQT "Run windeployqt6 after build"                  ON)
```

- [ ] **Step 10.2: Extend the manual fallback to copy imageformats**

Find the `else()` branch of the `if(SGT_RUN_WINDEPLOYQT AND WINDEPLOYQT6_EXECUTABLE)` block. After the existing `qwindows.dll` copy, add the following block (still inside the same `else()`):

```cmake
    set(QT6_IMAGEFORMATS "${_MINGW_PREFIX}/share/qt6/plugins/imageformats")
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:SGTDetector>/imageformats"
        COMMENT "Preparing imageformats plugin directory...")
    foreach(_fmt qjpeg qico qgif qsvg)
        add_custom_command(TARGET SGTDetector POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DSRC=${QT6_IMAGEFORMATS}/${_fmt}.dll"
                "-DDST=$<TARGET_FILE_DIR:SGTDetector>/imageformats/${_fmt}.dll"
                -P "${CMAKE_BINARY_DIR}/copy_if_exists.cmake"
            COMMENT "Copying Qt6 imageformats/${_fmt}.dll (if available)...")
    endforeach()

    set(QT6_STYLES "${_MINGW_PREFIX}/share/qt6/plugins/styles")
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:SGTDetector>/styles"
        COMMENT "Preparing styles plugin directory...")
    add_custom_command(TARGET SGTDetector POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            "-DSRC=${QT6_STYLES}/qmodernwindowsstyle.dll"
            "-DDST=$<TARGET_FILE_DIR:SGTDetector>/styles/qmodernwindowsstyle.dll"
            -P "${CMAKE_BINARY_DIR}/copy_if_exists.cmake"
        COMMENT "Copying Qt6 styles/qmodernwindowsstyle.dll (if available)...")
```

- [ ] **Step 10.3: Clean rebuild**

```bash
rm -rf build
cmake -B build -G Ninja
cmake --build build
```

Inspect `build/imageformats/` (if `windeployqt6` is found in this env it will populate the directory automatically; if not, our fallback should place `qjpeg.dll` there).

Expected: `build/imageformats/qjpeg.dll` exists.

- [ ] **Step 10.4: Full manual acceptance pass**

Launch:
```bash
./build/SGTDetector.exe
```

Verify each item against the spec's §7 acceptance list:

- Live page renders camera feed; HUD shows camera-online status, FPS, mode pills.
- Pressing `C` saves a capture; bottom status bar reports the id.
- Sidebar gallery icon switches to the gallery page; every prior capture renders with a **real thumbnail image** (root-cause fix from spec §3.5).
- Clicking a `ThumbCard` opens the modal; `←` / `→` step through captures; Esc closes; `Open folder` opens Windows Explorer at the capture's day directory.
- Bottom-left moon/sun toggle flips theme; relaunch — choice persists via `QSettings`.
- Search box filters cards by `id` substring and timestamp substring.
- `ctest --test-dir build --output-on-failure` passes all three tests
  (`CaptureStoreSmoke`, `ThemeSmoke`, `ThumbnailCacheSmoke`).

- [ ] **Step 10.5: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: deploy imageformats + modern style plugins"
```

---

## Verification matrix (spec → task)

| Spec section                                | Implemented in   |
|---------------------------------------------|------------------|
| §3.1 AppShell + Sidebar + QStackedWidget    | Tasks 3, 4       |
| §3.2 Theme tokens + QSS template + manager  | Task 1, 2        |
| §3.3 LivePage with HUD overlay              | Tasks 4, 8       |
| §3.3 ControlPanel cards + table             | Task 9           |
| §3.4 GalleryPage + ThumbCard + FlowLayout   | Task 5           |
| §3.4 CaptureDetailDialog                    | Task 7           |
| §3.5 Thumbnail root-cause fix (sync)        | Task 5           |
| §3.5 ThumbnailCache (async)                 | Task 6           |
| §3.6 OpenCV `imgcodecs`, resource, sources  | Tasks 1, 10      |
| §3.6 windeployqt + plugin fallback          | Task 10          |
| §3.7 main.cpp wiring                        | Tasks 2, 4       |
| §7 Acceptance (ctest + manual smoke)        | Tasks 1, 6, 10   |
| §4.3 Deletes (MainWindow / GalleryPanel / AppStyle) | Tasks 2, 4, 5 |
