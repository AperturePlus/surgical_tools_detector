# UI Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine the SGTDetector Qt6 Widgets UI across three independent layers — theme token consistency, layout/information hierarchy, and rendering performance — without restructuring the architecture.

**Architecture:** Layer 1 extends `ThemeTokens` and `base.qss` with missing primitives (chip backgrounds, scrollbars, focus rings, card variants) and fixes two hardcoded dark colors that break the light theme. Layer 2 adjusts margins, spacing, card hierarchy, HUD dot semantics, and the gallery header/empty-state. Layer 3 adds a diff-and-skip signature to `ControlPanel::setFrameResult` and removes the per-hover `QGraphicsDropShadowEffect` from `ThumbCard`. Each layer is a standalone commit.

**Tech Stack:** Qt 6 Widgets, C++17, CMake, OpenCV (imgcodecs already linked). Tests are plain `int main` smoke binaries registered via `add_test(NAME ... COMMAND ...)` — no GTest/QtTest framework.

## Global Constraints

- Namespace is `xcwj::ui` (rebranded from `sgt::`); do not reintroduce `sgt::`.
- Theme tokens are `QString` hex/rgba values; QSS templating uses `QString::replace("{{token}}", value)` in `ui/Theme.cpp::renderQss`.
- QSS template lives at `assets/qss/base.qss`, compiled into the binary as `:/qss/base.qss` via `assets/icons.qrc`.
- The `ThemeSmoke` test asserts no `{{` placeholders remain after `renderQss` — any new token added MUST be substituted in `renderQss` or this test fails.
- No new files; all changes are in existing files (spec §7.2).
- No signal/slot rewiring, no page-class restructuring, no QML.
- Build directory is `build/`; configure with CMake as shown in Task 1 Step 2.

---

## Task 1: Theme tokens — add fields and populate dark/light

**Files:**
- Modify: `ui/Theme.h` (struct `ThemeTokens`)
- Modify: `ui/Theme.cpp` (`dark()`, `light()`, `renderQss()`)
- Test: `tests/ThemeSmoke.cpp` (existing — extended in Step 1)

**Interfaces:**
- Consumes: nothing (this is the foundation task).
- Produces: `ThemeTokens` now has `chipBg`, `chipBorder`, `focusRing`, `scrollBar`, `scrollBarHover` fields, all populated for both themes and substituted in `renderQss`. Later tasks reference `{{chipBg}}`, `{{focusRing}}`, `{{scrollBar}}`, `{{scrollBarHover}}` in QSS.

- [ ] **Step 1: Extend the failing test**

Add assertions to `tests/ThemeSmoke.cpp` that the new tokens are present and substituted. Insert before the final `return 0;` (after the existing `lightQss` checks, line 41):

```cpp
    // New tokens must exist and be substituted (no leftover {{...}}).
    const QStringList newTokens = {
        dark.chipBg, dark.chipBorder, dark.focusRing, dark.scrollBar, dark.scrollBarHover,
        light.chipBg, light.chipBorder, light.focusRing, light.scrollBar, light.scrollBarHover,
    };
    for (const QString& tok : newTokens) {
        if (tok.isEmpty()) {
            std::cerr << "a new token is empty\n";
            return 1;
        }
    }
    if (!darkQss.contains(dark.chipBg)) {
        std::cerr << "dark chipBg not substituted into qss\n";
        return 1;
    }
    if (!lightQss.contains(light.chipBg)) {
        std::cerr << "light chipBg not substituted into qss\n";
        return 1;
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `build/`):
```bash
cmake --build . --target ThemeSmoke && ctest -R ThemeSmoke --output-on-failure
```
Expected: FAIL — compile error (`'const struct xcwj::ui::ThemeTokens' has no member named 'chipBg'`).

- [ ] **Step 3: Add fields to `ThemeTokens`**

In `ui/Theme.h`, add five fields to the struct, after `shadow;` and before the closing `};`:

```cpp
    QString chipBg;
    QString chipBorder;
    QString focusRing;
    QString scrollBar;
    QString scrollBarHover;
```

- [ ] **Step 4: Populate dark and light factories**

In `ui/Theme.cpp`, replace `dark()`:

```cpp
ThemeTokens dark()
{
    return {
        "dark",
        "#0E141B", "#161D26", "#1E2733", "#243140",
        "#E6EDF3", "#8B98A8",
        "#14B8A6", "#2DD4BF",
        "#38BDF8", "#F59E0B", "#EF4444",
        "rgba(0,0,0,0.35)",
        "rgba(14,20,27,0.72)", "#243140", "#38BDF8", "#2A3441", "#3A4654"
    };
}
```

Replace `light()`:

```cpp
ThemeTokens light()
{
    return {
        "light",
        "#F4F6F9", "#FFFFFF", "#FFFFFF", "#E2E8F0",
        "#0F172A", "#64748B",
        "#0F766E", "#0D9488",
        "#2563EB", "#D97706", "#DC2626",
        "rgba(15,23,42,0.12)",
        "rgba(255,255,255,0.82)", "#E2E8F0", "#2563EB", "#CBD5E1", "#94A3B8"
    };
}
```

- [ ] **Step 5: Substitute new tokens in `renderQss`**

In `ui/Theme.cpp` `renderQss`, add after the `s.replace("{{shadow}}", t.shadow);` line (before `return s;`):

```cpp
    s.replace("{{chipBg}}", t.chipBg);
    s.replace("{{chipBorder}}", t.chipBorder);
    s.replace("{{focusRing}}", t.focusRing);
    s.replace("{{scrollBar}}", t.scrollBar);
    s.replace("{{scrollBarHover}}", t.scrollBarHover);
```

- [ ] **Step 6: Run test to verify it passes**

Run:
```bash
cmake --build . --target ThemeSmoke && ctest -R ThemeSmoke --output-on-failure
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ui/Theme.h ui/Theme.cpp tests/ThemeSmoke.cpp
git commit -m "feat(theme): add chipBg/focusRing/scrollBar tokens for both themes"
```

---

## Task 2: Theme layer — fix hardcodes and add QSS primitives

**Files:**
- Modify: `assets/qss/base.qss`
- Modify: `ui/GalleryPage.cpp:29` (margins)
- Modify: `ui/SettingsPage.cpp:37` (margins)
- Test: `tests/ThemeSmoke.cpp` (existing — extended in Step 1)

**Interfaces:**
- Consumes: Task 1's new tokens (`{{chipBg}}`, `{{focusRing}}`, `{{scrollBar}}`, `{{scrollBarHover}}`).
- Produces: `base.qss` has `QFrame#CardFlat`, `QPushButton#Secondary`, scrollbar, and focus-ring rules. `QFrame#Card` gains `border-left: 3px solid {{accent}}`. `QFrame#StatusChip` and `QPushButton#ModePill` use `{{chipBg}}`. Gallery/Settings margins are `20,18,20,18`.

- [ ] **Step 1: Extend the test to guard against the hardcoded color**

In `tests/ThemeSmoke.cpp`, add after the `lightQss` checks (before the new-token block added in Task 1, or immediately after it — either is fine; place it right after the `lightQss.contains(light.accent)` check at line 41):

```cpp
    // HUD chips must not hardcode the dark background; they must use the token.
    if (darkQss.contains("rgba(14,20,27,0.72)")) {
        std::cerr << "dark qss still hardcodes chip background\n";
        return 1;
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cmake --build . --target ThemeSmoke && ctest -R ThemeSmoke --output-on-failure
```
Expected: FAIL — "dark qss still hardcodes chip background".

- [ ] **Step 3: Fix the two hardcoded chip backgrounds in `base.qss`**

In `assets/qss/base.qss`:

`QFrame#StatusChip` (line 52): change
```
    background: rgba(14,20,27,0.72);
```
to
```
    background: {{chipBg}};
```

`QPushButton#ModePill` (line 66): change
```
    background: rgba(14,20,27,0.72);
```
to
```
    background: {{chipBg}};
```

- [ ] **Step 4: Add card, secondary-button, scrollbar, and focus-ring rules**

Append to the end of `assets/qss/base.qss`:

```css
QFrame#Card { border-left: 3px solid {{accent}}; }
QFrame#CardFlat { background: transparent; border: none; }
QFrame#CardFlat QLabel#CardTitle { color: {{textSecondary}}; }

QPushButton#Secondary {
    background: transparent;
    border: 1px solid {{border}};
    color: {{textSecondary}};
    border-radius: 6px;
    padding: 8px 14px;
}
QPushButton#Secondary:hover { background: {{surface}}; color: {{textPrimary}}; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: {{scrollBar}};
    border-radius: 4px;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: {{scrollBarHover}};
}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page { background: none; height: 0; width: 0; }

QLineEdit:focus, QPushButton:focus, QToolButton:focus,
QSlider:focus { border: 1px solid {{focusRing}}; }
```

- [ ] **Step 5: Unify page margins**

In `ui/GalleryPage.cpp` line 29, change:
```cpp
    root->setContentsMargins(22, 18, 22, 18);
```
to:
```cpp
    root->setContentsMargins(20, 18, 20, 18);
```

In `ui/SettingsPage.cpp` line 37, change:
```cpp
    root->setContentsMargins(22, 18, 22, 18);
```
to:
```cpp
    root->setContentsMargins(20, 18, 20, 18);
```

- [ ] **Step 6: Run test to verify it passes**

Run:
```bash
cmake --build . --target ThemeSmoke && ctest -R ThemeSmoke --output-on-failure
```
Expected: PASS.

- [ ] **Step 7: Build the full app to confirm no regressions**

Run:
```bash
cmake --build . --target XunChaWeiJian
```
Expected: builds cleanly.

- [ ] **Step 8: Commit**

```bash
git add assets/qss/base.qss ui/GalleryPage.cpp ui/SettingsPage.cpp tests/ThemeSmoke.cpp
git commit -m "feat(theme): fix chip hardcodes, add scrollbar/focus/card primitives"
```

---

## Task 3: Layout layer — Card variant and ControlPanel Models card

**Files:**
- Modify: `ui/Card.h`
- Modify: `ui/Card.cpp`
- Modify: `ui/ControlPanel.cpp:91` (Models card construction)

**Interfaces:**
- Consumes: Task 2's `QFrame#CardFlat` QSS rule.
- Produces: `Card` has a `Variant` enum (`Standard`, `Flat`) and a two-arg constructor `Card(title, Variant, parent)`. The existing single-arg constructor still works (delegates to `Standard`).

- [ ] **Step 1: Add the Variant enum and constructor to `Card.h`**

Replace the entire contents of `ui/Card.h` with:

```cpp
#pragma once

#include <QFrame>

class QVBoxLayout;

namespace xcwj::ui {

class Card final : public QFrame {
    Q_OBJECT

public:
    enum class Variant { Standard, Flat };

    explicit Card(const QString& title, QWidget* parent = nullptr);
    Card(const QString& title, Variant variant, QWidget* parent = nullptr);

    QVBoxLayout* bodyLayout() const { return bodyLayout_; }

private:
    void init(const QString& title, Variant variant);

    QVBoxLayout* bodyLayout_ = nullptr;
};

} // namespace xcwj::ui
```

- [ ] **Step 2: Implement the variant in `Card.cpp`**

Replace the entire contents of `ui/Card.cpp` with:

```cpp
#include "ui/Card.h"

#include <QLabel>
#include <QVBoxLayout>

namespace xcwj::ui {

Card::Card(const QString& title, QWidget* parent)
    : Card(title, Variant::Standard, parent) {}

Card::Card(const QString& title, Variant variant, QWidget* parent)
    : QFrame(parent)
{
    init(title, variant);
}

void Card::init(const QString& title, Variant variant)
{
    setObjectName(variant == Variant::Flat ? "CardFlat" : "Card");
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(14, 12, 14, 14);
    layout->setSpacing(10);

    auto* titleLabel = new QLabel(title);
    titleLabel->setObjectName("CardTitle");
    layout->addWidget(titleLabel);

    bodyLayout_ = new QVBoxLayout();
    bodyLayout_->setContentsMargins(0, 0, 0, 0);
    bodyLayout_->setSpacing(8);
    layout->addLayout(bodyLayout_);
}

} // namespace xcwj::ui
```

- [ ] **Step 3: Switch the Models card to Flat**

In `ui/ControlPanel.cpp`, change line 91:
```cpp
    auto* modelsCard = new Card("Models");
```
to:
```cpp
    auto* modelsCard = new Card("Models", Card::Variant::Flat);
```

- [ ] **Step 4: Build and run smoke tests**

Run:
```bash
cmake --build . --target XunChaWeiJian ThemeSmoke && ctest --output-on-failure
```
Expected: builds cleanly; all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/Card.h ui/Card.cpp ui/ControlPanel.cpp
git commit -m "feat(ui): add Card::Flat variant, use it for Models card"
```

---

## Task 4: Layout layer — StatusChip dot color API and HUD wiring

**Files:**
- Modify: `ui/StatusChip.h`
- Modify: `ui/StatusChip.cpp`
- Modify: `ui/HudOverlay.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks (independent of tokens).
- Produces: `StatusChip::setDotColor(const QString& color)` applies a stylesheet background to the dot. `HudOverlay::setCameraStatus` derives the dot color (`danger` if text contains "unavailable", else `accent`); the FPS chip dot is set to `info` once in the constructor.

- [ ] **Step 1: Add `setDotColor` to `StatusChip.h`**

Replace the entire contents of `ui/StatusChip.h` with:

```cpp
#pragma once

#include <QFrame>
#include <QString>

class QLabel;

namespace xcwj::ui {

class StatusChip final : public QFrame {
    Q_OBJECT

public:
    explicit StatusChip(const QString& text, QWidget* parent = nullptr);

    void setText(const QString& text);
    void setDotColor(const QString& color);

private:
    QFrame* dot_ = nullptr;
    QLabel* textLabel_ = nullptr;
};

} // namespace xcwj::ui
```

- [ ] **Step 2: Implement `setDotColor` in `StatusChip.cpp`**

Replace the entire contents of `ui/StatusChip.cpp` with:

```cpp
#include "ui/StatusChip.h"

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>

namespace xcwj::ui {

StatusChip::StatusChip(const QString& text, QWidget* parent)
    : QFrame(parent)
{
    setObjectName("StatusChip");
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(10, 4, 12, 4);
    layout->setSpacing(8);

    dot_ = new QFrame();
    dot_->setObjectName("StatusChipDot");
    dot_->setFixedSize(8, 8);
    layout->addWidget(dot_, 0, Qt::AlignVCenter);

    textLabel_ = new QLabel(text);
    textLabel_->setObjectName("StatusChipText");
    layout->addWidget(textLabel_);
}

void StatusChip::setText(const QString& text)
{
    textLabel_->setText(text);
}

void StatusChip::setDotColor(const QString& color)
{
    dot_->setStyleSheet(QString("background: %1; border: none; border-radius: 4px;").arg(color));
}

} // namespace xcwj::ui
```

- [ ] **Step 3: Wire dot colors in `HudOverlay.cpp` and `.h`**

First add a member to store the last camera status. In `ui/HudOverlay.h`, add to the private section (after `CaptureFAB* captureButton_ = nullptr;`):
```cpp
    QString lastCameraStatus_;
```

In `ui/HudOverlay.cpp`, add the `ThemeManager` include after the existing includes (after `#include "ui/StatusChip.h"`):
```cpp
#include "ui/ThemeManager.h"
```

In the constructor, after `fpsChip_ = new StatusChip("FPS 0.0");` (line 28), set the FPS dot to `info`:
```cpp
    fpsChip_ = new StatusChip("FPS 0.0");
    fpsChip_->setDotColor(ThemeManager::instance().tokens().info);
```

At the end of the constructor (after the `connect(captureButton_, ...)` line), subscribe to theme changes so dots recolor when the theme flips:
```cpp
    connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, [this]() {
        const auto& t = ThemeManager::instance().tokens();
        fpsChip_->setDotColor(t.info);
        const QString camColor = lastCameraStatus_.contains("unavailable", Qt::CaseInsensitive)
            ? t.danger : t.accent;
        cameraChip_->setDotColor(camColor);
    });
```

Replace the `setCameraStatus` body (currently lines 48-51):
```cpp
void HudOverlay::setCameraStatus(const QString& text)
{
    cameraChip_->setText(text);
}
```
with:
```cpp
void HudOverlay::setCameraStatus(const QString& text)
{
    lastCameraStatus_ = text;
    cameraChip_->setText(text);
    const auto& tokens = ThemeManager::instance().tokens();
    const QString color = text.contains("unavailable", Qt::CaseInsensitive)
        ? tokens.danger
        : tokens.accent;
    cameraChip_->setDotColor(color);
}
```

- [ ] **Step 4: Build**

Run:
```bash
cmake --build . --target XunChaWeiJian
```
Expected: builds cleanly.

- [ ] **Step 5: Commit**

```bash
git add ui/StatusChip.h ui/StatusChip.cpp ui/HudOverlay.h ui/HudOverlay.cpp
git commit -m "feat(ui): semantic StatusChip dot color, wire camera/FPS in HUD"
```

---

## Task 5: Layout layer — Live page spacing and gallery header/empty-state

**Files:**
- Modify: `ui/LivePage.cpp:26` (handleWidth)
- Modify: `ui/ControlPanel.cpp:62` (spacing)
- Modify: `ui/GalleryPage.cpp` (header, empty state, count chip)

**Interfaces:**
- Consumes: Task 2's `QPushButton#Secondary` QSS rule; `IconLoader::load` (existing).
- Produces: Live splitter handleWidth 8; ControlPanel spacing 10; gallery header has no overflow button, a secondary Export button, and a count chip; empty state shows icon + two lines.

- [ ] **Step 1: Live page handle width**

In `ui/LivePage.cpp` line 26, change:
```cpp
    splitter->setHandleWidth(12);
```
to:
```cpp
    splitter->setHandleWidth(8);
```

- [ ] **Step 2: ControlPanel spacing**

In `ui/ControlPanel.cpp` line 62, change:
```cpp
    layout->setSpacing(12);
```
to:
```cpp
    layout->setSpacing(10);
```

- [ ] **Step 3: Gallery header — remove overflow, secondary Export, count chip**

In `ui/GalleryPage.cpp`, replace the header block (lines 32-53):
```cpp
    auto* header = new QHBoxLayout();
    header->setSpacing(10);
    titleLabel_ = new QLabel("Captures - 0");
    titleLabel_->setObjectName("AppTitle");
    header->addWidget(titleLabel_);

    searchEdit_ = new QLineEdit();
    searchEdit_->setPlaceholderText("Search id or date");
    searchEdit_->setMinimumWidth(260);
    header->addWidget(searchEdit_);
    header->addStretch();

    auto* exportButton = new QPushButton("Export all...");
    exportButton->setEnabled(false);
    exportButton->setToolTip("Export is planned for a later phase.");
    auto* overflow = new QToolButton();
    overflow->setText("...");
    overflow->setEnabled(false);
    overflow->setToolTip("More gallery actions are planned for a later phase.");
    header->addWidget(exportButton);
    header->addWidget(overflow);
    root->addLayout(header);
```
with:
```cpp
    auto* header = new QHBoxLayout();
    header->setSpacing(10);
    titleLabel_ = new QLabel("Captures");
    titleLabel_->setObjectName("AppTitle");
    header->addWidget(titleLabel_);

    countChip_ = new StatusChip("0");
    countChip_->setDotColor(ThemeManager::instance().tokens().accent);
    header->addWidget(countChip_);
    header->addSpacing(6);

    searchEdit_ = new QLineEdit();
    searchEdit_->setPlaceholderText("Search id or date");
    searchEdit_->setMinimumWidth(260);
    header->addWidget(searchEdit_);
    header->addStretch();

    auto* exportButton = new QPushButton("Export all...");
    exportButton->setObjectName("Secondary");
    exportButton->setEnabled(false);
    exportButton->setToolTip("Export is planned for a later phase.");
    header->addWidget(exportButton);
    root->addLayout(header);
```

Add the includes at the top of `ui/GalleryPage.cpp` (after the existing `#include "ui/GalleryFilterBar.h"` / before `#include "ui/ThumbCard.h"`, keeping alphabetical order):
```cpp
#include "ui/StatusChip.h"
#include "ui/ThemeManager.h"
```

Remove the now-unused `#include <QToolButton>` line from `ui/GalleryPage.cpp` (it was only used by the overflow button).

- [ ] **Step 4: Add `countChip_` member and update it in `rebuild`**

In `ui/GalleryPage.h`, add the include and member. First check the header:

`countChip_` is a `StatusChip*`. Add to `ui/GalleryPage.h`:
- forward declaration `class StatusChip;` in the `xcwj::ui` namespace block (or include `"ui/StatusChip.h"`).
- private member `StatusChip* countChip_ = nullptr;`

In `ui/GalleryPage.cpp` `rebuild()` (line 101), change:
```cpp
    titleLabel_->setText(QString("Captures - %1").arg(records_.size()));
```
to:
```cpp
    titleLabel_->setText("Captures");
    countChip_->setText(QString::number(records_.size()));
```

- [ ] **Step 5: Empty state — icon + two lines**

In `ui/GalleryPage.cpp`, replace the empty-state block (lines 118-127):
```cpp
    if (groups.empty()) {
        auto* empty = new QLabel(records_.empty()
            ? "No captures yet - press Capture or 'C' in Live."
            : "No captures match the current search.");
        empty->setObjectName("SubtleText");
        empty->setAlignment(Qt::AlignCenter);
        empty->setMinimumHeight(320);
        contentLayout_->addWidget(empty, 1);
        return;
    }
```
with:
```cpp
    if (groups.empty()) {
        auto* empty = new QWidget();
        auto* emptyLayout = new QVBoxLayout(empty);
        emptyLayout->setAlignment(Qt::AlignCenter);
        emptyLayout->setSpacing(8);
        const bool noCaptures = records_.empty();
        const auto& tokens = ThemeManager::instance().tokens();
        auto* icon = new QLabel();
        icon->setPixmap(IconLoader::load("nav-gallery", tokens.textSecondary, QSize(32, 32)).pixmap(QSize(32, 32)));
        icon->setAlignment(Qt::AlignCenter);
        auto* primary = new QLabel(noCaptures ? "No captures yet" : "No matches");
        primary->setObjectName("PanelTitle");
        primary->setAlignment(Qt::AlignCenter);
        auto* secondary = new QLabel(noCaptures
            ? "Press Capture or 'C' in Live."
            : "Try a different search or date range.");
        secondary->setObjectName("SubtleText");
        secondary->setAlignment(Qt::AlignCenter);
        emptyLayout->addWidget(icon);
        emptyLayout->addWidget(primary);
        emptyLayout->addWidget(secondary);
        empty->setMinimumHeight(320);
        contentLayout_->addWidget(empty, 1);
        return;
    }
```

Add the includes at the top of `ui/GalleryPage.cpp` (with the others from Step 3):
```cpp
#include "ui/IconLoader.h"
```

- [ ] **Step 6: Build**

Run:
```bash
cmake --build . --target XunChaWeiJian
```
Expected: builds cleanly. If `QToolButton` include removal causes an error elsewhere, restore it — but it was only used by the overflow button.

- [ ] **Step 7: Commit**

```bash
git add ui/LivePage.cpp ui/ControlPanel.cpp ui/GalleryPage.cpp ui/GalleryPage.h
git commit -m "feat(ui): gallery header/empty-state, live spacing rhythm"
```

---

## Task 6: Performance layer — ControlPanel table diff-and-skip

**Files:**
- Modify: `ui/ControlPanel.h` (add `lastSignature_` member)
- Modify: `ui/ControlPanel.cpp` (`setFrameResult`)

**Interfaces:**
- Consumes: nothing.
- Produces: `setFrameResult` skips the table rebuild when a computed signature is unchanged.

- [ ] **Step 1: Add the signature member**

In `ui/ControlPanel.h`, add the include at the top (after `#include <QElapsedTimer>`):
```cpp
#include <QString>
```
Add to the private section (after `QElapsedTimer tableUpdateTimer_;`):
```cpp
    QString lastSignature_;
```

- [ ] **Step 2: Implement diff-and-skip in `setFrameResult`**

In `ui/ControlPanel.cpp`, replace the body of `setFrameResult` (lines 145-182). The current body starts with the throttle gate and then rebuilds the table. Replace from `void ControlPanel::setFrameResult` through the closing brace with:

```cpp
void ControlPanel::setFrameResult(const DetectionFrameResult& result)
{
    if (tableUpdateTimer_.isValid() && tableUpdateTimer_.elapsed() < 200) {
        return;
    }
    if (tableUpdateTimer_.isValid()) {
        tableUpdateTimer_.restart();
    } else {
        tableUpdateTimer_.start();
    }

    // Build a cheap signature; skip the table rebuild if nothing changed.
    QString sig;
    sig.reserve(128);
    sig += "T"; sig += QString::number(result.toolDetections.size());
    sig += "G"; sig += QString::number(result.graspDetections.size());
    sig += "D"; sig += QString::number(result.defectResults.size());
    sig += "|";
    auto append = [&sig](const QString& mode, const QString& label, float score) {
        sig += mode + ":" + label + ":" + QString::number(score, 'f', 3) + ";";
    };
    for (const auto& d : result.toolDetections) {
        append("Tool", QString::fromStdString(d.label), d.score);
    }
    for (const auto& d : result.graspDetections) {
        append("Grasp", QString::fromStdString(d.label), d.score);
    }
    for (const auto& d : result.defectResults) {
        append("Defect", d.defective ? "Defective" : "Normal", d.defectScore);
    }
    if (sig == lastSignature_) {
        return;
    }
    lastSignature_ = sig;

    const int rows = static_cast<int>(result.toolDetections.size()
        + result.graspDetections.size()
        + result.defectResults.size());
    resultTable_->setRowCount(rows);

    int row = 0;
    auto addDetection = [this, &row](const QString& mode, const QString& label, float score) {
        resultTable_->setItem(row, 0, makeItem(mode));
        resultTable_->setItem(row, 1, makeItem(label));
        resultTable_->setItem(row, 2, makeItem(QString("%1%").arg(score * 100.0f, 0, 'f', 1)));
        ++row;
    };

    for (const auto& d : result.toolDetections) {
        addDetection("Tool", QString::fromStdString(d.label), d.score);
    }
    for (const auto& d : result.graspDetections) {
        addDetection("Grasp", QString::fromStdString(d.label), d.score);
    }
    for (const auto& d : result.defectResults) {
        const int defectRow = row;
        addDetection("Defect", d.defective ? "Defective" : "Normal", d.defectScore);
        if (d.defective) {
            colorRow(resultTable_, defectRow, QColor(ThemeManager::instance().tokens().warn));
        }
    }
}
```

- [ ] **Step 3: Build and run tests**

Run:
```bash
cmake --build . --target XunChaWeiJian && ctest --output-on-failure
```
Expected: builds cleanly; all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add ui/ControlPanel.h ui/ControlPanel.cpp
git commit -m "perf(ui): skip ControlPanel table rebuild when detections unchanged"
```

---

## Task 7: Performance layer — remove ThumbCard shadow effect

**Files:**
- Modify: `ui/ThumbCard.cpp` (delete `enterEvent`/`leaveEvent`)
- Modify: `assets/qss/base.qss` (hover background)

**Interfaces:**
- Consumes: Task 2's `QFrame#ThumbCard:hover` rule.
- Produces: `ThumbCard` no longer allocates a `QGraphicsDropShadowEffect` on hover; hover state is pure QSS.

- [ ] **Step 1: Add hover background to QSS**

In `assets/qss/base.qss`, find the `QFrame#ThumbCard:hover` rule (line 128):
```css
QFrame#ThumbCard:hover { border-color: {{accent}}; }
```
Change to:
```css
QFrame#ThumbCard:hover { border-color: {{accent}}; background: {{elevated}}; }
```

- [ ] **Step 2: Delete the effect overrides in `ThumbCard.cpp`**

In `ui/ThumbCard.cpp`, delete the `enterEvent` and `leaveEvent` methods (lines 64-78):
```cpp
void ThumbCard::enterEvent(QEnterEvent* event)
{
    QFrame::enterEvent(event);
    auto* effect = new QGraphicsDropShadowEffect(this);
    effect->setBlurRadius(18);
    effect->setOffset(0, 6);
    effect->setColor(QColor(0, 0, 0, 90));
    setGraphicsEffect(effect);
}

void ThumbCard::leaveEvent(QEvent* event)
{
    QFrame::leaveEvent(event);
    setGraphicsEffect(nullptr);
}
```

Remove the now-unused includes from `ui/ThumbCard.cpp`:
- `#include <QEnterEvent>` (only used by `enterEvent`)
- `#include <QGraphicsDropShadowEffect>` (only used by the effect)
- `#include <QMouseEvent>` — check first: it's used by `mousePressEvent` and `mouseDoubleClickEvent`, so KEEP it.

- [ ] **Step 3: Build**

Run:
```bash
cmake --build . --target XunChaWeiJian
```
Expected: builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add ui/ThumbCard.cpp assets/qss/base.qss
git commit -m "perf(ui): replace ThumbCard hover shadow with QSS hover state"
```

---

## Task 8: Final verification

**Files:**
- None (verification only).

- [ ] **Step 1: Full build and test**

Run from `build/`:
```bash
cmake --build . && ctest --output-on-failure
```
Expected: all targets build; `CaptureStoreSmoke`, `ThemeSmoke`, `ThumbnailCacheSmoke`, `AppSettingsSmoke` all PASS.

- [ ] **Step 2: Manual smoke (documented for the operator)**

Launch the app and verify, per layer:
- **Theme:** toggle dark/light (sidebar theme button); HUD chips, scrollbars, focus rings render correctly in both; no dark-on-light chip backgrounds.
- **Layout:** Live page Models card is flat (no border/accent bar); gallery header has no `...` button; empty gallery shows icon + two lines; count chip reflects record count.
- **Performance:** with a static camera scene, Live table does not rebuild (no flicker); gallery hover shows border + background highlight with no shadow; scrolling is smooth.

- [ ] **Step 3: No commit needed** — verification only. If any issue is found, fix in a new commit referencing the failing layer.
