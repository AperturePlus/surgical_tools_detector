# SGTDetector UI Modernization — Design

**Date:** 2026-05-12
**Status:** Approved (pre-implementation)
**Scope:** Restructure the Qt6 Widgets UI into a multi-page shell with a left
sidebar, redesign the live-detection page, replace the bottom gallery strip with
a dedicated gallery page, fix broken capture thumbnails, and add a
two-theme (dark / light) system.

---

## 1. Goals

1. **Fix capture thumbnails.** Today the gallery only shows the timestamp and
   detection counts because `QIcon(path)` silently fails when the Qt JPEG image
   plugin is not deployed. Captures must render as real thumbnails.
2. **Separate the gallery from live detection.** Today they share one window
   and feel cramped. Move the gallery to its own page and navigate between
   pages via a left sidebar.
3. **Modernize the visual language.** Replace the current ad-hoc QSS in
   `ui/AppStyle.cpp` with a tokenized theme system that supports a dark and a
   light theme, switchable at runtime and persisted in `QSettings`.
4. **Stay within Qt6 Widgets.** No QML migration; everything is `QWidget` +
   QSS so it stays consistent with the existing codebase and build.

## 2. Non-Goals

- Building a Settings page beyond a placeholder sidebar entry.
- Internationalization (English + Microsoft YaHei CJK fallback remains).
- Replacing the file-system + `index.json` capture store with a database.
- Implementing export / delete capture pipelines; the new
  `CaptureDetailDialog` exposes buttons but they remain stubs.
- Refactoring the detection engine, exporters, or capture store.

## 3. Architecture

### 3.1 Window structure

```
AppShell : QMainWindow                           (replaces ui/MainWindow)
├── Sidebar (QFrame, fixed 64 px wide)
│   ├── BrandMark
│   ├── NavButton "Live"      (camera glyph)
│   ├── NavButton "Gallery"   (image glyph)
│   ├── NavButton "Settings"  (gear glyph, disabled — Phase 2 placeholder)
│   ├── stretch
│   └── ThemeToggleButton     (sun / moon glyph)
└── ContentStack : QStackedWidget
    ├── LivePage    : QWidget
    └── GalleryPage : QWidget
```

- `Sidebar` uses a `QButtonGroup` in exclusive mode for nav buttons and emits
  `pageRequested(int index)`.
- `AppShell` owns the `cv::VideoCapture`, the `QTimer`, the `DetectionEngine`,
  and the `CaptureStore` (identical to today's `MainWindow`), so switching
  pages does **not** restart the camera.
- `AppShell` forwards each `DetectionFrameResult` to `LivePage` via a Qt
  signal `frameResultReady(const DetectionFrameResult&, uint8_t mask)`.
- `AppShell` forwards capture-saved events to `GalleryPage` via
  `captureSaved(const CaptureRecord&)` so the gallery refreshes incrementally
  rather than reloading every record on every snap.

### 3.2 Theme system

`ui/AppStyle.cpp` is replaced by:

- `ui/Theme.h` — declares a `ThemeTokens` struct:
  `bg, surface, elevated, border, textPrimary, textSecondary, accent,
   accentHover, info, warn, danger, shadow` (all `QString` hex / rgba).
- `ui/Theme.cpp` — exposes `Theme::dark()` and `Theme::light()` factories and
  `Theme::renderQss(const ThemeTokens&)` which loads a template file
  (`assets/qss/base.qss`, compiled into the binary as `:/qss/base.qss` via
  `assets/icons.qrc`) and substitutes `{{bg}}`, `{{accent}}`, … via
  `QString::replace`. No external templating engine.
- `ui/ThemeManager.{h,cpp}` — `QObject` singleton accessed via
  `ThemeManager::instance()`. Persists the chosen mode (`"dark"` / `"light"`)
  in `QSettings("SGT","Detector")`. Exposes:
  - `apply(QApplication* app)` — sets `qApp->setStyleSheet(...)`.
  - `toggle()` — flips mode, re-applies, emits `themeChanged(ThemeTokens)`.
- Widgets that draw outside QSS (notably `LivePreviewWidget`'s HUD overlay)
  subscribe to `themeChanged` and repaint.

**Dark palette (default):**
`bg #0E141B / surface #161D26 / elevated #1E2733 / border #243140 /
 textPrimary #E6EDF3 / textSecondary #8B98A8 / accent #14B8A6 /
 accentHover #2DD4BF / info #38BDF8 / warn #F59E0B / danger #EF4444`.

**Light palette:**
`bg #F4F6F9 / surface #FFFFFF / elevated #FFFFFF (shadow only) /
 border #E2E8F0 / textPrimary #0F172A / textSecondary #64748B /
 accent #0F766E / accentHover #0D9488 / info #2563EB / warn #D97706 /
 danger #DC2626`.

### 3.3 LivePage

```
LivePage : QWidget
└── QSplitter (Horizontal, non-collapsible)
    ├── VideoStage (QFrame, stretch 7)
    │   ├── LivePreviewWidget (back layer, fills the frame)
    │   └── HudOverlay (front layer, transparent QWidget)
    │       ├── top-left  : StatusChip × 2  (camera state, FPS)
    │       ├── top-right : ModePillBar     (Tool · Grasp · Defect)
    │       └── bottom-right : CaptureFAB   (circular primary button)
    └── ControlRail (QFrame, stretch 3)
        ├── Card "Thresholds"  — three sliders, values, reset link
        ├── Card "Live Data"   — QTableWidget (Mode | Label | Score)
        └── Card "Models"      — read-only model-name labels
```

- `HudOverlay` is a sibling `QWidget` with
  `setAttribute(Qt::WA_TransparentForMouseEvents, false)`, manually
  geometry-tracked to the `VideoStage` `resizeEvent`. Children use translucent
  background (`rgba(14,20,27,0.72)`), border-radius 12 px.
- `ModePillBar` and `ControlPanel`'s checkbox group are bound to the same
  bitmask source of truth (held by `ControlPanel`). The pill bar emits
  `toggleMode(uint8_t bit)`; `ControlPanel` applies it and re-emits
  `modeMaskChanged` as today.
- `ControlPanel` is restructured into the three cards above. The free-form
  `QPlainTextEdit` is replaced by a `QTableWidget` populated each frame with
  per-detection rows. Defect rows whose `defective==true` colour the row text
  with the `warn` token.
- The current top-level header (title, status chip, FPS chip, Capture button)
  in `MainWindow` is **removed**. Window title is the `QMainWindow` title;
  HUD owns status and capture; sidebar owns brand.

### 3.4 GalleryPage

```
GalleryPage : QWidget
├── HeaderBar
│   ├── "Captures · N" title
│   ├── QLineEdit (search by id / date)         [Phase 1: filter by substring]
│   ├── stretch
│   ├── QPushButton "Export all…"               [stub — disabled with tooltip]
│   └── QToolButton "⋮" (overflow)              [stub]
└── QScrollArea (vertical)
    └── content widget
        └── for each day group (sorted desc):
            ├── DateHeading "Today" / "2026-05-10"
            └── FlowLayout
                └── ThumbCard × records-of-the-day
```

- **`FlowLayout`** — based on the Qt examples FlowLayout (`QLayout` subclass)
  in `ui/FlowLayout.{h,cpp}`; reflows when the scroll area resizes.
- **`ThumbCard`** — `QFrame`, 192 × 158 px:
  - Top 192 × 120 thumbnail `QLabel`.
  - Bottom 38 px info strip: timestamp (HH:MM:SS) on the left, three badges
    (`T n`, `G n`, `D n`) on the right; badges with `n>0` for defect use
    `warn`; otherwise `textSecondary`.
  - `:hover` raises border to `accent`, applies a soft shadow via
    `QGraphicsDropShadowEffect` (cached).
  - Emits `activated(QString id)` on click; double-click also opens.
- **Empty state** — when `records.empty()`, gallery shows a centred icon +
  the line `"No captures yet — press Capture or 'C' in Live."`.
- **`CaptureDetailDialog`** — modal `QDialog`, ~80 % of window:
  - Left: large `QLabel` displaying `annotatedImagePath` (`KeepAspectRatio`,
    `SmoothTransformation`).
  - Bottom: filmstrip — `QHBoxLayout` of mini-thumbs for siblings; left/right
    arrow keys also navigate; Esc closes.
  - Right column: metadata (id, timestamp, mode mask via `modeText`,
    thresholds, counts), full JSON in a read-only `QPlainTextEdit`,
    `[Open folder]` button (`QDesktopServices::openUrl` on parent dir),
    `[Export…]` (stub, disabled).

### 3.5 Thumbnail loading (root-cause fix)

`ui/ThumbnailCache.{h,cpp}` — `QObject` singleton:

- API: `void requestThumbnail(const QString& id, const QString& path,
  std::function<void(QPixmap)> onReady, QSize size = {192,120})`.
- Internals:
  - LRU `QHash<QString, QPixmap>` keyed on `id+sizeHash`, capped at 256.
  - Cache miss → enqueue a job on a dedicated `QThread`-owned worker
    (`QThreadPool::globalInstance()` is fine; one worker is enough for
    file-system decode).
  - Worker uses `cv::imread(path, cv::IMREAD_COLOR)` and
    `cv::resize(...,INTER_AREA)`, then `matToPixmap` on the UI thread via
    `QMetaObject::invokeMethod(this, ..., Qt::QueuedConnection)`.
  - On UI thread the result is inserted into the cache and `onReady` is
    invoked.
- `ThumbCard` calls `requestThumbnail` in its constructor and again in
  `themeChanged` (placeholder background colour switches with the theme but
  the decoded pixmap is re-used).

This eliminates dependence on Qt's image-format plugins at render time;
OpenCV (already linked, including `imgcodecs` via `OpenCV_LIBS`) decodes the
JPG itself.

> **Note on existing `OpenCV` components:** `find_package(OpenCV REQUIRED
> COMPONENTS core imgproc videoio dnn)` does not include `imgcodecs`.
> The build currently works because `cv::imwrite` is used in
> `core/CaptureStore.cpp` and `imgcodecs` happens to be a transitive
> dependency of `videoio` on most builds — but we now depend on it
> explicitly. **CMake change:** add `imgcodecs` to the `COMPONENTS` list.

### 3.6 CMake & deployment changes

- Add `imgcodecs` to the `find_package(OpenCV ...)` component list.
- Append the new sources / headers to `SGT_SOURCES` / `SGT_HEADERS`:
  `AppShell, Sidebar, LivePage, GalleryPage, ThumbCard,
   CaptureDetailDialog, FlowLayout, ThumbnailCache, Theme, ThemeManager`.
- Remove `AppStyle.{h,cpp}` and `GalleryPanel.{h,cpp}` from the source
  lists; delete the files.
- Add a Qt resource: `qt_add_resources(SGTDetector "ui_assets" PREFIX "/"
  FILES assets/qss/base.qss assets/icons/*.svg)`.
- Flip the default of `SGT_RUN_WINDEPLOYQT` to `ON`.
- In the manual fallback branch (when `windeployqt6` isn't found), also copy
  `imageformats/{qjpeg.dll,qico.dll,qgif.dll,qsvg.dll}` and
  `styles/qmodernwindowsstyle.dll` (each via `copy_if_exists.cmake`, so a
  missing file is a warning rather than an error).

### 3.7 main.cpp

- Construct `QApplication` as today.
- Before showing the window:
  `ThemeManager::instance().apply(qApp);`
- Replace `MainWindow` instantiation with `AppShell`.

## 4. Files

### 4.1 New

| Path                                | Purpose                                    |
|-------------------------------------|--------------------------------------------|
| `ui/AppShell.{h,cpp}`               | Replaces `MainWindow`; sidebar + stack     |
| `ui/Sidebar.{h,cpp}`                | Left nav + theme toggle                    |
| `ui/LivePage.{h,cpp}`               | Live page container                        |
| `ui/HudOverlay.{h,cpp}`             | Translucent overlay on `LivePreviewWidget` |
| `ui/ModePillBar.{h,cpp}`            | Mode toggles in the HUD                    |
| `ui/StatusChip.{h,cpp}`             | Reusable pill (camera / FPS / etc.)        |
| `ui/CaptureFAB.{h,cpp}`             | Circular primary capture button            |
| `ui/Card.{h,cpp}`                   | Section card with accent left border       |
| `ui/GalleryPage.{h,cpp}`            | Replaces `GalleryPanel`                    |
| `ui/ThumbCard.{h,cpp}`              | One capture tile                           |
| `ui/CaptureDetailDialog.{h,cpp}`    | Modal capture detail                       |
| `ui/FlowLayout.{h,cpp}`             | Reflowing layout (Qt example)              |
| `ui/ThumbnailCache.{h,cpp}`         | Async OpenCV-backed thumb cache            |
| `ui/Theme.{h,cpp}`                  | Token + QSS templating                     |
| `ui/ThemeManager.{h,cpp}`           | Singleton, QSettings persistence           |
| `assets/qss/base.qss`               | Tokenized QSS template                     |
| `assets/icons/*.svg`                | Sidebar + HUD glyphs                       |
| `assets/icons.qrc`                  | Qt resource manifest                       |
| `tests/ThumbnailCacheSmoke.cpp`     | Smoke test for `cv::imread`-backed decode  |

### 4.2 Modified

| Path                       | Change                                                                  |
|----------------------------|-------------------------------------------------------------------------|
| `main.cpp`                 | Use `AppShell`, apply theme on startup                                  |
| `ui/ControlPanel.{h,cpp}`  | Card layout; remove mode checkbox group; replace text edit with table   |
| `ui/LivePreviewWidget.cpp` | Stop drawing on-canvas FPS/status (HUD owns it); honour theme tokens    |
| `CMakeLists.txt`           | Source list, OpenCV `imgcodecs`, `qt_add_resources`, deploy fallback    |

### 4.3 Deleted

- `ui/MainWindow.{h,cpp}` (logic moves to `AppShell`)
- `ui/GalleryPanel.{h,cpp}` (replaced by `GalleryPage`)
- `ui/AppStyle.{h,cpp}` (replaced by `Theme`)

## 5. Signal / slot map

```
DetectionEngine ── process ──> AppShell ── frameResultReady ──> LivePage
                                       ── captureSaved      ──> GalleryPage
LivePage::HudOverlay  ── toggleMode(bit)        ──> ControlPanel
LivePage::HudOverlay  ── captureRequested       ──> AppShell
ControlPanel          ── modeMaskChanged        ──> AppShell, HudOverlay
ControlPanel          ── thresholdsChanged      ──> AppShell → engine
Sidebar               ── pageRequested(int)     ──> ContentStack
Sidebar::ThemeToggle  ── clicked                ──> ThemeManager::toggle
ThemeManager          ── themeChanged(tokens)   ──> LivePreviewWidget,
                                                    ThumbCard placeholder
GalleryPage::ThumbCard ── activated(id)         ──> GalleryPage::openDetail
```

## 6. Build sequence (informational — full plan in writing-plans output)

1. Add `Theme` / `ThemeManager` + `base.qss` + resource manifest; verify the
   current `MainWindow` still launches with the new QSS.
2. Introduce `AppShell` + `Sidebar` + empty `LivePage` / `GalleryPage`;
   migrate `MainWindow`'s widgets verbatim into `LivePage` first.
3. Replace `GalleryPanel` with `GalleryPage` + `ThumbCard` + `FlowLayout`
   using a synchronous `cv::imread` decode (no cache yet).
4. Extract `ThumbnailCache`; wire async decode.
5. Add `CaptureDetailDialog`.
6. Build the HUD overlay; relocate Capture / status / FPS / mode pills.
7. Restructure `ControlPanel` into cards + table.
8. Deployment changes: `SGT_RUN_WINDEPLOYQT=ON`, fallback copies
   `imageformats/qjpeg.dll` etc.
9. Add `ThumbnailCacheSmoke` test.

## 7. Testing & acceptance

- `ctest` continues to pass (`CaptureStoreSmoke`).
- New `ThumbnailCacheSmoke`: loads a known JPG via the cache, asserts a
  non-null pixmap of the requested size.
- Manual smoke:
  - Launch → Live page shows video + HUD + control rail.
  - Press Capture (or `C`) → new `ThumbCard` appears on Gallery page.
  - Click the sidebar gallery icon → see thumbnails of pre-existing
    captures (the original bug).
  - Click a card → modal detail opens; ← / → navigates; Esc closes.
  - Toggle theme → entire UI switches; relaunch → choice persists.

## 8. Open questions

None at design time. Implementation may discover Qt-version specifics
(`QGraphicsDropShadowEffect` performance on some drivers; `FlowLayout`
behaviour inside `QScrollArea`) — these are tactical and resolved during
build.
