# SGTDetector Settings Page + Gallery Filter — Design

**Date:** 2026-05-13
**Status:** Approved (pre-implementation)
**Scope:** Replace the disabled Settings sidebar entry with a working Settings
page that persists user preferences via `QSettings`. Add a date-range filter
bar to the Gallery page. Wire `AppShell` / `main.cpp` to read the persisted
preferences on startup.

---

## 1. Goals

1. **Make the Settings sidebar entry functional.** Today it is a disabled
   placeholder (`Sidebar.cpp:36 settings->setEnabled(false)`). Users perceive
   it as a broken button.
2. **Persist user preferences across launches.** Currently only the theme
   choice is persisted (in `ThemeManager`). Camera index, default detection
   thresholds, default mode mask, and capture output directory reset every
   launch.
3. **Let users narrow the Gallery by date.** The existing search box filters
   by id/timestamp substring, which is awkward for "show me last week's
   captures". Add an explicit date filter bar with preset chips and an
   optional custom range.

## 2. Non-Goals

- **Live application of settings.** All Settings changes take effect on the
  next launch (the chosen "minimum Settings page" option). A future iteration
  can add hot-reload for camera id / thresholds.
- **Filtering by detection mode, defect status, or count.** Out of scope for
  this iteration; the chip bar leaves room for those columns later if needed.
- **Importing / exporting capture sets.** The "Export all..." button on the
  Gallery header remains a stub.
- **Per-camera multi-profile settings.** A single saved profile.
- **Internationalization.** English + Microsoft YaHei CJK fallback remains.

## 3. Architecture

### 3.1 Module overview

```
AppShell  (existing)
├── reads AppSettings on startup → seeds AppOptions
├── owns Sidebar (3 nav buttons all enabled)
└── owns QStackedWidget { LivePage, GalleryPage, SettingsPage }
                                                       (NEW)
                            └── owns GalleryFilterBar
                                (NEW)

core/AppSettings.{h,cpp}        ← QSettings facade            (NEW)
ui/SettingsPage.{h,cpp}         ← persisted-prefs UI          (NEW)
ui/GalleryFilterBar.{h,cpp}     ← chip bar + QDateEdits       (NEW)
ui/Sidebar.cpp                  ← drop setEnabled(false)
ui/GalleryPage.{h,cpp}          ← integrate filter bar
core/AppOptions  (in AppShell.h)← add `thresholds` field
main.cpp                        ← read AppSettings → AppOptions
tests/AppSettingsSmoke.cpp      ← QSettings round-trip test   (NEW)
```

### 3.2 `core/AppSettings`

A thin façade over `QSettings("SGT","Detector")`. Centralizes the key strings
and types so callers don't sprinkle `QSettings` calls across the codebase.

```cpp
namespace sgt {

class AppSettings {
public:
    AppSettings();                              // uses default org/app
    explicit AppSettings(QSettings::Format fmt, // for tests
                         const QString& path);

    int cameraId() const;
    void setCameraId(int id);

    uint8_t modeMask() const;
    void setModeMask(uint8_t mask);

    DetectionThresholds defaultThresholds() const;
    void setDefaultThresholds(const DetectionThresholds& t);

    QString captureDir() const;                 // empty if unset
    void setCaptureDir(const QString& dir);

    // Reset only the "DETECTION DEFAULTS" group (Reset to factory in UI).
    void resetDefaults();

private:
    std::unique_ptr<QSettings> settings_;
};

} // namespace sgt
```

Key map:

| Key                              | Type   | Factory default | Read by               |
|----------------------------------|--------|-----------------|-----------------------|
| `ui/theme`                       | string | `"dark"`        | ThemeManager (existing) |
| `app/cameraId`                   | int    | 0               | `main.cpp` startup    |
| `app/captureDir`                 | string | `""` → exeDir/captures | `main.cpp` startup |
| `defaults/modeMask`              | int    | `MODE_TOOL`     | `main.cpp` startup    |
| `defaults/threshold/tool`        | double | 0.6             | `main.cpp` startup    |
| `defaults/threshold/grasp`       | double | 0.55            | `main.cpp` startup    |
| `defaults/threshold/defect`      | double | 0.6             | `main.cpp` startup    |

`resetDefaults()` removes keys under `defaults/` (model defaults take over on
next read).

### 3.3 `ui/SettingsPage`

A `QWidget` page added to the existing `QStackedWidget` after `LivePage` and
`GalleryPage` (index 2 — same slot the placeholder occupied).

Layout (reuses `ui/Card`):

```
Settings                                              <AppTitle>
Preferences are saved automatically. Some changes apply on next launch.

┌── CAPTURE SOURCE ──────────────────────────────────────────┐
│  Camera index            [SpinBox 0..15]                   │
│  Applies on next launch.                                   │
└────────────────────────────────────────────────────────────┘
┌── DETECTION DEFAULTS ──────────────────────────────────────┐
│  Tool threshold     ▬▬▬●▬▬▬▬▬   60%                        │
│  Grasp threshold    ▬▬▬●▬▬▬▬▬   55%                        │
│  Defect threshold   ▬▬▬●▬▬▬▬▬   60%                        │
│  Active modes       [✓] Tool  [✓] Grasp  [✓] Defect        │
│  Applies on next launch.            [Reset to factory]     │
└────────────────────────────────────────────────────────────┘
┌── CAPTURE STORAGE ─────────────────────────────────────────┐
│  Output folder      [<path>           ] [Browse...]        │
│  Applies on next launch.                                   │
└────────────────────────────────────────────────────────────┘
```

Persistence semantics:
- Each control writes to `AppSettings` on `valueChanged` /
  `editingFinished` / `toggled` — no Save button.
- `Reset to factory` calls `AppSettings::resetDefaults()`, then re-reads and
  populates the threshold sliders + mode checkboxes. Camera id and capture
  dir are not affected.
- All three mode checkboxes turning off rolls back to Tool (matches existing
  `ControlPanel::toggleBit` invariant of "never empty mask"). The UI restores
  the Tool checkbox via `QSignalBlocker` so we don't re-emit.
- `Browse...` opens `QFileDialog::getExistingDirectory`. The selected path is
  shown in a read-only `QLineEdit` (paths can be long; users edit by
  re-browsing rather than typing).

### 3.4 `ui/GalleryFilterBar`

A `QFrame` placed below the Gallery header, above the scroll area.

```
[All]  [Today]  [Last 7]  [Last 30]  [Custom...]
                                          ↓ checked, expands inline:
                                      From [2026-04-13] To [2026-05-13]
```

- Five `QPushButton`s with `objectName == "FilterChip"`, in an exclusive
  `QButtonGroup`. Default checked: **All**.
- The two `QDateEdit`s live in the same row, hidden unless `Custom...` is
  checked. They use `QDate::currentDate()` as `to` and `currentDate().addDays(-30)`
  as `from` by default; both ranges are inclusive.
- Signal: `rangeChanged(QDate from, QDate to)`. `All` emits a pair of
  invalid dates → "no filter". `Today` emits `(today, today)`. `Last N` emits
  `(today - (N-1), today)`. `Custom` follows the DateEdits.
- The QSS class `QPushButton#FilterChip` mirrors `#ModePill` (rounded 14px,
  accent on checked). Defined in `assets/qss/base.qss`.

### 3.5 `ui/GalleryPage` integration

- The constructor instantiates a `GalleryFilterBar` and inserts it between
  the header layout and the scroll area.
- `GalleryPage` holds the current `(QDate from, QDate to)` (both invalid →
  no filter).
- `matchesFilter(record)` combines (a) the existing search-text filter and
  (b) the date range:

```cpp
bool GalleryPage::matchesFilter(const CaptureRecord& r) const {
    if (!matchesSearch(r))   return false;
    if (!matchesDateRange(r)) return false;
    return true;
}
```

`matchesDateRange` parses `r.timestamp.substr(0,10)` to `QDate`. If either
endpoint is invalid the range is open on that side; if both are invalid no
date filter applies.

- When `rangeChanged` fires, `GalleryPage::rebuild()` is called (the same
  method already triggered by `searchEdit_->textChanged`).
- Filter state is **not persisted** across sessions — galleries are
  session-fresh.

### 3.6 Sidebar wiring

`ui/Sidebar.cpp`:
- Remove the `settings->setEnabled(false)` line.
- Remove the "(coming soon)" hint from the tooltip; just `"Settings"`.
- The existing `QToolButton#NavButton:checked` QSS already lights up the
  accent border for the Settings entry — no QSS changes needed.

### 3.7 `core/AppOptions` and `main.cpp`

`AppOptions` (in `ui/AppShell.h`) gains:

```cpp
struct AppOptions {
    int cameraId = 0;
    bool cameraIdFromCli = false;          // NEW — tells main.cpp not to overwrite
    uint8_t modeMask = MODE_TOOL;
    DetectionThresholds thresholds;         // NEW — seeded from QSettings
    std::string toolModel;
    std::string graspModel;
    std::string defectModel;
};
```

`main.cpp` flow:

```cpp
AppSettings settings;
AppOptions opts = parseArgs(argc, argv);

if (!opts.cameraIdFromCli) opts.cameraId = settings.cameraId();
opts.modeMask    = settings.modeMask();
opts.thresholds  = settings.defaultThresholds();

QString captureDirPref = settings.captureDir();
fs::path captureDir = captureDirPref.isEmpty()
    ? (exeDir / "captures")
    : fs::path(captureDirPref.toStdString());

auto store = std::make_unique<sgt::CaptureStore>(captureDir);
```

`parseArgs` sets `opts.cameraIdFromCli = true` whenever the bare-numeric
camera arg is supplied, so a CLI camera id wins over the persisted setting.
Other CLI args (`--mode`, `--*-model`) still override `AppSettings`, but only
mode is currently persisted; models stay CLI-only for now.

`AppShell` constructor already uses `opts.modeMask`. It must additionally
seed `thresholds_` from `opts.thresholds`, then `engine_->setThresholds(thresholds_)`
before the first frame.

### 3.8 QSS additions

`assets/qss/base.qss` gains one rule, mirroring `#ModePill`:

```css
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
```

## 4. Files

### 4.1 New

| Path                                | Purpose                                  |
|-------------------------------------|------------------------------------------|
| `core/AppSettings.h`                | QSettings façade declaration             |
| `core/AppSettings.cpp`              | Implementation                           |
| `ui/SettingsPage.h`                 | Settings page widget                     |
| `ui/SettingsPage.cpp`               |                                          |
| `ui/GalleryFilterBar.h`             | Filter chip bar + custom QDateEdits      |
| `ui/GalleryFilterBar.cpp`           |                                          |
| `tests/AppSettingsSmoke.cpp`        | QSettings round-trip test                |

### 4.2 Modified

| Path                       | Change                                                      |
|----------------------------|-------------------------------------------------------------|
| `ui/AppShell.h`            | Add `thresholds`, `cameraIdFromCli` to `AppOptions`         |
| `ui/AppShell.cpp`          | Add SettingsPage to stack, seed `thresholds_` from opts     |
| `ui/Sidebar.cpp`           | Drop `setEnabled(false)`, clean tooltip                     |
| `ui/GalleryPage.h`         | Hold date-range filter, integrate `GalleryFilterBar`        |
| `ui/GalleryPage.cpp`       |                                                             |
| `assets/qss/base.qss`      | Add `QPushButton#FilterChip` and `QDateEdit` rules          |
| `main.cpp`                 | Read `AppSettings` into `AppOptions`; set `cameraIdFromCli` |
| `CMakeLists.txt`           | Register new sources/headers and `AppSettingsSmoke`         |

### 4.3 Deleted

None.

## 5. Signal / slot map

```
SettingsPage controls          ──valueChanged──> AppSettings (write)
                                                 (no UI feedback needed)
SettingsPage "Reset to factory"──clicked──> AppSettings::resetDefaults
                                            then SettingsPage repopulates

GalleryFilterBar chips         ──QButtonGroup::idClicked──> resolveRange
GalleryFilterBar QDateEdits    ──dateChanged──> resolveRange (only Custom)
GalleryFilterBar               ──rangeChanged(from,to)──> GalleryPage::rebuild
GalleryPage searchEdit         ──textChanged──> GalleryPage::rebuild  (existing)

Sidebar Settings button        ──clicked──> AppShell stack setCurrentIndex(2)
                                            (already wired via pageRequested)
```

## 6. Build sequence (informational — detailed steps in writing-plans output)

1. Add `core/AppSettings.{h,cpp}` + smoke test; verify keys round-trip.
2. Add `ui/SettingsPage.{h,cpp}` wired to AppSettings; show in stack;
   functional but Sidebar still disabled.
3. Drop `settings->setEnabled(false)` in Sidebar; manual smoke that the
   Settings page is reachable.
4. Add `ui/GalleryFilterBar.{h,cpp}`; insert into `GalleryPage`; wire
   `rangeChanged`; manual smoke filters cards by date.
5. Extend `AppOptions` (`thresholds`, `cameraIdFromCli`); update `main.cpp`
   and `AppShell` to honor persisted values.
6. Add QSS rules for `FilterChip` and `QDateEdit`.

## 7. Testing & acceptance

**Automated:**
- `AppSettingsSmoke`: writes camera id, thresholds, mode mask, capture dir
  into a temporary INI file; reconstructs `AppSettings(IniFormat, path)`;
  asserts all values read back. Calls `resetDefaults()`, asserts only
  `defaults/*` keys are gone, `app/*` survive.

**Manual:**
- Sidebar Settings icon is clickable; switching pages shows Settings.
- Change tool threshold slider in Settings → close app → relaunch → Live
  page's Tool threshold slider starts at the saved value.
- Change camera index in Settings → relaunch with no CLI args → app opens
  that camera index. Launch with `SGTDetector.exe 2` → that overrides the
  persisted index for this run, persisted setting unchanged.
- Reset to factory only resets the defaults card.
- Gallery: clicking "Today" hides all cards except today's; "Last 7" shows
  the past week; "Custom..." reveals the date pickers; "All" returns to
  full list.
- Search text and date filter compose (e.g. typing into search while
  "Last 7" is active narrows further).
- Theme toggle still works; nothing in Settings page is unstyled.

## 8. Risks

- **QSettings org/app collision with ThemeManager.** Both use
  `QSettings("SGT","Detector")`. Sharing the same registry/INI is fine; the
  key prefixes (`ui/theme` vs `app/*` vs `defaults/*`) don't collide.
- **Capture dir change without restart.** Users may expect the gallery to
  re-scan the new directory immediately. We document "applies on next
  launch"; this matches the user-approved minimum.
- **DateEdit locale.** `QDateEdit` defaults to the system locale. We force
  ISO format (`yyyy-MM-dd`) on both `from` and `to` to avoid ambiguous
  parsing across locales.

## 9. Open questions

None at design time.
