# SGTDetector UI Optimization — Design

**Date:** 2026-07-12
**Status:** Approved (pre-implementation)
**Scope:** Refine the existing Qt6 Widgets UI across three independent layers —
theme consistency, layout/information hierarchy, and rendering performance —
without restructuring the architecture. No new pages, no signal/slot rewiring,
no QML migration.

---

## 1. Goals

1. **Theme consistency.** HUD chips (`StatusChip`, `ModePill`) currently hardcode
   a dark `rgba(14,20,27,0.72)` background that breaks under the light theme.
   Extend the token system so every themed surface reads from a token.
2. **Visual refinement.** Tighten spacing rhythm, add missing primitives
   (scrollbars, focus rings, card hierarchy, empty states) so the UI reads as
   intentional rather than templated — while staying within the current teal /
   dark language.
3. **Rendering performance.** Eliminate two known hotspots: the per-frame
   `QTableWidget` rebuild in `ControlPanel`, and the per-hover
   `QGraphicsDropShadowEffect` allocation in `ThumbCard`.

## 2. Non-Goals

- Restructuring page classes, splitting files, or adding new widget
  hierarchies. Layout changes touch margins, spacing, and object names only.
- Changing the detection engine, capture store, or thumbnail cache threading
  model.
- Introducing virtualized `FlowLayout` or any layout-virtualization work.
- New features (export pipeline, delete, etc.). The disabled Export button is
  restyled but stays disabled.
- Internationalization changes.

## 3. Approach

Three independent layers, each a standalone commit that can be verified and
rolled back on its own:

1. **Theme layer** — token additions + QSS hardcode fixes + new primitives.
2. **Layout layer** — card variants, HUD grouping, gallery header, empty state,
   spacing rhythm.
3. **Performance layer** — table diff-and-skip, drop the shadow effect.

Build order is theme → layout → performance, because layout changes reference
the new tokens, and the performance layer touches the same `ControlPanel` /
`ThumbCard` files as layout, so doing it last avoids rebase churn.

## 4. Layer 1 — Theme

### 4.1 New `ThemeTokens` fields

Add to `ThemeTokens` in [ui/Theme.h](ui/Theme.h):

| Field         | Dark                       | Light                        | Reuse            |
|---------------|----------------------------|------------------------------|------------------|
| `chipBg`      | `rgba(14,20,27,0.72)`      | `rgba(255,255,255,0.82)`     | new              |
| `chipBorder`  | `{{border}}` value         | `{{border}}` value           | alias of border  |
| `focusRing`   | `#38BDF8`                  | `#2563EB`                    | alias of info   |
| `scrollBar`   | `#2A3441`                  | `#CBD5E1`                    | new              |
| `scrollBarHover` | `#3A4654`              | `#94A3B8`                    | new              |

`chipBorder` and `focusRing` are named separately from `border`/`info` so they
can diverge later without a search-and-replace; their initial values mirror the
existing tokens.

### 4.2 `renderQss` substitution

[ui/Theme.cpp](ui/Theme.cpp) `renderQss` gains `s.replace("{{chipBg}}", t.chipBg)`
and the four parallel lines for the new fields.

### 4.3 Hardcode fixes in `base.qss`

- [assets/qss/base.qss:52](assets/qss/base.qss#L52) `QFrame#StatusChip`
  `background: rgba(14,20,27,0.72)` → `background: {{chipBg}}`
- [assets/qss/base.qss:66](assets/qss/base.qss#L66) `QPushButton#ModePill`
  `background: rgba(14,20,27,0.72)` → `background: {{chipBg}}`

### 4.4 New QSS primitives

Append to `base.qss`:

- **Scrollbars** — `QScrollBar:vertical` / `:horizontal`: width/height 10px,
  groove transparent, handle `{{scrollBar}}` with 4px radius, hover
  `{{scrollBarHover}}`, no buttons (`add-line`/`sub-line` height 0).
- **Focus ring** — `QLineEdit:focus, QPushButton:focus, QToolButton:focus,
  QSlider:focus { border: 1px solid {{focusRing}}; }`. Applied to interactive
  controls only, not containers.
- **Secondary button** — `QPushButton#Secondary { background: transparent;
  border: 1px solid {{border}}; color: {{textSecondary}}; }` with
  `:hover { background: {{surface}}; color: {{textPrimary}}; }`. Used by the
  gallery Export button.

### 4.5 Page margin unification

- [ui/GalleryPage.cpp:29](ui/GalleryPage.cpp#L29) and
  [ui/SettingsPage.cpp:37](ui/SettingsPage.cpp#L37): change `22, 18, 22, 18` →
  `20, 18, 20, 18` to match [ui/LivePage.cpp:21](ui/LivePage.cpp#L21).

## 5. Layer 2 — Layout

### 5.1 Card variants

[ui/Card.cpp](ui/Card.cpp) gains a constructor flag or a second constructor
`Card(title, Variant variant, parent)` where `Variant { Standard, Flat }`.
- `Standard` (default): current `QFrame#Card` styling, plus a 3px left accent
  bar via a child `QFrame#CardAccent` (or a `border-left` in QSS — QSS
  `border-left` is simpler and avoids a child widget; use QSS).
- `Flat`: object name `CardFlat`, QSS `background: transparent; border: none;`
  with only the title and body, used for read-only groupings.

QSS additions:
```
QFrame#Card { border-left: 3px solid {{accent}}; }
QFrame#CardFlat { background: transparent; border: none; }
QFrame#CardFlat QLabel#CardTitle { color: {{textSecondary}}; }
```

`ControlPanel` changes the "Models" card from `Card` to `CardFlat`.

### 5.2 HUD chip grouping and semantic dots

[HudOverlay.cpp](ui/HudOverlay.cpp) top row stays one `QHBoxLayout` but the two
logical groups (camera+fps on the left, mode pills on the right) are visually
self-contained by the chip background tokens fixed in Layer 1.

`StatusChip` gains `void setDotColor(const QString& tokenOrColor)`:
- [ui/StatusChip.cpp](ui/StatusChip.cpp) stores the dot `QFrame` as a member and
  applies a stylesheet `background: <color>`.

`HudOverlay`/`LivePage` wiring:
- Camera online → `accent`; camera unavailable → `danger`.
- FPS chip dot → `info` (static, set once in the `HudOverlay` constructor).

The camera-status → dot-color derivation lives in **one place**:
`HudOverlay::setCameraStatus` (which both `LivePage::setCameraStatus` and
`AppShell`'s camera-status calls funnel through). It inspects the status text —
if it contains `"unavailable"`, set the dot to `danger`; otherwise `accent`.
`LivePage` and `AppShell` call sites are unchanged.

### 5.3 Gallery header cleanup

[ui/GalleryPage.cpp:32-53](ui/GalleryPage.cpp#L32):
- Remove the disabled `overflow` `QToolButton` ("...") — zero information value.
- `exportButton` → `setObjectName("Secondary")`.
- Replace title `"Captures - N"` with a `QLabel#AppTitle` "Captures" plus a
  small count chip (`QFrame#StatusChip`-style, read-only) showing `N` on the
  right of the title row.

### 5.4 Empty state

[ui/GalleryPage.cpp:118-127](ui/GalleryPage.cpp#L118): replace the single
`QLabel` with a vertical container — `nav-gallery` SVG (via `IconLoader`) at
32px in `textSecondary`, a primary line, and a secondary hint line, all
`Qt::AlignCenter`, minimum height 320 retained.

### 5.5 Live page spacing rhythm

- [ui/LivePage.cpp:26](ui/LivePage.cpp#L26) `splitter->setHandleWidth(12)` → `8`.
- [ui/ControlPanel.cpp:62](ui/ControlPanel.cpp#L62) `layout->setSpacing(12)` →
  `10`.

## 6. Layer 3 — Performance

### 6.1 ControlPanel table: diff-and-skip

[ui/ControlPanel.cpp:145](ui/ControlPanel.cpp#L145) `setFrameResult`:

- Keep the existing 200ms `tableUpdateTimer_` throttle.
- After the throttle gate, compute a cheap signature:
  `QString sig = "T" + counts + "|" + joined("label:score:mode" per row)`.
  Store `QString lastSignature_` as a member.
- If `sig == lastSignature_`, return without touching the table.
- Otherwise rebuild as today and update `lastSignature_`.

This makes the static-scene cost zero (one string build + compare per throttled
tick) and only rebuilds when a detection label/score/count actually changes.

### 6.2 ThumbCard: drop the shadow effect

[ui/ThumbCard.cpp:64-78](ui/ThumbCard.cpp#L64):
- Delete `enterEvent` and `leaveEvent` overrides (and the
  `QGraphicsDropShadowEffect` include).
- QSS already has `QFrame#ThumbCard:hover { border-color: {{accent}}; }`; add
  `background: {{elevated}};` to the `:hover` rule so the hover state still
  reads as a lift without a graphics effect.

### 6.3 LivePreviewWidget — verified, no change

[ui/LivePreviewWidget.cpp:57](ui/LivePreviewWidget.cpp#L57) `updatePixmap` uses
`FastTransformation` and is only called from `setResult` (per frame) and
`resizeEvent` (on resize). No redundant calls. Documented as
"verified, no change" to prevent well-meaning future edits.

## 7. Files

### 7.1 Modified

| Path                        | Change                                                        |
|-----------------------------|---------------------------------------------------------------|
| `ui/Theme.h`                | Add 5 token fields                                            |
| `ui/Theme.cpp`              | Populate tokens for dark/light; add `renderQss` replacements |
| `assets/qss/base.qss`       | Fix 2 hardcodes; add scrollbar, focus, secondary, card rules |
| `ui/Card.h` / `Card.cpp`    | Add `Flat` variant                                            |
| `ui/ControlPanel.cpp`       | "Models" → `CardFlat`; table diff-and-skip; spacing 10        |
| `ui/ControlPanel.h`         | Add `lastSignature_` member                                   |
| `ui/StatusChip.h` / `.cpp`  | Add `setDotColor`, store dot member                           |
| `ui/HudOverlay.cpp`         | Derive camera dot color from status; set FPS dot to info      |
| `ui/LivePage.cpp`           | handleWidth 8 (dot-color wiring stays in HudOverlay)          |
| `ui/GalleryPage.cpp`        | Remove overflow btn; secondary Export; count chip; empty state; margins |
| `ui/ThumbCard.cpp`          | Delete enter/leave effect overrides                           |

### 7.2 New

None. All changes are in existing files. (The empty-state icon reuses the
existing `nav-gallery` alias via `IconLoader`.)

### 7.3 Deleted

None.

## 8. Build sequence

1. **Theme layer** — tokens, `renderQss`, `base.qss` hardcodes + new primitives,
   page margins. Build, launch, toggle theme, confirm chips/scrollbars/focus
   look right in both themes.
2. **Layout layer** — `Card` variant, `ControlPanel` Models card, `StatusChip`
   dot color + wiring, gallery header, empty state, Live spacing. Build,
   launch, walk all three pages.
3. **Performance layer** — `ControlPanel` signature skip, `ThumbCard` effect
   removal. Build, launch, confirm Live table updates only on change and
   gallery hover/scroll is smooth.

Each layer is a separate commit.

## 9. Testing & acceptance

- `ctest` continues to pass (`CaptureStoreSmoke`, `ThumbnailCacheSmoke`).
- Manual smoke, per layer:
  - **Theme:** toggle dark/light; HUD chips, scrollbars, focus rings render
    correctly in both; no dark-on-light chip backgrounds.
  - **Layout:** Live page card hierarchy reads (Models is flat); gallery
    header has no `...` button; empty gallery shows icon + two lines; count
    chip reflects record count.
  - **Performance:** with a static camera scene, Live table does not rebuild
    (no flicker, CPU drops vs. before); gallery hover shows border highlight
    with no shadow allocation; scrolling a large gallery is smooth.
- No regressions: capture flow (`C` / Capture button) still writes a record
  and adds a `ThumbCard`; theme choice still persists across relaunch.

## 10. Open questions

None at design time. The `Card` variant is implemented as a constructor
argument rather than a subclass to keep the change minimal; if the QSS
`border-left` accent bar proves visually heavy it can be dropped to 2px or
removed during implementation without affecting the rest of the design.
