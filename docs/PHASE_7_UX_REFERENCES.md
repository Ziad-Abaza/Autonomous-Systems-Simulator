---
noteId: "7dda6aa0bf4911f1a29f1fbaabbd87c8"
tags: []

---

# Phase 7 — UX Reference Research

Proven interaction patterns from established 3D editors and simulation
tools, and how each is adapted to this application. We borrow *principles*,
not visuals. The product remains its own thing: a calm, dense, technical
simulation studio.

Constraints that shape adaptation:

- Immediate-mode pygame UI — no DOM/widgets; every pattern must be
  implementable with rects, text, and hit regions.
- Single-window desktop app, not a multi-pane dockable IDE.
- The domain is *environments as assets* (tracks + agent/RL config), not
  arbitrary scene graphs.

---

## Unity Editor

| Pattern | Why it works | Adaptation |
|---|---|---|
| **Hierarchy ⇄ Inspector ⇄ Scene triad** | Selection is one object model reflected in three places; users always know "what is selected" | We already have bidirectional selection (editor ⇄ inspector). Formalize it: a dedicated OUTLINE panel lists track points/entities/gates; clicking selects + focuses; selection state lives in the editor, both panels read it. |
| **Toolbar left / tools center / play right** | Spatially stable; muscle memory | Compact toolbar: tool cluster (Select/Move/Draw/Place) left, view commands (Frame All, Frame Selected, view toggles) center, Save + Simulate/Play right. |
| **F = Frame Selected, A/Shift+F = Frame All** | The single most-used navigation shortcut in any 3D tool | Implement `FrameSelected`/`FrameAll` as pure functions of the 2D editor transform (compute bounds → set offset+zoom). Same shortcuts. |
| **Project window as asset grid** | Assets (scenes/prefabs) are files; the library is just a browser over a directory | Track Library = grid browser over `presets/` + a user library dir of `*.sim.json`. Cards show name, description, version, modified time, thumbnail rendered from spline geometry. |
| **Inspector follows selection, collapses by category** | Progressive disclosure keeps a huge property space manageable | Keep PropertyRow model; group rows into collapsible sections; only show the section relevant to selection. |
| **Play mode visual distinction** | Edit vs. simulate must be unmistakable | Editor mode = 2D authoring canvas, simulation paused; Sim mode = 3D live. Clear mode color in the top bar. |

## Unreal Engine

| Pattern | Why it works | Adaptation |
|---|---|---|
| **Viewport overlay controls** (camera speed, view mode dropdown, "show" flags) | Keeps viewport uncluttered while exposing depth on demand | Editor VIEW menu: toggles for grid, curvature, direction, gates, width handles, collision footprints, LiDAR. Editor-only flags — never leak to observations. |
| **Outliner with folders + type icons + eye toggles** | Large scenes stay scannable | Outline groups: Control Points / Entities / Spawn / Checkpoints with visibility toggles and a filter box. |
| **Details panel with search + reset-to-default** | Fast access in a large property space | Property search field + per-row reset button. |
| **"Content Browser" = one place for all assets** | Assets shouldn't be reached through unrelated screens | Home screen browses tracks, datasets, and experiments as first-class assets with a uniform card/list treatment. |

## Blender

| Pattern | Why it works | Adaptation |
|---|---|---|
| **Mode-based interaction** (Object/Edit mode) | Same viewport, different verbs; prevents accidental edits | Editor has explicit *tools* (Select vs. Draw). Select mode never mutates geometry — fixes the "stray click inserts a point" defect. |
| **Status bar with coordinates + tool hints** | The bottom strip always answers "what mode am I in, what do clicks do now" | Bottom bar becomes contextual: mode + active tool + cursor world coords + zoom + relevant shortcut hints. |
| **Numeric transform readout near viewport** | Precise edits without opening a panel | Selected point/entity shows position in the status bar and in the inspector with units. |
| **N-panel (properties) / T-panel (tools)** | Tools on the left, properties on the right | Our layout: tools+hierarchy left rail (collapsible), inspector right rail. Matches both Blender and the existing inspector-left layout — we move it right to match convention? No: keep inspector docked **right** (Unity/Unreal convention) and hierarchy **left**. |

## Godot

| Pattern | Why it works | Adaptation |
|---|---|---|
| **Scene dock = tree of nodes** | Simple flat learning curve | The outline is a plain grouped tree, not a full scene graph — same visual grammar (indent, icon, name). |
| **"Editable children" + focused inspector** | You only ever edit the thing you selected | Inspector shows only the selected object's properties; unselected state shows environment summary + project actions. |
| **Bottom panels for secondary output** | Logs/validation out of the way but reachable | Validation issues surface as a bottom dock (or inspector VALIDATE section) with click-to-navigate to the offending setting. |

## CARLA / simulation tooling

| Pattern | Why it works | Adaptation |
|---|---|---|
| **Scenario/config travels with the map** | A scenario references its map; you never configure against nothing | Track assets carry their full EnvironmentProject (already true in `*.sim.json`); creating an experiment from a track carries the env fingerprint automatically. |
| **Recorder produces self-describing packages** | Replays/datasets must be portable and inspectable | Recording flow produces a directory: episode file + manifest (already nearly true — we add user-chosen destination + summary). Datasets keep `transitions_v1` (manifest.json + episodes.jsonl). |
| **Debug render toggles separate from sensor ground truth** | Visualization aids ≠ agent input | Editor visualization flags are strictly view-layer; documented boundary, never fed to observations (§38 of the phase spec). |

## BeamNG / racing-tool workflows

| Pattern | Why it works | Adaptation |
|---|---|---|
| **Track-as-first-class-asset** | In driving tools the track IS the document | `*.sim.json` is the document type; the studio's primary noun is "Track/Environment". Library, recents, duplicate, rename, delete all operate on it. |
| **World editor with snap/grid/align** | Track layout needs precision | Grid toggle exists; add snap-to-grid for control points/entities (0.5/1 m steps) as an editor option. |

## Cross-cutting principles adopted

1. **Selection is app-global** — one selection model; viewport, outline,
   inspector all reflect it. (Unity)
2. **Viewport dominance** — panels are collapsible; the canvas always
   wins space. (Blender/Unreal)
3. **Frame before you edit** — opening a track auto-frames it; F/A
   shortcuts recover orientation. (Unity)
4. **Mode color language** — edit vs. simulate vs. replay are visually
   distinct states, not just tabs. (Unreal "PIE" banner)
5. **Progressive disclosure** — basic properties visible, advanced behind
   collapsed sections; validation summarizes and deep-links. (Godot/Unity)
6. **Assets are files** — library = browser over real directories;
   no second persistence system. (Unity Project window)
7. **Status, not mystery** — bottom bar always shows mode, tool,
   coordinates, save state. (Blender)
8. **Dangerous actions confirm** — delete/quit-overwrite get a small
   confirmation, not a workflow-blocking wizard. (all)
