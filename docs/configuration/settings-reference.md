# Settings Reference

Studio-level settings — persisted in `<data_root>/studio_settings.json`
(default `<repo>/data/studio_settings.json`). Per-project settings live
in the `.sim.json` document — see
[Project Schema](project-schema.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `theme` | str | `"default_dark"` | Theme id — see palettes below |
| `data_root` | str | `null` → `data/` | Root for recordings & datasets (write-probed on save; invalid paths keep the old root) |
| `window_size` | `[w, h]` | `[1280, 720]` | Startup window size |
| `ui_scale` | float | `1.0` | UI scale, clamped to `[0.75, 2.0]` |
| `recent_tracks` | str[] | `[]` | Most-recent-first, max 10 entries |

`StudioSettingsService.load()` tolerates a missing or malformed file and
falls back to defaults.

## Themes

| Id | Palette | Kind |
|---|---|---|
| `default_dark` | Midnight | Dark |
| `ocean_dark` | Ocean | Dark |
| `ember_dark` | Ember | Dark |
| `paper_light` | Paper | Light |
| `solar_light` | Solar | Light |

Selected on the home **SETTINGS** page. Applied instantly and persisted.

## Where each setting lives

| Setting | Where | Persistence |
|---|---|---|
| Theme, data root, UI scale | Home → **SETTINGS** page | `studio_settings.json` |
| Window size | `--width`/`--height` CLI flags | `window_size` key |
| Server port | `--port` CLI flag (default 8765) | not persisted |
| Headless / num-envs | `--headless` / `--num-envs` CLI flags | not persisted |
| Track defaults | `--track` CLI flag or home pick | `recent_tracks` |
| Sensor obs inclusion | Inspector SENSORS → **In Observations** | `.sim.json` → `image_channels` |
| Reward/termination toggles | Inspector RWD / TRM tabs | `.sim.json` → `agent` |
| Scenario | Inspector SCN tab | `.sim.json` → `scenario_def` |

## Data root layout

```
<data_root>/
├── recordings/           # episode recordings (● Record, TCP agent runs)
│   └── <name>/episode.json + manifest.json
├── datasets/             # exported datasets / imported zips
└── studio_settings.json
```

The DATASETS home page and DATA tab scan this root plus `experiments/`.

## See also

- [Studio Home](../user-guide/studio-home.md) — the SETTINGS page itself
- [Project Schema](project-schema.md) — per-project `.sim.json` fields
- [Troubleshooting](../troubleshooting/troubleshooting.md)
