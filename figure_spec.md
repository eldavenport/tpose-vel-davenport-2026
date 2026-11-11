# Figure Style Specification — TPOSE-Vel Manuscript

This file defines the visual conventions for all figures in this project.
Claude should check each rendered PNG against these specs before declaring a figure done.

---

## Global Defaults

| Parameter | Value |
|---|---|
| `plt.rcParams['font.size']` | 14 |
| `savefig dpi` | 150 |
| `savefig bbox_inches` | `'tight'` |
| Grid alpha | 0.3 |
| Month boundary lines | gray dashed, `lw=0.8`, `alpha=0.5` |
| Month label text | gray, `fontsize=11`, `va='top'`, at 97% of y-axis max |

**Do not change without asking:** colorbar limits (vmin/vmax), normalization factors, cost multipliers, depth clip values.

---

## Color & Linestyle Conventions

| Dataset | Color | Solid | Dashed | Dotted |
|---|---|---|---|---|
| TPOSE-noVel | `C0` (blue) | State estimate | Forecast | Persistence |
| TPOSE-Vel | `C1` (orange) | State estimate | Forecast | Persistence |
| GLORYS | `C2` (green) | — | — | — |
| HYCOM | `C3` (red) | — | — | — |
| TAO climatology σ | black | `lw=1.5`, dashed | — | — |
| TAO climatology RMSE | black | `lw=1.5`, dashed | — | — |

Line weight: `lw=2` for all model/obs time series.

---

## Colormaps

| Data type | Colormap | Typical range |
|---|---|---|
| Zonal velocity (U) | `RdBu_r` | −1.232 to 1.232 m/s |
| Meridional velocity (V) | `RdBu_r` | −0.4107 to 0.4107 m/s |
| Velocity difference | `RdBu_r` | symmetric, ±10 dB or ±0.5 m/s |
| PSD (TPOSE-Vel / noVel) | `viridis` | 10⁻⁴ to 10⁻² m²s⁻²/cpd |
| PSD difference | `RdBu_r` | ±10 dB |
| SST / SSH | per-notebook (check existing savefig call) | — |

---

## Layout: RMSE / Correlation Time Series (UV_forecast, SSH_forecast, SST_forecast)

- `figsize=(14, 8)`
- `GridSpec(2, 2, width_ratios=[4, 1], hspace=0.35, wspace=0.15)`
- Left column: time series (rows = U, V)
- Right column: boxplot IQR summary
- `ax_ts[1].sharex(ax_ts[0])`
- Title on top-left axis only (time series row 0)
- x-label on bottom-left axis only: `'Days since beginning of simulation'`
- Legend: bottom-left axis, `loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3, borderaxespad=0`

**Boxplot style:**
- `width=0.6`
- Mean marker: `marker='D', ms=5, color='white', mec='black', mew=0.8, zorder=5`
- x-tick labels: `fontsize=12, rotation=45, ha='right'`
- y-axis label cleared (`set_ylabel('')`)
- y-grid only: `ax.grid(alpha=0.3, axis='y')`

---

## Layout: Per-Location RMSE (UV_clim_forecast)

- `figsize=(14, 11)`
- `GridSpec(3, 2, width_ratios=[4, 1], hspace=0.45, wspace=0.12)`
- 3 rows = 3 TAO mooring locations (0°N 170°W, 0°N 140°W, 0°N 110°W)
- y-label on time series column: `'{loc_title}\nRMSE (m/s)'`
- x-tick labels hidden on top 2 rows (`plt.setp(ax_ts.get_xticklabels(), visible=False)`)
- Title on `fig.axes[0]`, `fontsize=13`
- x-label on `fig.axes[4]`
- Legend on `fig.axes[4]`: `loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=11`
- Climatology baseline: black dashed horizontal line in boxplot at `np.nanmedian(clim[:, loc_idx])`

---

## Layout: Hovmöller (depth-time) Panels

- 2 columns (U, V) × N rows (models/obs)
- Depth y-axis: negative (surface at top), clipped to ~−300 m
- x-axis: days since start of simulation
- Shared colorbar per column, placed to the right
- Row labels (dataset name) inside or above each panel
- Location annotation (e.g., `0N, 170W`) in bottom-right corner of each panel, small fontsize

---

## Layout: Spectral (PSD) Panels

- 3 rows (longitudes: 170°W, 140°W, 110°W) × 3 cols (TPOSE-Vel, TPOSE-noVel, difference)
- x-axis: frequency (cycles per day)
- y-axis: depth (m), negative, surface at top
- Each panel has its own colorbar on the right
- Column titles above top row only

---

## Panel Annotations & Labels

- Suptitles: use explicit `y=` position to avoid overlap with column titles. Typical `y=1.01` to `1.03`.
- Location strings (e.g., `0N, 170W`) go in the **lower-right** corner of each panel, `fontsize` ~10–11.
- Do not add annotations that aren't already present in the existing figure.

---

## Checks Before Claiming Done

1. Font size ≥ 14 for axis labels; ≥ 11 for minor text (month labels, tick labels).
2. Legend is fully visible and not clipped by figure edge.
3. Colorbar label includes units.
4. No panel titles overlap with suptitle or neighboring panels.
5. All time series lines are visually distinguishable (color + linestyle).
6. Boxplot mean markers (white diamonds) are visible.
7. DPI = 150, bbox_inches = 'tight' in savefig call.
8. No extra whitespace gaps between panels beyond the specified hspace/wspace.
