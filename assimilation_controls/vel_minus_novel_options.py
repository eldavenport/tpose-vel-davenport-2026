"""Prototype layout options for (TPOSE-Vel - TPOSE-noVel) control adjustments.

Renders three candidate layouts for the atmospheric controls so we can judge
readability before committing one into tpose_vel_controls.ipynb.

Metric = signed time-mean adjustment (xx field, units of sigma), same as the
existing *_mean.png figures (cmocean balance).

Run:  ~/miniforge3/envs/tpose/bin/python vel_minus_novel_options.py
(from vel-assim-manuscript/assimilation_controls/)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mitgcm_assim.ctrls import load_controls_and_sensitivities

GRID = '/data/SO6/TPOSE_diags/tpose6/grid_6/'
# Current paths (from tiw-controls/tiw.py): Vel controls moved under diags/.
VEL = {  # season: (dir, final_iter)
    'Sep': ('/data/SO3/edavenport/tpose6/diags/sep2012/run_iter14', 14),
    'Nov': ('/data/SO3/edavenport/tpose6/diags/nov2012/run_iter20', 20),
    'Jan': ('/data/SO3/edavenport/tpose6/diags/jan2013/run_iter14', 14),
    'Mar': ('/data/SO3/edavenport/tpose6/diags/mar2013/run_iter16', 16),
}
NOVEL = {
    'Sep': ('/data/SO6/TPOSE/tpose6/sep2012/ITER7/', 7),
    'Nov': ('/data/SO6/TPOSE/tpose6/nov2012/ITER4/', 4),
    'Jan': ('/data/SO6/TPOSE/tpose6/jan2013/ITER7/', 7),
    'Mar': ('/data/SO6/TPOSE/tpose6/mar2013/ITER4/', 4),
}
SEASONS = ['Sep', 'Nov', 'Jan', 'Mar']

VARS = {  # atmospheric controls present in both sims
    'xx_precip': 'Precip.',
    'xx_uwind':  'U Wind',
    'xx_vwind':  'V Wind',
    'xx_swdown': 'Shortwave Rad.',
}

tao = [tuple(map(float, l.split())) for l in open('../tao_moorings.txt')
       if l.strip() and not l.strip().startswith('#')]
HILITE = {(220, 0), (190, 0), (250, 0)}

PROJ = ccrs.PlateCarree(central_longitude=180)


def load(table):
    out = {}
    for s, (d, it) in table.items():
        ds = load_controls_and_sensitivities(d, GRID, it)
        out[s] = ds[[v for v in ds.data_vars if str(v) in VARS]]
    return out


def adj(ds, var):
    """Signed time-mean adjustment map."""
    return ds[var].mean(dim='time')


def panel(fig, sub, da, cmap, vmin, vmax, col_title=None, row_label=None,
          left=False, bottom=False):
    ax = fig.add_subplot(sub, projection=PROJ)
    ax.set_aspect('auto')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.set_extent([float(da.XC.min()), float(da.XC.max()), -17, 17],
                  crs=ccrs.PlateCarree())
    im = da.plot(ax=ax, add_colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                 transform=ccrs.PlateCarree(), x='XC', y='YC')
    for lon, lat in tao:
        ax.plot(lon, lat, marker='o',
                color='red' if (lon, lat) in HILITE else 'white',
                markersize=4, markeredgecolor='black', markeredgewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.5, linestyle='--', crs=ccrs.PlateCarree())
    gl.left_labels, gl.bottom_labels = left, bottom
    gl.right_labels = gl.top_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 11}
    ax.set_title(col_title if col_title else '', fontsize=14, fontweight='bold')
    if row_label:
        ax.text(0.02, 0.97, row_label, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=1.0, edgecolor='gray', linewidth=0.5))
    return im


def sym(*arrs, pct=98):
    m = max(float(np.nanpercentile(np.abs(a.values), pct)) for a in arrs)
    return -m, m


# ---------------------------------------------------------------- load data
print('loading Vel...'); vel = load(VEL)
print('loading noVel...'); nov = load(NOVEL)

# precompute adjustment + difference fields
A_vel = {v: {s: adj(vel[s], v) for s in SEASONS} for v in VARS}
A_nov = {v: {s: adj(nov[s], v) for s in SEASONS} for v in VARS}
DIFF  = {v: {s: A_vel[v][s] - A_nov[v][s] for s in SEASONS} for v in VARS}

CMAP = cmocean.cm.balance

# ============================================================= OPTION 1
# Difference-only matrix: rows = variables, cols = seasons. Compact, same
# shape as the existing figures; shows only Vel - noVel.
print('rendering option 1 (difference-only matrix)...')
nr, nc = len(VARS), len(SEASONS)
fig = plt.figure(figsize=(17, 9))
gs = gridspec.GridSpec(nr, nc + 1, width_ratios=[1, 1, 1, 1, 0.03],
                       wspace=0.15, hspace=0.12)
for r, (var, label) in enumerate(VARS.items()):
    vmin, vmax = sym(*DIFF[var].values())
    for c, s in enumerate(SEASONS):
        im = panel(fig, gs[r, c], DIFF[var][s], CMAP, vmin, vmax,
                   col_title=s if r == 0 else None,
                   row_label=label if c == 0 else None,
                   left=(c == 0), bottom=(r == nr - 1))
    fig.colorbar(im, cax=fig.add_subplot(gs[r, nc]),
                 label='Vel $-$ noVel\n(units of $\\sigma$)')
fig.suptitle('Atmospheric control adjustment difference (TPOSE-Vel $-$ TPOSE-noVel)',
             y=0.97, fontsize=15)
plt.savefig('diff_atmos_option1_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================= OPTION 2
# Triptych for a single variable: rows = [Vel, noVel, Vel-noVel], cols =
# seasons. Rows 1-2 share one scale (comparable); row 3 its own diff scale.
print('rendering option 2 (U-wind triptych)...')
VAR, VLABEL = 'xx_uwind', 'U Wind'
adj_lim = sym(*A_vel[VAR].values(), *A_nov[VAR].values())
dif_lim = sym(*DIFF[VAR].values())
rows = [('TPOSE-Vel', A_vel[VAR], adj_lim),
        ('TPOSE-noVel', A_nov[VAR], adj_lim),
        ('Vel $-$ noVel', DIFF[VAR], dif_lim)]
fig = plt.figure(figsize=(17, 7.5))
gs = gridspec.GridSpec(3, nc + 1, width_ratios=[1, 1, 1, 1, 0.03],
                       wspace=0.15, hspace=0.12)
for r, (rlab, fields, (vmin, vmax)) in enumerate(rows):
    for c, s in enumerate(SEASONS):
        im = panel(fig, gs[r, c], fields[s], CMAP, vmin, vmax,
                   col_title=s if r == 0 else None,
                   row_label=rlab if c == 0 else None,
                   left=(c == 0), bottom=(r == 2))
    lbl = 'Vel $-$ noVel\n($\\sigma$)' if r == 2 else 'adjustment ($\\sigma$)'
    fig.colorbar(im, cax=fig.add_subplot(gs[r, nc]), label=lbl)
fig.suptitle(f'{VLABEL} control adjustment: TPOSE-Vel, TPOSE-noVel, and difference',
             y=1.00, fontsize=15)
plt.savefig('diff_atmos_option2_triptych_uwind.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================= OPTION 3
# Per-season, all variables: rows = variables, cols = [Vel, noVel, Diff] for
# one season (Sep). Vel/noVel share scale per row; Diff its own.
print('rendering option 3 (Sep, vars x [Vel/noVel/Diff])...')
S = 'Sep'
fig = plt.figure(figsize=(14, 11))
gs = gridspec.GridSpec(nr, 5, width_ratios=[1, 1, 0.03, 1, 0.03],
                       wspace=0.12, hspace=0.12)
col_titles = ['TPOSE-Vel', 'TPOSE-noVel', None, 'Vel $-$ noVel']
for r, (var, label) in enumerate(VARS.items()):
    a_lim = sym(A_vel[var][S], A_nov[var][S])
    d_lim = sym(DIFF[var][S])
    fields = [(A_vel[var][S], a_lim, 0), (A_nov[var][S], a_lim, 1),
              (DIFF[var][S], d_lim, 3)]
    for da, (vmin, vmax), gc in fields:
        im = panel(fig, gs[r, gc], da, CMAP, vmin, vmax,
                   col_title=col_titles[gc] if r == 0 else None,
                   row_label=label if gc == 0 else None,
                   left=(gc == 0), bottom=(r == nr - 1))
        if gc == 1:
            fig.colorbar(im, cax=fig.add_subplot(gs[r, 2]), label='$\\sigma$')
        if gc == 3:
            fig.colorbar(im, cax=fig.add_subplot(gs[r, 4]), label='$\\sigma$')
fig.suptitle(f'Atmospheric control adjustments, {S} 2012',
             y=0.93, fontsize=15)
plt.savefig('diff_atmos_option3_perseason_sep.png', dpi=150, bbox_inches='tight')
plt.close()

print('done:')
for f in ('diff_atmos_option1_matrix.png',
          'diff_atmos_option2_triptych_uwind.png',
          'diff_atmos_option3_perseason_sep.png'):
    print('  ', f)
