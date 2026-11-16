"""Figures: (TPOSE-Vel - TPOSE-noVel) control-adjustment triptychs.

Every figure has rows = [TPOSE-Vel, TPOSE-noVel, Vel - noVel]. Three axes vary:

  view    'map'      lat-lon map (all 6 control vars)
          'equator'  depth-longitude section at the equator (EUC), IC T/S
          '140W'     depth-latitude section at 140 W, +/-10 lat, IC T/S
          (sections clipped to the upper ~300 m)

  metric  'rms'   magnitude: sqrt(mean of xx^2); sequential cmap from 0
          'mean'  signed time/depth mean of xx; diverging balance cmap

  span    per-season -> columns = Sep/Nov/Jan/Mar
          all-season -> single column aggregated across the 4 seasons
                        (rms -> sqrt(mean of squares), mean -> mean)

Map reductions: atmos over 'time', IC T/S over upper-300 m 'Z'. Per-season
sections are instantaneous IC slices (signed); the rms/mean distinction only
applies to the all-season aggregate.

Self-contained (own data loads, current paths from tiw-controls/tiw.py).
Run: ~/miniforge3/envs/tpose/bin/python diff_triptychs.py
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mitgcm_assim.ctrls import load_controls_and_sensitivities

GRID = '/data/SO6/TPOSE_diags/tpose6/grid_6/'
VEL = {'Sep': ('/data/SO3/edavenport/tpose6/diags/sep2012/run_iter14', 14),
       'Nov': ('/data/SO3/edavenport/tpose6/diags/nov2012/run_iter20', 20),
       'Jan': ('/data/SO3/edavenport/tpose6/diags/jan2013/run_iter14', 14),
       'Mar': ('/data/SO3/edavenport/tpose6/diags/mar2013/run_iter16', 16)}
NOVEL = {'Sep': ('/data/SO6/TPOSE/tpose6/sep2012/ITER7/', 7),
         # ITER17 (averdy's latest run, written 2026-11-14). Supersedes ITER11;
         # earlier under-converged ITER4 is archived at nov2012/old/ITER4.
         'Nov': ('/data/SO6/TPOSE/tpose6/nov2012/ITER17/', 17),
         'Jan': ('/data/SO6/TPOSE/tpose6/jan2013/ITER7/', 7),
         'Mar': ('/data/SO6/TPOSE/tpose6/mar2013/ITER4/', 4)}
SEASONS = ['Sep', 'Nov', 'Jan', 'Mar']
ZTOP = slice(0, 39)   # upper ~300 m

# var: (label, sequential cmap for rms rows, kind)  -- atmos has time, ic has Z
MAP_VARS = {
    'xx_precip': ('Precip.',         cmocean.cm.rain,    'atmos'),
    'xx_uwind':  ('U Wind',          cmocean.cm.speed,   'atmos'),
    'xx_vwind':  ('V Wind',          cmocean.cm.speed,   'atmos'),
    'xx_swdown': ('Shortwave Rad.',  cmocean.cm.solar,   'atmos'),
    'xx_theta':  ('Initial T (0-300 m)', cmocean.cm.thermal, 'ic'),
    'xx_salt':   ('Initial S (0-300 m)', cmocean.cm.haline,  'ic'),
}
SECT_VARS = {
    'xx_theta': ('Initial T', cmocean.cm.thermal),
    'xx_salt':  ('Initial S', cmocean.cm.haline),
}
DIFF_CMAP = cmocean.cm.balance

# Physical units and prior-uncertainty (sigma) weight files per control.
# Physical adjustment = xx * sigma, sigma = 1/sqrt(weight). Weight files are
# big-endian float32 on the tpose6 grid (same sigma for both runs).
UNITS = {'xx_theta': '$^\\circ$C', 'xx_salt': 'psu', 'xx_uwind': 'm/s',
         'xx_vwind': 'm/s', 'xx_precip': 'm/s', 'xx_swdown': 'W/m$^2$'}
_WDIR = '/data/SO6/TPOSE/model_setup/linked_files/TP6/'
_WEIGHT = {  # var: (filename, is_3d)
    'xx_theta':  ('Argo_T_wt_fac10_tp6_66.bin', True),
    'xx_salt':   ('Argo_S_wt_fac10_tp6_66.bin', True),
    'xx_precip': ('ERA_rain_wt_tp6.bin', False),
    'xx_uwind':  ('ERA_u10m_wt_tp6.bin', False),
    'xx_vwind':  ('ERA_v10m_wt_tp6.bin', False),
    'xx_swdown': ('ERA_dsw_wt_tp6.bin', False),
}
_NX, _NY, _NZ = 1128, 336, 66


def _sigma(fname, is_3d):
    raw = np.fromfile(_WDIR + fname, dtype='>f4')
    wt = (raw.reshape((_NX, _NY, _NZ), order='F').transpose(2, 1, 0) if is_3d
          else raw.reshape((_NX, _NY), order='F').T)
    return np.where(wt > 0, 1.0 / np.sqrt(wt), np.nan)   # sigma = 1/sqrt(weight)


SIGMA = {v: _sigma(f, d3) for v, (f, d3) in _WEIGHT.items()}

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'vel_minus_novel')
os.makedirs(OUTDIR, exist_ok=True)
tao = [tuple(map(float, l.split()))
       for l in open(os.path.join(HERE, '..', 'tao_moorings.txt'))
       if l.strip() and not l.strip().startswith('#')]
HILITE = {(220, 0), (190, 0), (250, 0)}
PROJ = ccrs.PlateCarree(central_longitude=180)


def load(table, tries=4):
    want = set(MAP_VARS) | set(SECT_VARS)
    out = {}
    for s, (d, it) in table.items():
        for k in range(tries):
            ds = load_controls_and_sensitivities(d, GRID, it)
            sub = ds[[v for v in ds.data_vars if str(v) in want]]
            # grid read is flaky under load; retry if vars or coords missing
            if sub.data_vars and 'XC' in sub.coords:
                break
        else:
            raise RuntimeError(f'empty/gridless load after {tries} tries: {d}')
        out[s] = sub
    return out


def near(coord, target):
    # MITgcm coords are big-endian -> cast before pandas sees them.
    return int(np.abs(np.asarray(coord.values, dtype='f8') - target).argmin())


def base_field(ds, var, physical):
    da = ds[var]
    if not physical:
        return da
    return da.copy(data=da.values * SIGMA[var])  # physical adjustment = xx*sigma


def season_field(ds, var, view, metric, physical=False):
    """2D field for one season/variable/view (sigma units or physical)."""
    da = base_field(ds, var, physical)
    if view == 'map':
        base = da.isel(Z=ZTOP) if 'Z' in da.dims else da
        dim = 'Z' if 'Z' in da.dims else 'time'
        return np.sqrt((base ** 2).mean(dim)) if metric == 'rms' else base.mean(dim)
    if view == 'equator':
        return da.isel(Z=ZTOP, YC=near(da.YC, 0.0))
    if view == '140W':
        jlo, jhi = near(da.YC, -10.0), near(da.YC, 10.0)
        return da.isel(Z=ZTOP, XC=near(da.XC, 220.0),
                       YC=slice(min(jlo, jhi), max(jlo, jhi) + 1))


def aggregate(fields, metric):
    """Collapse a list of same-grid DataArrays across seasons."""
    arr = np.stack([f.values for f in fields], axis=0)
    agg = np.sqrt(np.nanmean(arr ** 2, 0)) if metric == 'rms' else np.nanmean(arr, 0)
    return fields[0].copy(data=agg)


def p98(a):
    return float(np.nanpercentile(a, 98))


# --------------------------------------------------------------- panels
def map_panel(fig, sub, da, cmap, vmin, vmax, col_title, row_label, left, bottom):
    ax = fig.add_subplot(sub, projection=PROJ)
    ax.set_aspect('auto')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.set_extent([float(da.XC.min()), float(da.XC.max()), -17, 17],
                  crs=ccrs.PlateCarree())
    im = da.plot.contourf(ax=ax, add_colorbar=False, cmap=cmap,
                          levels=np.linspace(vmin, vmax, 21), extend='both',
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
    _titles(ax, col_title, row_label)
    return im


def section_panel(fig, sub, da, xdim, cmap, vmin, vmax, col_title, row_label,
                  left, bottom):
    ax = fig.add_subplot(sub)
    im = da.plot.contourf(ax=ax, x=xdim, y='Z', cmap=cmap,
                          levels=np.linspace(vmin, vmax, 21), extend='both',
                          add_colorbar=False)
    ax.set_ylim(-315, 0)
    ax.axvline(220 if xdim == 'XC' else 0, color='k', ls='--', lw=0.6, alpha=0.5)
    ax.set_ylabel('Depth (m)' if left else '')
    if not left:
        ax.set_yticklabels([])
    ax.set_xlabel(('Longitude ($^\\circ$E)' if xdim == 'XC' else 'Latitude')
                  if bottom else '')
    if not bottom:
        ax.set_xticklabels([])
    _titles(ax, col_title, row_label, transax=True)
    return im


def _cbar(fig, sub, cmap, lim, label, nbins, symmetric):
    """Linear colorbar from a continuous Normalize so ticks are evenly spaced
    (a contourf colorbar is discrete + extended, which squishes ticks)."""
    sm = ScalarMappable(norm=Normalize(*lim), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=fig.add_subplot(sub), label=label, extend='both',
                 ticks=MaxNLocator(nbins=nbins, symmetric=symmetric))


def _titles(ax, col_title, row_label, transax=False):
    ax.set_title(col_title if col_title else '', fontsize=14, fontweight='bold')
    if row_label:
        ax.text(0.02, 0.97, row_label, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=1.0, edgecolor='gray', linewidth=0.5))


# --------------------------------------------------------------- renderer
VIEW_CFG = {  # per-column width, fig height, (hspace, wspace), x-dim
    # ~25% wider / shorter than square so maps read like the tropical Pacific
    'map':     (5.0, 6.0, (0.12, 0.15), 'XC'),
    'equator': (4.9, 6.4, (0.30, 0.18), 'XC'),
    '140W':    (3.0, 6.8, (0.30, 0.22), 'YC'),
}


def render(view, var, label, cmap, metric, magnitude, col_labels, V, N, outname,
           unit='units of $\\sigma$'):
    vname = label.split(' (')[0]  # e.g. 'Initial T (0-300 m)' -> 'Initial T'
    D = {c: V[c] - N[c] for c in col_labels}
    dmax = max(p98(np.abs(D[c].values)) for c in col_labels) or 1e-9
    if magnitude:
        vmax = max(p98(V[c].values) for c in col_labels)
        vmax = max(vmax, max(p98(N[c].values) for c in col_labels), 1e-9)
        row_lim, row_cmap = (0, vmax), cmap
        top_cb = f'RMS adjustment\n({unit})'
    else:
        amax = max(max(p98(np.abs(V[c].values)), p98(np.abs(N[c].values)))
                   for c in col_labels) or 1e-9
        row_lim, row_cmap = (-amax, amax), DIFF_CMAP
        top_cb = f'mean adjustment\n({unit})'

    rows = [('TPOSE-Vel', V, *row_lim, row_cmap),
            ('TPOSE-noVel', N, *row_lim, row_cmap),
            ('Vel $-$ noVel', D, -dmax, dmax, DIFF_CMAP)]

    percol, figh, (hs, ws), xdim = VIEW_CFG[view]
    nc = len(col_labels)
    fig = plt.figure(figsize=(percol * nc + 1.2, figh))
    gs = gridspec.GridSpec(3, nc + 1, width_ratios=[1] * nc + [0.04],
                           wspace=ws, hspace=hs)
    for r, (rlab, fields, vmin, vmax, cm) in enumerate(rows):
        for c, lab in enumerate(col_labels):
            args = dict(col_title=f'{lab} {vname}' if r == 0 else None,
                        row_label=rlab if c == 0 else None,
                        left=(c == 0), bottom=(r == 2))
            if view == 'map':
                im = map_panel(fig, gs[r, c], fields[lab], cm, vmin, vmax, **args)
            else:
                im = section_panel(fig, gs[r, c], fields[lab], xdim, cm,
                                   vmin, vmax, **args)
        if r == 0:  # spans 2 rows
            _cbar(fig, gs[0:2, nc], row_cmap, row_lim, top_cb,
                  nbins=5, symmetric=not magnitude)
        if r == 2:  # 1 row tall -> fewer ticks
            _cbar(fig, gs[2, nc], DIFF_CMAP, (-dmax, dmax),
                  f'Vel $-$ noVel\n({unit})',
                  nbins=3, symmetric=True)
    out = os.path.join(OUTDIR, outname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return outname


def fields_for(view, var, metric, vel, nov, per_season, physical=False):
    sf = lambda ds: season_field(ds, var, view, metric, physical)
    if per_season:
        cols = SEASONS
        V = {s: sf(vel[s]) for s in SEASONS}
        N = {s: sf(nov[s]) for s in SEASONS}
    else:
        cols = ['All seasons']
        V = {'All seasons': aggregate([sf(vel[s]) for s in SEASONS], metric)}
        N = {'All seasons': aggregate([sf(nov[s]) for s in SEASONS], metric)}
    return cols, V, N


if __name__ == '__main__':
    print('loading Vel...');   vel = load(VEL)
    print('loading noVel...'); nov = load(NOVEL)

    # Two unit systems: sigma-normalized (xx) and physical (xx * sigma).
    for physical in (False, True):
        uf = '_physical' if physical else ''
        unit = lambda var: UNITS[var] if physical else 'units of $\\sigma$'

        # ---- maps: per-season and all-season, both metrics
        for metric in ('rms', 'mean'):
            for var, (label, cmap, kind) in MAP_VARS.items():
                for per in (True, False):
                    cols, V, N = fields_for('map', var, metric, vel, nov, per,
                                            physical)
                    suffix = '' if per else '_allseason'
                    out = f'diff_{metric}_triptych_{var}{suffix}{uf}.png'
                    print('map', uf or 'sigma', metric, var,
                          'per_season' if per else 'allseason')
                    render('map', var, label, cmap, metric, metric == 'rms',
                           cols, V, N, out, unit(var))

        # ---- sections: per-season (signed slice) + all-season (both metrics)
        for view in ('equator', '140W'):
            for var, (label, cmap) in SECT_VARS.items():
                cols, V, N = fields_for(view, var, 'mean', vel, nov, True,
                                        physical)
                print('section', uf or 'sigma', view, var, 'per_season')
                render(view, var, label, cmap, 'mean', False, cols, V, N,
                       f'diff_{view}_section_{var}{uf}.png', unit(var))
                for metric in ('rms', 'mean'):
                    cols, V, N = fields_for(view, var, metric, vel, nov, False,
                                            physical)
                    print('section', uf or 'sigma', view, var, 'allseason',
                          metric)
                    render(view, var, label, cmap, metric, metric == 'rms',
                           cols, V, N,
                           f'diff_{view}_section_{var}_{metric}_allseason{uf}.png',
                           unit(var))
    print('done')
