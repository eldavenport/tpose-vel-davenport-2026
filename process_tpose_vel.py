#!/usr/bin/env python3
"""
process_tpose_vel.py

Processes TPOSE-Vel (TAO velocity assimilation) MITgcm outputs into
monthly NetCDF files. Applies proper linear blending over the 2-month
overlap regions between consecutive 4-month assimilation windows, following
the TPOSE product methodology (see Figure 2 of manuscript).

Blending rule:
  - 4 windows of 4 months each, staggered by 2 months → 2-month overlaps.
  - In overlap regions: outgoing window weight 1→0, incoming window 0→1
    (linear over the overlap duration).
  - Outside overlaps: single window, weight = 1.
  - If a day falls in two consecutive overlaps (boundary edge, 1–2 days),
    the later overlap pair takes precedence.

Output (written to OUTPUT_DIR):
  SSH_SST_YYYY-MM.nc   ETAN (SSH) and surface THETA (SST), 2D (time, YC, XC)
  T_YYYY-MM.nc         THETA (potential temperature),       3D (time, Z, YC, XC)
  S_YYYY-MM.nc         SALT  (salinity),                    3D (time, Z, YC, XC)
  U_YYYY-MM.nc         UVEL  (zonal velocity),              3D (time, Z, YC, XG)
  V_YYYY-MM.nc         VVEL  (meridional velocity),         3D (time, Z, YG, XC)

Run with:
    conda run -n tpose python3 process_tpose_vel.py

Files are skipped if they already exist, so the script is safe to restart.
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from xmitgcm import open_mdsdataset

# ─── Configuration ────────────────────────────────────────────────────────────

GRID_DIR   = '/data/SO6/TPOSE_diags/tpose6/grid_6/'
OUTPUT_DIR = '/data/SO3/edavenport/tpose6_processed/'

WINDOWS = [
    dict(name='sep2012',
         data_dir='/data/SO3/edavenport/tpose6/sep2012/velocity_assim/run_iter22/',
         ref_date='2012-09-01', delta_t=1200),
    dict(name='nov2012',
         data_dir='/data/SO3/edavenport/tpose6/nov2012/run_iter20/',
         ref_date='2012-11-01', delta_t=1200),
    dict(name='jan2013',
         data_dir='/data/SO3/edavenport/tpose6/jan2013/run_iter14/',
         ref_date='2013-01-01', delta_t=1200),
    dict(name='mar2013',
         data_dir='/data/SO3/edavenport/tpose6/mar2013/run_iter16/',
         ref_date='2013-03-01', delta_t=1200),
]

# 122 daily output files per window: iterations 72, 144, ..., 8784
ITERS = list(range(72, 72 * 123, 72))

# Calendar months to write out
OUTPUT_MONTHS = [
    '2012-09', '2012-10', '2012-11', '2012-12',
    '2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06',
]

COMPLEVEL = 4   # zlib compression level (1=fast/large, 9=slow/small; 4 is a good balance)


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_window(window, prefix):
    """Open one 4-month assimilation window lazily (dask-backed)."""
    ds = open_mdsdataset(
        data_dir=window['data_dir'],
        grid_dir=GRID_DIR,
        iters=ITERS,
        prefix=prefix,
        ref_date=window['ref_date'],
        delta_t=window['delta_t'],
    )
    # Cast coordinates to float, matching notebook convention
    for coord in ('XC', 'YC', 'Z', 'XG', 'YG'):
        if coord in ds.coords:
            ds[coord] = ds[coord].astype(float)
    # Chunk along time so each day is a separate dask task; keep spatial dims whole
    ds = ds.chunk({'time': 1})
    return ds


# ─── Blending weights ─────────────────────────────────────────────────────────

def build_weights(datasets):
    """
    Compute a (n_windows, n_times) weight matrix over the union of all window
    time axes. Returns (all_times, W) where:
      - all_times  : sorted pandas DatetimeIndex of all unique daily timestamps
      - W[i, j]    : weight of window i at all_times[j], values in [0, 1]
                     and sum_i W[i, j] == 1 for every j

    Algorithm (per time step):
      - If only 1 window covers that day: weight = 1.
      - If 2+ windows cover that day: use the highest-indexed consecutive pair
        (latest transition takes precedence at boundary days) with linear ramp.
    """
    win_times = [pd.DatetimeIndex(ds.time.values) for ds in datasets]

    all_times = pd.DatetimeIndex(sorted({t for wt in win_times for t in wt}))
    n_t = len(all_times)
    n_w = len(datasets)
    W   = np.zeros((n_w, n_t), dtype=float)

    # Precompute boolean coverage matrix: covered[i, j] = True if window i has all_times[j]
    covered = np.zeros((n_w, n_t), dtype=bool)
    for i, wt in enumerate(win_times):
        idx = all_times.get_indexer(wt)
        covered[i, idx] = True

    # Precompute overlap DatetimeIndex for each consecutive pair
    overlaps = {}
    for i in range(n_w - 1):
        ov = win_times[i].intersection(win_times[i + 1])
        overlaps[(i, i + 1)] = ov

    for j in range(n_t):
        covering = np.where(covered[:, j])[0]   # indices of windows that cover this day

        if len(covering) == 1:
            W[covering[0], j] = 1.0

        else:
            # Use the latest consecutive pair (highest indices), so that
            # boundary days where 3 windows overlap follow the most recent transition.
            i_out = int(covering[-2])
            i_in  = int(covering[-1])
            ov    = overlaps[(i_out, i_in)]
            N     = len(ov)
            t_val = all_times[j]
            pos   = ov.get_loc(t_val)           # 0-indexed position within overlap
            frac  = pos / (N - 1)               # 0 at overlap start, 1 at end
            W[i_out, j] = 1.0 - frac            # outgoing: 1 → 0
            W[i_in,  j] = frac                  # incoming: 0 → 1

    # Sanity check
    col_sums = W.sum(axis=0)
    bad = ~np.isclose(col_sums, 1.0)
    if bad.any():
        bad_dates = all_times[bad]
        print(f'  WARNING: weights do not sum to 1 at {bad.sum()} times:')
        for d in bad_dates[:5]:
            print(f'    {d.date()}  sum={col_sums[all_times.get_loc(d)]:.4f}')
    else:
        print(f'  Weights verified: sum=1 at all {n_t} times.')

    return all_times, W


# ─── Monthly selection ────────────────────────────────────────────────────────

def month_indices(all_times, month_str):
    """
    Return (month_times, tidxs) — the subset of all_times falling in month_str
    ('YYYY-MM'), and their integer positions in all_times.

    all_times uses raw MITgcm end-of-period timestamps (e.g. Sep 2 for the
    Sep 1 daily mean), so the selection window is shifted forward by 1 day to
    capture the right records before blend() shifts them back.
    """
    cal_start = pd.Timestamp(month_str + '-01')
    cal_end   = cal_start + pd.offsets.MonthEnd(1)   # last calendar day of month
    # Shift both bounds forward by 1 day to match raw MITgcm timestamps
    raw_start = cal_start + pd.Timedelta('1D')
    raw_end   = cal_end   + pd.Timedelta('1D')
    mask  = (all_times >= raw_start) & (all_times <= raw_end)
    return all_times[mask], np.where(mask)[0]


# ─── Blended DataArray construction ──────────────────────────────────────────

def blend(datasets, mitgcm_var, month_times, tidxs, W):
    """
    Return a lazily blended DataArray for `mitgcm_var` over `month_times`.

    For non-overlap months (one contributing window, weight=1 everywhere) this
    is just a .sel() with no arithmetic overhead.  For overlap months a
    time-varying weight DataArray is broadcast over the spatial dims.

    Boundary months (e.g. the transition from one overlap to the next) can have
    a window that only covers the first 1–2 days of the month.  In that case we
    select only those days from that window, then reindex (zero-pad) back to the
    full month before accumulating, so we never ask a window for times it
    doesn't have.
    """
    blended = None
    for i in range(len(datasets)):
        w_vals = W[i, tidxs]
        if not np.any(w_vals > 0):
            continue

        # Select only timestamps where this window has nonzero weight
        active       = w_vals > 0
        active_times = month_times[active]
        active_w     = w_vals[active]

        da = datasets[i][mitgcm_var].sel(time=active_times)

        if np.all(active_w == 1.0):
            contribution = da                       # common case: sole contributor
        else:
            w_da = xr.DataArray(
                active_w.astype('float32'),
                coords={'time': active_times},
                dims=['time'],
            )
            contribution = w_da * da

        # Reindex to full month_times if this window only covers a subset
        # (missing times get 0; the other window's contribution fills them)
        if not active.all():
            contribution = contribution.reindex(time=month_times, fill_value=0.0)

        blended = contribution if blended is None else blended + contribution

    # Shift timestamps back by 1 day: MITgcm labels daily-mean output at the
    # END of the averaging window (iter 72 = midnight Sep 2 for the Sep 1 mean).
    # Subtracting 1 day gives start-of-period labeling (Sep 1, Sep 2, ...).
    blended['time'] = blended['time'] - pd.Timedelta('1D')

    return blended


# ─── NetCDF encoding ──────────────────────────────────────────────────────────

def encoding_for(da, varname):
    """Chunked, compressed float32 encoding for to_netcdf."""
    return {varname: {
        'zlib':       True,
        'complevel':  COMPLEVEL,
        'dtype':      'float32',
        'chunksizes': (1,) + da.shape[1:],   # 1 time step, full spatial slice
    }}


# ─── Per-variable pipelines ───────────────────────────────────────────────────

def process_3d_var(var_label, mitgcm_var, datasets, all_times, W):
    """Write T / S / U / V monthly NetCDF files."""
    print(f'\n── {var_label}  ({mitgcm_var}) ──')
    for month_str in OUTPUT_MONTHS:
        outpath = os.path.join(OUTPUT_DIR, f'{var_label}_{month_str}.nc')
        if os.path.exists(outpath):
            print(f'  {month_str}: exists, skipping')
            continue

        m_times, tidxs = month_indices(all_times, month_str)
        if len(m_times) == 0:
            print(f'  {month_str}: no times in window, skipping')
            continue

        print(f'  {month_str}: blending {len(m_times)} days ...', flush=True)
        da = blend(datasets, mitgcm_var, m_times, tidxs, W)
        ds_out = da.to_dataset(name=var_label)
        ds_out.to_netcdf(outpath, encoding=encoding_for(da, var_label), format='NETCDF4')
        print(f'  {month_str}: wrote {outpath}')


def process_ssh_sst(ds_state, ds_surf, all_times_state, all_times_surf, W_state, W_surf):
    """Write SSH_SST monthly NetCDF files (ETAN + surface THETA)."""
    print('\n── SSH_SST ──')
    for month_str in OUTPUT_MONTHS:
        outpath = os.path.join(OUTPUT_DIR, f'SSH_SST_{month_str}.nc')
        if os.path.exists(outpath):
            print(f'  {month_str}: exists, skipping')
            continue

        print(f'  {month_str}: blending ...', flush=True)

        # SSH: ETAN from diag_surf
        m_times_surf, tidxs_surf = month_indices(all_times_surf, month_str)
        ssh = blend(ds_surf, 'ETAN', m_times_surf, tidxs_surf, W_surf)

        # SST: surface level (Z index 0) of THETA from diag_state
        m_times_state, tidxs_state = month_indices(all_times_state, month_str)
        sst = blend(ds_state, 'THETA', m_times_state, tidxs_state, W_state).isel(Z=0).drop_vars('Z')

        ds_out = xr.Dataset({'SSH': ssh, 'SST': sst})
        enc = {
            'SSH': {'zlib': True, 'complevel': COMPLEVEL, 'dtype': 'float32',
                    'chunksizes': (1,) + ssh.shape[1:]},
            'SST': {'zlib': True, 'complevel': COMPLEVEL, 'dtype': 'float32',
                    'chunksizes': (1,) + sst.shape[1:]},
        }
        ds_out.to_netcdf(outpath, encoding=enc, format='NETCDF4')
        print(f'  {month_str}: wrote {outpath}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Output directory: {OUTPUT_DIR}')

    # ── Load all 4 windows lazily ─────────────────────────────────────────
    print('\nOpening diag_state windows (lazy) ...')
    ds_state = [load_window(w, ['diag_state']) for w in WINDOWS]

    print('Opening diag_surf windows (lazy) ...')
    ds_surf  = [load_window(w, ['diag_surf'])  for w in WINDOWS]

    # ── Build blending weight matrices ────────────────────────────────────
    print('\nBuilding blend weights (diag_state) ...')
    all_times_state, W_state = build_weights(ds_state)

    print('Building blend weights (diag_surf) ...')
    all_times_surf,  W_surf  = build_weights(ds_surf)

    # ── Report time coverage ──────────────────────────────────────────────
    print(f'\nState coverage: {all_times_state[0].date()} → {all_times_state[-1].date()}'
          f'  ({len(all_times_state)} days)')
    print(f'Surf  coverage: {all_times_surf[0].date()} → {all_times_surf[-1].date()}'
          f'  ({len(all_times_surf)} days)')

    # Summarise overlap periods (informational)
    win_times = [pd.DatetimeIndex(ds.time.values) for ds in ds_state]
    for i in range(len(WINDOWS) - 1):
        ov = win_times[i].intersection(win_times[i + 1])
        print(f'  Overlap {WINDOWS[i]["name"]}↔{WINDOWS[i+1]["name"]}: '
              f'{ov[0].date()} → {ov[-1].date()}  ({len(ov)} days)')

    # ── SSH + SST ─────────────────────────────────────────────────────────
    process_ssh_sst(ds_state, ds_surf, all_times_state, all_times_surf, W_state, W_surf)

    # ── 3D state variables ────────────────────────────────────────────────
    process_3d_var('T', 'THETA', ds_state, all_times_state, W_state)
    process_3d_var('S', 'SALT',  ds_state, all_times_state, W_state)
    process_3d_var('U', 'UVEL',  ds_state, all_times_state, W_state)
    process_3d_var('V', 'VVEL',  ds_state, all_times_state, W_state)

    print('\nAll done.')


if __name__ == '__main__':
    main()
