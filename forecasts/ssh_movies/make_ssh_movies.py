#!/usr/bin/env python3
"""
Generate SSH evolution movies for the 4 forecast windows (jan, mar, may, jul 2013).

For each window, produces two movies in forecasts/ssh_movies/:
  {month}{year}_estimates.mp4  -- AVISO + state estimates (no forecasts)
  {month}{year}_forecasts.mp4  -- AVISO + forecasts

All panels within a movie share a single symmetric colorbar.
"""
import sys
import os
sys.path.insert(0, '/home/edavenport/analysis/vel-assim-manuscript')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from xmitgcm import open_mdsdataset
from forecasts.forecast_utils import get_forecast_params, load_hycom_daily

OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
AVISO_PATH = ('/home/edavenport/analysis/vel-assim-manuscript'
              '/forecasts/aviso_data/aviso_equatorial_pacific.nc')

WINDOWS = [
    pd.Timestamp('2013-01-01'),
    pd.Timestamp('2013-03-01'),
    pd.Timestamp('2013-05-01'),
    pd.Timestamp('2013-07-01'),
]

# Vel estimate dirs for windows that have them (may/jul have no vel estimate)
VEL_EST_DIRS = {
    pd.Timestamp('2013-01-01'): '/data/SO3/edavenport/tpose6/jan2013/run_iter14',
    pd.Timestamp('2013-03-01'): '/data/SO3/edavenport/tpose6/mar2013/run_iter16',
}

lonMin, lonMax = 180, 260
latMin, latMax = -10, 10


def open_tpose(data_dir, p):
    ds = open_mdsdataset(
        data_dir=data_dir,
        grid_dir=p.grid_dir,
        iters=p.intervals,
        prefix=['diag_surf'],
        ref_date=p.ref_date,
        delta_t=p.delta_t,
    )
    for coord in ['XC', 'YC', 'Z', 'XG', 'YG']:
        ds[coord] = ds[coord].astype(float)
    return ds


def extract_etan(ds, p):
    """Return (lon, lat, data) where data is (n_frames, ny, nx) anomaly array."""
    arr = (
        ds.ETAN
        .sel(XC=slice(lonMin, lonMax), YC=slice(latMin, latMax))
        .isel(time=p.eval_slice)
        .compute()
    )
    data = arr.values
    return arr.XC.values, arr.YC.values, data - np.nanmean(data)


def extract_aviso(win_start, win_end, p):
    """Return (lon, lat, data) where data is (n_frames, ny, nx) anomaly array."""
    ds = xr.open_dataset(AVISO_PATH, chunks={'time': 10})
    ds = (ds
          .assign_coords(longitude=(ds.longitude % 360))
          .sortby('longitude')
          .sortby('latitude'))
    arr = (
        ds.adt
        .sel(
            time=slice(win_start.strftime('%Y-%m-%d'), win_end.strftime('%Y-%m-%d')),
            latitude=slice(latMin, latMax),
            longitude=slice(lonMin, lonMax),
        )
        .isel(time=p.eval_slice)
        .compute()
    )
    data = arr.values
    return arr.longitude.values, arr.latitude.values, data - np.nanmean(data)


def extract_glorys(win_start, win_end, p):
    """Return (lon, lat, data) for GLORYS zos, or None if no data."""
    months = pd.date_range(
        start=win_start.to_period('M').to_timestamp(),
        end=win_end.to_period('M').to_timestamp(),
        freq='MS',
    )
    files = [
        f'/data/SO3/edavenport/tpose6/glorys_data/glorys_{m.year}_{m.month:02d}.nc'
        for m in months
        if os.access(f'/data/SO3/edavenport/tpose6/glorys_data/glorys_{m.year}_{m.month:02d}.nc', os.R_OK)
    ]
    if not files:
        return None
    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = (ds
          .assign_coords(longitude=(ds.longitude % 360))
          .sortby('longitude')
          .sortby('latitude'))
    arr = (
        ds.zos
        .sel(
            time=slice(win_start.strftime('%Y-%m-%d'), win_end.strftime('%Y-%m-%d')),
            latitude=slice(latMin, latMax),
            longitude=slice(lonMin, lonMax),
        )
        .isel(time=p.eval_slice)
        .compute()
    )
    data = arr.values
    return arr.longitude.values, arr.latitude.values, data - np.nanmean(data)


def extract_hycom(win_start, win_end, p):
    """Return (lon, lat, data) for HYCOM surf_el, or None if no data."""
    hycom = load_hycom_daily(win_start, win_end)
    if hycom is None:
        return None
    ds = hycom.assign_coords(lon=(hycom.lon % 360)).sortby('lon').sortby('lat')
    arr = (
        ds.surf_el
        .sel(
            lat=slice(latMin, latMax),
            lon=slice(lonMin, lonMax),
        )
        .isel(time=p.eval_slice)
        .compute()
    )
    data = arr.values
    return arr.lon.values, arr.lat.values, data - np.nanmean(data)


def make_movie(panels, titles, dates, out_path, fps=8):
    """
    panels : list of (lon, lat, data) tuples
        lon, lat : 1-D coordinate arrays
        data     : (n_frames, ny, nx) float array
    titles : list of str, one per panel
    dates  : list of str, one per frame (shown as suptitle)
    """
    n_panels = len(panels)
    n_frames = panels[0][2].shape[0]

    # Symmetric colorbar: 98th percentile of |anomaly| across all panels and frames
    all_vals = np.concatenate([d.ravel() for _, _, d in panels])
    finite   = all_vals[np.isfinite(all_vals)]
    vabs     = float(np.nanpercentile(np.abs(finite), 98))
    vmin, vmax = -vabs, vabs

    fig, axes_arr = plt.subplots(
        1, n_panels,
        figsize=(6 * n_panels, 3.8),
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(axes_arr[0])

    meshes = []
    for i, (ax, (lon, lat, data), title) in enumerate(zip(axes, panels, titles)):
        mesh = ax.pcolormesh(lon, lat, data[0],
                             vmin=vmin, vmax=vmax,
                             cmap='RdBu_r', shading='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude (°E)', fontsize=9)
        ax.set_xlim(lonMin, lonMax)
        ax.set_ylim(latMin, latMax)
        meshes.append(mesh)

    axes[0].set_ylabel('Latitude (°N)', fontsize=9)
    fig.colorbar(meshes[0], ax=axes, label='SSH anomaly (m)', shrink=0.85)
    date_title = fig.suptitle(dates[0], fontsize=12)

    def update(frame):
        for mesh, (_, _, data) in zip(meshes, panels):
            mesh.set_array(data[frame].ravel())
        date_title.set_text(dates[frame])
        return meshes

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   blit=False, interval=1000 // fps)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f'  Saved: {out_path}')


for win_start in WINDOWS:
    p     = get_forecast_params(win_start)
    label = f"{p.month_str}{p.year_str}"
    print(f'\n=== {label}: {win_start.date()} -> {p.end_date.date()} ({p.n_forecast_days} days) ===')

    dates_str = [d.strftime('%Y-%m-%d') for d in p.eval_dates]

    print('  Loading AVISO ...')
    av = extract_aviso(win_start, p.end_date, p)
    print(f'    shape: {av[2].shape}')

    print(f'  Loading noVel estimate: {p.noTAO_data_dir}')
    ne = extract_etan(open_tpose(p.noTAO_data_dir, p), p)
    print(f'    shape: {ne[2].shape}')

    vel_est_dir = VEL_EST_DIRS.get(win_start)
    if vel_est_dir:
        print(f'  Loading vel estimate: {vel_est_dir}')
        ve = extract_etan(open_tpose(vel_est_dir, p), p)
    else:
        ve = None
        print('  vel estimate: not available for this window')

    print('  Loading GLORYS ...')
    gl = extract_glorys(win_start, p.end_date, p)
    if gl is None:
        print('    GLORYS: no data for this window')

    print('  Loading HYCOM ...')
    hy = extract_hycom(win_start, p.end_date, p)
    if hy is None:
        print('    HYCOM: no data for this window')

    # Estimates movie: AVISO + all state estimates + reanalyses
    est_panels = [av, ne]
    est_titles = ['AVISO', 'TPOSE-noVel Est.']
    if ve is not None:
        est_panels.append(ve); est_titles.append('TPOSE-Vel Est.')
    if gl is not None:
        est_panels.append(gl); est_titles.append('GLORYS')
    if hy is not None:
        est_panels.append(hy); est_titles.append('HYCOM')
    print('  Generating estimates movie ...')
    make_movie(est_panels, est_titles, dates_str,
               os.path.join(OUT_DIR, f'{label}_estimates.mp4'))

print('\nAll movies done.')
