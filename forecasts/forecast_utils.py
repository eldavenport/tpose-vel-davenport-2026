"""
forecast_utils.py
-----------------
Shared utilities for the forecast comparison notebooks:
    SSH_forecast.ipynb, UV_forecast.ipynb, forecast_time_series.ipynb

Two public functions are provided:

    get_forecast_params(start_date)
        Build every derived parameter needed to run a forecast notebook
        for any 4-month window from Sep 2012 through Jul 2013.
        Handles the 2012 special cases:
          - sep2012: diags_iter7_daily/, itPerFile=48 (30-min timestep)
          - nov2012: diags_new/, itPerFile=72

    load_hycom_daily(start_date, end_date)
        Load HYCOM reanalysis from monthly netCDF files, subset to the
        date window, and daily-average (HYCOM is stored sub-daily).
        Returns an xarray.Dataset with variables surf_el, water_u, water_v.
        Longitude is in −180:180 convention. Depth is positive downward (m).
"""

import pandas as pd
import numpy as np
import xarray as xr
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Per-start-month TPOSE-noVel state estimate overrides for 2012 special cases
# ---------------------------------------------------------------------------
_TPOSE_NOVEL_OVERRIDES = {
    # sep2012 uses a longer timestep: 48 iterations/day (1800 s/step)
    ('sep', 2012): {
        'data_dir': '/data/SO6/TPOSE_diags/tpose6/sep2012/diags_iter7_daily/',
        'itPerFile': 48,
    },
    # nov2012 uses a different diagnostic folder
    ('nov', 2012): {
        'data_dir': '/data/SO6/TPOSE_diags/tpose6/nov2012/diags_jul2026/',
        'itPerFile': 72,
    },
    # jan2013 noVel moved from averdy's SO3 tree to the SO6 tree
    ('jan', 2013): {
        'data_dir': '/data/SO6/TPOSE_diags/tpose6/jan2013/diags_daily/',
        'itPerFile': 72,
    },
}

_GRID_DIR   = '/data/SO6/TPOSE_diags/tpose6/grid_6/'
_HYCOM_DIR  = '/data/SO3/edavenport/tpose6/hycom_data/'


# ---------------------------------------------------------------------------
# get_forecast_params
# ---------------------------------------------------------------------------

def get_forecast_params(start_date):
    """
    Build all derived forecast parameters for a 4-month run.

    Parameters
    ----------
    start_date : pd.Timestamp or str
        First day of the forecast window (should be the 1st of a month).
        Supported range: Sep 2012 – Jul 2013.

    Returns
    -------
    types.SimpleNamespace
        All parameters used by the forecast notebooks, ready to unpack.
        Key attributes:

        Path / directory
          noTAO_data_dir          – TPOSE-noVel state estimate data
          noTAO_forecast_data_dir – TPOSE-noVel forecast run
          vel_forecast_data_dir   – TPOSE-Vel forecast run
          grid_dir                – shared MITgcm grid directory

        open_mdsdataset arguments
          ref_date    – reference date string ('YYYY-MM-DD')
          itPerFile   – model iterations per output file (48 or 72)
          delta_t     – model timestep in seconds (1800 or 1200)
          num_diags   – number of diagnostic files to load
          intervals   – range object passed to open_mdsdataset iters=

        Time / axis helpers
          start_date, end_date
          month_str, day_str, year_str
          n_forecast_days, n_eval
          eval_slice   – slice(0, n_forecast_days)
          eval_dates   – DatetimeIndex of length n_eval
          eval_months  – sorted list of calendar months in the window
          days         – 1-based day index array (length n_forecast_days)
          eval_start_date

        Figure helpers
          month_bounds  – {label: day_number} vertical lines for figures
          month_centers – [(month_abbrev, center_day), ...] for x-axis labels
    """
    start_date = pd.Timestamp(start_date)
    end_date   = start_date + pd.DateOffset(months=4) - pd.Timedelta(days=1)

    month_str = start_date.strftime('%b').lower()   # e.g. 'sep'
    day_str   = start_date.strftime('%d')            # e.g. '01'
    year_str  = start_date.strftime('%Y')            # e.g. '2012'
    year_int  = start_date.year

    # ------------------------------------------------------------------
    # TPOSE-noVel state estimate directory and timestep
    # ------------------------------------------------------------------
    key = (month_str, year_int)
    if key in _TPOSE_NOVEL_OVERRIDES:
        cfg            = _TPOSE_NOVEL_OVERRIDES[key]
        noTAO_data_dir = cfg['data_dir']
        itPerFile      = cfg['itPerFile']
    elif year_int == 2012:
        # Other 2012 months (not sep/nov) use the standard 2012 path
        noTAO_data_dir = f'/data/SO6/TPOSE_diags/tpose6/{month_str}{year_str}/diags/'
        itPerFile      = 72
    else:
        # 2013 standard path
        noTAO_data_dir = f'/data/SO3/averdy/TPOSE6/{month_str}{year_str}/diags_daily/'
        itPerFile      = 72

    delta_t = 1440 / itPerFile * 60   # seconds per model step

    # ------------------------------------------------------------------
    # Forecast run directories (same naming convention for all windows)
    # ------------------------------------------------------------------
    noTAO_forecast_data_dir = (
        f'/data/SO3/edavenport/tpose6/forecasts/'
        f'{month_str}{day_str}{year_str}_tpose_noVel/'
    )
    vel_forecast_data_dir = (
        f'/data/SO3/edavenport/tpose6/forecasts/'
        f'{month_str}{day_str}{year_str}/'
    )

    # ------------------------------------------------------------------
    # Derived time quantities
    # ------------------------------------------------------------------
    ref_date        = start_date.strftime('%Y-%m-%d')
    n_forecast_days = (end_date - start_date).days
    n_eval          = n_forecast_days
    eval_start_date = start_date
    num_diags       = n_forecast_days + 1
    intervals       = range(itPerFile, itPerFile * num_diags, itPerFile)
    eval_slice      = slice(0, n_forecast_days)
    days            = np.arange(1, n_forecast_days + 1)
    eval_dates      = pd.date_range(eval_start_date, periods=n_eval)
    eval_months     = sorted(
        {(eval_start_date + pd.Timedelta(days=i)).month for i in range(n_eval)}
    )

    # ------------------------------------------------------------------
    # Month boundary labels and label-centre positions (for figures)
    # ------------------------------------------------------------------
    month_starts_in_window = pd.date_range(
        start_date + pd.offsets.MonthBegin(1), end_date, freq='MS'
    )
    month_bounds = {
        ms.strftime('%b 1'): (ms - start_date).days + 1
        for ms in month_starts_in_window
    }

    all_month_starts = [start_date] + list(month_starts_in_window)
    month_centers = []
    for i, ms in enumerate(all_month_starts):
        me = (
            all_month_starts[i + 1]
            if i + 1 < len(all_month_starts)
            else end_date + pd.Timedelta(days=1)
        )
        center_day = (ms - start_date).days + (me - ms).days // 2 + 1
        month_centers.append((ms.strftime('%b'), center_day))

    return SimpleNamespace(
        # dates
        start_date=start_date,
        end_date=end_date,
        # string helpers for path construction and figure labels
        month_str=month_str,
        day_str=day_str,
        year_str=year_str,
        # directories
        noTAO_data_dir=noTAO_data_dir,
        noTAO_forecast_data_dir=noTAO_forecast_data_dir,
        vel_forecast_data_dir=vel_forecast_data_dir,
        grid_dir=_GRID_DIR,
        # open_mdsdataset arguments
        ref_date=ref_date,
        itPerFile=itPerFile,
        delta_t=delta_t,
        num_diags=num_diags,
        intervals=intervals,
        # evaluation window
        n_forecast_days=n_forecast_days,
        n_eval=n_eval,
        eval_slice=eval_slice,
        eval_start_date=eval_start_date,
        days=days,
        eval_dates=eval_dates,
        eval_months=eval_months,
        # figure helpers
        month_bounds=month_bounds,
        month_centers=month_centers,
    )


# ---------------------------------------------------------------------------
# load_hycom_daily
# ---------------------------------------------------------------------------

def load_hycom_daily(start_date, end_date):
    """
    Load HYCOM reanalysis for a date window, returning daily-mean data.

    Combines the relevant monthly netCDF files from::

        /data/SO3/edavenport/tpose6/hycom_data/hycom_{YYYY}_{MM:02d}.nc

    Sub-daily timesteps are averaged to daily means via resample('1D').
    Missing monthly files are skipped; days with no data are NaN-filled
    by reindexing to the full requested date range.

    Parameters
    ----------
    start_date, end_date : pd.Timestamp

    Returns
    -------
    xr.Dataset or None
        Variables: ``surf_el`` (time, lat, lon),
                   ``water_u`` (time, depth, lat, lon),
                   ``water_v`` (time, depth, lat, lon).
        Coordinates: ``time`` (daily), ``depth`` (m, positive down),
                     ``lat`` (°N), ``lon`` (°E, −180:180 convention).
        Returns ``None`` if no HYCOM files exist for the requested window.
    """
    import os

    start_date = pd.Timestamp(start_date)
    end_date   = pd.Timestamp(end_date)

    # Only include files that actually exist on disk
    months = pd.date_range(
        start=start_date.to_period('M').to_timestamp(),
        end=end_date.to_period('M').to_timestamp(),
        freq='MS',
    )
    files = [
        _HYCOM_DIR + f'hycom_{m.year}_{m.month:02d}.nc'
        for m in months
        if os.path.exists(_HYCOM_DIR + f'hycom_{m.year}_{m.month:02d}.nc')
    ]

    if not files:
        return None

    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = ds.sel(
        time=slice(start_date.strftime('%Y-%m-%d'),
                   end_date.strftime('%Y-%m-%d'))
    )
    ds = ds.resample(time='1D').mean()

    # Reindex to the full requested daily range, NaN-filling any missing days
    full_index = pd.date_range(start_date, end_date, freq='1D')
    ds = ds.reindex(time=full_index)

    return ds
