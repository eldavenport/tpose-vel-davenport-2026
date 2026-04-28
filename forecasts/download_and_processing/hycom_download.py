"""
Download HYCOM GLBv0.08 reanalysis data via OPeNDAP/THREDDS.

Data is saved as monthly NetCDF files, one per month, with all requested variables.
Monthly chunking balances file size, restart capability, and typical access patterns.

Coordinate conventions
-----------------------
- Longitude input : 0–360 °E  (HYCOM GLBv0.08/expt_53.X uses 0–360 °E natively;
                    no conversion applied)
- Depth input     : 0 (surface) to negative values (e.g. –500 m); HYCOM uses
                    positive-downward depth, so conversion is applied automatically.
- Date strings    : "MM-DD-YYYY"

HYCOM dataset
-------------
Uses HYCOM + NCODA GLBv0.08/expt_53.X (1/12 °, 1994–2015) via the HYCOM THREDDS
server.  No authentication is required.

HYCOM OPeNDAP coordinate names
--------------------------------
    time, lat, lon, depth
"""

import calendar
from datetime import date, datetime
from pathlib import Path

import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THREDDS_BASE = "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/{year}"

VAR_MAP: dict[str, str] = {
    "T": "water_temp",
    "S": "salinity",
    "U": "water_u",
    "V": "water_v",
    "SSH": "surf_el",
}

# Variables that carry a depth dimension in HYCOM
DEPTH_VARS: frozenset[str] = frozenset({"T", "S", "U", "V"})


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O, fully unit-testable)
# ---------------------------------------------------------------------------

def parse_date(date_str: str) -> date:
    """Parse *date_str* in "MM-DD-YYYY" format and return a :class:`datetime.date`."""
    return datetime.strptime(date_str, "%m-%d-%Y").date()


def get_hycom_vars(variables: list[str]) -> list[str]:
    """Map user-facing variable names to HYCOM dataset variable names.

    Parameters
    ----------
    variables:
        Subset of ``["T", "S", "U", "V", "SSH"]``.

    Returns
    -------
    list of str
        Corresponding HYCOM variable names (``water_u``, ``surf_el``, etc.).
    """
    return [VAR_MAP[v] for v in variables]


def get_monthly_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Return ``(month_start, month_end)`` pairs for every month in [*start*, *end*].

    The first and last pairs are clipped to the supplied dates when the period
    does not begin/end on a month boundary.
    """
    ranges: list[tuple[date, date]] = []
    current = start.replace(day=1)
    while current <= end:
        days_in_month = calendar.monthrange(current.year, current.month)[1]
        month_end = current.replace(day=days_in_month)
        actual_start = max(current, start)
        actual_end = min(month_end, end)
        ranges.append((actual_start, actual_end))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return ranges


def build_thredds_url(year: int, base: str = THREDDS_BASE) -> str:
    """Return the THREDDS OPeNDAP URL for *year*."""
    return base.format(year=year)


def _select_subset(
    ds: xr.Dataset,
    depth_hycom_vars: list[str],
    nodepth_hycom_vars: list[str],
    time_slice: slice,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    depth_min_pos: float,
    depth_max_pos: float,
) -> xr.Dataset:
    """Select spatial/temporal subset, handling depth dimension correctly.

    ``surf_el`` has no depth axis; xarray raises if you attempt a depth
    selection on it.  This function separates depth-having and depth-free
    variables, selects each group independently, then merges.
    """
    spatial_time: dict = dict(
        time=time_slice,
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )
    parts: list[xr.Dataset] = []
    if depth_hycom_vars:
        parts.append(
            ds[depth_hycom_vars].sel(
                **spatial_time, depth=slice(depth_min_pos, depth_max_pos)
            )
        )
    if nodepth_hycom_vars:
        parts.append(ds[nodepth_hycom_vars].sel(**spatial_time))
    return xr.merge(parts) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_hycom(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    depth_min: float,
    depth_max: float,
    variables: list[str],
    time_start: str,
    time_end: str,
    output_dir: str = "hycom_data",
    thredds_base: str = THREDDS_BASE,
    chunk_days: int = 2,
) -> list[Path]:
    """Download HYCOM reanalysis data as monthly NetCDF files.

    One file is produced per calendar month; existing files are skipped so
    interrupted downloads can be resumed without re-downloading.  The annual
    OPeNDAP dataset for each year is opened once and reused for all months in
    that year, then closed when the download finishes.

    The HYCOM THREDDS server silently truncates large OPeNDAP responses and
    fills remaining values with zeros rather than raising an error.  To avoid
    this, each month is fetched in ``chunk_days``-day windows (default 2 days,
    ~0.9 GB per chunk for U+V at this domain size) and the chunks are
    concatenated before writing to disk.

    Parameters
    ----------
    lon_min, lon_max:
        Longitude bounds in 0–360 °E.
    lat_min, lat_max:
        Latitude bounds in –80–80 °N (HYCOM global extent).
    depth_min, depth_max:
        Depth bounds using the oceanographic sign convention:
        0 = surface, negative = deeper (e.g. ``depth_max=-500`` for 500 m).
        Only applied to variables with a depth dimension (T, S, U, V).
    variables:
        Variables to download; any non-empty subset of
        ``["T", "S", "U", "V", "SSH"]``.
    time_start, time_end:
        Date range in "MM-DD-YYYY" format (inclusive).
    output_dir:
        Directory for output files (created if it does not exist).
    thredds_base:
        OPeNDAP URL template with ``{year}`` placeholder; override to use a
        different HYCOM experiment.
    chunk_days:
        Number of days per OPeNDAP request.  Smaller values are safer against
        server truncation but require more round-trips.  Default is 2.

    Returns
    -------
    list of Path
        Paths to all files (downloaded this call + pre-existing skipped files).
    """
    from datetime import timedelta

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    depth_min_pos = min(abs(depth_min), abs(depth_max))
    depth_max_pos = max(abs(depth_min), abs(depth_max))

    depth_hycom_vars = [VAR_MAP[v] for v in variables if v in DEPTH_VARS]
    nodepth_hycom_vars = [VAR_MAP[v] for v in variables if v not in DEPTH_VARS]

    start = parse_date(time_start)
    end = parse_date(time_end)
    monthly_ranges = get_monthly_ranges(start, end)

    downloaded: list[Path] = []
    open_datasets: dict[int, xr.Dataset] = {}

    try:
        for month_start, month_end in monthly_ranges:
            filename = f"hycom_{month_start.year}_{month_start.month:02d}.nc"
            filepath = output_path / filename

            if filepath.exists():
                print(f"  Skipping {filename} (already exists)")
                downloaded.append(filepath)
                continue

            print(f"  Downloading {filename}  ({month_start} → {month_end}) …")

            year = month_start.year
            if year not in open_datasets:
                url = build_thredds_url(year, thredds_base)
                open_datasets[year] = xr.open_dataset(url, engine="pydap")

            ds = open_datasets[year]

            # Build day-by-day chunk windows for the month
            chunk_start = month_start
            chunks: list[xr.Dataset] = []
            while chunk_start <= month_end:
                chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), month_end)
                time_slice = slice(
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                )
                subset = _select_subset(
                    ds=ds,
                    depth_hycom_vars=depth_hycom_vars,
                    nodepth_hycom_vars=nodepth_hycom_vars,
                    time_slice=time_slice,
                    lat_min=float(lat_min),
                    lat_max=float(lat_max),
                    lon_min=float(lon_min),
                    lon_max=float(lon_max),
                    depth_min_pos=depth_min_pos,
                    depth_max_pos=depth_max_pos,
                )
                subset.load()
                chunks.append(subset)
                print(f"    chunk {chunk_start} → {chunk_end} ✓")
                chunk_start = chunk_end + timedelta(days=1)

            monthly_ds = xr.concat(chunks, dim="time")
            monthly_ds.to_netcdf(str(filepath))

            downloaded.append(filepath)
            print(f"    ✓ Saved {filename}")

    finally:
        for ds in open_datasets.values():
            ds.close()

    return downloaded
