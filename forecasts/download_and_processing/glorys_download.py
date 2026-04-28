"""
Download GLORYS12 reanalysis data from Copernicus Marine Service (CMEMS).

Data is saved as monthly NetCDF files, one per month, with all requested variables.
Monthly chunking balances file size (~100 MB–few GB), restart capability, and
typical access patterns.

Coordinate conventions
-----------------------
- Longitude input : 0–360 °E  (converted internally to –180–180 °E for CMEMS)
- Depth input     : 0 (surface) to negative values (e.g. –500 m); GLORYS uses
                    positive-downward depth, so conversion is applied automatically.
- Date strings    : "MM-DD-YYYY"

CMEMS credentials
-----------------
Credentials are read automatically when the user has run `copernicusmarine login`
(stores to ~/.copernicusmarine/). Alternatively pass *username*/*password* kwargs,
or set the environment variables:
    COPERNICUSMARINE_SERVICE_USERNAME
    COPERNICUSMARINE_SERVICE_PASSWORD
"""

import calendar
from datetime import date, datetime
from pathlib import Path

import copernicusmarine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

VAR_MAP: dict[str, str] = {
    "T": "thetao",
    "S": "so",
    "U": "uo",
    "V": "vo",
    "SSH": "zos",
}

# Variables that carry a depth dimension in GLORYS
DEPTH_VARS: frozenset[str] = frozenset({"T", "S", "U", "V"})


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O, fully unit-testable)
# ---------------------------------------------------------------------------

def parse_date(date_str: str) -> date:
    """Parse *date_str* in "MM-DD-YYYY" format and return a :class:`datetime.date`."""
    return datetime.strptime(date_str, "%m-%d-%Y").date()


def lon_360_to_180(lon: float) -> float:
    """Convert longitude from 0–360 °E to –180–180 °E.

    Uses the modulo formula so that 180 maps to –180 (antimeridian western edge).

    Examples
    --------
    >>> lon_360_to_180(0)
    0.0
    >>> lon_360_to_180(260)
    -100.0
    >>> lon_360_to_180(180)
    -180.0
    """
    return ((float(lon) + 180.0) % 360.0) - 180.0


def get_cmems_vars(variables: list[str]) -> list[str]:
    """Map user-facing variable names to CMEMS dataset variable names.

    Parameters
    ----------
    variables:
        Subset of ``["T", "S", "U", "V", "SSH"]``.

    Returns
    -------
    list of str
        Corresponding CMEMS variable names (``thetao``, ``so``, etc.).
    """
    return [VAR_MAP[v] for v in variables]


def get_monthly_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Return ``(month_start, month_end)`` pairs for every month in [*start*, *end*].

    The first and last pairs are clipped to the supplied dates when the period
    does not begin/end on a month boundary.

    Parameters
    ----------
    start, end:
        Inclusive date range.

    Returns
    -------
    list of (date, date)
        One tuple per calendar month, in chronological order.
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


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_glorys(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    depth_min: float,
    depth_max: float,
    variables: list[str],
    time_start: str,
    time_end: str,
    output_dir: str = "glorys_data",
    username: str | None = None,
    password: str | None = None,
    dataset_id: str = DATASET_ID,
) -> list[Path]:
    """Download GLORYS reanalysis data as monthly NetCDF files.

    One file is produced per calendar month; existing files are skipped so
    interrupted downloads can be resumed without re-downloading.

    Parameters
    ----------
    lon_min, lon_max:
        Longitude bounds in 0–360 °E.
    lat_min, lat_max:
        Latitude bounds in –90–90 °N.
    depth_min, depth_max:
        Depth bounds in metres using the oceanographic sign convention:
        0 = surface, negative = deeper (e.g. ``depth_max=-500`` for 500 m).
        Only applied to variables that carry a depth dimension (T, S, U, V).
    variables:
        Variables to download; any non-empty subset of
        ``["T", "S", "U", "V", "SSH"]``.
    time_start, time_end:
        Date range in "MM-DD-YYYY" format (inclusive).
    output_dir:
        Directory for output files (created if it does not exist).
    username, password:
        CMEMS credentials.  When *None*, the library uses cached credentials
        (``copernicusmarine login``) or the COPERNICUSMARINE_SERVICE_* env vars.
    dataset_id:
        CMEMS dataset identifier; override only if you need a different product.

    Returns
    -------
    list of Path
        Paths to all files (downloaded this call + pre-existing skipped files).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Coordinate conversion
    min_lon = lon_360_to_180(lon_min)
    max_lon = lon_360_to_180(lon_max)
    min_depth = min(abs(depth_min), abs(depth_max))
    max_depth = max(abs(depth_min), abs(depth_max))

    cmems_vars = get_cmems_vars(variables)
    need_depth = any(v in DEPTH_VARS for v in variables)

    start = parse_date(time_start)
    end = parse_date(time_end)
    monthly_ranges = get_monthly_ranges(start, end)

    downloaded: list[Path] = []

    for month_start, month_end in monthly_ranges:
        filename = f"glorys_{month_start.year}_{month_start.month:02d}.nc"
        filepath = output_path / filename

        if filepath.exists():
            print(f"  Skipping {filename} (already exists)")
            downloaded.append(filepath)
            continue

        print(f"  Downloading {filename}  ({month_start} → {month_end}) …")

        kwargs: dict = dict(
            dataset_id=dataset_id,
            variables=cmems_vars,
            start_datetime=month_start.strftime("%Y-%m-%dT00:00:00"),
            end_datetime=month_end.strftime("%Y-%m-%dT23:59:59"),
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=float(lat_min),
            maximum_latitude=float(lat_max),
            output_directory=str(output_path),
            output_filename=filename,
            force_download=True,
        )

        if need_depth:
            kwargs["minimum_depth"] = min_depth
            kwargs["maximum_depth"] = max_depth

        if username is not None:
            kwargs["username"] = username
        if password is not None:
            kwargs["password"] = password

        copernicusmarine.subset(**kwargs)
        downloaded.append(filepath)
        print(f"    ✓ Saved {filename}")

    return downloaded
