"""
Command-line script to download GLORYS12 reanalysis data for an arbitrary region.

Wraps :func:`forecasts.glorys_download.download_glorys` with argparse so that
region limits, variables, dates, and output directory can all be specified on
the command line without editing any code.

Usage examples
--------------
# Equatorial Pacific, SSH/U/V, Jul–Dec 2013 (notebook defaults)
python forecasts/glorys_download_region.py

# Custom region and date range
python forecasts/glorys_download_region.py \\
    --lon-min 150 --lon-max 290 \\
    --lat-min -20 --lat-max 20 \\
    --depth-min 0 --depth-max -1000 \\
    --variables U V \\
    --time-start 01-01-2010 --time-end 12-31-2010 \\
    --output-dir my_glorys_data

# All five variables for a specific month
python forecasts/glorys_download_region.py \\
    --variables T S U V SSH \\
    --time-start 03-01-2013 --time-end 03-31-2013
"""

import argparse
import sys
from pathlib import Path

# Allow running from the repo root or from within forecasts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecasts.download_and_processing.glorys_download import download_glorys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GLORYS12 reanalysis data for a configurable region.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Spatial bounds
    parser.add_argument(
        "--lon-min", type=float, default=180.0,
        help="Western longitude bound (0–360 °E).",
    )
    parser.add_argument(
        "--lon-max", type=float, default=260.0,
        help="Eastern longitude bound (0–360 °E).",
    )
    parser.add_argument(
        "--lat-min", type=float, default=-10.0,
        help="Southern latitude bound (°N, negative = south).",
    )
    parser.add_argument(
        "--lat-max", type=float, default=10.0,
        help="Northern latitude bound (°N).",
    )

    # Depth bounds
    parser.add_argument(
        "--depth-min", type=float, default=0.0,
        help="Minimum depth in metres (0 = surface, use 0 or positive).",
    )
    parser.add_argument(
        "--depth-max", type=float, default=-500.0,
        help="Maximum depth in metres (negative = deeper, e.g. -500).",
    )

    # Variables
    parser.add_argument(
        "--variables", nargs="+", default=["SSH", "U", "V"],
        choices=["T", "S", "U", "V", "SSH"],
        metavar="VAR",
        help="Variables to download. Choose from: T S U V SSH.",
    )

    # Time range
    parser.add_argument(
        "--time-start", default="07-01-2013",
        help="Start date in MM-DD-YYYY format (inclusive).",
    )
    parser.add_argument(
        "--time-end", default="12-31-2013",
        help="End date in MM-DD-YYYY format (inclusive).",
    )

    # Output
    parser.add_argument(
        "--output-dir", default="glorys_data",
        help="Directory where monthly NetCDF files will be saved.",
    )

    # Credentials (optional; copernicusmarine login caches these)
    parser.add_argument("--username", default=None, help="CMEMS username.")
    parser.add_argument("--password", default=None, help="CMEMS password.")

    # Dataset override
    parser.add_argument(
        "--dataset-id", default=None,
        help="CMEMS dataset ID (uses library default when omitted).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    kwargs = dict(
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        variables=args.variables,
        time_start=args.time_start,
        time_end=args.time_end,
        output_dir=args.output_dir,
        username=args.username,
        password=args.password,
    )
    if args.dataset_id is not None:
        kwargs["dataset_id"] = args.dataset_id

    print("GLORYS download configuration:")
    print(f"  Region     : lon [{args.lon_min}, {args.lon_max}] °E, "
          f"lat [{args.lat_min}, {args.lat_max}] °N")
    print(f"  Depth      : {args.depth_min} – {args.depth_max} m")
    print(f"  Variables  : {args.variables}")
    print(f"  Period     : {args.time_start} → {args.time_end}")
    print(f"  Output dir : {args.output_dir}")
    print()

    files = download_glorys(**kwargs)

    print(f"\nDownloaded {len(files)} file(s):")
    for f in files:
        size_mb = Path(f).stat().st_size / 1e6 if Path(f).exists() else 0
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
