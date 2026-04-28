"""
Download NOAA OISST v2.1 daily SST for the equatorial Pacific (1993-2012).
Called by oisst_download.sh — do not run directly.

Uses NOAA CoastWatch ERDDAP OPeNDAP server (±180 longitude convention).
Downloads one year at a time (~42 MB/year) to avoid ERDDAP request size limits,
then concatenates into a single output file (~860 MB).
"""
import sys
from pathlib import Path

import xarray as xr
from tqdm import tqdm

OUTPUT_DIR  = Path(sys.argv[1])
OUTPUT_FILE = OUTPUT_DIR / "oisst_equatorial_pacific_2012to2013.nc"

LON_MIN    = -180.0
LON_MAX    = -90.0
LAT_MIN    = -10.0
LAT_MAX    = 10.0
YEARS      = range(2012, 2014)
ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180"

if OUTPUT_FILE.exists():
    print(f"Output already exists: {OUTPUT_FILE}")
    sys.exit(0)

print("Opening OISST ERDDAP dataset …")
ds = xr.open_dataset(ERDDAP_URL, engine="netcdf4",
                     drop_variables=["anom", "err", "ice"])

yearly_chunks = []
try:
    with tqdm(YEARS, unit="year", ncols=72) as pbar:
        for year in pbar:
            pbar.set_description(str(year))
            subset = ds[["sst"]].sel(
                time=slice(f"{year}-01-01", f"{year}-12-31"),
                latitude=slice(LAT_MIN, LAT_MAX),
                longitude=slice(LON_MIN, LON_MAX),
            )
            if "zlev" in subset.dims:
                subset = subset.squeeze("zlev", drop=True)
            subset.load()
            yearly_chunks.append(subset)
            tqdm.write(f"  {year} ✓  ({subset.nbytes / 1e6:.0f} MB)")
finally:
    ds.close()

print("Concatenating and saving …")
combined = xr.concat(yearly_chunks, dim="time")
combined.attrs = {}
combined.to_netcdf(str(OUTPUT_FILE))
print(f"Saved {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1e6:.0f} MB)")
