import sys
from pathlib import Path

sys.path.insert(0, str(Path("__file__").resolve().parent))

from forecasts.download_and_processing.glorys_download import download_glorys

OUTPUT_DIR = '/data/SO3/edavenport/tpose6/glorys_T_data'

download_glorys(
    lon_min=180,
    lon_max=260,
    lat_min=-10,
    lat_max=10,
    depth_min=0,
    depth_max=-500,
    variables=["T"],
    time_start="10-01-2012",
    time_end="12-31-2013",
    output_dir=OUTPUT_DIR,
)