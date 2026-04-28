#!/usr/bin/env bash
# Download NOAA OISST v2.1 daily SST for the equatorial Pacific (1993-2012).
#
# Parameters:
#   Longitude  : 180–270 °E (stored as -180 to -90 °)
#   Latitude   : 10 °S – 10 °N
#   Variable   : SST (sst)
#   Period     : 01 Jan 1993 – 31 Dec 2012
#   Output     : /data/SO3/edavenport/tpose6/oisst_data/
#
# Uses the tpose conda environment. Downloads one month per request
# (data is small ~3 MB/month; no chunking required).
# Existing monthly files are skipped so interrupted runs can be resumed.

set -euo pipefail

OUTPUT_DIR="/data/SO3/edavenport/tpose6/oisst_data"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda run -n tpose python "$SCRIPT_DIR/oisst_download.py" "$OUTPUT_DIR"

echo "Done."
