"""Unit tests for hycom_download.py.  No network access required."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import date
from pathlib import Path
from unittest.mock import patch

from forecasts.download_and_processing.hycom_download import (
    parse_date,
    get_hycom_vars,
    get_monthly_ranges,
    build_thredds_url,
    _select_subset,
    download_hycom,
    THREDDS_BASE,
)


# ---------------------------------------------------------------------------
# Helpers / pure functions
# ---------------------------------------------------------------------------

def test_parse_date_format():
    assert parse_date("09-01-2012") == date(2012, 9, 1)


def test_parse_date_wrong_format_raises():
    with pytest.raises(ValueError):
        parse_date("2012-09-01")


def test_get_hycom_vars_mapping():
    assert get_hycom_vars(["U", "V", "SSH"]) == ["water_u", "water_v", "surf_el"]


def test_build_thredds_url_embeds_year():
    assert build_thredds_url(2012) == THREDDS_BASE.format(year=2012)


def test_build_thredds_url_custom_base():
    assert build_thredds_url(2013, base="http://host/{year}") == "http://host/2013"


def test_monthly_ranges_complete_months():
    ranges = get_monthly_ranges(date(2012, 9, 1), date(2012, 10, 31))
    assert ranges == [
        (date(2012, 9, 1), date(2012, 9, 30)),
        (date(2012, 10, 1), date(2012, 10, 31)),
    ]


def test_monthly_ranges_crosses_year():
    ranges = get_monthly_ranges(date(2012, 12, 1), date(2013, 1, 31))
    assert ranges == [
        (date(2012, 12, 1), date(2012, 12, 31)),
        (date(2013, 1, 1), date(2013, 1, 31)),
    ]


def test_monthly_ranges_mid_month_clipping():
    ranges = get_monthly_ranges(date(2012, 9, 15), date(2012, 10, 10))
    assert ranges == [
        (date(2012, 9, 15), date(2012, 9, 30)),
        (date(2012, 10, 1), date(2012, 10, 10)),
    ]


def test_monthly_ranges_leap_february():
    ranges = get_monthly_ranges(date(2012, 2, 1), date(2012, 2, 29))
    assert ranges == [(date(2012, 2, 1), date(2012, 2, 29))]


# ---------------------------------------------------------------------------
# _select_subset — tested against a real (in-memory) xarray Dataset so that
# actual selection behaviour is verified, not just mock calls.
# ---------------------------------------------------------------------------

@pytest.fixture
def small_ds():
    """Small in-memory dataset matching HYCOM variable/coordinate names."""
    times = pd.date_range("2012-09-01", "2012-09-30", freq="D")
    lats = np.linspace(-10.0, 10.0, 9)
    lons = np.linspace(180.0, 260.0, 9)
    depths = np.array([0.0, 2.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    n_t, n_d, n_y, n_x = len(times), len(depths), len(lats), len(lons)
    return xr.Dataset(
        {
            "surf_el":   (["time", "lat", "lon"],         np.zeros((n_t, n_y, n_x))),
            "water_u":   (["time", "depth", "lat", "lon"], np.zeros((n_t, n_d, n_y, n_x))),
            "water_v":   (["time", "depth", "lat", "lon"], np.zeros((n_t, n_d, n_y, n_x))),
            "water_temp":(["time", "depth", "lat", "lon"], np.zeros((n_t, n_d, n_y, n_x))),
            "salinity":  (["time", "depth", "lat", "lon"], np.zeros((n_t, n_d, n_y, n_x))),
        },
        coords={"time": times, "lat": lats, "lon": lons, "depth": depths},
    )


def test_select_subset_depth_limited(small_ds):
    result = _select_subset(
        ds=small_ds,
        depth_hycom_vars=["water_u"],
        nodepth_hycom_vars=[],
        time_slice=slice("2012-09-01", "2012-09-07"),
        lat_min=-10.0, lat_max=10.0,
        lon_min=180.0, lon_max=260.0,
        depth_min_pos=0.0, depth_max_pos=100.0,
    )
    assert result["water_u"].depth.values.max() <= 100.0
    # Depths beyond 100 m should not be present
    assert 1000.0 not in result["water_u"].depth.values


def test_select_subset_ssh_has_no_depth_dim(small_ds):
    result = _select_subset(
        ds=small_ds,
        depth_hycom_vars=[],
        nodepth_hycom_vars=["surf_el"],
        time_slice=slice("2012-09-01", "2012-09-07"),
        lat_min=-10.0, lat_max=10.0,
        lon_min=180.0, lon_max=260.0,
        depth_min_pos=0.0, depth_max_pos=500.0,
    )
    assert "depth" not in result["surf_el"].dims


def test_select_subset_mixed_vars_merge(small_ds):
    """SSH and 3-D vars together should merge without error."""
    result = _select_subset(
        ds=small_ds,
        depth_hycom_vars=["water_u", "water_v"],
        nodepth_hycom_vars=["surf_el"],
        time_slice=slice("2012-09-01", "2012-09-07"),
        lat_min=-10.0, lat_max=10.0,
        lon_min=180.0, lon_max=260.0,
        depth_min_pos=0.0, depth_max_pos=500.0,
    )
    assert "water_u" in result and "water_v" in result and "surf_el" in result
    assert "depth" not in result["surf_el"].dims


# ---------------------------------------------------------------------------
# download_hycom — xr.open_dataset mocked to return the in-memory small_ds
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_open(small_ds):
    # Intercept only THREDDS URLs; let local-file opens pass through to real xarray.
    # patch("hycom_download.xr.open_dataset") modifies the global xr module object,
    # so without the side_effect guard the test's own xr.open_dataset calls (used to
    # read written files) would also return small_ds instead of the actual file.
    _real_open = xr.open_dataset

    def _side_effect(path_or_url, **kwargs):
        if isinstance(path_or_url, str) and path_or_url.startswith("http"):
            return small_ds
        return _real_open(path_or_url, **kwargs)

    with patch("hycom_download.xr.open_dataset", side_effect=_side_effect) as m:
        yield m


def _base_call(tmp_path, **overrides):
    kwargs = dict(
        lon_min=180, lon_max=260,
        lat_min=-10, lat_max=10,
        depth_min=0, depth_max=-500,
        variables=["U", "V", "SSH"],
        time_start="09-01-2012",
        time_end="09-30-2012",
        output_dir=str(tmp_path),
    )
    kwargs.update(overrides)
    return download_hycom(**kwargs)


def test_one_file_per_month(tmp_path, patched_open):
    result = _base_call(tmp_path, time_end="10-31-2012")
    assert len(result) == 2
    assert (tmp_path / "hycom_2012_09.nc").exists()
    assert (tmp_path / "hycom_2012_10.nc").exists()


def test_file_naming(tmp_path, patched_open):
    _base_call(tmp_path)
    assert (tmp_path / "hycom_2012_09.nc").exists()


def test_existing_file_skipped(tmp_path, patched_open):
    (tmp_path / "hycom_2012_09.nc").touch()
    _base_call(tmp_path, time_end="10-31-2012")
    assert (tmp_path / "hycom_2012_10.nc").exists()
    # September was skipped — its file should remain as an empty touch file (0 bytes)
    assert (tmp_path / "hycom_2012_09.nc").stat().st_size == 0


def test_returns_paths_including_skipped(tmp_path, patched_open):
    (tmp_path / "hycom_2012_09.nc").touch()
    result = _base_call(tmp_path, time_end="10-31-2012")
    assert [p.name for p in result] == ["hycom_2012_09.nc", "hycom_2012_10.nc"]


def test_depth_limited_in_written_file(tmp_path, patched_open):
    """Depth selection should be reflected in the actual saved NetCDF."""
    _base_call(tmp_path, variables=["U", "V"], depth_max=-100)
    ds = xr.open_dataset(tmp_path / "hycom_2012_09.nc")
    assert ds.depth.values.max() <= 100.0
    ds.close()


def test_ssh_only_no_depth_in_written_file(tmp_path, patched_open):
    """SSH-only output must not have a depth dimension."""
    _base_call(tmp_path, variables=["SSH"])
    ds = xr.open_dataset(tmp_path / "hycom_2012_09.nc")
    assert "depth" not in ds.dims
    ds.close()


def test_year_boundary_opens_dataset_per_year(tmp_path, small_ds):
    """When download spans two years, open_dataset is called once per year."""
    with patch("hycom_download.xr.open_dataset", return_value=small_ds) as mock_open:
        _base_call(tmp_path, time_start="12-01-2012", time_end="01-31-2013")
        assert mock_open.call_count == 2


def test_dataset_opened_once_per_year_within_same_year(tmp_path, patched_open):
    """All months in the same year share one open_dataset call."""
    _base_call(tmp_path, time_end="10-31-2012")
    assert patched_open.call_count == 1


def test_thredds_url_passed_to_open_dataset(tmp_path, patched_open):
    _base_call(tmp_path)
    call_url = patched_open.call_args.args[0]
    assert "2012" in call_url
    assert "GLBv0.08" in call_url
