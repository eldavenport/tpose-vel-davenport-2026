"""Unit tests for glorys_download.py. No CMEMS credentials required."""

import pytest
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

from forecasts.download_and_processing.glorys_download import (
    parse_date,
    lon_360_to_180,
    get_cmems_vars,
    get_monthly_ranges,
    download_glorys,
)


def test_lon_antimeridian():
    # 180 °E must map to −180 so CMEMS gets a valid western boundary
    assert lon_360_to_180(180.0) == pytest.approx(-180.0)

def test_lon_greater_than_180():
    assert lon_360_to_180(260.0) == pytest.approx(-100.0)

def test_lon_less_than_180_unchanged():
    assert lon_360_to_180(100.0) == pytest.approx(100.0)


def test_parse_date_format():
    assert parse_date("09-01-2012") == date(2012, 9, 1)

def test_parse_date_wrong_format_raises():
    with pytest.raises(ValueError):
        parse_date("2012-09-01")


def test_get_cmems_vars_mapping():
    assert get_cmems_vars(["U", "V", "SSH"]) == ["uo", "vo", "zos"]


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

def test_monthly_ranges_target_period():
    # Sep 2012 – Jun 2013 = 10 months
    ranges = get_monthly_ranges(date(2012, 9, 1), date(2013, 6, 30))
    assert len(ranges) == 10
    assert ranges[0] == (date(2012, 9, 1), date(2012, 9, 30))
    assert ranges[-1] == (date(2013, 6, 1), date(2013, 6, 30))

def test_monthly_ranges_leap_february():
    ranges = get_monthly_ranges(date(2012, 2, 1), date(2012, 2, 29))
    assert ranges == [(date(2012, 2, 1), date(2012, 2, 29))]


# --- download_glorys (copernicusmarine.subset mocked) ---

@pytest.fixture
def mock_cm():
    with patch("glorys_download.copernicusmarine") as m:
        m.subset.return_value = MagicMock()
        yield m

def _base_call(tmp_path, mock_cm, **overrides):
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
    return download_glorys(**kwargs)

def test_one_call_per_month(tmp_path, mock_cm):
    _base_call(tmp_path, mock_cm, time_end="10-31-2012")
    assert mock_cm.subset.call_count == 2

def test_longitude_converted(tmp_path, mock_cm):
    _base_call(tmp_path, mock_cm)
    kw = mock_cm.subset.call_args.kwargs
    assert kw["minimum_longitude"] == pytest.approx(-180.0)
    assert kw["maximum_longitude"] == pytest.approx(-100.0)

def test_depth_converted_to_positive(tmp_path, mock_cm):
    _base_call(tmp_path, mock_cm)
    kw = mock_cm.subset.call_args.kwargs
    assert kw["minimum_depth"] == pytest.approx(0.0)
    assert kw["maximum_depth"] == pytest.approx(500.0)

def test_no_depth_for_ssh_only(tmp_path, mock_cm):
    _base_call(tmp_path, mock_cm, variables=["SSH"])
    kw = mock_cm.subset.call_args.kwargs
    assert "minimum_depth" not in kw and "maximum_depth" not in kw

def test_variables_mapped_to_cmems_names(tmp_path, mock_cm):
    _base_call(tmp_path, mock_cm)
    assert mock_cm.subset.call_args.kwargs["variables"] == ["uo", "vo", "zos"]

def test_file_naming_and_datetime_format(tmp_path, mock_cm):
    _base_call(tmp_path, mock_cm)
    kw = mock_cm.subset.call_args.kwargs
    assert kw["output_filename"] == "glorys_2012_09.nc"
    assert kw["start_datetime"] == "2012-09-01T00:00:00"
    assert kw["end_datetime"] == "2012-09-30T23:59:59"

def test_existing_file_skipped(tmp_path, mock_cm):
    (tmp_path / "glorys_2012_09.nc").touch()
    _base_call(tmp_path, mock_cm, time_end="10-31-2012")
    assert mock_cm.subset.call_count == 1
    assert mock_cm.subset.call_args.kwargs["output_filename"] == "glorys_2012_10.nc"

def test_returns_paths_including_skipped(tmp_path, mock_cm):
    (tmp_path / "glorys_2012_09.nc").touch()
    result = _base_call(tmp_path, mock_cm, time_end="10-31-2012")
    assert [p.name for p in result] == ["glorys_2012_09.nc", "glorys_2012_10.nc"]

def test_credentials_forwarded(tmp_path, mock_cm):
    _base_call(tmp_path, mock_cm, username="u", password="p")
    kw = mock_cm.subset.call_args.kwargs
    assert kw["username"] == "u" and kw["password"] == "p"

def test_credentials_omitted_by_default(tmp_path, mock_cm):
    _base_call(tmp_path, mock_cm)
    kw = mock_cm.subset.call_args.kwargs
    assert "username" not in kw and "password" not in kw
