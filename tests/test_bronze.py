"""Tests for ingest.bronze (pure helpers; no Spark session required)."""

from __future__ import annotations

from ingest.bronze import CAST_MAP, EVENT_DATE_COL, INGEST_TS_COL, RENAME_MAP, snake_case


def test_rename_map() -> None:
    assert RENAME_MAP["VendorID"] == "vendor_id"
    assert RENAME_MAP["tpep_pickup_datetime"] == "pickup_datetime"
    assert RENAME_MAP["tpep_dropoff_datetime"] == "dropoff_datetime"
    assert RENAME_MAP["RatecodeID"] == "rate_code_id"
    assert RENAME_MAP["PULocationID"] == "pu_location_id"
    assert RENAME_MAP["DOLocationID"] == "do_location_id"


def test_cast_map_covers_core_columns() -> None:
    for name in ("pickup_datetime", "dropoff_datetime", "trip_distance", "total_amount"):
        assert name in CAST_MAP
    assert CAST_MAP["pickup_datetime"] == "timestamp"
    assert CAST_MAP["trip_distance"] == "double"
    assert CAST_MAP["vendor_id"] == "long"


def test_snake_case() -> None:
    assert snake_case("PULocationID") == "pulocation_id"
    assert snake_case("VendorID") == "vendor_id"
    assert snake_case("trip_distance") == "trip_distance"
    assert snake_case("camelCaseCol") == "camel_case_col"


def test_derived_column_names() -> None:
    assert INGEST_TS_COL == "ingest_ts"
    assert EVENT_DATE_COL == "event_date"
