"""Tests for the NYC TLC trace downloader.

The network call itself is injected so these tests never touch the network.
What is worth pinning is the URL construction, the month enumeration across a
year boundary, and the caching rule that stops a rerun re-fetching hundreds of
megabytes.
"""

import pytest

from src.data.download import months_range, tlc_url


class RecordingFetcher:
    """Stands in for the real HTTP download and records what was asked for."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, url: str, dest) -> None:
        self.calls.append(url)
        dest.write_bytes(b"parquet-bytes")


# --- URL construction -------------------------------------------------------


def test_url_matches_the_published_tlc_layout():
    assert tlc_url(2024, 1) == (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    )


def test_url_zero_pads_single_digit_months():
    assert tlc_url(2024, 3).endswith("yellow_tripdata_2024-03.parquet")
    assert tlc_url(2024, 12).endswith("yellow_tripdata_2024-12.parquet")


def test_url_supports_other_tlc_datasets():
    assert tlc_url(2024, 1, dataset="green").endswith(
        "green_tripdata_2024-01.parquet"
    )


def test_url_rejects_an_invalid_month():
    for bad_month in (0, 13):
        with pytest.raises(ValueError):
            tlc_url(2024, bad_month)


# --- month enumeration ------------------------------------------------------


def test_months_range_is_inclusive_of_both_ends():
    assert months_range("2024-01", "2024-03") == [(2024, 1), (2024, 2), (2024, 3)]


def test_months_range_crosses_a_year_boundary():
    assert months_range("2023-11", "2024-02") == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


def test_months_range_of_a_single_month():
    assert months_range("2024-01", "2024-01") == [(2024, 1)]


def test_months_range_rejects_a_reversed_interval():
    with pytest.raises(ValueError):
        months_range("2024-03", "2024-01")


# --- download and caching ---------------------------------------------------


def test_download_writes_the_file_and_returns_its_path(tmp_path):
    from src.data.download import download_month

    fetcher = RecordingFetcher()

    path = download_month(2024, 1, dest_dir=tmp_path, fetch=fetcher)

    assert path.exists()
    assert path.name == "yellow_tripdata_2024-01.parquet"
    assert len(fetcher.calls) == 1


def test_download_skips_a_file_that_is_already_present(tmp_path):
    # Reruns of the pipeline are routine; re-fetching ~50 MB per month each
    # time would make the one-command acceptance path needlessly slow.
    from src.data.download import download_month

    fetcher = RecordingFetcher()
    download_month(2024, 1, dest_dir=tmp_path, fetch=fetcher)

    download_month(2024, 1, dest_dir=tmp_path, fetch=fetcher)

    assert len(fetcher.calls) == 1


def test_download_refetches_when_forced(tmp_path):
    from src.data.download import download_month

    fetcher = RecordingFetcher()
    download_month(2024, 1, dest_dir=tmp_path, fetch=fetcher)

    download_month(2024, 1, dest_dir=tmp_path, fetch=fetcher, force=True)

    assert len(fetcher.calls) == 2


def test_download_creates_the_destination_directory(tmp_path):
    from src.data.download import download_month

    nested = tmp_path / "data" / "raw"

    path = download_month(2024, 1, dest_dir=nested, fetch=RecordingFetcher())

    assert path.exists()


def test_a_failed_download_does_not_leave_a_partial_file(tmp_path):
    # A half-written parquet that looks cached would poison every later run.
    from src.data.download import download_month

    def failing_fetch(url, dest):
        dest.write_bytes(b"partial")
        raise ConnectionError("network died mid-transfer")

    with pytest.raises(ConnectionError):
        download_month(2024, 1, dest_dir=tmp_path, fetch=failing_fetch)

    assert list(tmp_path.glob("*.parquet")) == []
