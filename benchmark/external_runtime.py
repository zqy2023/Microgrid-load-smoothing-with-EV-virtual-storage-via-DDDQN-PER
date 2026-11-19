"""Prepare the historical SGSC/AEMO sensitivity windows from an explicit input."""
import copy
import hashlib

import numpy as np
import pandas as pd

from scenario_runtime import checksum

SHA256 = "2037f235cd3e02b041a7ae7f17283d99a60597e9b14e3b34559f98e1ffa91ff8"
SEASONS = {
    "autumn": ("2013-03-01", "2013-06-01"),
    "winter": ("2013-06-01", "2013-09-01"),
    "spring": ("2013-09-01", "2013-12-01"),
    "summer": ("2013-12-01", "2014-03-01"),
}


def prepared_cases(archive, path):
    if checksum(path) != SHA256:
        raise ValueError("The external study requires the SHA-bound aligned SGSC/AEMO table.")
    external = pd.read_csv(path, parse_dates=["timestamp"], float_precision="round_trip")
    assert list(external.columns) == ["timestamp", "load_kw", "price_aud_mwh"]
    assert not external.isnull().any().any() and len(external) == 40704
    index = pd.DatetimeIndex(external["timestamp"])
    assert index.equals(pd.date_range("2013-01-01", periods=len(external), freq="15min"))
    measured = external[["load_kw", "price_aud_mwh"]].to_numpy(np.float64)
    assert np.isfinite(measured).all() and np.array_equal(measured[0::2], measured[1::2])
    cases = {}
    for season, (start, nominal_end) in SEASONS.items():
        first, stop = map(int, index.searchsorted([pd.Timestamp(start), pd.Timestamp(nominal_end)]))
        available_prefix = min(stop, len(external) - 5) - first
        size = (available_prefix // 96) * 96
        assert size > 0
        extended = size + 5
        selected = slice(first, first + extended)
        data = copy.copy(archive)
        data.timestamps = index[selected].to_numpy(copy=True)
        data.price = measured[selected, 1].copy()
        data.load = measured[selected, 0].copy() / 300.0
        data.ev = archive.ev[:extended]
        assert data.train_stats is archive.train_stats
        assert data.timestamps[0] == np.datetime64(start)
        row = first + size
        cases[season] = (data, size, dict(
            season=season,
            start=start,
            end_exclusive=str(index[row]),
            nominal_end_exclusive=nominal_end,
            end_trimmed_rows=stop-row,
            evaluation_rows=size,
            complete_days=size // 96,
            liveness_rows=5,
            external_rows=[first, row],
            ev_proxy_archive_rows=[0, size],
            ev_proxy_archive_first_timestamp=str(archive.timestamps[0]),
            source_load_mean_kw=float(data.load[:size].mean()),
            source_price_mean_aud_mwh=float(data.price[:size].mean()),
        ))
    return cases


def trace_binding(trace):
    digest = hashlib.sha256()
    for key, value in sorted(trace.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, "<i8").tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()
