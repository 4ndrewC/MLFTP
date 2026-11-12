"""Read Spark Engine frame logs (research/frametime/DATA_FORMAT.md) with numpy.

    from framelog import read_log
    header, rec = read_log("research/frametime/data/sessions/s0001.sfl")
    rec["t_total_us"], rec["n_steps"], header["session"]["tags"]

The field table in each file's JSON header drives the numpy dtype, so this
reader needs no changes when fields are added (the format version changes).
"""
import json
import struct

import numpy as np

MAGICS = {b"SPKFLOG\0": "framelog", b"SPKFEAT\0": "features"}
SUPPORTED_VERSIONS = (1, 2)  # v2 = v1 layout with reserved0 -> q_aa and a fingerprint in the header
SUPPORTED_VERSION = 2
_NP = {"u8": "<u1", "u16": "<u2", "u32": "<u4", "f32": "<f4"}


def read_table(path):
    """Return (magic, version, record_size, header_dict, raw_rows_bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24 or data[:8] not in MAGICS:
        raise ValueError(f"{path}: not a Spark table file")
    version, rec_size, hlen, data_off = struct.unpack_from("<4I", data, 8)
    header = json.loads(data[24:24 + hlen].decode("utf-8"))
    n = (len(data) - data_off) // rec_size  # a trailing partial row (crash) is ignored
    return MAGICS[data[:8]], version, rec_size, header, data[data_off:data_off + n * rec_size]


def frame_dtype(header):
    fields = header["fields"]
    return np.dtype({"names": [f["name"] for f in fields],
                     "formats": [_NP[f["type"]] for f in fields],
                     "offsets": [f["offset"] for f in fields],
                     "itemsize": header["record_size"]})


def read_log(path):
    """Return (header, structured numpy array of FrameRecords)."""
    kind, version, rec_size, header, rows = read_table(path)
    if kind != "framelog":
        raise ValueError(f"{path}: not a frame log")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"{path}: frame log version {version}, reader supports {SUPPORTED_VERSIONS}")
    dt = frame_dtype(header)
    assert dt.itemsize == rec_size
    return header, np.frombuffer(rows, dtype=dt)


def read_features(path):
    """Return (header, float32 array [frames, n_features]) from a --log-features dump."""
    kind, version, rec_size, header, rows = read_table(path)
    if kind != "features":
        raise ValueError(f"{path}: not a feature dump")
    n = len(header["features"])
    assert rec_size == 4 * n
    return header, np.frombuffer(rows, dtype="<f4").reshape(-1, n)
