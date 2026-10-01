"""Extract emission arrays from GTAP12a co2.har into co2_arrays.npz.

co2.har layout (newer GEMPACK):
  header card:  '    REFULL <desc>'  (desc contains header name context)
  name record:  '    ' + int32(1) + int32(ndim) + 12-char name +
                ndim*int32(1) + ndim*12-char set names
  dim records:  '    ' + int32(1) + int32(count) + int32(count) +
                count*12-char element names  (one per dim, in order)
  per block:    meta rec '    ' + 14*int32 ([..., reg_lo, reg_hi, ...])
                followed by data rec '    ' + (fuel*act*nreg)*float32
Headers:
  MDF (FUEL,ACTS,REG) emissions from domestic product, current production
  MMF (FUEL,ACTS,REG) emissions from imported product, current production
  MDP (FUEL,REG)      emissions private consumption of domestic product
  MMP (FUEL,REG)      emissions private consumption of imported product
Units: Mt CO2. Data stored as float32, tiled C-order over (fuel, act, reg)
blocks; block boundaries come from the meta records (reg ranges).
"""
import json
import struct
import numpy as np

path = "extracted/GTAP12a/GTAP/2023/co2.har"
with open(path, "rb") as f:
    raw = f.read()

pos, n, recs = 0, len(raw), []
while pos + 4 <= n:
    (ln,) = struct.unpack_from("<i", raw, pos)
    pos += 4
    recs.append(raw[pos:pos + ln])
    pos += ln + 4


def parse_name_record(b):
    """'    ' + int32(1) + int32(ndim) + 12c name + ndim*int32 + ndim*12c sets."""
    body = b[8:]
    ndim = struct.unpack_from("<i", body, 4)[0]
    s = body[8:]
    name = s[0:12].decode("ascii", "replace").strip()
    off = 12 + ndim * 4
    sets = [s[off + k * 12:off + (k + 1) * 12].decode("ascii", "replace").strip()
            for k in range(ndim)]
    return name, sets, ndim


def parse_dim_record(b):
    """'    ' + int32(1) + int32(count) + int32(count) + count*12-char names."""
    body = b[8:]
    d1 = struct.unpack_from("<i", body, 0)[0]
    d2 = struct.unpack_from("<i", body, 4)[0]
    if d1 != d2:
        return None
    txt = body[8:].decode("ascii", "replace")
    names = [txt[k * 12:(k + 1) * 12].strip() for k in range(d2)]
    ok = all(s and all(c.isalnum() or c == "_" for c in s) and s == s.lower()
             for s in names)
    return names if ok else None


def is_meta_rec(b):
    if len(b) < 20 or b[:4] != b"    ":
        return False
    cnt = struct.unpack_from("<i", b, 4)[0]
    return cnt in (14, 12) and len(b[8:]) == cnt * 4


def parse_meta(b):
    cnt = struct.unpack_from("<i", b, 4)[0]
    ints = [struct.unpack_from("<i", b, 8 + 4 * k)[0] for k in range(cnt)]
    # observed layout: [1, nf, 1, na, reg_lo, reg_hi, 1, 1, ...]
    return ints


headers = []
i = 0
while i < len(recs):
    r = recs[i]
    if len(r) >= 10 and r[4:10] in (b"REFULL", b"RESPSE"):
        is_re = r[4:10] == b"REFULL"
        desc = r[10:].decode("ascii", "replace").strip()
        i += 1
        if not is_re:
            # RESPSE headers (gov/invest consumption): skip their records
            # until the next REFULL/1CFULL header
            while i < len(recs) and not (len(recs[i]) >= 10 and
                                         recs[i][4:10] in (b"REFULL", b"1CFULL")):
                i += 1
            continue
        if "GTAP data format" in desc:
            i += 1
            continue
        b = recs[i]
        name, sets, ndim = parse_name_record(b)
        i += 1
        dims = []
        while i < len(recs):
            nm = parse_dim_record(recs[i])
            if nm is None:
                break
            dims.append(nm)
            i += 1
        # blocks: meta record + float32 data record, until next header
        data_chunks = []
        block_ranges = []
        while i < len(recs):
            if len(recs[i]) >= 10 and recs[i][4:10] in (b"REFULL", b"1CFULL",
                                                        b"RESPSE"):
                break
            if is_meta_rec(recs[i]) and i + 1 < len(recs):
                ints = parse_meta(recs[i])
                block_ranges.append((ints[4], ints[5]))
                db = recs[i + 1]
                payload = db[8:]
                nf = len(payload) // 4
                data_chunks.append(np.frombuffer(payload[:nf * 4], dtype="<f4"))
                i += 2
                continue
            db = recs[i]
            payload = db[8:]
            # data record: payload large and not a dims/meta/name record
            if len(payload) > 64 and len(payload) % 4 == 0 and \
                    not is_meta_rec(db) and parse_dim_record(db) is None:
                nf = len(payload) // 4
                arr = np.frombuffer(payload[:nf * 4], dtype="<f4")
                data_chunks.append(arr)
                i += 1
                continue
            i += 1
        print(f"{name:4s} ndim={ndim} dims={[len(d) for d in dims]} "
              f"blocks={len(data_chunks)} ranges={block_ranges[:3]}...", flush=True)
        headers.append({"name": name, "desc": desc, "sets": sets, "dims": dims,
                        "chunks": data_chunks, "ranges": block_ranges})
    else:
        i += 1

out = {}
meta = {}
for h in headers:
    if not h["chunks"]:
        continue
    sizes = [len(d) for d in h["dims"]]
    total = int(np.prod(sizes))
    arr = np.concatenate(h["chunks"])
    assert arr.size == total, (h["name"], arr.size, total)
    # blocks tile over reg ranges (REG outermost), within block C-order
    # (act, fuel) — verified empirically: only this order yields plausible
    # regional totals (chn/usa/ind/rus top)
    a = arr.reshape(tuple(reversed(sizes))).transpose(
        tuple(range(len(sizes) - 1, -1, -1))).astype(np.float64)
    out[h["name"]] = a
    meta[h["name"]] = {"sets": h["sets"], "dims": sizes,
                       "desc": h["desc"]}
    print(f'  {h["name"]}: shape={a.shape} sum={a.sum():.1f} Mt CO2 '
          f'(sets {h["sets"]})')

# element names (dims) for app use — take the LONGEST dim record per set
# (MDF sets list is garbled: trailing junk token; dims order = FUEL,ACTS,REG)
setnames = {}
for h in headers:
    if not h["dims"]:
        continue
    if len(h["dims"]) == 3:
        cand = {"FUEL": h["dims"][0], "ACTS": h["dims"][1], "REG": h["dims"][2]}
    else:
        cand = {"FUEL": h["dims"][0], "REG": h["dims"][1]}
    for k, v in cand.items():
        if k not in setnames or len(v) > len(setnames[k]):
            setnames[k] = v
meta["sets"] = setnames
print("sets:", {k: len(v) for k, v in setnames.items()})

np.savez_compressed("co2_arrays.npz", **out)
with open("co2_meta.json", "w") as f:
    json.dump(meta, f, indent=1)
print("saved co2_arrays.npz, co2_meta.json")
