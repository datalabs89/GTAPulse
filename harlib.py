"""GEMPACK HAR reader for GTAP 12a — verified against basedata 2023.

Layout (learned via hex analysis):
  Record framing: 4-byte len (or -1, 4-byte len, 8-byte pad), payload, 4-byte len.
  All records begin with 4 spaces.
  Header group = [4-char name] + [header card] + [body records...]:
    - header card: '    ' + htype(6) + coords(2) + desc(56) + trailing ints.
      For sets (1CFULL): tail = (nRec?, nElem, elemWidth).
    - set body: '    ' + (1, nElem, width) + fixed-width element names
    - FULL body: set-label records, element-name records (counter 1,1,1),
      dims record ('    ' + 23 + (7, d1, d2, ..., trailing 1s)),
      then alternating colhdr (even counter, gives element range) and
      data records (odd counter): '    ' + counter + float32 values.
  Element storage order: first-dim fastest? Verified: reshape order='F'
  for (8, 65, 163) EVFB gives correct region totals; blocks = 65*... no,
  blocks were ~7800 f32; order='F' validated against known magnitudes.
"""
import struct
import numpy as np


def read_records(path):
    recs = []
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b or len(b) < 4:
                break
            (n1,) = struct.unpack("<i", b)
            if n1 == -1:
                (ln,) = struct.unpack("<i", f.read(4))
                f.read(8)
            else:
                ln = n1
            data = f.read(ln)
            f.read(4)
            recs.append(data)
    return recs


def _is_name_rec(r):
    return len(r) == 4 and not r.startswith(b"    ")


class Header:
    def __init__(self, name, desc, htype):
        self.name = name
        self.desc = desc
        self.htype = htype
        self.dims = []
        self.names = []
        self.data = None

    def __repr__(self):
        shape = getattr(self.data, "shape", None)
        return f"<{self.name} {self.htype} dims={self.dims} shape={shape}>"


def read_har(path):
    recs = read_records(path)
    headers = []
    i = 0
    n = len(recs)
    while i < n:
        r = recs[i]
        if _is_name_rec(r):
            name = r.decode("ascii", "replace").strip()
            card = recs[i + 1]
            htype = card[4:10].decode("ascii", "replace").strip()
            desc = card[10:72].decode("ascii", "replace").strip()
            h = Header(name, desc, htype)
            j = i + 2
            k = j
            while k < n and not _is_name_rec(recs[k]):
                k += 1
            body = recs[j:k]
            if htype.startswith("1C"):
                try:
                    _, nElem, width = struct.unpack("<3i", card[-12:])
                except struct.error:
                    nElem, width = 0, 12
                h.dims = [nElem]
                for rr in body:
                    if len(rr) >= 16:
                        c0 = struct.unpack("<i", rr[4:8])[0]
                        c1 = struct.unpack("<i", rr[8:12])[0]
                        c2 = struct.unpack("<i", rr[12:16])[0] if len(rr) >= 16 else 0
                        if c0 == 1 and c1 == nElem and c2 == nElem and len(rr) - 16 >= nElem * width:
                            txt = rr[16:16 + nElem * width].decode("ascii", "replace")
                            h.names = [txt[m:m + width].strip()
                                       for m in range(0, nElem * width, width)]
                            break
                i = k
            else:
                # dims record: counter == max (5, 23, ...), ints[0] == 7
                dims = None
                for rr in body:
                    if len(rr) >= 16:
                        ints = [struct.unpack("<i", rr[m:m + 4])[0]
                                for m in range(8, len(rr) - 3, 4)]
                        if ints and ints[0] == 7 and len(ints) > 2:
                            cand = ints[1:]
                            trail = 0
                            for v in reversed(cand):
                                if v == 1:
                                    trail += 1
                                else:
                                    break
                            dims = cand[:len(cand) - trail]
                            break
                h.dims = dims or []
                # data records: odd counter, after the dims record's position.
                # Find dims record index first.
                dims_idx = -1
                for bi, rr in enumerate(body):
                    if len(rr) >= 16:
                        ints = [struct.unpack("<i", rr[m:m + 4])[0]
                                for m in range(8, len(rr) - 3, 4)]
                        if ints and ints[0] == 7:
                            dims_idx = bi
                            break
                vals = []
                for rr in body[dims_idx + 1 if dims_idx >= 0 else 0:]:
                    if len(rr) < 16:
                        continue
                    c = struct.unpack("<i", rr[4:8])[0]
                    if c % 2 != 1:
                        continue
                    n32 = (len(rr) - 8) // 4
                    if n32 <= 0 or (len(rr) - 8) % 4 != 0:
                        continue
                    vals.append(np.frombuffer(rr[8:8 + n32 * 4], dtype="<f4").astype(np.float64))
                data = np.concatenate(vals) if vals else np.array([])
                if data.size and dims and int(np.prod(dims)) == data.size:
                    h.data = data.reshape(dims, order="F")
                elif data.size:
                    h.data = data
                i = k
            headers.append(h)
        else:
            i += 1
    return headers


def load(path):
    return read_har(path)
