"""Minimal GEMPACK HAR reader — enough for GTAP basedata inspection."""
import struct
import numpy as np


def _read_records(f):
    recs = []
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


def _parse_header(data):
    h = {}
    h["name"] = data[0:10].decode("ascii", "replace").strip()
    h["long_name"] = data[10:42].decode("ascii", "replace").strip()
    h["type"] = data[50:53].decode("ascii", "replace").strip().upper()
    date = data[53:72].decode("ascii", "replace").strip()
    h["date"] = date
    return h


def _parse_dim(data):
    # one int per element, 4-byte each
    n = len(data) // 4
    return list(struct.unpack("<%di" % n, data))


class Header:
    def __init__(self, name, long_name, htype, date):
        self.name = name
        self.long_name = long_name
        self.type = htype  # RE, 1C, 2I, FULL etc.
        self.date = date
        self.dims = []
        self.names = []  # set element names (for FULL headers: first dim element names)
        self.sets = []  # set names per dim
        self.data = None
        self.records = []  # raw text records for 1C headers

    def __repr__(self):
        shape = getattr(self.data, "shape", None)
        return f"<HAR {self.name} type={self.type} shape={shape} sets={self.sets}>"


def read_har(path, verbose=False):
    with open(path, "rb") as f:
        recs = _read_records(f)
    headers = []
    i = 0
    cur = None
    while i < len(recs):
        r = recs[i]
        if len(r) >= 72 and r[10:14] in (b"1CFU", b"2IFU", b"REFU", b"REEM", b"1CSE", b"2DRE", b"1CDS"):
            h = _parse_header(r)
            cur = Header(h["name"], h["long_name"], h["type"], h["date"])
            headers.append(cur)
            i += 1
            # next record: dimension info
            if i < len(recs):
                d = recs[i]
                if len(d) % 4 == 0 and len(d) > 0:
                    cur.dims = _parse_dim(d)
                    i += 1
            # remaining records until next header: element names / data
            body = []
            while i < len(recs):
                nr = recs[i]
                if len(nr) >= 72 and nr[10:14] in (b"1CFU", b"2IFU", b"REFU", b"REEM", b"1CSE", b"2DRE", b"1CDS"):
                    break
                body.append(nr)
                i += 1
            if cur.type in ("1C", "2I"):  # set header: body is element name records
                for b_ in body:
                    txt = b_.decode("ascii", "replace")
                    cur.names.extend([x.strip() for x in txt.split() if x.strip()])
            elif body:
                # FULL/RE: dims record then float data records
                arr = np.frombuffer(b"".join(body), dtype="<f8")
                cur.data = arr
        else:
            i += 1
    return headers


def header_dict(headers):
    return {h.name.upper(): h for h in headers}


if __name__ == "__main__":
    import sys

    hs = read_har(sys.argv[1])
    for h in hs:
        print(h, "names[:5]=", h.names[:5] if h.names else None)
