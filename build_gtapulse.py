"""Build GTAPulse.html: single-file RunGTAP-like app with GTAP 12a 2023 dataset injected.

Reads app_dataset.json, rounds values, embeds as JSON inside the HTML template.
Never requires the JSON to be opened by hand.
"""
import json
import numpy as np

d = json.load(open("app_dataset.json"))
meta = d["meta"]

# round to 3 decimals to shrink payload (unit: juta USD)
def rnd(a, nd=3):
    return np.round(np.array(a, dtype=float), nd).tolist()

years = {}
for yr, data in d["years"].items():
    years[yr] = {k: rnd(v) for k, v in data.items()}

payload = json.dumps({"meta": meta, "years": years}, separators=(",", ":"))
print("payload:", len(payload) // 1024, "KB")

with open("GTAPulse_template.html", "r", encoding="utf-8") as f:
    html = f.read()

if "/*__DATA__*/null" not in html:
    raise SystemExit("placeholder /*__DATA__*/null not found in template")
html = html.replace("/*__DATA__*/null", payload, 1)

with open("GTAPulse.html", "w", encoding="utf-8") as f:
    f.write(html)
print("written GTAPulse.html:", len(html) // 1024, "KB")
