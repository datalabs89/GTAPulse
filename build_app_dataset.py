"""Build the RunGTAP-like app dataset from GTAP 12a basedata, all years.

Pipeline per year: basedata.har -> aggregate 65x163x8 -> 11 sectors x 16 regions
x 5 factors -> app_dataset.json with {"years": {year: data}} (embedded into the
HTML app at build time; user can pick the basedata year in the Data tab).
"""
import struct
import json
import numpy as np
from harlib import read_har

ROOT = "extracted/GTAP12a/GTAP/"
YEARS = ["2004", "2007", "2011", "2014", "2017", "2019", "2023"]

data_by_year = {}
for YEAR in YEARS:
    BASE = ROOT + YEAR + "/"
    SETS = {}
    for h in read_har(BASE + "sets.har"):
        SETS[h.name] = h.names
    flows = {}
    for h in read_har(BASE + "basedata.har"):
        if h.data is not None and h.name not in flows:
            flows[h.name] = h

    reg, comm, endw = SETS["REG"], SETS["COMM"], SETS["ENDW"]

SEC_MAP = {
    "AgriFood": ["pdr","wht","gro","v_f","osd","c_b","pfb","ocr","ctl","oap","rmk","wol","frs","fsh",
                 "cmt","omt","vol","mil","sgr","ofd","pcr"],
    "Energy": ["coa","oil","gas","g_n","g_c","omn"],
    "FerMet": ["i_s","nfm","nmm"],
    "ChemRub": ["chm","b_t","ppp","bph","rpp"],
    "Textile": ["tex","wap","lea"],
    "LightMfg": ["fmp","mvh","otn","ele","omf","lum"],
    "HeavyMfg": ["fmf","p_c","eeq","ome"],
    "UtilConst": ["ely","gdt","wtr","cns","dwe"],
    "TradeTrans": ["trd","otp","wtp","atp","cmn","whs","afs"],
    "FinBusSvc": ["ofi","isr","ins","obs","rsa"],
    "PubServ": ["osg","edu","hht","oxt","ros"],
}
SEC_OF = {}
for sec, lst in SEC_MAP.items():
    for c in lst:
        SEC_OF.setdefault(c, sec)
SECTORS = list(SEC_MAP.keys())

REG_MAP = {
    "Indonesia": ["idn"],
    "SEAsia": ["tha","vnm","mys","sgp","phl","lao","khm","brn","xmm","xse"],
    "China": ["chn","hkg","twn"],
    "EastAsia": ["jpn","kor","mng","xea"],
    "SouthAsia": ["ind","bgd","lka","pak","npl","afg","xsa"],
    "NAmerica": ["usa","can","mex","xna"],
    "LatinAmer": ["bra","arg","chl","col","per","ven","ecu","bol","ury","pry","xsm","cri","gtm","hnd","nic","pan","slv","xca","dom","hti","jam","pri","tto","xcb"],
    "EU27": ["aut","bel","bgr","hrv","cyp","cze","dnk","est","fin","fra","deu","hun","irl","ita","lva","ltu","lux","mlt","nld","pol","prt","svk","svn","esp","swe"],
    "UK": ["gbr"],
    "RestEurope": ["che","nor","isl","xer","rou","alb","srb","blr","ukr","xee"],
    "Russia": ["rus"],
    "CAsia": ["kaz","kgz","tjk","uzb","arm","aze","geo"],
    "MENA": ["tur","isr","sau","are","irn","irq","jor","kwt","omn","qat","bhr","lbn","pse","xws","egy","mar","dza","tun","xbg","xnf","xme","mau","sym"],
    "SSA": ["zaf","nga","gha","ken","eth","sen","tza","uga","cmr","cic","xac","zmb","zwe","moz","mdg","mwi","ago","bdm","nbl","bwa","nam","ben","bfa","civ","gin","mli","mrt","ner","tgo","xwf","caf","tcd","cog","cod","gnq","gab","stp","bdi","com","mus","rwa","sdn","xec","swz","mad"],
    "Oceania": ["aus","nzl","xoc"],
}
REG_OF = {}
for r, lst in REG_MAP.items():
    for c in lst:
        REG_OF.setdefault(c, r)
REGIONS = list(REG_MAP.keys())
for c in reg:
    if c not in REG_OF:
        REG_OF[c] = "RestofWorld"
REGIONS.append("RestofWorld")

FAC_ORDER = {  # ENDW order -> model factors
    "Land": "Land",
    "tech_aspros": "SklLab",
    "clerks": "UnskLab",
    "service_shop": "UnskLab",
    "off_mgr_pros": "SklLab",
    "ag_othlowsk": "UnskLab",
    "Capital": "Capital",
    "NatlRes": "NatRes",
}
FACTORS = ["Land", "UnskLab", "SklLab", "Capital", "NatRes"]

SI = {s: i for i, s in enumerate(SECTORS)}
RI = {r: i for i, r in enumerate(REGIONS)}
FI = {f: i for i, f in enumerate(FACTORS)}
nS, nR, nF = len(SECTORS), len(REGIONS), len(FACTORS)


def agg2_to(arr, c_idx, r_idx, out_shape):
    out = np.zeros(out_shape)
    Ci, Ri = np.meshgrid(c_idx, r_idx, indexing="ij")
    np.add.at(out, (Ci.ravel(), Ri.ravel()), arr.ravel())
    return out


def agg3_to(arr, a_map, b_map, c_map, out_shape):
    out = np.zeros(out_shape)
    Ai, Bi, Ci = np.meshgrid(a_map, b_map, c_map, indexing="ij")
    np.add.at(out, (Ai.ravel(), Bi.ravel(), Ci.ravel()), arr.ravel())
    return out


co2 = json.load(open("co2_dataset.json"))
data_by_year = {}
for YEAR in YEARS:
    BASE = ROOT + YEAR + "/"
    SETS = {}
    for h in read_har(BASE + "sets.har"):
        SETS[h.name] = h.names
    flows = {}
    for h in read_har(BASE + "basedata.har"):
        if h.data is not None and h.name not in flows:
            flows[h.name] = h

    reg, comm, endw = SETS["REG"], SETS["COMM"], SETS["ENDW"]
    c_idx = [SI[SEC_OF[c]] for c in comm]
    r_idx = [RI[REG_OF[r]] for r in reg]

    data = {}

    # factor payments (8,65,163) -> (5,nS,nR)
    evfb = flows["EVFB"].data
    fac_idx8 = [FI[FAC_ORDER[e]] for e in endw]
    fac_pay = np.zeros((nF, nS, nR))
    Fi8, Ci65, R163 = np.meshgrid(fac_idx8, c_idx, r_idx, indexing="ij")
    np.add.at(fac_pay, (Fi8.ravel(), Ci65.ravel(), R163.ravel()), evfb.ravel())
    data["EVFB"] = fac_pay.tolist()

    data["VDPP"] = agg2_to(flows["VDPP"].data, c_idx, r_idx, (nS, nR)).tolist()
    data["VDGB"] = agg2_to(flows["VDGB"].data, c_idx, r_idx, (nS, nR)).tolist()
    data["VDIB"] = agg2_to(flows["VDIB"].data, c_idx, r_idx, (nS, nR)).tolist()
    data["VXSB"] = agg3_to(flows["VXSB"].data, c_idx, r_idx, r_idx, (nS, nR, nR)).tolist()
    data["VMSB"] = agg3_to(flows["VMSB"].data, c_idx, r_idx, r_idx, (nS, nR, nR)).tolist()
    data["VDFB"] = agg3_to(flows["VDFB"].data, c_idx, c_idx, r_idx, (nS, nS, nR)).tolist()
    data["VMFB"] = agg3_to(flows["VMFB"].data, c_idx, c_idx, r_idx, (nS, nS, nR)).tolist()

    vst = flows["VST"].data
    marg = np.zeros((3, nR)); Mi, Ri2 = np.meshgrid(np.arange(3), r_idx, indexing="ij")
    np.add.at(marg, (Mi.ravel(), Ri2.ravel()), vst.ravel())
    data["VST"] = marg.tolist()

    popv = np.zeros(nR); np.add.at(popv, r_idx, flows["POP"].data)
    data["POP"] = popv.tolist()

    # emissions (CO2) — pre-aggregated from co2.har 2023, applied to all years
    # (emission factors evolve slowly; sectoral structure is per reference year)
    data.update(co2["data"])

    data_by_year[YEAR] = data
    print(f"{YEAR}: aggregated")

out = {
    "meta": {
        "source": "GTAP 12a Data Base (reference years 2004-2023)",
        "unit": "juta USD",
        "sectors": SECTORS,
        "regions": REGIONS,
        "factors": FACTORS,
        "margins": ["otp", "wtp", "atp"],
        "populationUnit": "juta jiwa",
        "years": YEARS,
        "defaultYear": "2023",
    },
    "years": data_by_year,
}
with open("app_dataset.json", "w") as f:
    json.dump(out, f)
print("app_dataset.json:", len(json.dumps(out)) // 1024, "KB;",
      f"{len(YEARS)} tahun x {nS} sektor x {nR} region x {nF} faktor")
