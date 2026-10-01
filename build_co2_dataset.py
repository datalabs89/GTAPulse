"""Aggregate co2.har emissions (5 fuel x 65 act x 163 reg) to the app's
11 sectors x 16 regions, and compute EV (welfare) inputs. Writes
co2_agg.json with the same sector/region naming as app_dataset.json.

Aggregation logic:
- Production emissions: MDF[f,a,r] = CO2 from using domestic fuel f in
  activity a, region r; MMF = same but from imported fuel. Sum over fuels,
  map act->app sector, reg->app region.
- Final-demand emissions: MDP/MMP (private consumption of dom/imp fuel).
- Per-sector emission coefficient (Mt CO2 per juta USD output) can be
  combined with model output changes to get emission changes.
Also maps the fuel set to app sectors for consistency reporting.
"""
import json
import numpy as np

d = np.load("co2_arrays.npz", allow_pickle=False)
meta = json.load(open("co2_meta.json"))

FUEL = meta["sets"]["FUEL"]    # 5 fuels: coa oil gas p_c gdt
ACTS = meta["sets"]["ACTS"]    # 65 activities
REGS = meta["sets"]["REG"]     # 163 regions

# app aggregation maps (must mirror build_app_dataset.py)
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
        SEC_OF[c] = sec
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
        REG_OF[c] = r
REGIONS = list(REG_MAP.keys())
missing_r = [r for r in REGS if r not in REG_OF]
for r in missing_r:
    REG_OF[r] = "RestofWorld"
REGIONS.append("RestofWorld")

missing_a = [a for a in ACTS if a not in SEC_OF]
print("acts not mapped:", missing_a)
missing_reg = [r for r in REGS if r not in REG_OF]
print("regs not mapped:", missing_reg)

SI = {s: i for i, s in enumerate(SECTORS)}
RI = {r: i for i, r in enumerate(REGIONS)}
a_idx = [SI[SEC_OF[a]] for a in ACTS]
r_idx = [RI[REG_OF[r]] for r in REGS]
nS, nR = len(SECTORS), len(REGIONS)


def agg_ar(arr2d):
    """(65,163) -> (nS,nR)."""
    out = np.zeros((nS, nR))
    Ai, Ri = np.meshgrid(a_idx, r_idx, indexing="ij")
    np.add.at(out, (Ai.ravel(), Ri.ravel()), arr2d.ravel())
    return out


# production emissions by fuel-using activity (Mt CO2)
MDF, MMF = d["MDF"], d["MMF"]           # (5 fuel, 65 act, 163 reg)
MDP, MMP = d["MDP"], d["MMP"]           # (5 fuel, 163 reg)

prodDom = MDF.sum(axis=0)               # (65,163) dom fuel use
prodImp = MMF.sum(axis=0)               # imported fuel use
prodTot = prodDom + prodImp
consDom = MDP.sum(axis=0)               # (163,) private, dom fuel
consImp = MMP.sum(axis=0)               # private, imported fuel
consTot = consDom + consImp

# consTot aggregation (163 -> nR)
ec = np.zeros(nR)
np.add.at(ec, r_idx, consTot)

emis = {
    "EMIS_PROD": agg_ar(prodTot).round(3).tolist(),
    "EMIS_CONS": [round(float(x), 3) for x in ec],
    "EMIS_PROD_DOM": agg_ar(prodDom).round(3).tolist(),
    "EMIS_PROD_IMP": agg_ar(prodImp).round(3).tolist(),
}

out = {
    "meta": {
        "source": "GTAP 12a co2.har (2023)",
        "unit": "Mt CO2",
        "sectors": SECTORS,
        "regions": REGIONS,
        "fuelSet": FUEL,
    },
    "data": emis,
}
with open("co2_dataset.json", "w") as f:
    json.dump(out, f)
size = len(json.dumps(out)) // 1024
print(f"co2_dataset.json: {size} KB; {nS} sectors x {nR} regions")
# sanity: top emitters
tot = np.array(emis["EMIS_PROD"]).sum(axis=0)
order = np.argsort(-tot)[:6]
for oi in order:
    print(f"  {REGIONS[oi]:12s} prod emis: {tot[oi]:9.1f} Mt CO2")
