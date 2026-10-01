# RunMiniGTAP — Mini CGE berbasis GTAP 12a dalam satu file HTML

Aplikasi web single-file (mirip RunGTAP) untuk simulasi CGE (Computable General Equilibrium)
berbasis data GTAP 12a agregat: 11 sektor x 16 region x 5 faktor.

## Fitur

- **Data (ViewHAR)** — telusuri header .har hasil agregasi + pilih tahun basedata
  (2004, 2007, 2011, 2014, 2017, 2019, 2023)
- **Agregasi** — penjelasan metode agregasi sektor (penjumlahan nilai USD), region, dan faktor
- **Closure & Shocks** — shock endowmen (acak tanah, tenaga kerja, modal, dsb.),
  produktivitas sektor, dan tarif impor (per sektor, tujuan, sumber)
- **Solve** — solver Armington mini-CGE (full employment, numeraire Upah Indonesia)
  + analisis sensitivitas elastisitas
- **Hasil** — dampak output, harga, upah, kesejahteraan (EV), dan emisi CO2
- **Banding** — bandingkan hasil antar-simulasi
- **Export GAMS** — file .gms berisi set, data, kalibrasi, dan shock
- Panduan langkah (stepper) untuk alur simulasi

## Cara pakai

Buka `rungtap.html` langsung di browser (Chrome/Edge). Tidak perlu server.

1. **Data** — pilih tahun basedata, telusuri header jika perlu
2. **Closure & Shocks** — tambah shock (mis. tarif 10% sektor FerMet di Indonesia)
3. **Solve** — klik "Jalankan Model"
4. **Hasil** — lihat dampak; **Banding** untuk membandingkan skenario

## Rebuild dari sumber

```bash
# 1. Agregasi basedata GTAP 12a (perlu folder extracted/ berisi .har per tahun)
python build_app_dataset.py

# 2. Ekstraksi emisi CO2 dari co2.har
python build_co2_dataset.py

# 3. Build HTML final
python build_rungtap.py   # -> rungtap.html
```

## Struktur

| File | Keterangan |
|---|---|
| `rungtap.html` | Aplikasi final (data ter-embed, ~730 KB) |
| `rungtap_template.html` | Template (placeholder `/*__DATA__*/null`) |
| `build_rungtap.py` | Suntik dataset JSON ke template |
| `build_app_dataset.py` | Agregasi basedata.har -> 11x16x5, 7 tahun |
| `build_co2_dataset.py` | Agregasi co2.har -> emisi per region/sektor |
| `app_dataset.json` | Dataset agregat multi-tahun |
| `harlib.py` | Pembaca file HAR GEMPACK (Python) |

## Basis data

GTAP 12a Database (Purdue University), agregat 11x16x5, tahun 2004-2023.
Emisi CO2 dari header emis_CO2_prod/emis_CO2_cons (co2.har).
