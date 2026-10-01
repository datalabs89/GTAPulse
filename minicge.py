"""MiniCGE solver prototype - global Armington CGE (teaching model).

Standard CGE closure: full employment. Given wages wf:
  1. Cost pricing: ps[s,r] = shVA[s,r]*cva + sum_i aic[i,s,r]*parm[i,r]  (unit cost / productivity)
  2. Income: inc[r] = wf.Endow + tariff revenue + transfers (transf pins base income to base final demand)
  3. CD final demand: qFin = shFin * inc / parm
  4. Leontief intermediates: DI[s,r] = sum_u aic[s,u,r] * Y[u,r]
  5. Armington split of composite demand qTot = DI + qFin into domestic / import variety
  6. Export demand: expTarget * (ps/pwX)^(-SIG_X)
  7. Output: Y = qDom + expD  -- solved EXACTLY per region as a linear system
     Y[s] = sd[s]*pwrD[s]*(sum_u aic[s,u]*Y[u] + qFin[s]) + expD[s]   (contraction, rho(A)<0.67)
  8. Outer loop: factor demand from CES vs endowments -> wf update (numeraire: Land IDN)

Base replication is EXACT by construction:
  - cost = shVA + colsum(aic) = 1  => ps = 1
  - inc = Endow.sum + (finTot - Endow.sum) = finTot => qFin = finD_total (incl. imported finals)
  - CA0 = INTsum + finTot;  shDom*CA0 = domIn + finD (domestic variety), shImp*CA0 = Mtot0
    (exact because INTsum - impInt = domIn and impFin = Mtot0 - impInt)
  - expTarget = Y0 - (domIn + finD)  => Y = Y0
  - factor demand = thetaF * VA = EVFB = endowments
"""
import json
import numpy as np

d = json.load(open("app_dataset.json"))
data = d["data"]; meta = d["meta"]
S = meta["sectors"]; R = meta["regions"]; F = meta["factors"]
nS, nR, nF = len(S), len(R), len(F)

EVFB = np.array(data["EVFB"])          # (F, S, R) factor payments
VXSB = np.array(data["VXSB"])          # (S, src, dst)
VMSB = np.array(data["VMSB"])          # (S, src, dst)
VDFB = np.array(data["VDFB"])          # (S_in, S_user, R) domestic intermediates
VMFB = np.array(data["VMFB"])          # (S_in, S_user, R) imported intermediates
VDPP = np.array(data["VDPP"]); VDGB = np.array(data["VDGB"]); VDIB = np.array(data["VDIB"])

SI = {s: i for i, s in enumerate(S)}
RI = {r: i for i, r in enumerate(R)}
FI = {f: i for i, f in enumerate(F)}
idx = np.arange(nR)

# ---- base quantities (all prices = 1 at base) ----
VA = EVFB.sum(axis=0)                  # (S,R) value added per sector
INT = VDFB + VMFB                      # (S_in, S_user, R) composite intermediates
INTU = INT.sum(axis=0)                 # (S_user,R) total intermediate input of user
Y0 = VA + INTU                         # (S,R) base gross output (cost side)

INTsum = INT.sum(axis=1)               # (S_in,R) composite good used as intermediates
impInt = VMFB.sum(axis=1)              # (S,R) imported intermediates
domIn = VDFB.sum(axis=1)               # (S,R) domestic-variety intermediates

Mtot0 = np.transpose(VMSB, (0, 2, 1)).sum(axis=2)   # (S,dst) total imports
impFin = np.maximum(Mtot0 - impInt, 0.0)            # (S,R) imported final demand
finD = VDPP + VDGB + VDIB              # (S,R) domestic final demand
finTot = finD + impFin                 # (S,R) TOTAL final demand (composite)

DD0 = domIn + finD                     # (S,R) base demand for domestic variety
CA0 = INTsum + finTot                  # (S,R) base composite absorption
# SAM residuals: in some cells domestic-variety absorption exceeds output
# (services: margins/valuation). Cap domestic sales at output; the gap
# flows into imports so shares still sum to exactly 1.
qDom0 = np.minimum(DD0, Y0)            # (S,R) base sales of domestic variety
shDom = np.where(CA0 > 1e-9, qDom0 / np.maximum(CA0, 1e-9), 0.5)
shImp = 1.0 - shDom

expTarget = np.maximum(Y0 - qDom0, 0.0)  # (S,R) export target (residual)

Xrow0 = VXSB.sum(axis=2) - VXSB[:, idx, idx]        # (S,src) exports fob
VMSBt = np.transpose(VMSB, (0, 2, 1))               # (S, dst, src)
thetaMn = np.where(Mtot0[:, :, None] > 1e-12,
                   VMSBt / np.maximum(Mtot0[:, :, None], 1e-12), 0.0)

aic = np.where(Y0 > 1e-9, INT / np.maximum(Y0[None, :, :], 1e-9), 0.0)   # (in, user, R)
shVA = np.where(Y0 > 1e-9, VA / np.maximum(Y0, 1e-9), 0.0)
EndowR = EVFB.sum(axis=1)              # (F,R)
thetaF = np.where(EVFB.sum(axis=0)[:, :, None] > 1e-9,
                  np.transpose(EVFB, (1, 2, 0)) /
                  np.maximum(EVFB.sum(axis=0)[:, :, None], 1e-9), 0.0)
shFin = np.where(finTot.sum(axis=0) > 1e-9,
                 finTot / np.maximum(finTot.sum(axis=0)[None, :], 1e-9), 0.0)

SIG_ARM = 2.0   # Armington dom/import
SIG_X = 4.0     # export demand
SIG_SRC = 4.0   # import source substitution
SIG_VA = 0.8    # factor substitution in VA


def ces_index(shares, prices, sigma):
    s1 = 1.0 - sigma
    p = np.maximum(prices, 1e-12)
    if abs(s1) < 1e-9:
        return np.exp(np.sum(shares * np.log(p), axis=-1))
    z = np.sum(shares * np.power(p, s1), axis=-1)
    return np.power(np.maximum(z, 1e-12), 1.0 / s1)


shockEndow = np.zeros((nF, nR)); shockEndow[FI["SklLab"], RI["Indonesia"]] = 0.10
shockProd = np.zeros((nS, nR))
shockProd[SI["LightMfg"], RI["Indonesia"]] = 0.02
shockProd[SI["HeavyMfg"], RI["Indonesia"]] = 0.02
shockTariff = np.zeros((nS, nR, nR))   # (S, dst, src)


def solve(maxiter=800, damp=0.3, tol=1e-6, verbose=True):
    wf = np.ones((nF, nR))
    ps = np.ones((nS, nR))
    Yq = Y0.copy()
    Ynew = np.zeros_like(Yq)
    EndowEff = EndowR * (1.0 + shockEndow)
    aprod = 1.0 + shockProd
    num = FI["Land"], RI["Indonesia"]
    mQty0 = Mtot0[:, :, None] * thetaMn          # (S,dst,src) base import quantities
    transf = finTot.sum(axis=0) - EndowR.sum(axis=0)

    for it in range(maxiter):
        for inner in range(300):
            wfc = np.broadcast_to(wf.T[None, :, :], (nS, nR, nF))
            cva = ces_index(thetaF, wfc, SIG_VA)
            pms = ps[:, None, :] * (1.0 + shockTariff)
            pimp = ces_index(thetaMn, pms, SIG_SRC)
            parm = shDom * ps + shImp * pimp
            cost = shVA * cva + np.sum(aic * parm[:, None, :], axis=0)
            ps_new = np.where(Y0 > 1e-9, cost / aprod, 1.0)
            e_ps = float(np.max(np.abs(ps_new - ps) / np.maximum(ps, 1e-9)))
            ps = ps + 0.4 * (ps_new - ps)

            tarifRev = np.sum(mQty0 * pms * (shockTariff /
                              np.maximum(1.0 + shockTariff, 1e-9)), axis=(0, 2))
            inc = np.sum(wf * EndowR, axis=0) + tarifRev + transf
            qFin = (shFin * inc[None, :]) / np.maximum(parm, 1e-9)

            pwrD = np.power(np.maximum(ps, 1e-12) / np.maximum(parm, 1e-12), -SIG_ARM)
            pwX = np.sum(Xrow0 * ps, axis=1) / np.maximum(Xrow0.sum(axis=1), 1e-12)
            pwrX = np.power(np.maximum(ps, 1e-12) / np.maximum(pwX, 1e-12)[:, None], -SIG_X)
            expD = expTarget * pwrX
            # exact linear solve per region:
            # Y[s] = sd[s]*pwrD[s]*(sum_u aic[s,u]*Y[u] + qFin[s]) + expD[s]
            for r in range(nR):
                sd = shDom[:, r] * pwrD[:, r]                      # (S,) row scaling
                Arow = sd[:, None] * aic[:, :, r]                  # (in, user)
                M = np.eye(nS) - Arow
                Ynew[:, r] = np.linalg.solve(M, sd * qFin[:, r] + expD[:, r])
            e_Y = float(np.max(np.abs(Ynew - Yq) / np.maximum(Yq, 1e-9)))
            Yq = Ynew.copy()
            if e_ps < 1e-10 and e_Y < 1e-10:
                break

        # factor market clearing
        wfc = np.broadcast_to(wf.T[None, :, :], (nS, nR, nF))
        cva = ces_index(thetaF, wfc, SIG_VA)
        VAq = shVA * Yq / aprod
        XF = thetaF * VAq[:, :, None] * np.power(cva[:, :, None] / np.maximum(wfc, 1e-12), SIG_VA)
        FD = XF.sum(axis=0).T
        exc = (FD - EndowEff) / np.maximum(EndowEff, 1e-9)
        eF = float(np.max(np.abs(exc[EndowEff > 1e-6])))
        wf = wf * np.exp(damp * np.clip(exc, -0.5, 0.5))
        wf = wf / wf[num]
        if verbose and it % 100 == 0:
            print(f"it {it}: eF={eF:.2e} (inner {inner})")
        if eF < tol:
            if verbose:
                print(f"converged it {it}: eF={eF:.2e}")
            break
    return wf, Yq, ps


def verify_base():
    global shockEndow, shockProd
    sE, sP = shockEndow.copy(), shockProd.copy()
    shockEndow = np.zeros_like(sE); shockProd = np.zeros_like(sP)
    wf, Yq, ps = solve(maxiter=2000, verbose=False)
    shockEndow = sE; shockProd = sP
    print("base replication: max |dY/Y| =", f"{float(np.max(np.abs(Yq/Y0-1))):.2e}",
          " max |dw| =", f"{float(np.max(np.abs(wf-1))):.2e}",
          " max |dps| =", f"{float(np.max(np.abs(ps-1))):.2e}")


if __name__ == "__main__":
    verify_base()
    wf, Yq, ps = solve()
    i = RI["Indonesia"]
    print("\nResults (shock: IDN SklLab +10%, prod LightMfg/HeavyMfg +2%):")
    for f in F:
        print(f"  {f:8s} wage {(wf[FI[f], i]-1)*100:+7.2f}%")
    for s in S:
        j = SI[s]
        print(f"  {s:10s} output {(Yq[j, i]/Y0[j, i]-1)*100:+7.2f}%")
    print(f"  real GDP proxy {(Yq[:, i].sum()/Y0[:, i].sum()-1)*100:+.2f}%")
