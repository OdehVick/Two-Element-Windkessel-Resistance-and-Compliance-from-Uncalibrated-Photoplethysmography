"""
phe_discover.py — step 1 of the phenylephrine timing fix.

Problem
-------
The current analysis takes a 120 s window from the MIDDLE of each case and
relates the resulting reference resistance to whether phenylephrine was given
at any point during surgery. If the drug was administered before that window,
the reference resistance is measured under its effect. The positive control in
section 3.8 would then be circular: phenylephrine raises resistance, resistance
is measured afterwards, and resistance "predicts" phenylephrine.

Fix
---
A landmark design. The analysis window must precede the first dose, and the
outcome must be administration within a defined horizon after the window.

This script establishes what is available before any analysis is written:
  1. which infusion tracks encode phenylephrine
  2. how many EXP-B cases carry them
  3. the distribution of first-dose times
  4. how many cases have enough drug-free recording before the first dose

Run:
    !python phe_discover.py
"""

import numpy as np
import pandas as pd

TRKS = "https://api.vitaldb.net/trks"
CASES = "https://api.vitaldb.net/cases"
CO_TRACKS = ["EV1000/CO", "Vigileo/CO", "CardioQ/CO"]

# window requirements, matching the main analysis
WIN_S = 120.0          # analysis window length
GAP_S = 60.0           # gap between window end and first dose
SKIP_START_S = 600.0   # ignore induction
FS = 500.0


def main():
    trks = pd.read_csv(TRKS)
    names = sorted(trks.tname.unique())

    print("=" * 74)
    print("1. CANDIDATE PHENYLEPHRINE TRACKS")
    print("=" * 74)
    cand = [n for n in names
            if any(k in n.upper() for k in ("PHEN", "PHE_", "/PHE", "NEOSYN"))]
    if not cand:
        print("   none matched the name filter; listing all infusion-like tracks")
        cand = [n for n in names
                if any(k in n.upper() for k in ("RATE", "VOL", "CE", "CP"))
                and "Orchestra" in n]
    for n in cand:
        print(f"   {n:34s} n_cases = {trks[trks.tname == n].caseid.nunique()}")

    # EXP-B cohort
    art = set(trks[trks.tname == "SNUADC/ART"].caseid)
    ple = set(trks[trks.tname == "SNUADC/PLETH"].caseid)
    co = set()
    for t in CO_TRACKS:
        co |= set(trks[trks.tname == t].caseid)
    expb = art & ple & co
    print(f"\n   EXP-B cohort: {len(expb)}")

    print("\n" + "=" * 74)
    print("2. OVERLAP WITH EXP-B")
    print("=" * 74)
    best = None
    for n in cand:
        ids = set(trks[trks.tname == n].caseid) & expb
        if ids:
            print(f"   {n:34s} {len(ids):5d} of {len(expb)}")
            if best is None or len(ids) > len(best[1]):
                best = (n, ids)
    if best is None:
        print("   NO phenylephrine infusion track overlaps EXP-B.")
        print("   The landmark design is not possible from waveform tracks.")
        print("   Fall back to the clinical table (intraop_phe, total dose only),")
        print("   and report the absence of timing as a limitation.")
        return

    track, ids = best
    print(f"\n   using: {track}  (n = {len(ids)})")

    print("\n" + "=" * 74)
    print("3. FIRST-DOSE TIMING")
    print("=" * 74)
    import vitaldb

    rows = []
    sample = sorted(ids)[:120]          # sample for the timing distribution
    for k, cid in enumerate(sample, 1):
        try:
            v = vitaldb.load_case(cid, [track], 1.0)      # 1 Hz is enough
        except Exception:
            continue
        if v is None or v.size == 0:
            continue
        x = v[:, 0]
        pos = np.flatnonzero(np.isfinite(x) & (x > 0))
        rows.append(dict(caseid=cid,
                         dur_s=len(x),
                         t_first=float(pos[0]) if pos.size else np.nan,
                         any_dose=bool(pos.size)))
        if k % 30 == 0:
            print(f"   scanned {k}/{len(sample)}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("   no usable infusion data")
        return

    exposed = df[df.any_dose]
    print(f"\n   scanned {len(df)} cases; {len(exposed)} received the drug "
          f"({100*len(exposed)/len(df):.1f}%)")
    if len(exposed):
        t = exposed.t_first / 60.0
        print(f"   first dose at {t.median():.1f} min "
              f"[{t.quantile(.25):.1f}, {t.quantile(.75):.1f}]")
        need = SKIP_START_S + WIN_S + GAP_S
        ok = int((exposed.t_first >= need).sum())
        print(f"\n   cases with >= {need/60:.1f} min of drug-free recording "
              f"before the first dose: {ok} of {len(exposed)} "
              f"({100*ok/len(exposed):.1f}%)")
        print("   (these are the exposed cases usable in a landmark design)")

    print("\n" + "=" * 74)
    print("4. UNEXPOSED CONTROLS")
    print("=" * 74)
    un = df[~df.any_dose]
    print(f"   never exposed: {len(un)}")
    print("   for these the landmark is set at the same clock time as the")
    print("   median exposed case, so that both arms are measured at a")
    print("   comparable point in the operation.")

    print("\n" + "=" * 74)
    print("SUMMARY FOR THE NEXT STEP")
    print("=" * 74)
    print(f"   track          : {track}")
    print(f"   EXP-B overlap  : {len(ids)}")
    if len(exposed):
        print(f"   usable exposed : ~{ok} in the {len(sample)}-case sample")
        print(f"   scale to full  : ~{int(ok*len(ids)/max(len(df),1))} expected")
    print("\n   If the usable exposed count is under ~40, the landmark analysis")
    print("   will be underpowered and the honest report is that the temporal")
    print("   ordering cannot be established in these data.")


if __name__ == "__main__":
    main()
