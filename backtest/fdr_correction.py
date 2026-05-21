#!/usr/bin/env python3
"""
Study #4: FDR / Bonferroni correction on V4 passers.

V4 ran 1,596 statistical tests. At α=0.05, ~80 false positives are expected
by chance. The 100 "passers" likely include false positives.

Apply Benjamini-Hochberg FDR correction to identify the truly statistically
robust configs.
"""
import json, glob
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def load_v4_results():
    rows = []
    for f in glob.glob(str(_ROOT / 'logs/research/val_v4_all/*.json')):
        try:
            r = json.load(open(f))
        except: continue
        if r.get('full_2yr_trades', 0) < 10: continue
        rows.append(r)
    return rows


def benjamini_hochberg(rows, alpha=0.05, key='mc_p_value'):
    """Apply BH-FDR. Returns rows with 'bh_significant' = True/False."""
    # Sort by p ascending
    rows = sorted(rows, key=lambda r: r.get(key, 1.0))
    n = len(rows)
    # Find largest k such that p_k <= (k/n)*alpha
    cutoff_p = 0.0
    for k, r in enumerate(rows, 1):
        threshold = (k / n) * alpha
        if r.get(key, 1.0) <= threshold:
            cutoff_p = r.get(key, 1.0)
    # Mark all with p<=cutoff as significant
    for r in rows:
        r['bh_significant'] = r.get(key, 1.0) <= cutoff_p if cutoff_p > 0 else False
    return rows, cutoff_p


def bonferroni(rows, alpha=0.05, key='mc_p_value'):
    """Apply Bonferroni: significant if p < alpha/n."""
    n = len(rows)
    threshold = alpha / n
    for r in rows:
        r['bonf_significant'] = r.get(key, 1.0) < threshold
    return rows, threshold


def main():
    rows = load_v4_results()
    print(f"V4 configs with trades: {len(rows)}")

    # Original "passer" count (WF >= 6/10, MC p <= 0.05, slip-0.05 >= 1.0)
    original_passers = [
        r for r in rows
        if r.get('wf_wins', 0) >= 6
        and r.get('mc_p_value', 1.0) <= 0.05
        and (r.get('slip_full_2yr', {}) or {}).get('0.05', 0) >= 1.0
    ]
    print(f"Original V4 passers (no correction): {len(original_passers)}")

    # Apply BH-FDR to MC p-values
    rows_bh, bh_cutoff = benjamini_hochberg(rows, alpha=0.05, key='mc_p_value')
    print(f"\nBenjamini-Hochberg FDR (q=0.05):")
    print(f"  p-value cutoff: {bh_cutoff:.5f}")
    bh_passers = [r for r in rows_bh if r.get('bh_significant')]
    print(f"  configs with significant MC: {len(bh_passers)} / {len(rows)}")

    # Apply Bonferroni
    rows_bonf, bonf_cutoff = bonferroni(rows, alpha=0.05, key='mc_p_value')
    print(f"\nBonferroni (alpha=0.05/{len(rows)}):")
    print(f"  p-value cutoff: {bonf_cutoff:.6f}")
    bonf_passers = [r for r in rows_bonf if r.get('bonf_significant')]
    print(f"  configs with significant MC: {len(bonf_passers)} / {len(rows)}")

    # Now intersect with WF + slippage to get final BH-corrected passers
    bh_full = [
        r for r in rows_bh
        if r.get('bh_significant')
        and r.get('wf_wins', 0) >= 6
        and (r.get('slip_full_2yr', {}) or {}).get('0.05', 0) >= 1.0
    ]
    print(f"\nBH-corrected QUAD passers (BH-MC + WF + slip-0.05): {len(bh_full)}")

    bonf_full = [
        r for r in rows_bonf
        if r.get('bonf_significant')
        and r.get('wf_wins', 0) >= 6
        and (r.get('slip_full_2yr', {}) or {}).get('0.05', 0) >= 1.0
    ]
    print(f"Bonferroni-corrected QUAD passers: {len(bonf_full)}")

    # Show top 20 BH-corrected by slip-0.05 PF
    bh_full.sort(key=lambda r: -(r.get('slip_full_2yr', {}) or {}).get('0.05', 0))
    print(f"\nTop 15 BH-corrected configs (by slip-0.05 PF):")
    print(f"  {'strat':<14} {'sym':<6} {'sl':<5} {'rr':<5} {'mc_p':<8} {'wf_wins':<8} {'slip05':<7}")
    for r in bh_full[:15]:
        slip = (r.get('slip_full_2yr', {}) or {}).get('0.05', 0)
        print(f"  {r['strategy']:<14} {r['symbol']:<6} {r['sl']:<5} {r['rr']:<5} "
              f"{r.get('mc_p_value', 0):<8.4f} {r.get('wf_wins', 0)}/{r.get('wf_total', 0):<6} "
              f"{slip:<7.3f}")

    # By symbol breakdown
    from collections import defaultdict
    bh_by_sym = defaultdict(int)
    for r in bh_full:
        bh_by_sym[r['symbol']] += 1
    bonf_by_sym = defaultdict(int)
    for r in bonf_full:
        bonf_by_sym[r['symbol']] += 1
    print(f"\nBH-corrected QUAD passers by symbol:")
    for s in sorted(set(list(bh_by_sym.keys()) + list(bonf_by_sym.keys()))):
        print(f"  {s:<8} BH={bh_by_sym[s]:<4}  Bonf={bonf_by_sym[s]:<4}")

    # Save full list
    out = _ROOT / 'logs/research/studies_2026_04_26/fdr_corrected_passers.json'
    with open(out, 'w') as f:
        json.dump({
            'meta': {
                'n_tests': len(rows),
                'original_passers': len(original_passers),
                'bh_cutoff_p': bh_cutoff,
                'bonf_cutoff_p': bonf_cutoff,
                'bh_quad_passers': len(bh_full),
                'bonf_quad_passers': len(bonf_full),
            },
            'bh_passers': [
                {k: v for k, v in r.items() if k not in ('slip_full_2yr', 'slip_holdout', 'slip_bear')}
                for r in bh_full
            ],
            'bonf_passers': [
                {k: v for k, v in r.items() if k not in ('slip_full_2yr', 'slip_holdout', 'slip_bear')}
                for r in bonf_full
            ],
        }, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
