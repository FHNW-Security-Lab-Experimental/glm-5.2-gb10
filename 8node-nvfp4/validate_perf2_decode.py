#!/usr/bin/env python3
"""PERF#2 offline correctness gate (FIX 3) — pure-torch, CPU, no cluster.

Proves the Approach-2 union top-k (each rank's LOCAL top-2048 -> fixed-size all_gather
-> re-top-k by strict (score desc, global_pos asc)) yields the EXACT global top-K SET,
and is rank-invariant, across: distinct scores, boundary ties, rank-owns-<K, and S<K.

This validates the SELECTION LOGIC (argsort/gather), device-independent. It does NOT
test the in-container top_k_per_row_decode kernel's own tie-break — that is the separate
in-container caveat in PERF2-DECODE-SPEC.md FIX 3.
"""
import torch

NEG = float("-inf")


def _strict_topk(scores, gpos, k):
    """Top-k of [rows,M] by strict order (score desc, gpos asc). Invalid lanes carry
    score=-inf, gpos=-1 and are demoted. Returns (sel_scores, sel_gpos) each [rows,k]
    with gpos=-1 where the selected lane was invalid. Mirrors the patch exactly."""
    poskey = torch.where(gpos >= 0, gpos, torch.full_like(gpos, (1 << 62)))
    o1 = torch.argsort(poskey, dim=1, stable=True)
    sc1 = torch.gather(scores, 1, o1)
    o2 = torch.argsort(sc1, dim=1, descending=True, stable=True)
    order = torch.gather(o1, 1, o2)
    k = min(k, scores.shape[1])
    sel = order[:, :k]
    vsel = torch.gather(scores, 1, sel)
    gsel = torch.gather(gpos, 1, sel)
    gsel = torch.where(vsel == NEG, torch.full_like(gsel, -1), gsel)
    return vsel, gsel


def approach2(glogits, N, K, rank_order=None):
    """Simulate the full Approach-2 path for global logits [rows,S] sharded N ways.
    rank_order lets us permute the all_gather rank concatenation order to test
    rank-invariance. Returns selected global positions [rows,K] (sorted by the
    strict order; -1 for short rows)."""
    rows, S = glogits.shape
    ranks = list(range(N)) if rank_order is None else rank_order
    cand_sc, cand_pos = [], []
    for r in ranks:
        loc = glogits[:, r::N]                       # owned cols: global c*N+r
        Lloc = loc.shape[1]
        gpos_r = (torch.arange(Lloc) * N + r).unsqueeze(0).expand(rows, -1).contiguous()
        vs, gs = _strict_topk(loc, gpos_r, K)        # rank r local top-K (pad to K)
        if vs.shape[1] < K:                          # pad short ranks to K with invalid
            padn = K - vs.shape[1]
            vs = torch.cat([vs, torch.full((rows, padn), NEG)], dim=1)
            gs = torch.cat([gs, torch.full((rows, padn), -1, dtype=gs.dtype)], dim=1)
        cand_sc.append(vs); cand_pos.append(gs)
    allsc = torch.cat(cand_sc, dim=1)                # [rows, N*K] fixed size
    allpos = torch.cat(cand_pos, dim=1)
    _, gsel = _strict_topk(allsc, allpos, K)
    return gsel


def brute(glogits, K):
    """Reference: global top-K of [rows,S] by the SAME strict (score desc, gpos asc).
    Pads to exactly K columns with -1 (as the real topk_indices buffer is) so the order
    comparison is shape-fair when S<K."""
    rows, S = glogits.shape
    gpos = torch.arange(S).unsqueeze(0).expand(rows, -1).contiguous()
    _, gsel = _strict_topk(glogits, gpos, K)
    if gsel.shape[1] < K:
        gsel = torch.cat(
            [gsel, torch.full((rows, K - gsel.shape[1]), -1, dtype=gsel.dtype)], dim=1)
    return gsel


def _set_rows(t):
    return [frozenset(int(x) for x in row.tolist() if x >= 0) for row in t]


def check(name, glogits, N, K):
    a = approach2(glogits, N, K)
    b = brute(glogits, K)
    set_ok = _set_rows(a) == _set_rows(b)
    ord_ok = torch.equal(a, b)                       # identical selection+order (no kernel involved)
    # rank-invariance: permuted all_gather order -> identical result
    perm = approach2(glogits, N, K, rank_order=list(range(N - 1, -1, -1)))
    inv_ok = torch.equal(a, perm)
    print(f"  [{name}] set={'PASS' if set_ok else 'FAIL'} "
          f"order={'PASS' if ord_ok else 'FAIL'} rank_invariant={'PASS' if inv_ok else 'FAIL'}")
    return set_ok and ord_ok and inv_ok


def main():
    torch.manual_seed(0)
    K, N = 2048, 8
    ok = True
    # 1) distinct random scores, large S (subset lemma should be exact)
    ok &= check("distinct/large-S", torch.randn(4, 50000), N, K)
    # 2) small K/N for dense brute coverage
    ok &= check("distinct/small", torch.randn(8, 1000), 4, 64)
    # 3) heavy ties straddling the K boundary (quantized scores -> many equal)
    g = (torch.randn(6, 20000) * 4).round()          # integer-valued -> many exact ties
    ok &= check("dense-ties", g, N, K)
    # 4) all high scores on one rank's class (rank r=3 owns the top)
    g = torch.full((4, 20000), -5.0); g[:, 3::N] += 10.0
    ok &= check("skewed-to-one-rank", g, N, K)
    # 5) a rank owns < K columns (S/N < K)
    ok &= check("rank-owns-<K", torch.randn(4, N * 100), N, K)   # ~100 cols/rank << 2048
    # 6) S < K globally (early decode)
    ok &= check("S<K", torch.randn(4, 50), N, K)
    # 7) ties + short together
    g = (torch.randn(4, N * 50) * 3).round()
    ok &= check("ties+short", g, N, K)
    print("PERF#2 OFFLINE GATE:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
