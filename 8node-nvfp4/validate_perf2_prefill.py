#!/usr/bin/env python3
"""PERF#2-for-PREFILL offline correctness gate — pure-torch, CPU, no cluster.

#2 replaces item-C's "all_gatherv the full shard-K → reassemble global → top-k over the
full global K" (superlinear) with the PERF#2 candidate-union: each rank scores
chunk-queries × its LOCAL shard-K, takes its LOCAL top-2048 (carrying global key ids),
the group all-gathers only those FIXED-SIZE candidates, unions, and re-selects the exact
global top-2048 — then owned-masks. This proves that union == the global top-K SET the
old reassembly path would have produced, with the prefill specifics decode lacked:

  - PER-QUERY CAUSAL RANGES: prefill query q attends to keys [0, ke[q]); ke varies per
    query (and is < K early in a sequence). The decode gate used one length for all rows.
  - RAGGED interleave-1 sharding: owner(p)=p%N, local(p)=p//N, per request/query.

It validates the SELECTION LOGIC (argsort/gather/union over causal ranges), device-
independent. It does NOT test the in-container fp8_fp4_mqa_logits / top_k_per_row_prefill
kernels — that's the separate in-container K-equality + set-equality-vs-N=1 gate
(PREFILL-ALLGATHER-SPEC.md §5).
"""
import torch

NEG = float("-inf")
BIG = (1 << 62)


def _strict_topk(scores, pos, k):
    """Top-k of [rows,M] by strict (score desc, global_pos asc). Invalid lanes carry
    score=-inf, pos=-1 and are demoted. Returns (sel_scores, sel_pos), pos=-1 where the
    chosen lane was invalid. Mirrors the patch's two-stable-sort exactly."""
    poskey = torch.where(pos >= 0, pos, torch.full_like(pos, BIG))
    o1 = torch.argsort(poskey, dim=1, stable=True)
    sc1 = torch.gather(scores, 1, o1)
    o2 = torch.argsort(sc1, dim=1, descending=True, stable=True)
    order = torch.gather(o1, 1, o2)
    k = min(k, scores.shape[1])
    sel = order[:, :k]
    vs = torch.gather(scores, 1, sel)
    ps = torch.gather(pos, 1, sel)
    ps = torch.where(vs == NEG, torch.full_like(ps, -1), ps)
    return vs, ps


def _pad(vs, ps, K):
    if vs.shape[1] >= K:
        return vs, ps
    n = vs.shape[1]
    return (torch.cat([vs, torch.full((vs.shape[0], K - n), NEG)], 1),
            torch.cat([ps, torch.full((ps.shape[0], K - n), -1, dtype=ps.dtype)], 1))


def candidate_union(glogits, ke, N, K, rank_order=None):
    """Approach-2 prefill: per rank, per query, LOCAL top-K over owned keys in [0,ke[q]);
    all-gather (concat) the FIXED-SIZE [Q, K] candidates; union → global top-K.
    rank_order permutes the concat order to test rank-invariance."""
    Q, S = glogits.shape
    ranks = list(range(N)) if rank_order is None else rank_order
    col = torch.arange(S).unsqueeze(0)                      # [1,S] global key id
    in_range = col < ke.unsqueeze(1)                        # [Q,S] causal
    cand_sc, cand_pos = [], []
    for r in ranks:
        owned = (col % N == r) & in_range                  # rank r's owned keys in causal range
        sc = torch.where(owned, glogits, torch.full_like(glogits, NEG))
        ps = torch.where(owned, col.expand(Q, S), torch.full((Q, S), -1))
        vs, p = _strict_topk(sc, ps, K)                    # LOCAL top-K per query
        vs, p = _pad(vs, p, K)                             # fixed [Q,K] (kernel pads with -1)
        cand_sc.append(vs); cand_pos.append(p)
    allsc = torch.cat(cand_sc, 1)                           # [Q, N*K] fixed-size all-gather
    allpos = torch.cat(cand_pos, 1)
    _, gsel = _strict_topk(allsc, allpos, K)
    return _pad(torch.zeros_like(gsel, dtype=torch.float), gsel, K)[1] if gsel.shape[1] < K else gsel


def brute(glogits, ke, K):
    """Reference: per-query global top-K over the causal range [0,ke[q]), same strict order."""
    Q, S = glogits.shape
    col = torch.arange(S).unsqueeze(0).expand(Q, S)
    valid = col < ke.unsqueeze(1)
    sc = torch.where(valid, glogits, torch.full_like(glogits, NEG))
    ps = torch.where(valid, col, torch.full((Q, S), -1))
    _, gsel = _strict_topk(sc, ps, K)
    return _pad(torch.zeros_like(gsel, dtype=torch.float), gsel, K)[1] if gsel.shape[1] < K else gsel


def _own(c, r, N):
    """Owned compressed-key count: # of keys p in [0,c) with p%N==r. Mirrors the source's
    _own(gt,rk) = gt//N + clamp((gt%N)-rk, 0, 1). This is the per-token LOCAL causal count."""
    return (c // N) + (1 if (c % N) > r else 0)


def check_localization():
    """Validate the load-bearing localization formula + the lp->gp mapping the patch relies on:
    for every causal count c and rank r, {lp*N+r : lp in [0,_own(c,r))} must equal exactly the
    owned keys {p in [0,c) : p%N==r}. If _own is off-by-one or gp=lp*N+r is wrong, this FAILS."""
    ok = True
    bad = None
    for N in (2, 3, 4, 8):
        for r in range(N):
            for c in range(0, 3000):
                formula = set(lp * N + r for lp in range(_own(c, r, N)))
                truth = set(p for p in range(c) if p % N == r)
                if formula != truth:
                    bad = (N, r, c); ok = False; break
            if not ok: break
        if not ok: break
    print("  [localization _own(c,r) + gp=lp*N+r]", "PASS" if ok else ("FAIL @ %r" % (bad,)))
    return ok


def check_topk_equiv():
    """The patch's union uses torch.topk (O(M), light temps) instead of the strict two-stable
    sort. Validate they pick the SAME top-k SCORES (set-exact on distinct scores; score-multiset
    -exact even on dense ties) and that torch.topk is rank-invariant (deterministic on identical
    inputs). This justifies the swap that removes the small-context slowdown + the heavy argsort
    temporaries (the 512k-wedge candidate)."""
    torch.manual_seed(1)
    ok = True
    for trial in range(300):
        M = 16384
        sc = torch.randn(4, M) if trial % 3 else (torch.randn(4, M) * 3).round()  # mix distinct + dense ties
        pos = torch.arange(M).unsqueeze(0).expand(4, M).contiguous()
        k = 2048
        vts, sts = torch.topk(sc, k, dim=1)
        vst, _ = _strict_topk(sc, pos, k)
        # same top-k SCORES (sorted) ?
        if not torch.allclose(vts.sort(dim=1, descending=True).values,
                              vst.sort(dim=1, descending=True).values):
            ok = False; break
        # rank-invariance: torch.topk on the same input is deterministic
        vts2, _ = torch.topk(sc, k, dim=1)
        if not torch.equal(vts, vts2):
            ok = False; break
    print("  [torch.topk union == strict top-k SCORES (distinct + dense-ties) + deterministic]",
          "PASS" if ok else "FAIL")
    return ok


def _sets(t):
    return [frozenset(int(x) for x in row.tolist() if x >= 0) for row in t]


def check(name, glogits, ke, N, K):
    a = candidate_union(glogits, ke, N, K)
    b = brute(glogits, ke, K)
    set_ok = _sets(a) == _sets(b)
    ord_ok = torch.equal(a, b)
    inv_ok = torch.equal(a, candidate_union(glogits, ke, N, K, rank_order=list(range(N - 1, -1, -1))))
    print("  [%s] set=%s order=%s rank_invariant=%s" %
          (name, "PASS" if set_ok else "FAIL", "PASS" if ord_ok else "FAIL",
           "PASS" if inv_ok else "FAIL"))
    return set_ok and inv_ok


def main():
    torch.manual_seed(0)
    K, N = 2048, 8
    ok = True
    # 0) the localization formula + lp->gp mapping (the load-bearing trap)
    ok &= check_localization()
    # 0b) the torch.topk union is score-equivalent to the strict selection (the perf fix)
    ok &= check_topk_equiv()
    # 1) big context, full causal range (ke=S), distinct scores
    ok &= check("full-range/large", torch.randn(8, 60000), torch.full((8,), 60000), N, K)
    # 2) CAUSAL: each query attends to a different prefix length (the prefill-specific case)
    ke = torch.tensor([50, 2048, 4097, 10000, 33333, 59999, 60000, 1])
    ok &= check("causal/varying-ke", torch.randn(8, 60000), ke, N, K)
    # 3) dense ties straddling the K boundary, causal
    g = (torch.randn(6, 40000) * 4).round()
    ok &= check("dense-ties/causal", g, torch.tensor([40000, 39000, 5000, 2049, 2048, 2047]), N, K)
    # 4) all high scores on one rank's class, causal
    g = torch.full((4, 40000), -5.0); g[:, 3::N] += 10.0
    ok &= check("skewed-to-rank", g, torch.tensor([40000, 20000, 2050, 100]), N, K)
    # 5) ke < K everywhere (early decode-of-prefill: fewer keys than topk)
    ok &= check("ke<K", torch.randn(5, 5000), torch.tensor([10, 500, 1000, 2047, 16]), N, K)
    # 6) L%N coverage: per-query ranges hitting every remainder mod N
    ke = torch.tensor([4096 + m for m in range(8)])
    ok &= check("L%N-sweep", torch.randn(8, 8000), ke, N, K)
    # 7) small N/K dense brute coverage + causal
    ok &= check("small/causal", torch.randn(10, 1000), torch.tensor([7, 64, 128, 200, 333, 500, 640, 800, 999, 1000]), 4, 64)
    print("PERF#2-PREFILL OFFLINE GATE:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
