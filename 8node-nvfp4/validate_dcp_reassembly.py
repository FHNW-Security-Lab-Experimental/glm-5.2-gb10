#!/usr/bin/env python3
"""Harness #2 — DCP logits all-gather reassembly identity check (item B, runs anywhere, no cluster).

The single most likely item-B bug (per ITEM-B-DESIGN.md / the mapping workflow) is getting the
all_gather reassembly column order wrong, which fails SILENTLY (plausible-but-wrong output). This
encodes each global position into its logit value and asserts the reassembly reconstructs global
order exactly, across N in {2,4,8} and the ragged-tail cases S % N != 0.

Mapping (cp_kv_cache_interleave_size=1): owner(p)=p%N, local(p)=p//N; rank r's local logits
column c == global position c*N + r. all_gather is rank-major along the column axis. Reassembly:
  gathered[num_rows, N*Lmax] -> view(num_rows, N, Lmax) -> transpose(1,2) -> reshape(num_rows, Lmax*N)
so flat column c*N + r == global p.
"""
import torch

NEG = float("-inf")


def reassemble(local_per_rank, N, Lmax, num_rows):
    # Simulate GroupCoordinator.all_gather(dim=1): rank-major concat -> [num_rows, N*Lmax].
    gathered = torch.stack(local_per_rank, dim=0)          # [N, num_rows, Lmax]
    gathered = gathered.permute(1, 0, 2).reshape(num_rows, N * Lmax)  # rank-major
    # Reassemble to global column order p = c*N + r.
    return gathered.view(num_rows, N, Lmax).transpose(1, 2).reshape(num_rows, Lmax * N)


def owned_writeback(global_topk, N, r):
    # §3: owned -> local p//N, else -1 (preserve -1 sentinel).
    owned = (global_topk % N == r) & (global_topk >= 0)
    local = torch.div(global_topk, N, rounding_mode="floor")
    return torch.where(owned, local, torch.full_like(global_topk, -1))


def test_reassembly(N, S, num_rows=3):
    Lmax = (S + N - 1) // N
    locals_ = []
    for r in range(N):
        col = torch.full((num_rows, Lmax), NEG)
        c, p = 0, r
        while p < S:
            col[:, c] = float(p)   # encode the global position into the value
            c += 1
            p += N
        locals_.append(col)
    g = reassemble(locals_, N, Lmax, num_rows)
    want = torch.arange(Lmax * N, dtype=torch.float32).expand(num_rows, -1).clone()
    want[:, S:] = NEG
    ok = torch.equal(g, want)
    if not ok:
        bad = (g != want).nonzero()[:3]
        print(f"  N={N} S={S}: FAIL at {bad.tolist()} (got {g[0, bad[0,1]].item() if len(bad) else '?'})")
    return ok


def test_owned_union(N, S, topk=2048):
    # The global top-k set (here: the highest-scoring = last S positions, arbitrary subset).
    k = min(topk, S)
    global_topk = torch.arange(S - k, S, dtype=torch.int64)  # a known global set
    union = set()
    for r in range(N):
        wb = owned_writeback(global_topk, N, r)
        kept = wb[wb >= 0]
        union |= {int(c) * N + r for c in kept.tolist()}   # map local back to global
    ok = union == set(global_topk.tolist())
    if not ok:
        print(f"  N={N} S={S}: owned-union mismatch ({len(union)} vs {k})")
    return ok


def build_reassembly_index(L, N):
    # PREFILL ragged shard->global permutation (Approach A, item C).
    # L: int64 [N, R] owned compressed-key counts per (rank, req).
    # Returns int64 [total] s.t. K_global[g] = K_all[idx[g]], where K_all is the
    # all_gatherv output (rank-major: rank0's rows | rank1's rows | ...), each rank's
    # block per-request-concat in ascending owned-key order; global order is
    # per-request-concat, keys ascending. owner(p)=p%N, local(p)=p//N (interleave=1).
    L = L.to(torch.int64)
    R = L.shape[1]
    per_rank_total = L.sum(dim=1)                       # [N]
    rank_base = torch.zeros(N, dtype=torch.int64)
    rank_base[1:] = torch.cumsum(per_rank_total[:-1], dim=0)
    req_off = torch.zeros((N, R), dtype=torch.int64)
    req_off[:, 1:] = torch.cumsum(L[:, :-1], dim=1)
    parts = []
    for q in range(R):
        glen = int(L[:, q].sum().item())
        if glen == 0:
            continue
        p = torch.arange(glen, dtype=torch.int64)
        owner = p % N
        local = p // N
        parts.append(rank_base[owner] + req_off[owner, q] + local)
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64)


def test_prefill_reassembly(N, glens):
    # glens: global per-request key lengths (ragged). Encode each key as q*BIG+p.
    R = len(glens)
    L = torch.zeros((N, R), dtype=torch.int64)
    for q, g in enumerate(glens):
        for r in range(N):
            L[r, q] = (g - r + N - 1) // N if g > r else 0  # #{p<g : p%N==r}
    BIG = 10_000_000
    # K_all in all_gatherv rank-major order (each rank: per-req-concat owned keys asc):
    rows = []
    for r in range(N):
        for q, g in enumerate(glens):
            i = 0
            p = r
            while p < g:
                rows.append(q * BIG + p)  # value encodes (req, global key)
                i += 1
                p += N
    kq_all = torch.tensor(rows, dtype=torch.int64)
    idx = build_reassembly_index(L, N)
    got = kq_all[idx]
    ref = torch.tensor(
        [q * BIG + p for q, g in enumerate(glens) for p in range(g)], dtype=torch.int64
    )
    ok = got.shape == ref.shape and torch.equal(got, ref)
    if not ok:
        print(f"  N={N} glens={glens}: FAIL (got {got.shape} vs ref {ref.shape})")
    return ok


def main():
    all_ok = True
    print("== reassembly identity ==")
    for N in (2, 4, 8):
        for S in (N, N + 1, 2048, 524289):
            r = test_reassembly(N, S)
            print(f"  N={N} S={S} -> {'PASS' if r else 'FAIL'}")
            all_ok &= r
    print("== owned->local write-back union == global top-k ==")
    for N in (2, 4, 8):
        for S in (N + 1, 2048, 524289):
            r = test_owned_union(N, S)
            print(f"  N={N} S={S} -> {'PASS' if r else 'FAIL'}")
            all_ok &= r
    print("== prefill ragged shard->global K reassembly (item C) ==")
    cases = [
        (2, [5]), (2, [5, 8]), (4, [10, 3, 7]), (8, [9, 16, 1]),
        (8, [2048, 4097, 100]), (4, [524289]), (8, [33, 64, 65, 7, 8]),
    ]
    for N, glens in cases:
        r = test_prefill_reassembly(N, glens)
        print(f"  N={N} glens={glens} -> {'PASS' if r else 'FAIL'}")
        all_ok &= r
    print("REASSEMBLY+WRITEBACK:", "ALL PASS" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
