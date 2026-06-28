#!/usr/bin/env python3
"""LSE-recombine unit test (validation #1) — runs INSIDE the GLM container, no DCP, no downtime.

This is the foundation check for the sparse-aware context-parallel KV work: it proves the sparse
kernel's (out, lse) pair composes correctly via the LSE-weighted merge that vLLM's DCP recombine
(cp_lse_ag_out_rs / dcp_a2a_lse_reduce) performs across KV shards. If this passes, the LSE
base/layout is trustworthy and the remaining DCP corruption is purely the top-k selection + slot
mapping (item B) — not the recombine.

Method: attend a query over the FULL key set, then over two DISJOINT subsets, and check that the
LSE-weighted merge of the two partials reconstructs the full attention to fp32 tolerance.

Run:  sudo docker exec -i vllm-glm52 python3 /workspace/validate_lse_recombine.py
"""
import torch

# The bound, GB10-portable Triton sparse prefill kernel (returns out, max_logits, lse base-e).
from vllm.v1.attention.backends.mla.sm12x_sparse_mla_attn import (
    flash_mla_sparse_fwd_triton,
)


def _make_indices(per_query_lists, s_q, device):
    maxk = max(len(x) for x in per_query_lists)
    idx = torch.full((s_q, 1, maxk), -1, device=device, dtype=torch.int32)
    for i, lst in enumerate(per_query_lists):
        if lst:
            idx[i, 0, : len(lst)] = torch.tensor(lst, device=device, dtype=torch.int32)
    return idx


def _attn(q, kv, per_query_lists, sm_scale, d_v):
    idx = _make_indices(per_query_lists, q.shape[0], q.device)
    out, _max_logits, lse = flash_mla_sparse_fwd_triton(q, kv, idx, sm_scale, d_v=d_v)
    return out, lse  # out [s_q, h_q, d_v] bf16 ; lse [s_q, h_q] f32 base-e


def main():
    torch.manual_seed(0)
    device = "cuda"
    s_q, h_q, d_qk, d_v = 4, 128, 576, 512
    s_kv = 96
    sm_scale = 1.0 / (d_qk**0.5)

    q = torch.randn(s_q, h_q, d_qk, device=device, dtype=torch.bfloat16)
    kv = torch.randn(s_kv, 1, d_qk, device=device, dtype=torch.bfloat16)

    full = list(range(s_kv))
    even = [i for i in range(s_kv) if i % 2 == 0]
    odd = [i for i in range(s_kv) if i % 2 == 1]

    out_full, lse_full = _attn(q, kv, [full] * s_q, sm_scale, d_v)
    out_a, lse_a = _attn(q, kv, [even] * s_q, sm_scale, d_v)
    out_b, lse_b = _attn(q, kv, [odd] * s_q, sm_scale, d_v)

    # LSE-weighted merge of the two disjoint-subset partials (base-e), exactly the
    # math cp_lse_ag_out_rs applies across CP ranks:
    #   out = sum_s out_s * exp(lse_s) / sum_s exp(lse_s)
    #   lse = logsumexp_s(lse_s)
    la = lse_a.unsqueeze(-1).float()
    lb = lse_b.unsqueeze(-1).float()
    m = torch.maximum(la, lb)
    wa = torch.exp(la - m)
    wb = torch.exp(lb - m)
    merged = (out_a.float() * wa + out_b.float() * wb) / (wa + wb)
    merged_lse = (m + torch.log(wa + wb)).squeeze(-1)

    err_out = (merged - out_full.float()).abs().max().item()
    err_lse = (merged_lse - lse_full.float()).abs().max().item()
    # bf16 output has ~1e-2 granularity; lse is f32.
    ok_out = err_out < 2e-2
    ok_lse = err_lse < 1e-3
    print(f"max|out_merged - out_full| = {err_out:.4e}  (tol 2e-2) -> {'ok' if ok_out else 'FAIL'}")
    print(f"max|lse_merged - lse_full| = {err_lse:.4e}  (tol 1e-3) -> {'ok' if ok_lse else 'FAIL'}")
    print("LSE-RECOMBINE:", "PASS" if (ok_out and ok_lse) else "FAIL")
    raise SystemExit(0 if (ok_out and ok_lse) else 1)


if __name__ == "__main__":
    main()
