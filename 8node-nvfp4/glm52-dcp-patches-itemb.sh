#!/usr/bin/env bash
# glm52-dcp-patches.sh — EXPERIMENTAL context-parallel (DCP) patches for fp8 + 1M.
#
# Layers ON TOP of glm52-sparse-patches.sh (run this AFTER it, inside each
# container, before `vllm serve`). It is the in-place half of the sparse-aware
# context-parallel KV work; the kernel half (LSE return, item C) lives in the
# bind-mounted kernels/flashmla_sparse.py and needs no patch here.
#
#   *** DO NOT ENABLE IN PRODUCTION. ***
# With ONLY item A (this script) + item C (kernels) the engine BOOTS under
# --decode-context-parallel-size>1 and the attention recombine is wired
# (mla_attention.py already calls cp_lse_ag_out_rs with the LSE), BUT the DSA
# lightning indexer still selects a per-rank LOCAL top-2048 instead of the global
# top-2048 (item B is not done). That produces SILENTLY WRONG attention — no
# crash, /health stays 200. Gate any rollout on the global-top-k SET-EQUALITY
# test (validation #2 in 1M-SPARSE-CP-KV-PLAN.md), never on /health alone.
#
# Items implemented here:  A  (relax the fp8-DCP query gate in mla_attention.py)
# Items NOT here (TODO):   B  (distributed cross-rank global top-2048, indexer)
#                          D  (co-shard / replicate-indexer decision)
# Items elsewhere (done):  C, E  (kernels/flashmla_sparse.py + vLLM's existing
#                                 cp_lse_ag_out_rs/dcp_a2a_lse_reduce wiring)
#
# Idempotent, anchor-guarded against vLLM ab66606. Aborts if the source differs.
set -euo pipefail

if [[ "${CONFIRM_GLM52_DCP:-}" != "YES" ]]; then
  echo "glm52-dcp-patches.sh: refusing to run without CONFIRM_GLM52_DCP=YES" >&2
  echo "  (this is experimental 1M context-parallel work; see 1M-SPARSE-CP-KV-PLAN.md)" >&2
  exit 2
fi

PYDIST="${GLM52_VLLM_DIST:-/usr/local/lib/python3.12/dist-packages}"
VLLM="$PYDIST/vllm"
MLA_ATTENTION="$VLLM/model_executor/layers/attention/mla_attention.py"

echo "== glm52-dcp-patches.sh (EXPERIMENTAL) =="
echo "   vLLM dist: $VLLM"

# ---- Step A: relax the fp8 gate on the MLA decode-context-parallel path ----
# Two coupled edits in mla_attention.py:
#  (1) Under DCP, keep the query as the UNQUANTIZED (ql_nope, q_pe) tuple so the
#      subsequent torch.cat(mqa_q, dim=-1) + all_gather operate on a 2-tuple.
#      (The fp8 quant-query collapse turns mqa_q into a single tensor that cat
#       cannot consume.) The KV cache stays fp8_ds_mla; only the bf16 query is
#       all-gathered, so no precision is lost.
#  (2) Drop the hard `assert not fp8_attention` that forbids fp8 KV under DCP.
python3 - "$MLA_ATTENTION" <<'PY'
import sys

path = sys.argv[1]
src = open(path).read()

SENTINEL = "# GLM52_DCP_FP8_GATE"
if SENTINEL in src:
    print("   mla_attention.py already DCP-patched (sentinel present) — skipping")
    sys.exit(0)

# Anchor 1: the fp8 quant-query selector. Only take the fp8 collapse when NOT
# under DCP, so DCP keeps the tuple form.
anchor1 = (
    "            if fp8_attention and self.impl.supports_quant_query_input:\n"
)
repl1 = (
    "            " + SENTINEL + "\n"
    "            # Under DCP keep the query as a tuple (see glm52-dcp-patches.sh);\n"
    "            # only collapse to the quantized fp8 query when NOT context-parallel.\n"
    "            if (\n"
    "                fp8_attention\n"
    "                and self.impl.supports_quant_query_input\n"
    "                and self.impl.dcp_world_size <= 1\n"
    "            ):\n"
)

# Anchor 2: the hard fp8 refusal on the DCP branch.
anchor2 = (
    '                assert not fp8_attention, "DCP not support fp8 kvcache now."\n'
)
repl2 = (
    "                " + SENTINEL + "\n"
    "                # fp8 KV is allowed under DCP: only the bf16 query is\n"
    "                # all-gathered below; the KV cache stays fp8_ds_mla.\n"
)

if anchor1 not in src or anchor2 not in src:
    print(
        "   FATAL: mla_attention.py DCP anchors not found — source differs from "
        "ab66606; aborting (refuse to patch unknown source).",
        file=sys.stderr,
    )
    sys.exit(1)

src = src.replace(anchor1, repl1, 1).replace(anchor2, repl2, 1)
open(path, "w").write(src)
print("   patched mla_attention.py (fp8 allowed under DCP; query kept as tuple)")
PY

echo "== glm52-dcp item A: done"

# ---- Step B: sparse-aware global top-k under DCP (item B) ----
# Localize the indexer logits-kernel seq_lens (per-rank shard), all_gather the local
# logits -> reassemble GLOBAL column order (col c*N+r == global pos p) -> global top-k
# -> write owned-local p//N (else -1). See ITEM-B-SPEC.md. Anchor-guarded vs ab66606.
DCP_IDX="$VLLM/v1/attention/backends/mla/indexer.py"
DCP_SAI="$VLLM/model_executor/layers/sparse_attn_indexer.py"
echo "== glm52-dcp item B: patching indexer.py + sparse_attn_indexer.py"
python3 - "$DCP_IDX" "$DCP_SAI" <<'PYB'
import sys, py_compile
SENT = "# GLM52_DCP_ITEMB"

def patch_indexer(path):
    src = open(path).read()
    if SENT in src:
        print("   indexer.py already item-B patched — skip"); return
    a1 = ("    decode_lens: torch.Tensor\n"
          "    requires_padding: bool\n"
          "    schedule_metadata: torch.Tensor\n")
    r1 = a1 + ("    " + SENT + " LOCAL 2D seq_lens + schedule for the per-rank logits kernel\n"
               "    dcp_local_seq_lens: \"torch.Tensor | None\" = None\n"
               "    dcp_local_schedule_metadata: \"torch.Tensor | None\" = None\n")
    a2 = ("        self.scheduler_metadata_buffer = torch.empty(\n"
          "            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device\n"
          "        )\n")
    r2 = a2 + ("        " + SENT + " second schedule buffer for the LOCAL-seq_lens logits kernel\n"
               "        self.dcp_scheduler_metadata_buffer = torch.empty(\n"
               "            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device\n"
               "        )\n")
    a3 = ("            decode_metadata = DeepSeekV32IndexerDecodeMetadata(\n"
          "                block_table=block_table,\n"
          "                seq_lens=seq_lens,\n"
          "                decode_lens=decode_lens,\n"
          "                requires_padding=requires_padding,\n"
          "                schedule_metadata=self.scheduler_metadata_buffer,\n"
          "            )\n")
    r3 = ("            " + SENT + " localize GLOBAL 2D seq_lens to owned counts for the\n"
          "            # per-rank logits kernel (expand-then-localize); global top-k keeps GLOBAL.\n"
          "            dcp_local_seq_lens = None\n"
          "            dcp_local_schedule_metadata = None\n"
          "            try:\n"
          "                from vllm.distributed.parallel_state import get_dcp_group as _gdg\n"
          "                _dcp = _gdg(); _N = _dcp.world_size; _r = _dcp.rank_in_group\n"
          "            except Exception:\n"
          "                _N = 1; _r = 0\n"
          "            if _N > 1:\n"
          "                _I = self.vllm_config.parallel_config.cp_kv_cache_interleave_size\n"
          "                assert _I == 1, \"GLM52_DCP item B requires cp_kv_cache_interleave_size==1\"\n"
          "                _g = seq_lens\n"
          "                _rounds = _g // (_N * _I)\n"
          "                _rem = torch.clamp((_g % (_N * _I)) - _r * _I, min=0, max=_I)\n"
          "                dcp_local_seq_lens = (_rounds * _I + _rem).to(_g.dtype).contiguous()\n"
          "                if current_platform.is_cuda() and has_deep_gemm():\n"
          "                    self.dcp_scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(\n"
          "                        dcp_local_seq_lens,\n"
          "                        self.kv_cache_spec.storage_block_size,\n"
          "                        self.num_sms,\n"
          "                    )\n"
          "                    dcp_local_schedule_metadata = self.dcp_scheduler_metadata_buffer\n"
          "            decode_metadata = DeepSeekV32IndexerDecodeMetadata(\n"
          "                block_table=block_table,\n"
          "                seq_lens=seq_lens,\n"
          "                decode_lens=decode_lens,\n"
          "                requires_padding=requires_padding,\n"
          "                schedule_metadata=self.scheduler_metadata_buffer,\n"
          "                dcp_local_seq_lens=dcp_local_seq_lens,\n"
          "                dcp_local_schedule_metadata=dcp_local_schedule_metadata,\n"
          "            )\n")
    for n, a in [("E1", a1), ("E2", a2), ("E3", a3)]:
        if a not in src:
            print("   FATAL indexer.py anchor " + n + " not found", file=sys.stderr); sys.exit(1)
    src = src.replace(a1, r1, 1).replace(a2, r2, 1).replace(a3, r3, 1)
    open(path, "w").write(src); print("   patched indexer.py (E1/E2/E3)")

def patch_sai(path):
    src = open(path).read()
    if SENT in src:
        print("   sparse_attn_indexer.py already item-B patched — skip"); return
    a_r1 = ("        else:\n"
            "            logits = fp8_fp4_paged_mqa_logits(\n"
            "                (padded_q_quant_cast, padded_q_scale),\n"
            "                kv_cache,\n"
            "                weights[:num_padded_tokens],\n"
            "                seq_lens,\n"
            "                decode_metadata.block_table,\n"
            "                decode_metadata.schedule_metadata,\n"
            "                max_model_len=max_model_len,\n"
            "                clean_logits=False,\n"
            "            )\n"
            "        num_rows = logits.shape[0]\n"
            "        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]\n")
    r_r1 = ("        else:\n"
            "            " + SENT + " feed LOCAL seq_lens+schedule so each rank scores its shard;\n"
            "            # the global top-k below runs over the reassembled GLOBAL logits.\n"
            "            _glm52_lsl = (\n"
            "                decode_metadata.dcp_local_seq_lens\n"
            "                if decode_metadata.dcp_local_seq_lens is not None\n"
            "                else seq_lens\n"
            "            )\n"
            "            _glm52_lsched = (\n"
            "                decode_metadata.dcp_local_schedule_metadata\n"
            "                if decode_metadata.dcp_local_schedule_metadata is not None\n"
            "                else decode_metadata.schedule_metadata\n"
            "            )\n"
            "            logits = fp8_fp4_paged_mqa_logits(\n"
            "                (padded_q_quant_cast, padded_q_scale),\n"
            "                kv_cache,\n"
            "                weights[:num_padded_tokens],\n"
            "                _glm52_lsl,\n"
            "                decode_metadata.block_table,\n"
            "                _glm52_lsched,\n"
            "                max_model_len=max_model_len,\n"
            "                clean_logits=False,\n"
            "            )\n"
            "        " + SENT + " all_gather local logits -> reassemble GLOBAL column order (c*N+r==p)\n"
            "        try:\n"
            "            from vllm.distributed.parallel_state import get_dcp_group as _glm52_gdg\n"
            "            _glm52_dcp = _glm52_gdg()\n"
            "            _glm52_N = _glm52_dcp.world_size\n"
            "            _glm52_r = _glm52_dcp.rank_in_group\n"
            "        except Exception:\n"
            "            _glm52_dcp = None; _glm52_N = 1; _glm52_r = 0\n"
            "        if _glm52_N > 1:\n"
            "            _glm52_Lmax = (max_model_len + _glm52_N - 1) // _glm52_N\n"
            "            _glm52_L = logits.shape[1]\n"
            "            if _glm52_L < _glm52_Lmax:\n"
            "                _glm52_pad = logits.new_full(\n"
            "                    (logits.shape[0], _glm52_Lmax - _glm52_L), float(\"-inf\")\n"
            "                )\n"
            "                _glm52_lp = torch.cat([logits, _glm52_pad], dim=1).contiguous()\n"
            "            else:\n"
            "                _glm52_lp = logits[:, :_glm52_Lmax].contiguous()\n"
            "            _glm52_g = _glm52_dcp.all_gather(_glm52_lp, dim=1)\n"
            "            _glm52_g = _glm52_g.view(\n"
            "                logits.shape[0], _glm52_N, _glm52_Lmax\n"
            "            ).transpose(1, 2)\n"
            "            logits = _glm52_g.reshape(\n"
            "                logits.shape[0], _glm52_Lmax * _glm52_N\n"
            "            ).contiguous()\n"
            "        num_rows = logits.shape[0]\n"
            "        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]\n")
    a_r2 = ("        if decode_metadata.requires_padding:\n"
            "            # if padded, we need to unpack\n")
    r_r2 = ("        " + SENT + " topk_indices are GLOBAL positions; keep this rank's owned\n"
            "        # ones rewritten to local p//N (else -1, skipped by the gather kernel).\n"
            "        if _glm52_N > 1:\n"
            "            _glm52_owned = (topk_indices % _glm52_N == _glm52_r) & (topk_indices >= 0)\n"
            "            _glm52_loc = torch.div(topk_indices, _glm52_N, rounding_mode=\"floor\")\n"
            "            topk_indices.copy_(\n"
            "                torch.where(_glm52_owned, _glm52_loc, torch.full_like(topk_indices, -1))\n"
            "            )\n"
            "        if decode_metadata.requires_padding:\n"
            "            # if padded, we need to unpack\n")
    for n, a in [("R1", a_r1), ("R2", a_r2)]:
        if a not in src:
            print("   FATAL sparse_attn_indexer.py anchor " + n + " not found", file=sys.stderr); sys.exit(1)
    src = src.replace(a_r1, r_r1, 1).replace(a_r2, r_r2, 1)
    open(path, "w").write(src); print("   patched sparse_attn_indexer.py (R1/R2)")

idx, sai = sys.argv[1], sys.argv[2]
patch_indexer(idx); patch_sai(sai)
py_compile.compile(idx, doraise=True); py_compile.compile(sai, doraise=True)
print("   item B py_compile OK")
PYB
echo "== glm52-dcp item B: done"

# ---- Step C: prefill indexer K all-gather under DCP (item C) ----
# The prefill indexer gathers each rank's KV shard (LOCAL cu_seq_lens), all_gatherv's the
# shard-K across the DCP group, reassembles to GLOBAL per-request key order, runs the prefill
# logits/top-k over the full K (GLOBAL cu_seqlen_ks/ke), and owned-masks the prefill top-k.
# See PREFILL-ALLGATHER-SPEC.md. Reassembly validated offline (validate_dcp_reassembly.py).
echo "== glm52-dcp item C: patching prefill indexer (K all-gather)"
python3 - "$DCP_IDX" "$DCP_SAI" <<'PYC'
import sys, py_compile
SENT = "# GLM52_DCP_ITEMC"

def patch_indexer(path):
    src = open(path).read()
    if SENT in src:
        print("   indexer.py already item-C patched — skip"); return
    a1 = ("    num_reqs: int\n    skip_kv_gather: bool = False\n")
    r1 = a1 + ("    " + SENT + " per-rank-LOCAL gather metadata (None when DCP off)\n"
               "    dcp_local_cu_seq_lens: \"torch.Tensor | None\" = None\n"
               "    dcp_local_total_seq_lens: int = 0\n"
               "    dcp_local_seq_lens_allranks: \"torch.Tensor | None\" = None\n")
    a2 = ("    cu_seq_lens[:1] = 0\n"
          "    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])\n")
    r2 = a2 + ("    " + SENT + " per-rank-local cu_seq_lens + all-ranks owned counts (owner=p%N, local=p//N)\n"
               "    dcp_local_cu_seq_lens = None\n"
               "    dcp_local_total_seq_lens = 0\n"
               "    dcp_local_seq_lens_allranks = None\n"
               "    try:\n"
               "        from vllm.distributed.parallel_state import get_dcp_group as _gdg\n"
               "        _N = _gdg().world_size; _r = _gdg().rank_in_group\n"
               "    except Exception:\n"
               "        _N = 1; _r = 0\n"
               "    if _N > 1:\n"
               "        assert compress_ratio == 1, \"GLM52_DCP item C assumes compress_ratio==1\"\n"
               "        _gc = compressed_seq_lens[start_idx:end_idx]\n"
               "        _gc_cpu = compressed_seq_lens_cpu[start_idx:end_idx]\n"
               "        def _own(gt, rk):\n"
               "            return (gt // _N) + torch.clamp((gt % _N) - rk, min=0, max=1)\n"
               "        _loc = _own(_gc, _r)\n"
               "        dcp_local_cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)\n"
               "        dcp_local_cu_seq_lens[:1] = 0\n"
               "        torch.cumsum(_loc, dim=0, out=dcp_local_cu_seq_lens[1:])\n"
               "        dcp_local_total_seq_lens = int(_own(_gc_cpu, _r).sum().item())\n"
               "        dcp_local_seq_lens_allranks = torch.stack(\n"
               "            [_own(_gc_cpu, rr) for rr in range(_N)], dim=0\n"
               "        ).to(torch.int32)\n")
    a3 = ("        num_reqs=num_reqs,\n")
    r3 = a3 + ("        dcp_local_cu_seq_lens=dcp_local_cu_seq_lens,\n"
               "        dcp_local_total_seq_lens=dcp_local_total_seq_lens,\n"
               "        dcp_local_seq_lens_allranks=dcp_local_seq_lens_allranks,\n")
    for n, a in [("C1", a1), ("C2", a2), ("C3", a3)]:
        if a not in src:
            print("   FATAL indexer.py anchor " + n + " not found", file=sys.stderr); sys.exit(1)
    src = src.replace(a1, r1, 1).replace(a2, r2, 1).replace(a3, r3, 1)
    open(path, "w").write(src); print("   patched indexer.py (C1/C2/C3)")

def patch_sai(path):
    src = open(path).read()
    if SENT in src:
        print("   sparse_attn_indexer.py already item-C patched — skip"); return
    a4 = ("        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(\n"
          "            values_spec,\n"
          "            scales_spec,\n"
          "        )\n"
          "        for chunk in prefill_metadata.chunks:\n")
    r4 = ("        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(\n"
          "            values_spec,\n"
          "            scales_spec,\n"
          "        )\n"
          "        " + SENT + " DCP group + ragged shard->global K reassembly for prefill.\n"
          "        try:\n"
          "            from vllm.distributed.parallel_state import get_dcp_group as _glm52c_gdg\n"
          "            _glm52c_dcp = _glm52c_gdg()\n"
          "            _glm52c_N = _glm52c_dcp.world_size\n"
          "            _glm52c_r = _glm52c_dcp.rank_in_group\n"
          "        except Exception:\n"
          "            _glm52c_dcp = None; _glm52c_N = 1; _glm52c_r = 0\n"
          "        if _glm52c_N > 1:\n"
          "            _glm52c_lcap = (total_seq_lens + _glm52c_N - 1) // _glm52c_N\n"
          "            _glm52c_lv, _glm52c_ls = _gather_workspace_shapes(\n"
          "                _glm52c_lcap, head_dim, fp8_dtype, use_fp4_cache\n"
          "            )\n"
          "            k_quant_local_full, k_scale_local_full = workspace_manager.get_simultaneous(\n"
          "                _glm52c_lv, _glm52c_ls\n"
          "            )\n"
          "            def _glm52c_reasm_idx(L, N, device):\n"
          "                L = L.to(torch.int64); R = L.shape[1]\n"
          "                prt = L.sum(dim=1)\n"
          "                rb = torch.zeros(N, dtype=torch.int64); rb[1:] = torch.cumsum(prt[:-1], dim=0)\n"
          "                ro = torch.zeros((N, R), dtype=torch.int64); ro[:, 1:] = torch.cumsum(L[:, :-1], dim=1)\n"
          "                parts = []\n"
          "                for q in range(R):\n"
          "                    glen = int(L[:, q].sum().item())\n"
          "                    if glen == 0:\n"
          "                        continue\n"
          "                    p = torch.arange(glen, dtype=torch.int64); ow = p % N; lo = p // N\n"
          "                    parts.append(rb[ow] + ro[ow, q] + lo)\n"
          "                out = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64)\n"
          "                return out.to(device)\n"
          "        for chunk in prefill_metadata.chunks:\n")
    a5 = ("            if not chunk.skip_kv_gather:\n"
          "                ops.cp_gather_indexer_k_quant_cache(\n"
          "                    kv_cache,\n"
          "                    k_quant,\n"
          "                    k_scale,\n"
          "                    chunk.block_table,\n"
          "                    chunk.cu_seq_lens,\n"
          "                )\n")
    r5 = ("            if not chunk.skip_kv_gather:\n"
          "                if _glm52c_N <= 1:\n"
          "                    ops.cp_gather_indexer_k_quant_cache(\n"
          "                        kv_cache,\n"
          "                        k_quant,\n"
          "                        k_scale,\n"
          "                        chunk.block_table,\n"
          "                        chunk.cu_seq_lens,\n"
          "                    )\n"
          "                else:\n"
          "                    " + SENT + " gather shard (LOCAL cu_seq_lens) -> all_gatherv -> reassemble global\n"
          "                    _lt = chunk.dcp_local_total_seq_lens\n"
          "                    _kq_loc = k_quant_local_full[:_lt]\n"
          "                    _ks_loc = k_scale_local_full[:_lt]\n"
          "                    if _lt > 0:\n"
          "                        ops.cp_gather_indexer_k_quant_cache(\n"
          "                            kv_cache, _kq_loc, _ks_loc,\n"
          "                            chunk.block_table, chunk.dcp_local_cu_seq_lens,\n"
          "                        )\n"
          "                    _sizes = [int(chunk.dcp_local_seq_lens_allranks[rr].sum().item())\n"
          "                              for rr in range(_glm52c_N)]\n"
          "                    _kq_all = _glm52c_dcp.all_gatherv(_kq_loc, dim=0, sizes=_sizes)\n"
          "                    _ks_all = _glm52c_dcp.all_gatherv(_ks_loc, dim=0, sizes=_sizes)\n"
          "                    _ridx = _glm52c_reasm_idx(\n"
          "                        chunk.dcp_local_seq_lens_allranks, _glm52c_N, k_quant.device\n"
          "                    )\n"
          "                    torch.index_select(_kq_all, 0, _ridx, out=k_quant)\n"
          "                    torch.index_select(_ks_all, 0, _ridx, out=k_scale)\n")
    a6 = ("            ops.top_k_per_row_prefill(\n"
          "                logits,\n"
          "                chunk.cu_seqlen_ks,\n"
          "                chunk.cu_seqlen_ke,\n"
          "                topk_indices,\n"
          "                num_rows,\n"
          "                logits.stride(0),\n"
          "                logits.stride(1),\n"
          "                topk_tokens,\n"
          "            )\n")
    r6 = a6 + ("            " + SENT + " prefill topk are GLOBAL positions; keep owned -> local p//N else -1.\n"
               "            if _glm52c_N > 1:\n"
               "                _ti = topk_indices\n"
               "                _ow = (_ti % _glm52c_N == _glm52c_r) & (_ti >= 0)\n"
               "                _lo = torch.div(_ti, _glm52c_N, rounding_mode=\"floor\")\n"
               "                _ti.copy_(torch.where(_ow, _lo, torch.full_like(_ti, -1)))\n")
    for n, a in [("C4", a4), ("C5", a5), ("C6", a6)]:
        if a not in src:
            print("   FATAL sparse_attn_indexer.py anchor " + n + " not found", file=sys.stderr); sys.exit(1)
    src = src.replace(a4, r4, 1).replace(a5, r5, 1).replace(a6, r6, 1)
    open(path, "w").write(src); print("   patched sparse_attn_indexer.py (C4/C5/C6)")

idx, sai = sys.argv[1], sys.argv[2]
patch_indexer(idx); patch_sai(sai)
py_compile.compile(idx, doraise=True); py_compile.compile(sai, doraise=True)
print("   item C py_compile OK")
PYC
echo "== glm52-dcp item C: done"
echo
echo "== DONE (items A+B+C). Gate go-live on the top-k SET-EQUALITY test, never /health. =="
