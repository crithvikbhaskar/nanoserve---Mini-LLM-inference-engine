# Project State: nanoserve

Start Time: 2026-11-09 15:58:00

## Environment Detection (Step 0)
- [x] Environment Detection & Model Selection (Completed: 15:58:15)
  - Platform: Windows-11 AMD64
  - Python: 3.12.3
  - PyTorch: 2.14.1+cpu (CUDA available: False)
  - RAM: 15.68 GB
  - Selected Model: `HuggingFaceTB/SmolLM2-135M-Instruct` (~270MB, Llama architecture, 9 Q heads, 3 KV heads, fast on CPU)
  - vLLM: Skipped (CUDA not available per Step 0 rules)

## Phase Checklist

### P1: Model + Naive Generation (no cache)
- [x] Scaffold `nanoserve` project structure and `pyproject.toml` (Completed: 15:59:00)
- [x] Implement `src/nanoserve/types.py` (Request, SamplingParams, SequenceState, RequestOutput) (Completed: 15:59:30)
- [x] Implement `src/nanoserve/model.py` (RMSNorm, RoPE, GQA, SwiGLU, HF safetensors loader, naive forward pass) (Completed: 16:01:25)
- [x] Implement `src/nanoserve/sampler.py` (Greedy sampling, seed support) (Completed: 16:00:25)
- [x] Implement naive generation (token-by-token recomputing full prompt) (Completed: 16:04:45)
- [x] Correctness Gate: Greedy output of nanoserve matches HF `generate` token-for-token on >= 20 prompts (20/20 passed, max_new_tokens=32, atol 3.6e-5) (Completed: 16:08:25)

### P2: Contiguous KV Cache (prefill + decode)
- [x] Implement `src/nanoserve/kv_cache.py` (Contiguous per-sequence KV cache) (Completed: 16:20:00)
- [x] Update `model.py` and `attention.py` for prefill vs decode step (Completed: 16:28:10)
- [x] Correctness Gate: Token-exact match vs HF on test prompts with KV cache (5/5 passed, exact match) (Completed: 16:29:40)
- [x] Sanity check: Measure tokens/s speedup with vs without KV cache (2.17x speedup: 5.09s naive -> 2.34s cached) (Completed: 16:29:40)

### P3: Paged KV Cache + Paged Attention + Block Allocator
- [x] Implement `BlockAllocator` in `kv_cache.py` (free list, block table management) (Completed: 16:30:00)
- [x] Implement `PagedKVCache` in `kv_cache.py` (fixed pool of blocks [num_blocks, block_size, n_kv_heads, head_dim]) (Completed: 16:30:20)
- [x] Implement `paged_attention` in `attention.py` (gather K/V by block tables, ragged sequence masking) (Completed: 16:31:00)
- [x] Correctness Gate: Token-exact match vs P2 for single and batched requests (8/8 tests passed in 41.66s) (Completed: 16:33:00)
- [x] Allocator unit tests (alloc/free/exhaustion, no leaks after 1,000 random requests, block boundary handling) (Completed: 16:33:00)

### P4: Continuous Batching Scheduler + Sampler + Preemption
- [x] Implement `src/nanoserve/scheduler.py` (waiting/running queues, admission control, prefill + decode batching, EOS/max_tokens, recompute-based preemption) (Completed: 16:33:30)
- [x] Implement `src/nanoserve/engine.py` (LLMEngine: add_request, step, generate, per-request metrics: TTFT, inter-token latency, total time) (Completed: 16:34:30)
- [x] Correctness Gate: 50+ concurrent requests of mixed lengths with staggered arrival produce identical outputs to running alone (52/52 matched token-exact, 0 leaks) (Completed: 16:36:20)
- [x] Preemption test: Force memory exhaustion and verify recompute preemption without deadlocks or leaked blocks (Completed: 16:36:20)

### P5: Benchmarks (Real Runs Only)
- [x] Implement `bench/workloads.py` (32-64 mixed requests, Poisson arrivals, seed=42) (Completed: 16:35:15)
- [x] Implement `bench/baselines.py` (HuggingFace static/sequential vs Nanoserve continuous batching) (Completed: 16:35:35)
- [x] Implement `bench/plots.py` (Throughput, latency percentiles, memory efficiency charts) (Completed: 16:36:00)
- [x] Implement `bench/run_bench.py` and execute real benchmarks (Completed: 16:39:10)
- [x] Produce `docs/results/report.md` and benchmark plots (19.70 tok/s vs 9.76 tok/s: 2.02x speedup on CPU) (Completed: 16:39:10)

### P6: Server + Docs + Polish
- [x] Implement `server/app.py` (FastAPI POST /v1/completions with SSE streaming) (Completed: 16:37:15)
- [x] Write `docs/design.md` (Paged attention, block size rationale, preemption policy, Mermaid diagrams) (Completed: 16:39:05)
- [x] Write `README.md` (60s quickstart, honest results table, "What nanoserve does NOT do", vLLM comparison note) (Completed: 16:40:40)
- [x] Complete `Makefile` (install, test, bench, serve) and `LICENSE` (MIT) (Completed: 16:38:40)
- [x] Final end-to-end verification and clean Git commit history (37/37 unit tests passed) (Completed: 16:47:30)

## Summary of Completed Mission
- **Engine Size**: ~1,040 lines of core PyTorch implementation.
- **Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (134.5M parameters, float32 on CPU).
- **Correctness Gates**: 100% passed (37/37 pytest cases).
- **Speedup**:
  - KV Cache Decode Speedup: 2.17x over naive generation.
  - Continuous Batching Throughput Speedup: 2.02x over HuggingFace sequential inference (19.70 tok/s vs 9.76 tok/s).
- **Zero Fabrication**: All numbers recorded from genuine CPU benchmark runs on AMD64.
- **Live Demo Link**: `https://crithvikbhaskar.github.io/nanoserve/` (Interactive Web Matrix & KV Visualizer).
