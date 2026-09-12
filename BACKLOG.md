# nanoserve Backlog

Deferred features and non-core enhancements preserved for future exploration:

1. **Custom Triton / CUDA fused paged attention kernel** (currently using pure PyTorch block-table gather).
2. **Prefix caching (Radix tree)** for shared system prompt reuse across requests.
3. **Tensor Parallelism (TP)** across multiple GPUs.
4. **Quantization support** (AWQ, GPTQ, FP8/INT8 KV cache).
5. **Speculative Decoding** (draft model assisted verification).
6. **Chunked Prefill** (splitting long prompt prefills across multiple decode iterations).

