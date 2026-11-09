# nanoserve Architecture & Design

> A readable, tested, ~1k-line LLM inference engine that reimplements the core ideas of vLLM (paged KV cache + continuous batching), verified token-exact against HuggingFace and benchmarked on throughput, latency, and memory.

---

## 1. High-Level Architecture

The nanoserve architecture isolates each stage of high-throughput LLM serving into dedicated, composable PyTorch modules:

```mermaid
flowchart TD
    subgraph ClientLayer["API & Client Layer"]
        Req["Client Requests (Prompt, SamplingParams)"]
        FastAPI["FastAPI App (/v1/completions)"]
        Web["Web UI & KV Visualizer"]
    end

    subgraph EngineLayer["LLMEngine"]
        Sched["Continuous Batching Scheduler"]
        Alloc["BlockAllocator (Physical Pool)"]
        PagedCache["PagedKVCache [Pool of Fixed Blocks]"]
        Model["NanoserveModel (Decoder, RoPE, GQA, SwiGLU)"]
        Attn["Paged Attention (Block Gather)"]
        Sampler["Sampler (Greedy, Temp, Top-K, Top-P)"]
    end

    Req --> FastAPI
    Web --> FastAPI
    FastAPI --> Sched
    Sched -->|Admit Waiting| Alloc
    Sched -->|Batch Running Sequences| Model
    Alloc -->|Assign Physical Blocks| PagedCache
    Model --> Attn
    Attn -->|Gather via Block Tables| PagedCache
    Model --> Sampler
    Sampler -->|Next Token| Sched
```

---

## 2. Paged KV Cache & Memory Management

### The Problem of Contiguous Allocation
In traditional inference setups (e.g., standard HuggingFace static batching), memory for Key and Value activations must be pre-allocated contiguously for each sequence's *maximum* possible length ($S_{max} = 2,048$ or $8,192$). Because actual generation lengths vary widely, this causes:
1. **Severe Internal Fragmentation**: 60–80% of reserved GPU/RAM memory is never written to.
2. **Artificial Concurrency Limits**: The batch size is capped by peak memory reservations rather than active memory usage.

### Nanoserve Solution: Virtual Memory for KV Activations
Inspired by vLLM, nanoserve implements virtual memory paging for KV activations:

```mermaid
flowchart LR
    subgraph LogicalSequence["Logical Sequence (Tokens 0..34)"]
        L0["Block 0: Tokens 0..15"]
        L1["Block 1: Tokens 16..31"]
        L2["Block 2: Tokens 32..34 (3/16 used)"]
    end

    subgraph BlockTable["Sequence Block Table"]
        BT0["Logical 0 -> Physical 7"]
        BT1["Logical 1 -> Physical 2"]
        BT2["Logical 2 -> Physical 19"]
    end

    subgraph PhysicalPool["PagedKVCache Pool [num_blocks, 16, n_kv_heads, head_dim]"]
        P2["Physical Block 2"]
        P7["Physical Block 7"]
        P19["Physical Block 19"]
        Free["Free Blocks List: [0, 1, 3, 4, 5, 6, 8, ...]"]
    end

    L0 --> BT0 --> P7
    L1 --> BT1 --> P2
    L2 --> BT2 --> P19
```

- **Block Size Choice ($B = 16$)**: 16 tokens balances gather efficiency with minimal internal fragmentation (at most 15 tokens wasted per sequence, $< 1$ block).
- **Physical Memory Pool**: Allocated once at engine startup as a continuous tensor `[num_layers, num_blocks, block_size, n_kv_heads, head_dim]`.
- **Zero Memory Compaction**: When a request finishes, its physical blocks return immediately to the free list in $O(1)$ time.

---

## 3. Continuous Batching Scheduler

Instead of waiting for an entire batch of requests to reach their respective `EOS` tokens before admitting new requests (static batching), nanoserve schedules inference at the **iteration level**:

```mermaid
sequenceDiagram
    autonumber
    actor C as Client
    participant S as Scheduler
    participant A as BlockAllocator
    participant M as NanoserveModel
    participant K as PagedKVCache

    C->>S: add_request(req_1)
    S->>A: Allocate prompt blocks
    A-->>S: Block IDs [3, 4]
    S->>M: Prefill step (req_1)
    M->>K: Write prompt K/Vs into blocks [3, 4]
    M-->>S: Token 1 sampled

    Note over S: Continuous Decode Iterations
    C->>S: add_request(req_2) (Staggered Arrival)
    S->>A: Allocate prompt blocks for req_2
    A-->>S: Block IDs [7]
    S->>M: Combined Step: Prefill req_2 + Decode req_1
    M->>K: Paged attention gathers req_1 and writes req_2
    M-->>S: Next tokens for both requests
```

### Preemption Policy: Recomputation
If memory is temporarily exhausted (all physical blocks allocated):
1. The scheduler selects the youngest running sequence (`last_admitted`).
2. Its block table is freed and its blocks are returned to the free list.
3. The request's state is set to `PREEMPTED` and it is prepended to the front of the `waiting` queue.
4. When memory frees up, the request is readmitted and its KV cache is recomputed cleanly.

---

## 4. Codebase Line Count & Design Discipline

The core engine is strictly maintained at ~1,000 lines of readable PyTorch code:

| File | Purpose | Lines of Code |
| :--- | :--- | :--- |
| `src/nanoserve/model.py` | Llama/Qwen2 decoder: RMSNorm, RoPE, GQA, SwiGLU | ~340 |
| `src/nanoserve/kv_cache.py` | Contiguous & Paged KV cache, BlockAllocator | ~180 |
| `src/nanoserve/attention.py` | Paged attention decode with block table gather | ~75 |
| `src/nanoserve/scheduler.py` | Continuous batching queues, admission, preemption | ~110 |
| `src/nanoserve/sampler.py` | Greedy, temperature, top-k, top-p with RNG seed | ~80 |
| `src/nanoserve/engine.py` | High-level LLMEngine coordinating inference & metrics | ~160 |
| `src/nanoserve/types.py` | Request, SamplingParams, SequenceState, RequestOutput | ~95 |
| **Total Core Engine** | **Complete Mini LLM Inference Engine** | **~1,040 lines** |

