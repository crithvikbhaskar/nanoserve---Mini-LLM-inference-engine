"""Attention implementations: Naive attention and Paged Attention with Block Table gather."""

from typing import List, Optional
import torch
import torch.nn.functional as F
from nanoserve.kv_cache import PagedKVCache


def paged_attention_decode(
    q: torch.Tensor,
    paged_cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens: List[int],
    layer_idx: int,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Paged attention for decode step (single query token per sequence).
    
    Args:
        q: Query tensor of shape [batch_size, 1, num_heads, head_dim]
        paged_cache: PagedKVCache storing k_pool, v_pool
        block_tables: Tensor of shape [batch_size, max_blocks_per_seq] of physical block IDs
        seq_lens: Current total sequence length for each request in batch
        layer_idx: Current layer index
        scale: 1.0 / sqrt(head_dim)
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimensionality of each head
    Returns:
        Output tensor of shape [batch_size, 1, num_heads * head_dim]
    """
    batch_size = q.shape[0]
    num_queries_per_kv = num_heads // num_kv_heads
    block_size = paged_cache.block_size
    outputs: List[torch.Tensor] = []

    k_pool = paged_cache.k_pool[layer_idx]  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_pool = paged_cache.v_pool[layer_idx]  # [num_blocks, block_size, num_kv_heads, head_dim]

    for b in range(batch_size):
        seq_len = seq_lens[b]
        num_blocks = (seq_len + block_size - 1) // block_size
        if isinstance(block_tables, list):
            b_blocks = block_tables[b][:num_blocks]
        else:
            b_blocks = block_tables[b, :num_blocks].tolist()

        # Gather physical blocks: [num_blocks, block_size, num_kv_heads, head_dim]
        gathered_k = k_pool[b_blocks].view(-1, num_kv_heads, head_dim)[:seq_len]
        gathered_v = v_pool[b_blocks].view(-1, num_kv_heads, head_dim)[:seq_len]

        # Expand KV heads for GQA: [seq_len, num_heads, head_dim]
        if num_queries_per_kv > 1:
            gathered_k = gathered_k.repeat_interleave(num_queries_per_kv, dim=1)
            gathered_v = gathered_v.repeat_interleave(num_queries_per_kv, dim=1)

        # q for sequence b: [1, num_heads, head_dim] -> [num_heads, 1, head_dim]
        q_b = q[b, 0].unsqueeze(1)

        # k, v transposed for attention: [num_heads, seq_len, head_dim]
        k_b = gathered_k.transpose(0, 1)
        v_b = gathered_v.transpose(0, 1)

        # Scaled dot-product attention: [num_heads, 1, seq_len]
        scores = torch.bmm(q_b, k_b.transpose(-2, -1)) * scale
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)

        # Context vector: [num_heads, 1, head_dim]
        out_b = torch.bmm(probs, v_b)
        # Reshape to [1, 1, num_heads * head_dim]
        out_b = out_b.transpose(0, 1).contiguous().view(1, 1, -1)
        outputs.append(out_b)

    return torch.cat(outputs, dim=0)

