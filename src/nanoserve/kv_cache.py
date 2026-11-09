"""KV Cache implementations: Contiguous per-sequence cache and Paged KV cache with BlockAllocator."""

from typing import Dict, List, Optional, Tuple
import torch


class ContiguousKVCache:
    """Contiguous per-sequence KV cache for prefill + decode acceleration (Phase 2)."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.v_cache: List[Optional[torch.Tensor]] = [None] * num_layers

    def update(
        self, k: torch.Tensor, v: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a given layer.
        
        Args:
            k: [B, S, num_kv_heads, head_dim]
            v: [B, S, num_kv_heads, head_dim]
            layer_idx: Layer index
        Returns:
            Updated (all_k, all_v) of shape [B, total_S, num_kv_heads, head_dim]
        """
        past_k = self.k_cache[layer_idx]
        past_v = self.v_cache[layer_idx]

        if past_k is None:
            # Prefill phase: store initial sequence
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
            return k, v
        else:
            # Decode phase: append along sequence dimension (dim 1)
            all_k = torch.cat([past_k, k], dim=1)
            all_v = torch.cat([past_v, v], dim=1)
            self.k_cache[layer_idx] = all_k
            self.v_cache[layer_idx] = all_v
            return all_k, all_v

    def get_seq_len(self) -> int:
        """Returns currently cached sequence length (from layer 0)."""
        if self.k_cache[0] is None:
            return 0
        return self.k_cache[0].shape[1]

    def clear(self):
        """Reset cache."""
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers


class OutOfBlocksError(Exception):
    """Raised when BlockAllocator runs out of free memory blocks."""
    pass


class BlockAllocator:
    """Manages allocation and deallocation of fixed-size physical KV blocks (Phase 3)."""

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Free list of block IDs initialized in FIFO order
        self.free_blocks: List[int] = list(range(num_blocks))
        # Ref-counts for future copy-on-write / prefix-caching support
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def allocate(self) -> int:
        """Allocate a single free block ID."""
        if not self.free_blocks:
            raise OutOfBlocksError(
                f"KV Cache exhausted: all {self.num_blocks} blocks are allocated."
            )
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int):
        """Free a block ID back to the pool."""
        if block_id not in self.ref_counts or self.ref_counts[block_id] == 0:
            return  # Already freed

        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)

    def free_table(self, block_table: List[int]):
        """Free all blocks referenced by a sequence's block table."""
        for block_id in block_table:
            self.free(block_id)
        block_table.clear()


class PagedKVCache:
    """Fixed pool of physical blocks for Paged Attention (Phase 3)."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Physical memory pool for Keys and Values
        # Shape: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.k_pool = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_pool = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

    def write_slot(
        self,
        layer_idx: int,
        block_id: int,
        offset_in_block: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write a single token's K and V into the physical block slot.
        
        Args:
            layer_idx: Layer index
            block_id: Physical block ID
            offset_in_block: Slot index [0 .. block_size-1]
            k: [num_kv_heads, head_dim]
            v: [num_kv_heads, head_dim]
        """
        self.k_pool[layer_idx, block_id, offset_in_block] = k
        self.v_pool[layer_idx, block_id, offset_in_block] = v

    def write_slots(
        self,
        layer_idx: int,
        block_table: List[int],
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write a sequence of tokens into the paged cache starting at start_pos.
        
        Args:
            layer_idx: Layer index
            block_table: Physical block table for sequence
            start_pos: Starting token index in sequence
            k: [S, num_kv_heads, head_dim]
            v: [S, num_kv_heads, head_dim]
        """
        S = k.shape[0]
        for i in range(S):
            pos = start_pos + i
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = block_table[block_idx]
            self.k_pool[layer_idx, block_id, offset] = k[i]
            self.v_pool[layer_idx, block_id, offset] = v[i]

