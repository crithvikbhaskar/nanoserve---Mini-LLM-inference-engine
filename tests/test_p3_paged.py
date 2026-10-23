"""P3 Correctness Gate: BlockAllocator unit tests and Paged Attention token-exact match."""

import random
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nanoserve.kv_cache import BlockAllocator, PagedKVCache, OutOfBlocksError
from nanoserve.model import NanoserveModel
from tests.test_p2_kv_cache import cached_generate


# ============================================================================
# 1. BlockAllocator Unit Tests
# ============================================================================

def test_block_allocator_basic():
    """Verify basic allocate and free functionality."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    assert allocator.num_free_blocks == 10
    assert allocator.num_allocated_blocks == 0

    b0 = allocator.allocate()
    b1 = allocator.allocate()
    assert allocator.num_free_blocks == 8
    assert allocator.num_allocated_blocks == 2
    assert b0 != b1

    allocator.free(b0)
    assert allocator.num_free_blocks == 9
    assert allocator.num_allocated_blocks == 1

    allocator.free(b1)
    assert allocator.num_free_blocks == 10
    assert allocator.num_allocated_blocks == 0


def test_block_allocator_exhaustion():
    """Verify allocator raises OutOfBlocksError when capacity is exceeded."""
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    allocated = [allocator.allocate() for _ in range(4)]
    assert allocator.num_free_blocks == 0

    with pytest.raises(OutOfBlocksError):
        allocator.allocate()

    # Free one and allocate again
    allocator.free(allocated[0])
    assert allocator.num_free_blocks == 1
    new_b = allocator.allocate()
    assert new_b == allocated[0]


def test_block_allocator_free_table():
    """Verify free_table frees all blocks in a sequence's block table."""
    allocator = BlockAllocator(num_blocks=16, block_size=16)
    table = [allocator.allocate() for _ in range(5)]
    assert allocator.num_free_blocks == 11

    allocator.free_table(table)
    assert len(table) == 0
    assert allocator.num_free_blocks == 16


def test_block_allocator_stress_no_leaks():
    """Verify zero memory leaks over 1,000 random allocate and free cycles."""
    random.seed(42)
    num_blocks = 64
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=16)
    active_tables: dict[int, list[int]] = {}

    req_id = 0
    for _ in range(1000):
        action = random.choice(["alloc", "free", "alloc", "grow"])
        if action == "alloc" or not active_tables:
            req_id += 1
            num_req_blocks = random.randint(1, 4)
            if allocator.num_free_blocks >= num_req_blocks:
                active_tables[req_id] = [allocator.allocate() for _ in range(num_req_blocks)]
        elif action == "grow":
            target_req = random.choice(list(active_tables.keys()))
            if allocator.num_free_blocks >= 1:
                active_tables[target_req].append(allocator.allocate())
        else:
            target_req = random.choice(list(active_tables.keys()))
            allocator.free_table(active_tables[target_req])
            del active_tables[target_req]

    # Clean up all remaining tables
    for table in active_tables.values():
        allocator.free_table(table)

    assert allocator.num_free_blocks == num_blocks, f"Memory leak! Expected {num_blocks} free, got {allocator.num_free_blocks}"
    assert allocator.num_allocated_blocks == 0


# ============================================================================
# 2. Paged Attention Generation Helper
# ============================================================================

def paged_generate(
    model: NanoserveModel,
    input_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int,
    block_size: int = 16,
) -> list[int]:
    """Generate using PagedKVCache and paged attention decode."""
    allocator = BlockAllocator(num_blocks=64, block_size=block_size)
    config = model.config
    paged_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=allocator.num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # 1. Prefill phase: allocate enough blocks for prompt
    prompt_len = len(input_ids)
    num_prompt_blocks = (prompt_len + block_size - 1) // block_size
    block_table = [allocator.allocate() for _ in range(num_prompt_blocks)]

    x = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(
            x,
            paged_cache=paged_cache,
            block_tables=[block_table],
            seq_lens=[prompt_len],
        )

    generated = []
    next_token = int(torch.argmax(logits[0, -1, :]).item())
    generated.append(next_token)
    cur_seq_len = prompt_len + 1

    # 2. Decode phase: single token at a time with paged attention
    for _ in range(max_new_tokens - 1):
        if next_token == eos_token_id:
            break

        # Check if new block is needed for cur_seq_len
        needed_blocks = (cur_seq_len + block_size - 1) // block_size
        while len(block_table) < needed_blocks:
            block_table.append(allocator.allocate())

        x_step = torch.tensor([[next_token]], dtype=torch.long)
        with torch.no_grad():
            logits = model(
                x_step,
                paged_cache=paged_cache,
                block_tables=[block_table],
                seq_lens=[cur_seq_len],
            )

        next_token = int(torch.argmax(logits[0, 0, :]).item())
        generated.append(next_token)
        cur_seq_len += 1

    allocator.free_table(block_table)
    return generated


# ============================================================================
# 3. P3 Correctness Gate: Token-Exact Match vs P2 and HF
# ============================================================================

TEST_PROMPTS = [
    "The quick brown fox jumps over",
    "In machine learning, gradient descent is",
    "The capital of Australia is Canberra, which",
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = NanoserveModel.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return model, tokenizer


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_paged_vs_contiguous_token_exact(model_and_tokenizer, prompt):
    """GATE: Paged KV cache must match Contiguous KV cache token-for-token."""
    model, tokenizer = model_and_tokenizer
    max_new_tokens = 24
    eos_token_id = model.config.eos_token_id

    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()

    cached_tokens = cached_generate(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
    )
    paged_tokens = paged_generate(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, block_size=16
    )

    assert paged_tokens == cached_tokens, (
        f"Prompt '{prompt}' mismatch between Paged and Contiguous KV cache:\n"
        f"Contiguous: {cached_tokens}\n"
        f"Paged:      {paged_tokens}"
    )


def test_paged_boundary_crossing(model_and_tokenizer):
    """Verify that paged attention crosses block boundaries seamlessly."""
    model, tokenizer = model_and_tokenizer
    # Use small block_size=4 to force multiple block crossings within 16 tokens
    prompt = "Artificial neural networks can learn complex"
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()
    max_new_tokens = 20
    eos_token_id = model.config.eos_token_id

    cached_tokens = cached_generate(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
    )
    paged_tokens = paged_generate(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, block_size=4
    )

    assert paged_tokens == cached_tokens, "Block boundary crossing produced token mismatch!"

