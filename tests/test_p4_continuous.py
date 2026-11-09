"""P4 Correctness Gate: Continuous batching scheduler, concurrency, and preemption."""

import time
import pytest
import torch
from nanoserve.types import SamplingParams
from nanoserve.engine import LLMEngine
from nanoserve.kv_cache import BlockAllocator
from nanoserve.scheduler import Scheduler
from nanoserve.types import Request


BASE_PROMPTS = [
    "The capital of France is",
    "In physics, gravity is",
    "Python is a popular programming language because",
    "The speed of light in vacuum is approximately",
    "Photosynthesis converts sunlight into",
    "Machine learning models are trained using",
    "The largest planet in our solar system is",
    "Water boils at 100 degrees Celsius when",
]


def test_concurrent_requests_token_exact():
    """GATE: 50+ concurrent requests of mixed lengths produce identical outputs to running each alone."""
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    # Pre-generate 52 prompts by repeating base prompts
    test_prompts = [BASE_PROMPTS[i % len(BASE_PROMPTS)] for i in range(52)]
    max_new_tokens = 16
    params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)

    # 1. First run: individual reference generation (each alone)
    engine_ref = LLMEngine(
        model_name_or_path=model_id,
        num_blocks=256,
        block_size=16,
        max_batch_size=1,
    )
    # Get reference for unique prompts
    unique_refs = {}
    for p in set(test_prompts):
        res = engine_ref.generate([p], sampling_params=params)
        unique_refs[p] = res[0].output_token_ids

    # 2. Continuous batching run with 52 requests and staggered arrival
    engine_concurrent = LLMEngine(
        model_name_or_path=model_id,
        num_blocks=256,
        block_size=16,
        max_batch_size=8,
    )

    req_ids = []
    # Add first 16 immediately
    for i in range(16):
        req_id = f"stagger_{i}"
        engine_concurrent.add_request(req_id, test_prompts[i], params)
        req_ids.append(req_id)

    # Step engine and inject remaining requests in staggered waves
    curr_idx = 16
    step_count = 0
    while engine_concurrent.has_unfinished_requests():
        engine_concurrent.step()
        step_count += 1
        # Every 2 steps, inject another 6 requests if available
        if step_count % 2 == 0 and curr_idx < len(test_prompts):
            batch_end = min(curr_idx + 6, len(test_prompts))
            for i in range(curr_idx, batch_end):
                req_id = f"stagger_{i}"
                engine_concurrent.add_request(req_id, test_prompts[i], params)
                req_ids.append(req_id)
            curr_idx = batch_end

    # Check: All 52 requests must match reference token-for-token
    mismatches = 0
    for i, req_id in enumerate(req_ids):
        req = engine_concurrent.all_requests[req_id]
        expected = unique_refs[req.prompt]
        actual = req.output_token_ids
        if actual != expected:
            mismatches += 1

    assert mismatches == 0, f"{mismatches}/52 requests mismatched reference outputs!"

    # Check: Zero memory leaks after all 52 requests complete
    assert engine_concurrent.allocator.num_free_blocks == engine_concurrent.allocator.num_blocks, (
        f"Memory leak! Free blocks: {engine_concurrent.allocator.num_free_blocks} vs total: {engine_concurrent.allocator.num_blocks}"
    )


def test_scheduler_preemption_under_memory_pressure():
    """Verify that scheduler preempts requests gracefully when memory is constrained and resumes them."""
    allocator = BlockAllocator(num_blocks=4, block_size=4)  # Very tight memory: only 4 blocks
    scheduler = Scheduler(allocator=allocator, max_batch_size=4)

    # Create 3 requests that each need 2 blocks for prompt
    r1 = Request("r1", "prompt 1", [1, 2, 3, 4, 5])  # 5 tokens -> needs 2 blocks
    r2 = Request("r2", "prompt 2", [6, 7, 8, 9, 10]) # 5 tokens -> needs 2 blocks
    r3 = Request("r3", "prompt 3", [11, 12, 13, 14, 15]) # 5 tokens -> needs 2 blocks

    scheduler.add_request(r1)
    scheduler.add_request(r2)
    scheduler.add_request(r3)

    # Step 1: Allocates r1 (2 blocks) and r2 (2 blocks). Total allocated = 4 blocks.
    # r3 should remain waiting because 0 free blocks remain.
    res1 = scheduler.schedule()
    assert len(res1.prefills) == 2
    assert r1 in scheduler.running
    assert r2 in scheduler.running
    assert r3 in scheduler.waiting
    assert allocator.num_free_blocks == 0

    # Add tokens to r1 and r2 so r2 crosses block boundary (needs 3 blocks)
    for _ in range(4):
        r1.append_token(100)
        r2.append_token(101)

    # Step 2: r2 now needs another block, but 0 are free!
    # Scheduler should preempt r2 (youngest running request), free its 2 blocks, and let r1 continue
    res2 = scheduler.schedule()
    assert len(res2.preempted) == 1
    assert res2.preempted[0].request_id == "r2"
    assert r2 in scheduler.waiting  # Returned to waiting queue
    assert r1 in scheduler.running  # r1 continues running

    # Complete r1
    r1.state = r1.state.FINISHED_LENGTH
    # Step 3: r1 finishes and frees its blocks; r2 can now be readmitted from waiting queue
    res3 = scheduler.schedule()
    assert r1 not in scheduler.running
    assert r2 in scheduler.running or r3 in scheduler.running

