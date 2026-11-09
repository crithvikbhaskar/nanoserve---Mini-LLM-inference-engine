"""P2 Correctness Gate: Token-exact match with Contiguous KV cache and speedup verification."""

import time
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanoserve.kv_cache import ContiguousKVCache
from nanoserve.model import NanoserveModel

TEST_PROMPTS = [
    "The quick brown fox jumps over",
    "In machine learning, gradient descent is",
    "The capital of Australia is Canberra, which",
    "Photosynthesis is the biological process where",
    "Artificial neural networks are inspired by biological neurons in",
]


@pytest.fixture(scope="module")
def shared_models():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()
    nanoserve_model = NanoserveModel.from_pretrained(model_id, dtype=torch.float32)
    nanoserve_model.eval()
    return tokenizer, hf_model, nanoserve_model


def cached_generate(
    model: NanoserveModel, input_ids: list[int], max_new_tokens: int, eos_token_id: int
) -> list[int]:
    """Generation using Contiguous KV Cache (prefill once, then single-token decode steps)."""
    cache = ContiguousKVCache(num_layers=model.config.num_hidden_layers)
    generated = []

    # 1. Prefill phase: process full prompt tokens
    x = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x, kv_cache=cache)

    next_token = int(torch.argmax(logits[0, -1, :]).item())
    generated.append(next_token)

    # 2. Decode phase: process one token at a time with past KV cached
    for _ in range(max_new_tokens - 1):
        if next_token == eos_token_id:
            break
        x_step = torch.tensor([[next_token]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x_step, kv_cache=cache)
        next_token = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(next_token)

    return generated


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_cached_vs_hf_token_exact(shared_models, prompt):
    tokenizer, hf_model, nanoserve_model = shared_models
    max_new_tokens = 32
    eos_token_id = nanoserve_model.config.eos_token_id

    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()

    # HuggingFace reference
    with torch.no_grad():
        hf_out = hf_model.generate(
            torch.tensor([input_ids]),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )
    hf_tokens = hf_out[0, len(input_ids):].tolist()

    # Nanoserve with Contiguous KV Cache
    cached_tokens = cached_generate(
        nanoserve_model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
    )

    assert cached_tokens == hf_tokens, (
        f"Prompt '{prompt}' mismatch with KV cache:\n"
        f"HF:     {hf_tokens}\n"
        f"Cached: {cached_tokens}"
    )


def test_kv_cache_speedup(shared_models):
    """Sanity check: verify that contiguous KV cache provides noticeable speedup over naive O(S^2)."""
    tokenizer, _, nanoserve_model = shared_models
    prompt = "In computer science, algorithms are fundamental procedures that solve problems by"
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()
    max_new_tokens = 24
    eos_token_id = nanoserve_model.config.eos_token_id

    # Time naive generation (from P1)
    from tests.test_p1_correctness import naive_generate
    t0 = time.perf_counter()
    naive_tokens = naive_generate(nanoserve_model, input_ids, max_new_tokens, eos_token_id)
    t_naive = time.perf_counter() - t0

    # Time cached generation
    t0 = time.perf_counter()
    cached_tokens = cached_generate(nanoserve_model, input_ids, max_new_tokens, eos_token_id)
    t_cached = time.perf_counter() - t0

    assert naive_tokens == cached_tokens, "Tokens between naive and cached must match exactly"
    speedup = t_naive / t_cached
    print(f"\n[KV Cache Benchmark] Naive: {t_naive:.3f}s | Cached: {t_cached:.3f}s | Speedup: {speedup:.2f}x")
    assert speedup > 1.2, f"Expected KV cache speedup > 1.2x, got {speedup:.2f}x"

