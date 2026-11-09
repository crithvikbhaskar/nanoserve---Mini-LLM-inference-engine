"""P1 Correctness Gate: Verify token-exact match between naive Nanoserve generation and HuggingFace generate."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanoserve.model import NanoserveModel

PROMPTS = [
    "Hello, my name is",
    "The sky is",
    "2 + 2 = ",
    "Python is a ",
    "The capital of Japan is",
    "In machine learning, gradient descent is",
    "The French revolution began in",
    "Quantum computing leverages the principles of",
    "To make a classic Margherita pizza, one needs",
    "The quick brown fox jumps over",
    "The capital of Australia is Canberra, which",
    "Photosynthesis is the biological process where",
    "Albert Einstein was awarded the Nobel Prize in Physics for",
    "Write a short summary explaining why sorting algorithms have a theoretical lower bound of O(n log n):",
    "Explain how continuous batching improves GPU utilization in LLM serving compared to static batching:",
    "Once upon a time in a distant kingdom, there lived an astronomer who",
    "The human cardiovascular system is responsible for",
    "In software engineering, SOLID design principles represent",
    "Database indexing with B+ trees ensures efficient retrieval because",
    "The Apollo 11 moon landing was accomplished by",
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


def naive_generate(model: NanoserveModel, input_ids: list[int], max_new_tokens: int, eos_token_id: int) -> list[int]:
    """Naive generation without KV cache (re-evaluates full sequence at each step)."""
    current_tokens = list(input_ids)
    generated = []

    for _ in range(max_new_tokens):
        x = torch.tensor([current_tokens], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)  # [1, S, V]
        next_token = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(next_token)
        current_tokens.append(next_token)
        if next_token == eos_token_id:
            break

    return generated


@pytest.mark.parametrize("prompt_idx, prompt", enumerate(PROMPTS))
def test_naive_vs_hf_token_exact(shared_models, prompt_idx, prompt):
    tokenizer, hf_model, nanoserve_model = shared_models
    max_new_tokens = 32  # 32 tokens per prompt to test across 20 prompts cleanly on CPU
    eos_token_id = nanoserve_model.config.eos_token_id

    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()

    # 1. HuggingFace greedy generation
    with torch.no_grad():
        hf_output = hf_model.generate(
            torch.tensor([input_ids]),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )
    hf_generated_tokens = hf_output[0, len(input_ids):].tolist()

    # 2. Nanoserve naive greedy generation
    nanoserve_generated_tokens = naive_generate(
        nanoserve_model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
    )

    # Assert token-exact match
    assert nanoserve_generated_tokens == hf_generated_tokens, (
        f"Prompt {prompt_idx} ('{prompt}') mismatch:\n"
        f"HF:        {hf_generated_tokens}\n"
        f"Nanoserve: {nanoserve_generated_tokens}"
    )

