"""Baselines for honest inference benchmarking: HuggingFace static/sequential vs Nanoserve continuous batching."""

import time
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoserve.engine import LLMEngine
from nanoserve.types import SamplingParams


def run_huggingface_baseline(
    workload: List[Tuple[str, str, SamplingParams, float]],
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
) -> Dict[str, float]:
    """Run standard HuggingFace causal LM generation sequentially."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    total_tokens = 0
    t0 = time.perf_counter()
    per_request_latencies = []

    for req_id, prompt, params, _ in workload:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        req_start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=params.max_new_tokens,
                do_sample=False,
                pad_token_id=model.config.eos_token_id,
            )

        req_elapsed = time.perf_counter() - req_start
        gen_len = outputs.shape[1] - input_ids.shape[1]
        total_tokens += gen_len
        per_request_latencies.append(req_elapsed)

    total_wall_time = time.perf_counter() - t0
    throughput = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

    return {
        "engine": "HuggingFace (sequential/static)",
        "total_requests": len(workload),
        "total_tokens": total_tokens,
        "wall_time_s": total_wall_time,
        "throughput_tokens_per_s": throughput,
        "mean_latency_s": sum(per_request_latencies) / len(per_request_latencies),
    }


def run_nanoserve_baseline(
    workload: List[Tuple[str, str, SamplingParams, float]],
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    max_batch_size: int = 8,
    num_blocks: int = 256,
) -> Dict[str, float]:
    """Run Nanoserve with continuous batching scheduler and paged KV cache."""
    engine = LLMEngine(
        model_name_or_path=model_name,
        num_blocks=num_blocks,
        block_size=16,
        max_batch_size=max_batch_size,
    )

    t0 = time.perf_counter()

    # Sort workload by simulated arrival time
    sorted_workload = sorted(workload, key=lambda x: x[3])
    workload_idx = 0
    sim_time = 0.0

    # Add first wave
    while workload_idx < len(sorted_workload) and sorted_workload[workload_idx][3] <= 0.5:
        r_id, prompt, params, _ = sorted_workload[workload_idx]
        engine.add_request(r_id, prompt, params)
        workload_idx += 1

    ttfts = []
    itls = []
    step_num = 0

    while engine.has_unfinished_requests():
        engine.step()
        step_num += 1

        # Simulate arrival in continuous batches every 2 steps
        if step_num % 2 == 0 and workload_idx < len(sorted_workload):
            batch_to_add = min(4, len(sorted_workload) - workload_idx)
            for _ in range(batch_to_add):
                r_id, prompt, params, _ = sorted_workload[workload_idx]
                engine.add_request(r_id, prompt, params)
                workload_idx += 1

    total_wall_time = time.perf_counter() - t0
    total_tokens = 0
    latencies = []

    for req in engine.all_requests.values():
        num_gen = len(req.output_token_ids)
        total_tokens += num_gen
        if req.first_token_time and req.arrival_time:
            ttfts.append((req.first_token_time - req.arrival_time) * 1000.0)
        if req.finish_time and req.arrival_time:
            dur = req.finish_time - req.arrival_time
            latencies.append(dur)
            if num_gen > 1:
                itls.append((dur * 1000.0 - ttfts[-1]) / (num_gen - 1))

    throughput = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

    return {
        "engine": "Nanoserve (Continuous Batching + Paged KV)",
        "total_requests": len(workload),
        "total_tokens": total_tokens,
        "wall_time_s": total_wall_time,
        "throughput_tokens_per_s": throughput,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "mean_itl_ms": sum(itls) / len(itls) if itls else 0.0,
        "preemptions": engine.scheduler.num_preemptions,
    }

