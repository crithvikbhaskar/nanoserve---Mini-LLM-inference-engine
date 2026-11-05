"""Synthetic and standard workloads for benchmarking inference engines."""

import random
from typing import List, Tuple
from nanoserve.types import SamplingParams

BENCH_PROMPTS = [
    "The process of photosynthesis in green plants involves",
    "Explain the theory of general relativity and how mass bends space-time",
    "In modern computer architectures, cache coherence protocols such as MESI",
    "The Roman Empire expanded across the Mediterranean basin by utilizing",
    "Gradient descent optimization in deep neural networks relies on the chain rule",
    "Quantum entanglement describes a physical phenomenon where pairs of particles",
    "The industrial revolution transformed agrarian economies into industrial powerhouses",
    "In distributed systems, the CAP theorem states that a distributed datastore can only provide",
    "DNA replication is a semiconservative process in which each strand of the double helix",
    "Microservice architectures contrast with monolithic architectures primarily in their",
    "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth",
    "Convolutional neural networks are specialized for processing data that has a known grid-like topology",
    "Transformer architectures rely entirely on self-attention mechanisms to compute representations",
    "Operating systems manage hardware resources through abstraction layers including virtual memory",
    "TCP provides reliable, ordered, and error-checked delivery of a stream of octets between applications",
    "The mitochondria is often referred to as the powerhouse of the cell because",
]


def generate_workload(
    num_requests: int = 64,
    min_tokens: int = 16,
    max_tokens: int = 32,
    seed: int = 42,
) -> List[Tuple[str, str, SamplingParams, float]]:
    """Generate a reproducible workload with mixed prompts and Poisson arrival timestamps.
    
    Returns:
        List of (request_id, prompt, sampling_params, arrival_time)
    """
    rng = random.Random(seed)
    workload = []
    current_time = 0.0

    for i in range(num_requests):
        prompt = rng.choice(BENCH_PROMPTS)
        max_new_tokens = rng.randint(min_tokens, max_tokens)
        params = SamplingParams(
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )
        # Inter-arrival interval drawn from exponential distribution (Poisson process, lambda=4 req/s)
        inter_arrival = rng.expovariate(4.0)
        current_time += inter_arrival
        workload.append((f"bench_{i:03d}", prompt, params, current_time))

    return workload

