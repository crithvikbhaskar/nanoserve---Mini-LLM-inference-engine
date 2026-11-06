"""Generate honest publication-quality benchmark plots from real run data."""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt


def plot_throughput_comparison(
    hf_throughput: float,
    nanoserve_throughput: float,
    output_path: Path,
):
    """Bar chart comparing tokens/sec throughput."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    engines = ["HuggingFace (Sequential)", "Nanoserve (Continuous Batching)"]
    throughputs = [hf_throughput, nanoserve_throughput]
    colors = ["#94a3b8", "#3b82f6"]

    bars = ax.bar(engines, throughputs, color=colors, width=0.45, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylabel("Throughput (tokens / second)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Throughput on CPU (SmolLM2-135M)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f} tok/s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    speedup = nanoserve_throughput / hf_throughput if hf_throughput > 0 else 1.0
    ax.text(
        0.5,
        0.88,
        f"Continuous Batching Speedup: {speedup:.2f}x",
        transform=ax.transAxes,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e0f2fe", edgecolor="#0284c7"),
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_latency_metrics(
    ttft_ms: float,
    itl_ms: float,
    output_path: Path,
):
    """Bar chart showing Time-To-First-Token and Inter-Token Latency."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    metrics = ["TTFT\n(Time to First Token)", "ITL\n(Inter-Token Latency)"]
    values = [ttft_ms, itl_ms]
    colors = ["#f59e0b", "#10b981"]

    bars = ax.bar(metrics, values, color=colors, width=0.4, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylabel("Latency (milliseconds)", fontsize=11, fontweight="bold")
    ax.set_title("Nanoserve Latency Breakdown (SmolLM2-135M CPU)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_memory_fragmentation(
    output_path: Path,
):
    """Illustrate internal memory fragmentation: Paged KV cache vs Contiguous pre-allocated cache."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    seq_lengths = [10, 25, 40, 65, 80, 100, 128]
    max_context = 128
    block_size = 16

    contiguous_allocated = [max_context] * len(seq_lengths)
    paged_allocated = [((l + block_size - 1) // block_size) * block_size for l in seq_lengths]

    ax.plot(seq_lengths, contiguous_allocated, label="Contiguous (Pre-allocated Max Context)", color="#ef4444", linewidth=2.5, linestyle="--")
    ax.plot(seq_lengths, paged_allocated, label="Paged KV Cache (Block size = 16)", color="#3b82f6", linewidth=2.5, marker="o")
    ax.plot(seq_lengths, seq_lengths, label="Theoretical Optimum (Exact Length)", color="#10b981", linewidth=1.5, linestyle=":")

    ax.fill_between(seq_lengths, paged_allocated, contiguous_allocated, color="#ef4444", alpha=0.15, label="Contiguous Waste / Internal Fragmentation")
    ax.set_xlabel("Actual Generated Sequence Length (tokens)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Memory Reserved (tokens)", fontsize=11, fontweight="bold")
    ax.set_title("Memory Efficiency: Paged KV Cache vs Contiguous Allocation", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

