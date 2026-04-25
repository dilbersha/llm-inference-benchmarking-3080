"""
Visualization suite for experiment results.

Generates publication-quality charts from experiment JSON data.
Run: python -m src.experiments.visualize
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Style ────────────────────────────────────────────────────────────
plt.style.use("dark_background")
COLORS = {
    "primary": "#00D4FF",
    "secondary": "#FF6B6B",
    "tertiary": "#50FA7B",
    "quaternary": "#FFB86C",
    "purple": "#BD93F9",
    "pink": "#FF79C6",
    "gray": "#6272A4",
}
PALETTE = list(COLORS.values())


def _setup_chart(title: str, xlabel: str, ylabel: str, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15, color="white")
    ax.set_xlabel(xlabel, fontsize=12, color="#ccc")
    ax.set_ylabel(ylabel, fontsize=12, color="#ccc")
    ax.tick_params(colors="#999")
    ax.grid(True, alpha=0.15, color="#555")
    for spine in ax.spines.values():
        spine.set_color("#444")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    return fig, ax


def plot_token_confidence(data_path: str, output_dir: str = "./reports/charts"):
    """Generate token confidence charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Chart 1: Skip Rate vs Threshold (by profile) ─────────────
    fig, ax = _setup_chart(
        "Token Confidence: Skip Rate vs Threshold",
        "Confidence Threshold",
        "Skip Rate (%)",
    )

    for i, profile in enumerate(["confident", "natural", "uncertain"]):
        trials = [t for t in data["trials"]
                  if t["config"]["profile"] == profile
                  and t["config"]["temperature"] == 0.3
                  and t["config"]["vocab_size"] == 32000]
        trials.sort(key=lambda x: x["config"]["threshold"])

        thresholds = [t["config"]["threshold"] for t in trials]
        skip_rates = [t["metrics"]["skip_rate"] * 100 for t in trials]

        ax.plot(thresholds, skip_rates, "o-", color=PALETTE[i],
                label=f"{profile}", linewidth=2, markersize=6)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(out / "token_confidence_skip_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ token_confidence_skip_rate.png")

    # ── Chart 2: Skip Rate vs Temperature (at threshold=0.9) ─────
    fig, ax = _setup_chart(
        "Temperature Effect on Token Confidence",
        "Temperature",
        "Skip Rate (%)",
    )

    for i, profile in enumerate(["confident", "natural"]):
        trials = [t for t in data["trials"]
                  if t["config"]["profile"] == profile
                  and t["config"]["threshold"] == 0.9
                  and t["config"]["vocab_size"] == 32000]
        trials.sort(key=lambda x: x["config"]["temperature"])

        temps = [t["config"]["temperature"] for t in trials]
        skip_rates = [t["metrics"]["skip_rate"] * 100 for t in trials]

        ax.plot(temps, skip_rates, "s-", color=PALETTE[i],
                label=f"{profile}", linewidth=2, markersize=8)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(out / "token_confidence_temperature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ token_confidence_temperature.png")

    # ── Chart 3: Vocab Size Impact ────────────────────────────────
    fig, ax = _setup_chart(
        "Vocabulary Size Impact on Confidence",
        "Confidence Threshold",
        "Skip Rate (%)",
    )

    for i, (vs, label) in enumerate([(32000, "32K (Llama)"), (128256, "128K (GPT-4)")]):
        trials = [t for t in data["trials"]
                  if t["config"]["profile"] == "confident"
                  and t["config"]["temperature"] == 0.3
                  and t["config"]["vocab_size"] == vs]
        trials.sort(key=lambda x: x["config"]["threshold"])

        thresholds = [t["config"]["threshold"] for t in trials]
        skip_rates = [t["metrics"]["skip_rate"] * 100 for t in trials]

        ax.plot(thresholds, skip_rates, "D-", color=PALETTE[i],
                label=f"vocab={label}", linewidth=2, markersize=6)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(out / "token_confidence_vocab_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ token_confidence_vocab_size.png")


def plot_head_pruning(data_path: str, output_dir: str = "./reports/charts"):
    """Generate attention head pruning charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Chart: Quality vs Prune Fraction (by model size) ──────────
    fig, ax = _setup_chart(
        "Attention Head Pruning: Quality vs Heads Removed",
        "Fraction of Heads Pruned",
        "Quality Score (cosine similarity)",
    )

    for i, model in enumerate(["small", "medium", "large"]):
        trials = [t for t in data["trials"] if t["config"]["model"] == model]
        trials.sort(key=lambda x: x["config"]["prune_fraction"])

        fractions = [t["config"]["prune_fraction"] for t in trials]
        quality = [t["metrics"]["quality_score"] for t in trials]

        ax.plot(fractions, quality, "o-", color=PALETTE[i],
                label=f'{model} ({trials[0]["config"]["layers"]}L/{trials[0]["config"]["heads"]}H)',
                linewidth=2, markersize=6)

    # Mark the "cliff" region
    ax.axvspan(0, 0.15, alpha=0.1, color=PALETTE[1], label="Critical zone")
    ax.axhline(y=0.46, color="#666", linestyle="--", alpha=0.5, label="Quality plateau")

    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_xlim(-0.02, 0.82)
    fig.tight_layout()
    fig.savefig(out / "head_pruning_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ head_pruning_quality.png")

    # ── Chart: KL Divergence vs Prune Fraction ────────────────────
    fig, ax = _setup_chart(
        "Information Loss from Head Pruning",
        "Fraction of Heads Pruned",
        "KL Divergence (lower = better)",
    )

    for i, model in enumerate(["small", "medium", "large"]):
        trials = [t for t in data["trials"]
                  if t["config"]["model"] == model
                  and "kl_divergence" in t["metrics"]
                  and t["config"]["prune_fraction"] > 0]
        trials.sort(key=lambda x: x["config"]["prune_fraction"])

        fractions = [t["config"]["prune_fraction"] for t in trials]
        kl = [t["metrics"]["kl_divergence"] for t in trials]

        ax.plot(fractions, kl, "^-", color=PALETTE[i],
                label=f'{model}', linewidth=2, markersize=6)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "head_pruning_kl_div.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ head_pruning_kl_div.png")


def plot_kv_cache(data_path: str, output_dir: str = "./reports/charts"):
    """Generate KV cache eviction benchmark charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Chart: Quality vs Memory Saved (by policy) ────────────────
    fig, ax = _setup_chart(
        "KV Cache Eviction: Quality vs Memory Saved",
        "Memory Saved (%)",
        "Quality Retained (attention mass)",
    )

    policy_colors = {
        "full": COLORS["gray"],
        "window": COLORS["secondary"],
        "h2o": COLORS["primary"],
        "snapkv": COLORS["purple"],
        "pyramid": COLORS["quaternary"],
        "adaptive": COLORS["tertiary"],
    }

    for policy_name in ["window", "h2o", "snapkv", "pyramid", "adaptive"]:
        trials = [r for r in data["results"]
                  if r["policy_name"] == policy_name
                  and r["memory_saved_pct"] > 0]

        if not trials:
            continue

        mem_saved = [r["memory_saved_pct"] for r in trials]
        quality = [r["quality_score"] for r in trials]

        ax.scatter(mem_saved, quality,
                   color=policy_colors.get(policy_name, "#fff"),
                   label=policy_name, s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_xlim(-5, 100)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out / "kv_cache_quality_vs_memory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ kv_cache_quality_vs_memory.png")


def plot_quantization_sensitivity(data_path: str, output_dir: str = "./reports/charts"):
    """Generate quantization sensitivity heatmap — the LinkedIn chart."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Process each model
    for model_name in ["12L-768H", "24L-1024H"]:
        trials = [t for t in data["trials"] if t["config"]["model"] == model_name]
        if not trials:
            continue

        num_layers = trials[0]["config"]["num_layers"]
        bit_widths = sorted(set(t["config"]["bits"] for t in trials))
        bit_widths_no16 = [b for b in bit_widths if b < 16]

        # Build sensitivity matrix [layers × bits]
        matrix = np.zeros((num_layers, len(bit_widths_no16)))
        for t in trials:
            if t["config"]["bits"] >= 16:
                continue
            layer_idx = t["config"]["layer_idx"]
            bit_idx = bit_widths_no16.index(t["config"]["bits"])
            matrix[layer_idx, bit_idx] = t["metrics"]["sensitivity"]

        # ── Heatmap ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, max(6, num_layers * 0.35)))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        im = ax.imshow(
            matrix,
            cmap="RdYlGn_r",  # Red=sensitive, Green=safe
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=max(0.5, matrix.max()),
        )

        ax.set_xticks(range(len(bit_widths_no16)))
        ax.set_xticklabels([f"{b}-bit" for b in bit_widths_no16], color="#ccc")
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"L{i}" for i in range(num_layers)], color="#ccc", fontsize=8)

        ax.set_xlabel("Quantization Precision", fontsize=12, color="#ccc")
        ax.set_ylabel("Transformer Layer", fontsize=12, color="#ccc")
        ax.set_title(
            f"Quantization Sensitivity Map ({model_name})",
            fontsize=14, fontweight="bold", color="white", pad=15,
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Sensitivity (higher = more fragile)", color="#ccc")
        cbar.ax.tick_params(colors="#999")

        # Annotate cells with values
        for i in range(num_layers):
            for j in range(len(bit_widths_no16)):
                val = matrix[i, j]
                color = "white" if val > 0.3 else "#333"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6 if num_layers > 16 else 7, color=color)

        fig.tight_layout()
        fname = f"quant_sensitivity_{model_name.replace('-', '_')}.png"
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {fname}")

    # ── Line chart: sensitivity by layer position ─────────────────
    fig, ax = _setup_chart(
        "Quantization Sensitivity by Layer Position",
        "Layer Position (0=first, 1=last)",
        "Sensitivity (1 - cosine similarity)",
    )

    for i, bits in enumerate([2, 3, 4, 8]):
        trials = [t for t in data["trials"]
                  if t["config"]["bits"] == bits
                  and t["config"]["model"] == "24L-1024H"]
        trials.sort(key=lambda x: x["config"]["layer_position"])

        positions = [t["config"]["layer_position"] for t in trials]
        sensitivity = [t["metrics"]["sensitivity"] for t in trials]

        ax.plot(positions, sensitivity, "o-", color=PALETTE[i],
                label=f"{bits}-bit", linewidth=2, markersize=4)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    fig.tight_layout()
    fig.savefig(out / "quant_sensitivity_by_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ quant_sensitivity_by_position.png")


def plot_speculative_decoding(data_path: str, output_dir: str = "./reports/charts"):
    """Generate self-speculative decoding charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Speedup vs Quality by model ──────────────────────────────
    fig, ax = _setup_chart(
        "Self-Speculative Decoding: Speedup vs Quality",
        "Cosine Similarity to Full Model",
        "Effective Speedup (×)",
    )

    for i, model in enumerate(["12L-768H", "24L-1024H", "32L-2048H"]):
        trials = [t for t in data["trials"] if t["config"]["model"] == model]
        trials.sort(key=lambda x: x["config"]["exit_layer"])

        quality = [t["metrics"]["cosine_similarity"] for t in trials]
        speedup = [t["metrics"]["effective_speedup"] for t in trials]

        ax.plot(quality, speedup, "o-", color=PALETTE[i],
                label=model, linewidth=2, markersize=5)

    ax.axhline(y=1.0, color="#666", linestyle="--", alpha=0.5, label="Break-even")
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444")
    fig.tight_layout()
    fig.savefig(out / "speculative_speedup_vs_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ speculative_speedup_vs_quality.png")

    # ── Acceptance rate by exit position ──────────────────────────
    fig, ax = _setup_chart(
        "Draft Token Acceptance Rate by Exit Layer",
        "Exit Layer Position (0=first, 1=last)",
        "Acceptance Rate (%)",
    )

    for i, model in enumerate(["12L-768H", "24L-1024H", "32L-2048H"]):
        trials = [t for t in data["trials"] if t["config"]["model"] == model]
        trials.sort(key=lambda x: x["config"]["exit_position"])

        positions = [t["config"]["exit_position"] for t in trials]
        acceptance = [t["metrics"]["acceptance_rate"] * 100 for t in trials]

        ax.plot(positions, acceptance, "s-", color=PALETTE[i],
                label=model, linewidth=2, markersize=5)

    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(out / "speculative_acceptance_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ speculative_acceptance_rate.png")


def plot_transfer_benchmark(data_path: str, output_dir: str = "./reports/charts"):
    """Generate memory transfer benchmark charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = _setup_chart(
        "PCIe Transfer: Pinned vs Paged Memory (RTX 3080)",
        "Transfer Size (MB)",
        "Bandwidth (GB/s)",
    )

    configs = [
        ("gpu_to_cpu", False, "GPU→CPU (paged)", COLORS["secondary"], "o-"),
        ("gpu_to_cpu", True, "GPU→CPU (pinned)", COLORS["primary"], "o--"),
        ("cpu_to_gpu", False, "CPU→GPU (paged)", COLORS["quaternary"], "s-"),
        ("cpu_to_gpu", True, "CPU→GPU (pinned)", COLORS["tertiary"], "s--"),
    ]

    for direction, pinned, label, color, marker in configs:
        trials = [t for t in data["trials"]
                  if t["config"]["direction"] == direction
                  and t["config"]["use_pinned_memory"] == pinned]
        trials.sort(key=lambda x: x["config"]["size_mb"])

        sizes = [t["config"]["size_mb"] for t in trials]
        bw = [t["metrics"]["bandwidth_gbps_mean"] for t in trials]

        ax.plot(sizes, bw, marker, color=color, label=label, linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#444")
    fig.tight_layout()
    fig.savefig(out / "transfer_bandwidth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ transfer_bandwidth.png")


def plot_reasoning_waste(data_path: str, output_dir: str = "./reports/charts"):
    """Generate reasoning token waste charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Answer correctness by tokens removed ─────────────────────
    fig, ax = _setup_chart(
        "Reasoning Token Waste: Can You Remove Thinking Tokens?",
        "Tokens Removed (%)",
        "Answer Still Correct (%)",
    )

    for i, difficulty in enumerate(["easy", "medium", "hard"]):
        for j, strategy in enumerate(["keep_first", "importance_sample"]):
            trials = [t for t in data["trials"]
                      if t["config"]["difficulty"] == difficulty
                      and t["config"]["strategy"] == strategy
                      and t["config"]["chain_length"] == 200]
            trials.sort(key=lambda x: x["config"]["keep_fraction"])

            removed_pct = [t["metrics"]["tokens_removed_pct"] for t in trials]
            correct = [t["metrics"]["answer_correct"] * 100 for t in trials]

            ls = "-" if strategy == "keep_first" else "--"
            label = f"{difficulty} ({strategy.replace('_', ' ')})"
            ax.plot(removed_pct, correct, f"o{ls}", color=PALETTE[i * 2 + j],
                    label=label, linewidth=2, markersize=5)

    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", ncol=2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xlim(-5, 95)
    fig.tight_layout()
    fig.savefig(out / "reasoning_waste_correctness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ reasoning_waste_correctness.png")

    # ── Strategy comparison ──────────────────────────────────────
    fig, ax = _setup_chart(
        "Truncation Strategy Comparison (Medium Difficulty)",
        "Tokens Removed (%)",
        "Quality (Cosine Similarity)",
    )

    for i, strategy in enumerate(["keep_first", "keep_last", "remove_middle", "importance_sample"]):
        trials = [t for t in data["trials"]
                  if t["config"]["difficulty"] == "medium"
                  and t["config"]["strategy"] == strategy
                  and t["config"]["chain_length"] == 200]
        trials.sort(key=lambda x: x["config"]["keep_fraction"])

        removed = [t["metrics"]["tokens_removed_pct"] for t in trials]
        quality = [t["metrics"]["quality_cosine"] for t in trials]

        ax.plot(removed, quality, "o-", color=PALETTE[i],
                label=strategy.replace("_", " "), linewidth=2, markersize=5)

    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_xlim(-5, 95)
    fig.tight_layout()
    fig.savefig(out / "reasoning_strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ reasoning_strategy_comparison.png")


def generate_all_charts():
    """Generate all visualization charts."""
    print("Generating experiment visualizations...")
    print()

    # Find latest experiment files
    exp_dir = Path("./reports/experiments")
    charts_dir = "./reports/charts"

    tc_files = sorted(exp_dir.glob("token_confidence_*.json"))
    ah_files = sorted(exp_dir.glob("attention_head_*.json"))
    qs_files = sorted(exp_dir.glob("quantization_sensitivity_*.json"))
    sd_files = sorted(exp_dir.glob("self_speculative_*.json"))
    tb_files = sorted(exp_dir.glob("memory_transfer_*.json"))
    rw_files = sorted(exp_dir.glob("reasoning_token_waste_*.json"))
    kv_file = Path("./reports/kv_cache/initial_results.json")

    if tc_files:
        print("Token Confidence:")
        plot_token_confidence(str(tc_files[-1]), charts_dir)

    if ah_files:
        print("\nAttention Head Pruning:")
        plot_head_pruning(str(ah_files[-1]), charts_dir)

    if kv_file.exists():
        print("\nKV Cache Eviction:")
        plot_kv_cache(str(kv_file), charts_dir)

    if qs_files:
        print("\nQuantization Sensitivity:")
        plot_quantization_sensitivity(str(qs_files[-1]), charts_dir)

    if sd_files:
        print("\nSelf-Speculative Decoding:")
        plot_speculative_decoding(str(sd_files[-1]), charts_dir)

    if tb_files:
        print("\nMemory Transfer:")
        plot_transfer_benchmark(str(tb_files[-1]), charts_dir)

    if rw_files:
        print("\nReasoning Token Waste:")
        plot_reasoning_waste(str(rw_files[-1]), charts_dir)

    print(f"\n✅ All charts saved to {charts_dir}/")


if __name__ == "__main__":
    generate_all_charts()

