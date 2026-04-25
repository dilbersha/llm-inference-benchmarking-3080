"""
Experiment 3A: Self-Speculative Decoding Benchmark

Question: If you use early transformer layers as a "draft model"
instead of a separate small model, which exit layer gives the
best speedup-vs-quality tradeoff?

Simulates LayerSkip-style self-speculation by comparing outputs
from early layers against the full model's output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


def create_layer_representations(
    num_layers: int,
    hidden_dim: int,
    seq_len: int,
    num_sequences: int = 20,
    seed: int = 42,
) -> torch.Tensor:
    """
    Simulate per-layer hidden states for multiple sequences.

    In a real transformer, earlier layers capture local patterns
    and later layers capture global/semantic patterns. We model this
    with increasing refinement.

    Returns: [num_sequences, num_layers, seq_len, hidden_dim]
    """
    torch.manual_seed(seed)

    # Base signal (the "correct" final representation)
    final = torch.randn(num_sequences, seq_len, hidden_dim)

    # Build representations from first to last layer
    # Earlier layers = more noise, less signal
    representations = torch.zeros(num_sequences, num_layers, seq_len, hidden_dim)

    for layer in range(num_layers):
        layer_ratio = layer / (num_layers - 1)

        # Signal strength increases with depth
        signal_strength = layer_ratio ** 0.7  # Sublinear: early layers contribute more than nothing
        noise_strength = 1.0 - signal_strength

        # Build this layer's representation
        signal = final * signal_strength
        noise = torch.randn_like(final) * noise_strength * 0.5

        # Add layer-specific pattern (some layers are more "useful" than others)
        if layer % 4 == 0:
            # Every 4th layer adds a big refinement (mimics attention pattern changes)
            signal = signal + torch.randn_like(final) * 0.1

        representations[:, layer] = signal + noise

    return representations


def compute_speculative_metrics(
    representations: torch.Tensor,
    exit_layer: int,
    full_layers: int,
) -> dict:
    """
    Compare outputs from exit_layer vs full model.

    Measures:
    - Acceptance rate: how often early exit agrees with full model
    - Quality retention: cosine similarity of representations
    - Theoretical speedup: layers_skipped / total_layers
    """
    num_seq = representations.shape[0]

    # Full model output (last layer)
    full_output = representations[:, -1]  # [num_seq, seq_len, hidden_dim]
    # Early exit output
    exit_output = representations[:, exit_layer]

    # Token-level acceptance: argmax agreement
    # Project to "logits" (simplified: just the hidden states as logits)
    full_tokens = full_output.sum(dim=-1).argmax(dim=-1)  # pseudo-tokens
    exit_tokens = exit_output.sum(dim=-1).argmax(dim=-1)

    # Per-sequence acceptance rate
    acceptance_rates = []
    for i in range(num_seq):
        # For self-speculative: draft N tokens, verify all at once
        # Acceptance = fraction of drafted tokens that match full model
        full_logits = full_output[i]  # [seq_len, hidden_dim]
        exit_logits = exit_output[i]

        # Cosine similarity per position
        cos_per_pos = F.cosine_similarity(full_logits, exit_logits, dim=-1)

        # Accept if similarity > threshold (0.9 = very similar)
        accepted = (cos_per_pos > 0.9).float().mean().item()
        acceptance_rates.append(accepted)

    avg_acceptance = sum(acceptance_rates) / len(acceptance_rates)

    # Overall quality: representation similarity
    cos_sim = F.cosine_similarity(
        full_output.reshape(num_seq, -1),
        exit_output.reshape(num_seq, -1),
        dim=-1,
    ).mean().item()

    # MSE
    mse = F.mse_loss(exit_output, full_output).item()

    # Theoretical speedup
    layers_saved = full_layers - exit_layer - 1
    compute_saved_pct = layers_saved / full_layers * 100

    # Effective speedup (accounting for verification cost)
    # In self-speculation: draft with exit_layer, verify with remaining layers
    # Speedup = (N * draft_cost + 1 * verify_cost) vs (N * full_cost)
    # where draft_cost ≈ exit_layer/full_layers, verify_cost ≈ 1
    draft_tokens = 4  # Typical: draft 4 tokens at a time
    draft_cost = exit_layer / full_layers
    verify_cost = 1.0  # Full forward pass to verify
    naive_cost = draft_tokens * 1.0  # N full forward passes
    spec_cost = draft_tokens * draft_cost + verify_cost * (1 - avg_acceptance) + avg_acceptance
    effective_speedup = naive_cost / max(spec_cost, 0.01)

    return {
        "acceptance_rate": avg_acceptance,
        "cosine_similarity": cos_sim,
        "mse": mse,
        "compute_saved_pct": compute_saved_pct,
        "effective_speedup": effective_speedup,
        "layers_saved": layers_saved,
    }


def run_speculative_decoding_experiment():
    """Run self-speculative decoding benchmark."""
    runner = ExperimentRunner(
        name="self_speculative_decoding",
        description=(
            "Benchmarks self-speculative decoding (LayerSkip-style) across "
            "exit layers, model sizes, and draft token counts. Measures "
            "acceptance rate, quality retention, and effective speedup."
        ),
    )

    model_configs = [
        {"name": "12L-768H", "layers": 12, "hidden": 768},
        {"name": "24L-1024H", "layers": 24, "hidden": 1024},
        {"name": "32L-2048H", "layers": 32, "hidden": 2048},
    ]

    for model_cfg in model_configs:
        print(f"\nModel: {model_cfg['name']}")
        num_layers = model_cfg["layers"]

        reps = create_layer_representations(
            num_layers=num_layers,
            hidden_dim=model_cfg["hidden"],
            seq_len=64,
            num_sequences=20,
        )

        # Test every possible exit layer
        for exit_layer in range(1, num_layers - 1):
            exit_pct = exit_layer / num_layers

            config = {
                "model": model_cfg["name"],
                "num_layers": num_layers,
                "hidden_dim": model_cfg["hidden"],
                "exit_layer": exit_layer,
                "exit_position": round(exit_pct, 3),
            }

            with runner.trial(config) as trial:
                result = compute_speculative_metrics(reps, exit_layer, num_layers)
                for k, v in result.items():
                    trial.record(k, v)

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_speculative_decoding_experiment()
