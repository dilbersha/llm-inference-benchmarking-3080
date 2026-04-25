"""
Experiment 5A: Per-Layer Quantization Sensitivity Map

Question: Which transformer layers are fragile to quantization
and which can be aggressively compressed?

Produces a "sensitivity heatmap" — the visual that gets LinkedIn engagement.

Simulates per-layer quantization by adding noise proportional to
the precision loss, measuring output divergence from baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


def simulate_layer_quantization(
    weights: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """
    Simulate quantization of a weight tensor to N bits.

    Instead of actual quantization (which needs specific formats),
    we add calibrated noise that matches the quantization error
    distribution at each bit width.

    Quantization error ≈ Uniform(-Δ/2, Δ/2) where Δ = range / 2^bits
    """
    if bits >= 16:
        return weights  # No quantization

    w_range = weights.max() - weights.min()
    delta = w_range / (2 ** bits)

    # Add uniform noise matching quantization error
    noise = (torch.rand_like(weights) - 0.5) * delta
    return weights + noise


def create_synthetic_transformer(
    num_layers: int,
    hidden_dim: int,
    intermediate_dim: int,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """
    Create synthetic transformer layer weights.

    Each layer has:
    - q_proj, k_proj, v_proj, o_proj (attention)
    - gate_proj, up_proj, down_proj (MLP)

    Weight distributions mimic real models:
    - Early layers: smaller magnitude, more uniform
    - Middle layers: moderate magnitude
    - Late layers: larger magnitude, more concentrated
    """
    torch.manual_seed(seed)
    layers = []

    for i in range(num_layers):
        layer_ratio = i / max(num_layers - 1, 1)

        # Scale weights by layer depth (mimics real LLM weight distributions)
        base_scale = 0.02 * (1 + layer_ratio * 0.5)

        # Add some outliers in certain layers (mimics activation outliers)
        outlier_scale = 0.1 if (i % 4 == 0) else 0.0

        layer = {}
        for name, shape in [
            ("q_proj", (hidden_dim, hidden_dim)),
            ("k_proj", (hidden_dim, hidden_dim)),
            ("v_proj", (hidden_dim, hidden_dim)),
            ("o_proj", (hidden_dim, hidden_dim)),
            ("gate_proj", (intermediate_dim, hidden_dim)),
            ("up_proj", (intermediate_dim, hidden_dim)),
            ("down_proj", (hidden_dim, intermediate_dim)),
        ]:
            w = torch.randn(*shape) * base_scale
            # Add outliers
            if outlier_scale > 0:
                outlier_mask = torch.rand_like(w) < 0.001
                w[outlier_mask] *= outlier_scale / base_scale * 10
            layer[name] = w

        layers.append(layer)

    return layers


def measure_layer_sensitivity(
    layers: list[dict[str, torch.Tensor]],
    layer_idx: int,
    bits: int,
    input_data: torch.Tensor,
) -> dict:
    """
    Quantize one layer to `bits` precision, run a forward-like pass,
    and measure output divergence from the full-precision baseline.
    """
    # Baseline: full precision forward through all layers
    x_baseline = input_data.clone()
    for i, layer in enumerate(layers):
        # Simplified forward: x = x @ W_q + x @ W_v (attention-like)
        # Then x = x @ gate_proj * x @ up_proj (MLP-like)
        attn_out = x_baseline @ layer["q_proj"].T + x_baseline @ layer["v_proj"].T
        attn_out = F.layer_norm(attn_out, [attn_out.shape[-1]])
        mlp_out = F.silu(x_baseline @ layer["gate_proj"].T) * (x_baseline @ layer["up_proj"].T)
        mlp_out = mlp_out @ layer["down_proj"].T
        x_baseline = F.layer_norm(x_baseline + attn_out + mlp_out, [x_baseline.shape[-1]])

    # Quantized: same pass but layer_idx is quantized
    x_quant = input_data.clone()
    for i, layer in enumerate(layers):
        if i == layer_idx:
            # Quantize this layer's weights
            q_layer = {k: simulate_layer_quantization(v, bits) for k, v in layer.items()}
        else:
            q_layer = layer

        attn_out = x_quant @ q_layer["q_proj"].T + x_quant @ q_layer["v_proj"].T
        attn_out = F.layer_norm(attn_out, [attn_out.shape[-1]])
        mlp_out = F.silu(x_quant @ q_layer["gate_proj"].T) * (x_quant @ q_layer["up_proj"].T)
        mlp_out = mlp_out @ q_layer["down_proj"].T
        x_quant = F.layer_norm(x_quant + attn_out + mlp_out, [x_quant.shape[-1]])

    # Measure divergence
    cos_sim = F.cosine_similarity(
        x_baseline.reshape(1, -1),
        x_quant.reshape(1, -1),
    ).item()

    mse = F.mse_loss(x_quant, x_baseline).item()

    # KL divergence of output distributions
    base_probs = F.softmax(x_baseline.mean(dim=0), dim=-1).clamp(min=1e-10)
    quant_probs = F.softmax(x_quant.mean(dim=0), dim=-1).clamp(min=1e-10)
    kl_div = F.kl_div(quant_probs.log(), base_probs, reduction="sum").item()

    # Weight statistics for this layer
    layer = layers[layer_idx]
    all_weights = torch.cat([v.reshape(-1) for v in layer.values()])
    weight_range = (all_weights.max() - all_weights.min()).item()
    weight_std = all_weights.std().item()
    outlier_pct = ((all_weights.abs() > 3 * all_weights.std()).float().mean().item()) * 100

    return {
        "cosine_similarity": cos_sim,
        "mse": mse,
        "kl_divergence": kl_div,
        "sensitivity": 1.0 - cos_sim,  # Higher = more sensitive
        "weight_range": weight_range,
        "weight_std": weight_std,
        "outlier_pct": outlier_pct,
    }


def run_quantization_sensitivity_experiment():
    """Run the per-layer quantization sensitivity experiment."""
    runner = ExperimentRunner(
        name="quantization_sensitivity",
        description=(
            "Per-layer quantization sensitivity mapping. For each layer, "
            "quantize to [2,3,4,8] bits and measure output divergence. "
            "Produces a sensitivity heatmap showing which layers are fragile."
        ),
    )

    # Model configurations
    model_configs = [
        {"name": "12L-768H", "layers": 12, "hidden": 768, "intermediate": 3072},
        {"name": "24L-1024H", "layers": 24, "hidden": 1024, "intermediate": 4096},
    ]

    bit_widths = [2, 3, 4, 6, 8, 16]
    seq_len = 32  # Sequence length for test input

    for model_cfg in model_configs:
        print(f"\nModel: {model_cfg['name']}")
        layers = create_synthetic_transformer(
            num_layers=model_cfg["layers"],
            hidden_dim=model_cfg["hidden"],
            intermediate_dim=model_cfg["intermediate"],
        )

        # Create test input
        torch.manual_seed(42)
        test_input = torch.randn(seq_len, model_cfg["hidden"]) * 0.1

        for layer_idx in range(model_cfg["layers"]):
            for bits in bit_widths:
                config = {
                    "model": model_cfg["name"],
                    "num_layers": model_cfg["layers"],
                    "hidden_dim": model_cfg["hidden"],
                    "layer_idx": layer_idx,
                    "bits": bits,
                    "layer_position": layer_idx / model_cfg["layers"],
                }

                with runner.trial(config) as trial:
                    result = measure_layer_sensitivity(
                        layers, layer_idx, bits, test_input
                    )
                    for k, v in result.items():
                        trial.record(k, v)

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_quantization_sensitivity_experiment()
