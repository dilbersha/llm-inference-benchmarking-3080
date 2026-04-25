"""
Experiment 6A: Reasoning Token Waste Measurement

Question: How many "thinking" tokens do models actually need?
Can we truncate CoT reasoning and still get the right answer?

Simulates reasoning chains and measures the impact of progressive
truncation on answer correctness.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


def generate_reasoning_chain(
    difficulty: str,
    chain_length: int = 100,
    hidden_dim: int = 256,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate a reasoning chain as a sequence of hidden states.

    The "answer" depends on states accumulated through the chain.
    Earlier states build context, later states refine the answer.

    difficulty controls how much of the chain is actually needed:
    - "easy": answer crystallizes early (~30% of chain)
    - "medium": answer needs ~60% of chain
    - "hard": answer needs >80% of chain

    Returns: (chain: [chain_length, hidden_dim], answer: [hidden_dim])
    """
    torch.manual_seed(seed)

    chain = torch.zeros(chain_length, hidden_dim)

    # Build reasoning chain with decreasing importance
    if difficulty == "easy":
        critical_fraction = 0.3
    elif difficulty == "medium":
        critical_fraction = 0.6
    else:
        critical_fraction = 0.85

    critical_steps = int(chain_length * critical_fraction)

    # Critical steps: each contributes meaningfully to the answer
    for i in range(critical_steps):
        step_importance = 1.0 / (i + 1)  # Zipf-like: early steps matter most
        chain[i] = torch.randn(hidden_dim) * step_importance

    # Non-critical steps: redundant rephrasing / filler
    for i in range(critical_steps, chain_length):
        # Copy earlier patterns with noise (the "overthinking" phase)
        src_idx = i % critical_steps
        chain[i] = chain[src_idx] * 0.1 + torch.randn(hidden_dim) * 0.05

    # Ground-truth answer: sum of all critical steps
    answer = chain[:critical_steps].sum(dim=0)
    answer = F.normalize(answer, dim=0)

    return chain, answer


def truncate_and_evaluate(
    chain: torch.Tensor,
    answer: torch.Tensor,
    keep_fraction: float,
    strategy: str = "keep_first",
) -> dict:
    """
    Truncate reasoning chain and measure answer quality.

    Strategies:
    - "keep_first": keep first N% (remove end)
    - "keep_last": keep last N% (remove beginning)
    - "remove_middle": keep first 10% + last (N-10)%
    - "importance_sample": keep highest-norm states
    """
    chain_length = chain.shape[0]
    keep_count = max(1, int(chain_length * keep_fraction))

    if strategy == "keep_first":
        kept = chain[:keep_count]

    elif strategy == "keep_last":
        kept = chain[-keep_count:]

    elif strategy == "remove_middle":
        # Keep first 10% + last portion
        first_count = max(1, chain_length // 10)
        remaining = keep_count - first_count
        if remaining > 0:
            kept = torch.cat([chain[:first_count], chain[-remaining:]])
        else:
            kept = chain[:keep_count]

    elif strategy == "importance_sample":
        # Keep states with highest L2 norm (most "meaningful")
        norms = chain.norm(dim=-1)
        _, top_indices = norms.topk(keep_count)
        top_indices = top_indices.sort().values  # Maintain order
        kept = chain[top_indices]

    else:
        kept = chain[:keep_count]

    # Compute answer from truncated chain
    truncated_answer = kept.sum(dim=0)
    truncated_answer = F.normalize(truncated_answer, dim=0)

    # Quality metrics
    cos_sim = F.cosine_similarity(
        answer.unsqueeze(0), truncated_answer.unsqueeze(0)
    ).item()

    mse = F.mse_loss(truncated_answer, answer).item()

    # Is the answer "correct"? (cosine sim > 0.95)
    correct = 1.0 if cos_sim > 0.95 else 0.0

    return {
        "quality_cosine": cos_sim,
        "quality_mse": mse,
        "answer_correct": correct,
        "tokens_kept": keep_count,
        "tokens_removed": chain_length - keep_count,
        "tokens_removed_pct": (1 - keep_fraction) * 100,
    }


def run_reasoning_waste_experiment():
    """Run reasoning token waste measurement."""
    runner = ExperimentRunner(
        name="reasoning_token_waste",
        description=(
            "Measures how many reasoning tokens can be removed without "
            "changing the answer. Tests multiple truncation strategies "
            "across easy/medium/hard problems."
        ),
    )

    difficulties = ["easy", "medium", "hard"]
    chain_lengths = [50, 100, 200, 500]
    keep_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    strategies = ["keep_first", "keep_last", "remove_middle", "importance_sample"]

    for difficulty in difficulties:
        for chain_len in chain_lengths:
            chain, answer = generate_reasoning_chain(
                difficulty=difficulty,
                chain_length=chain_len,
                seed=42,
            )

            for strategy in strategies:
                for keep_frac in keep_fractions:
                    config = {
                        "difficulty": difficulty,
                        "chain_length": chain_len,
                        "keep_fraction": keep_frac,
                        "strategy": strategy,
                    }

                    with runner.trial(config) as trial:
                        result = truncate_and_evaluate(
                            chain, answer, keep_frac, strategy
                        )
                        for k, v in result.items():
                            trial.record(k, v)

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_reasoning_waste_experiment()
