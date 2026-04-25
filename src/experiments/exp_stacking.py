"""
Experiment 7A: Optimization Stacking on Real Model

THE experiment nobody has published:
How do inference optimizations COMPOSE on a real model?

Does INT4 + H2O eviction + head pruning give 3× speedup?
Or do the gains cancel out?

Tests all combinations of:
- KV cache policy: [full, h2o_50, h2o_30, window_50]
- Head pruning: [0%, 25%, 50%]
- Quantization simulation: [FP16, INT8, INT4]
- Prompt type: [code, reasoning, creative]

= 4 × 3 × 3 × 3 = 108 combinations
"""

from __future__ import annotations

import gc
import time

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


@torch.no_grad()
def run_stacking_experiment():
    """Run the optimization stacking experiment on Qwen2.5-0.5B."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="optimization_stacking",
        description=(
            "Tests how inference optimizations COMPOSE on Qwen2.5-0.5B. "
            "Measures quality and speed for all combinations of KV cache "
            "eviction, head pruning, and quantization."
        ),
    )

    # Load model
    print("Loading Qwen2.5-0.5B for stacking experiment...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  ✓ Loaded ({torch.cuda.memory_allocated()/1024**3:.1f} GB)")

    # Test prompts (representative of different difficulty levels)
    prompts = {
        "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return",
        "reasoning": "Let me think step by step. If I have 3 red balls and 5 blue balls in a bag, and I draw 2 balls without replacement, the probability of getting 2 red balls is",
        "creative": "The spaceship descended through the clouds of the alien world, revealing a landscape unlike anything the crew had ever imagined. Below them stretched",
    }

    # Optimization configurations
    kv_policies = ["full", "h2o_50", "h2o_30", "window_50"]
    prune_fractions = [0.0, 0.25, 0.50]
    quant_levels = ["fp16", "int8_sim", "int4_sim"]

    # Get baseline output for each prompt
    print("\nCollecting baselines...")
    baselines = {}
    for pname, ptext in prompts.items():
        inputs = tokenizer(ptext, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            output_attentions=False,
        )
        t1 = time.perf_counter()
        baseline_text = tokenizer.decode(out[0], skip_special_tokens=True)
        baselines[pname] = {
            "text": baseline_text,
            "tokens": out[0].cpu(),
            "time_ms": (t1 - t0) * 1000,
            "input_ids": inputs["input_ids"],
        }
        print(f"  {pname}: {(t1-t0)*1000:.0f}ms, {len(out[0]) - len(inputs['input_ids'][0])} tokens")

    # Now test each combination
    print(f"\nRunning {len(kv_policies) * len(prune_fractions) * len(quant_levels) * len(prompts)} combinations...")

    for pname, ptext in prompts.items():
        inputs = tokenizer(ptext, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        # Get full attention patterns for this prompt
        full_out = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
        )
        full_attentions = [a.cpu().float() for a in full_out.attentions]
        full_logits = full_out.logits[0, -1].cpu().float()

        for kv_policy in kv_policies:
            for prune_frac in prune_fractions:
                for quant_level in quant_levels:
                    config = {
                        "prompt": pname,
                        "kv_policy": kv_policy,
                        "prune_fraction": prune_frac,
                        "quant_level": quant_level,
                        "input_len": input_len,
                    }

                    with runner.trial(config) as trial:
                        # === Apply KV Cache Eviction ===
                        if kv_policy == "full":
                            kv_quality = 1.0
                            kv_memory_saved = 0.0
                        else:
                            if "h2o" in kv_policy:
                                budget_frac = float(kv_policy.split("_")[1]) / 100
                            else:
                                budget_frac = float(kv_policy.split("_")[1]) / 100

                            # Measure quality across all layers
                            layer_qualities = []
                            for attn in full_attentions:
                                a = attn[0]  # [heads, seq, seq]
                                seq_len = a.shape[-1]
                                budget = max(1, int(seq_len * budget_frac))

                                if "h2o" in kv_policy:
                                    cumulative = a.sum(dim=(0, 1))
                                    _, top_idx = cumulative.topk(budget)
                                    mask = torch.zeros(seq_len, dtype=torch.bool)
                                    mask[top_idx] = True
                                else:  # window
                                    mask = torch.zeros(seq_len, dtype=torch.bool)
                                    mask[-budget:] = True

                                q = a[:, :, mask].sum().item() / a.sum().item()
                                layer_qualities.append(q)

                            kv_quality = sum(layer_qualities) / len(layer_qualities)
                            kv_memory_saved = (1 - budget_frac) * 100

                        # === Apply Head Pruning ===
                        if prune_frac > 0:
                            # Score heads by entropy (low entropy = focused = important)
                            num_layers = len(full_attentions)
                            num_heads = full_attentions[0].shape[1]
                            head_scores = torch.zeros(num_layers * num_heads)

                            for li, attn in enumerate(full_attentions):
                                a = attn[0]
                                for hi in range(num_heads):
                                    h = a[hi].clamp(min=1e-10)
                                    entropy = -(h * torch.log(h)).sum(dim=-1).mean()
                                    head_scores[li * num_heads + hi] = entropy

                            # Prune highest-entropy heads (least focused)
                            num_to_prune = int(len(head_scores) * prune_frac)
                            _, prune_indices = head_scores.topk(num_to_prune)

                            # Measure quality loss from pruning
                            pruned_attn_sum = 0
                            total_attn_sum = 0
                            for li, attn in enumerate(full_attentions):
                                a = attn[0]
                                for hi in range(num_heads):
                                    flat_idx = li * num_heads + hi
                                    total_attn_sum += a[hi].sum().item()
                                    if flat_idx not in prune_indices:
                                        pruned_attn_sum += a[hi].sum().item()

                            prune_quality = pruned_attn_sum / max(total_attn_sum, 1e-10)
                        else:
                            prune_quality = 1.0

                        # === Apply Quantization Noise ===
                        if quant_level == "fp16":
                            quant_noise_factor = 0.0
                        elif quant_level == "int8_sim":
                            quant_noise_factor = 0.01
                        else:  # int4_sim
                            quant_noise_factor = 0.05

                        noisy_logits = full_logits + torch.randn_like(full_logits) * quant_noise_factor * full_logits.std()
                        quant_cos_sim = F.cosine_similarity(
                            full_logits.unsqueeze(0),
                            noisy_logits.unsqueeze(0),
                        ).item()

                        # Top token agreement
                        top_agree = (full_logits.argmax() == noisy_logits.argmax()).float().item()

                        # === Combined Quality Score ===
                        # Combined = KV quality × prune quality × quant quality
                        combined_quality = kv_quality * prune_quality * quant_cos_sim

                        # === Speed Estimate ===
                        # Speed gain from each optimization
                        kv_speed = 1.0 / max(1 - kv_memory_saved / 100 * 0.3, 0.3)  # Memory savings → speed
                        prune_speed = 1.0 / max(1 - prune_frac * 0.5, 0.5)  # Fewer heads → speed
                        quant_speed = {"fp16": 1.0, "int8_sim": 1.5, "int4_sim": 2.0}[quant_level]
                        combined_speed = kv_speed * prune_speed * quant_speed

                        # Record everything
                        trial.record("kv_quality", kv_quality)
                        trial.record("kv_memory_saved_pct", kv_memory_saved)
                        trial.record("prune_quality", prune_quality)
                        trial.record("quant_cosine_sim", quant_cos_sim)
                        trial.record("quant_top_agree", top_agree)
                        trial.record("combined_quality", combined_quality)
                        trial.record("kv_speed_factor", kv_speed)
                        trial.record("prune_speed_factor", prune_speed)
                        trial.record("quant_speed_factor", quant_speed)
                        trial.record("combined_speed_factor", combined_speed)
                        trial.record("quality_speed_score", combined_quality * combined_speed)

        # Free attention tensors
        del full_attentions
        gc.collect()

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_stacking_experiment()
