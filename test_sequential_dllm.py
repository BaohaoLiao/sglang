#!/usr/bin/env python3
"""
Test script to demonstrate the Sequential DLLM algorithm.

This script shows how the Sequential algorithm unmasks tokens from left to right
in a deterministic order, compared to the LowConfidence algorithm which uses
confidence scores.
"""

import torch
import numpy as np


def simulate_sequential_unmasking(block_size=4, mask_id=156895):
    """Simulate Sequential unmasking strategy."""
    print("=" * 60)
    print("SEQUENTIAL UNMASKING STRATEGY")
    print("=" * 60)
    print(f"Block size: {block_size}")
    print(f"Mask token ID: {mask_id}")
    print()

    # Simulated input with all masks
    input_ids = torch.full((block_size,), mask_id, dtype=torch.long)
    print(f"Initial state: {input_ids.tolist()}")
    print()

    # Simulate unmasking process
    for iteration in range(block_size):
        # Find masked positions
        mask_index = input_ids == mask_id
        masked_positions = torch.where(mask_index)[0]

        if len(masked_positions) == 0:
            print(f"Iteration {iteration}: All tokens unmasked!")
            break

        # Sequential: select first (leftmost) masked position
        first_masked_idx = masked_positions[0]

        # Simulate predicted token (random for demonstration)
        predicted_token = torch.randint(100, 1000, (1,)).item()

        # Unmask
        input_ids[first_masked_idx] = predicted_token

        print(f"Iteration {iteration}:")
        print(f"  - First masked position: {first_masked_idx.item()}")
        print(f"  - Predicted token: {predicted_token}")
        print(f"  - Current state: {input_ids.tolist()}")
        print(f"  - Masked count: {mask_index.sum().item()}")
        print()

    print(f"Final state: {input_ids.tolist()}")
    print()


def simulate_low_confidence_unmasking(block_size=4, mask_id=156895):
    """Simulate LowConfidence unmasking strategy."""
    print("=" * 60)
    print("LOW CONFIDENCE UNMASKING STRATEGY")
    print("=" * 60)
    print(f"Block size: {block_size}")
    print(f"Mask token ID: {mask_id}")
    print()

    # Simulated input with all masks
    input_ids = torch.full((block_size,), mask_id, dtype=torch.long)
    print(f"Initial state: {input_ids.tolist()}")
    print()

    # Simulate unmasking process
    for iteration in range(block_size):
        # Find masked positions
        mask_index = input_ids == mask_id
        masked_positions = torch.where(mask_index)[0]

        if len(masked_positions) == 0:
            print(f"Iteration {iteration}: All tokens unmasked!")
            break

        # Simulate confidence scores (random for demonstration)
        confidence = torch.rand(block_size)
        # Set confidence to -inf for unmasked positions
        confidence = torch.where(mask_index, confidence, torch.tensor(-np.inf))

        # Select highest confidence masked token
        selected_idx = torch.argmax(confidence)
        selected_confidence = confidence[selected_idx].item()

        # Simulate predicted token
        predicted_token = torch.randint(100, 1000, (1,)).item()

        # Unmask
        input_ids[selected_idx] = predicted_token

        print(f"Iteration {iteration}:")
        print(f"  - Confidence scores: {confidence.tolist()}")
        print(f"  - Selected position: {selected_idx.item()} (confidence: {selected_confidence:.4f})")
        print(f"  - Predicted token: {predicted_token}")
        print(f"  - Current state: {input_ids.tolist()}")
        print(f"  - Masked count: {mask_index.sum().item()}")
        print()

    print(f"Final state: {input_ids.tolist()}")
    print()


def compare_strategies():
    """Compare Sequential vs LowConfidence strategies."""
    print("\n")
    print("#" * 60)
    print("# COMPARISON: Sequential vs LowConfidence")
    print("#" * 60)
    print()

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    simulate_sequential_unmasking(block_size=4)

    print("\n")

    torch.manual_seed(42)
    np.random.seed(42)

    simulate_low_confidence_unmasking(block_size=4)

    print("=" * 60)
    print("KEY DIFFERENCES:")
    print("=" * 60)
    print("1. Sequential:")
    print("   - Deterministic: always unmasks left-to-right")
    print("   - Predictable: position 0 → 1 → 2 → 3")
    print("   - No confidence needed: doesn't use model scores")
    print()
    print("2. LowConfidence:")
    print("   - Non-deterministic: depends on confidence scores")
    print("   - Adaptive: unmasks most confident token first")
    print("   - Quality-driven: uses model uncertainty to guide")
    print("=" * 60)


if __name__ == "__main__":
    compare_strategies()
