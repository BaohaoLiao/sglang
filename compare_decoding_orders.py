#!/usr/bin/env python3
"""
Compare Sequential vs LowConfidence Decoding Orders

This script demonstrates the key difference between the two algorithms:
- Sequential: Always [0, 1, 2, 3, ...] (deterministic)
- LowConfidence: Variable (e.g., [2, 0, 3, 1, ...]) based on confidence scores

Usage:
    python compare_decoding_orders.py --model /path/to/llada2 --prompt "Hello world"
"""

import argparse
import sglang


def compare_algorithms(model_path, prompt, block_size=4):
    """Run the same prompt with both algorithms and show decoding orders."""

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         Sequential vs LowConfidence - Decoding Order             ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    print(f"Prompt: '{prompt}'")
    print(f"Block size: {block_size}")
    print()

    # Run with SequentialVerbose
    print("=" * 70)
    print("RUNNING WITH: SequentialVerbose")
    print("=" * 70)
    print("Expected: Decoding order will be [0, 1, 2, 3, ...]")
    print()

    llm_seq = sglang.Engine(
        model_path=model_path,
        dllm_algorithm="SequentialVerbose",
        dllm_block_size=block_size,
    )

    output_seq = llm_seq.generate(prompt, {"temperature": 1.0, "max_new_tokens": block_size})

    print("\n" + "=" * 70)
    print("RUNNING WITH: LowConfidenceVerbose")
    print("=" * 70)
    print("Expected: Decoding order will vary based on confidence scores")
    print()

    llm_conf = sglang.Engine(
        model_path=model_path,
        dllm_algorithm="LowConfidenceVerbose",
        dllm_block_size=block_size,
    )

    output_conf = llm_conf.generate(prompt, {"temperature": 1.0, "max_new_tokens": block_size})

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Sequential output:     {output_seq}")
    print(f"LowConfidence output:  {output_conf}")
    print()
    print("Key Observations:")
    print("1. Sequential always unmasks left-to-right (0→1→2→3)")
    print("2. LowConfidence unmasks highest confidence first (varies)")
    print("3. Final outputs may differ due to different decoding orders")
    print("=" * 70)


def show_pattern_only():
    """Show the decoding pattern difference without running a model."""

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         Sequential vs LowConfidence - Pattern Comparison         ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    print("\n" + "─" * 70)
    print("Sequential Algorithm")
    print("─" * 70)
    print("Strategy: Unmask from left to right")
    print("Decoding order: DETERMINISTIC")
    print()
    print("Block 1: [0, 1, 2, 3]")
    print("Block 2: [4, 5, 6, 7]")
    print("Block 3: [8, 9, 10, 11]")
    print()
    print("✓ Same order every time")
    print("✓ Predictable")
    print("✓ Simple to understand")

    print("\n" + "─" * 70)
    print("LowConfidence Algorithm")
    print("─" * 70)
    print("Strategy: Unmask highest confidence token first")
    print("Decoding order: ADAPTIVE")
    print()
    print("Run 1: [2, 0, 3, 1]  (position 2 had highest confidence)")
    print("Run 2: [1, 3, 0, 2]  (position 1 had highest confidence)")
    print("Run 3: [0, 2, 1, 3]  (position 0 had highest confidence)")
    print()
    print("✓ Adapts to model confidence")
    print("✓ Potentially better quality")
    print("✓ Order varies per generation")

    print("\n" + "─" * 70)
    print("When to Use Each")
    print("─" * 70)
    print("Sequential:")
    print("  • Need deterministic behavior")
    print("  • Want reproducible results")
    print("  • Debugging position-specific issues")
    print()
    print("LowConfidence:")
    print("  • Want best quality output")
    print("  • Model confidence matters")
    print("  • Production use cases")
    print("─" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Sequential vs LowConfidence decoding orders"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model (required for actual generation)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Translate to French: Hello",
        help="Prompt for generation"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4,
        help="DLLM block size"
    )
    parser.add_argument(
        "--pattern-only",
        action="store_true",
        help="Just show the pattern difference (no model needed)"
    )

    args = parser.parse_args()

    if args.pattern_only or not args.model:
        # Show pattern without running model
        show_pattern_only()

        if not args.model:
            print("\n" + "=" * 70)
            print("To run actual comparison with model:")
            print("=" * 70)
            print("python compare_decoding_orders.py \\")
            print("  --model /path/to/llada2 \\")
            print("  --prompt \"Your prompt here\"")
            print("=" * 70)
    else:
        # Run actual comparison with model
        compare_algorithms(args.model, args.prompt, args.block_size)


if __name__ == "__main__":
    main()
