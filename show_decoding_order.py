#!/usr/bin/env python3
"""
Show Decoding Order for SGLang Sequential DLLM

Usage:
    python show_decoding_order.py --model /path/to/llada2 --prompt "Hello world"
"""

import argparse
import sglang


def show_decoding_order_simple():
    """Simple example showing decoding order."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         Sequential DLLM - Decoding Order Visualization           ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    # For Sequential algorithm, the decoding order is ALWAYS deterministic:
    # Position 0, then 1, then 2, then 3, etc.

    block_size = 4

    print("\nSequential Algorithm Decoding Order:")
    print("=" * 70)
    print("\nFor a block of size 4, tokens are ALWAYS unmasked in this order:\n")

    print("Block: [MASK, MASK, MASK, MASK]")
    print()

    for i in range(block_size):
        state = ['MASK'] * block_size
        for j in range(i + 1):
            state[j] = 'TOKEN'

        print(f"Iteration {i}: [" + ", ".join(f"{s:5s}" for s in state) + f"]  ← Unmask position {i}")

    print()
    print("Pattern: Always left-to-right (0 → 1 → 2 → 3)")
    print("=" * 70)


def generate_with_tracking(model_path, prompt, algorithm="SequentialVerbose", block_size=4):
    """Generate with decoding order tracking."""
    print(f"\nInitializing SGLang Engine...")
    print(f"  Model: {model_path}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Block size: {block_size}")
    print()

    try:
        llm = sglang.Engine(
            model_path=model_path,
            dllm_algorithm=algorithm,
            dllm_block_size=block_size,
        )

        print(f"Generating with prompt: '{prompt}'")
        print("-" * 70)

        outputs = llm.generate(
            prompt,
            {
                "temperature": 1.0,
                "max_new_tokens": 16  # Generate 4 blocks of 4 tokens
            }
        )

        print("-" * 70)
        print(f"\nFinal output: {outputs}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure you have:")
        print("  1. A valid LLaDA2 model path")
        print("  2. SequentialVerbose algorithm installed")
        print("  3. Sufficient GPU memory")


def compare_deterministic():
    """Show that Sequential is deterministic."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              Sequential vs LowConfidence Comparison              ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    print("\nDecoding Order Patterns:\n")

    print("Sequential (Deterministic):")
    print("-" * 70)
    print("Run 1: Unmask order = [0, 1, 2, 3]")
    print("Run 2: Unmask order = [0, 1, 2, 3]  ← Same!")
    print("Run 3: Unmask order = [0, 1, 2, 3]  ← Same!")
    print("\n✓ Always identical order")
    print()

    print("LowConfidence (Adaptive):")
    print("-" * 70)
    print("Run 1: Unmask order = [2, 0, 3, 1]  (by confidence)")
    print("Run 2: Unmask order = [1, 3, 0, 2]  ← Different!")
    print("Run 3: Unmask order = [0, 2, 1, 3]  ← Different!")
    print("\n✓ Order depends on model confidence scores")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show Sequential DLLM decoding order"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to LLaDA2 model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Translate to French: Hello",
        help="Prompt for generation"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SequentialVerbose",
        choices=["Sequential", "SequentialVerbose", "LowConfidence", "LowConfidenceVerbose"],
        help="DLLM algorithm to use"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4,
        help="DLLM block size"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Just show the decoding order pattern without running model"
    )

    args = parser.parse_args()

    if args.simple:
        # Just show the pattern, no model needed
        show_decoding_order_simple()
        compare_deterministic()
    elif args.model:
        # Run actual generation with tracking
        generate_with_tracking(
            args.model,
            args.prompt,
            args.algorithm,
            args.block_size
        )
    else:
        # Show help and examples
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║         SGLang Sequential DLLM - Decoding Order Tool             ║
╚═══════════════════════════════════════════════════════════════════╝

Usage Examples:

1. Show decoding order pattern (no model needed):
   python show_decoding_order.py --simple

2. Generate with Sequential (verbose output):
   python show_decoding_order.py \\
     --model /path/to/llada2 \\
     --prompt "Hello world" \\
     --algorithm SequentialVerbose

3. Generate with LowConfidence (verbose with confidence scores):
   python show_decoding_order.py \\
     --model /path/to/llada2 \\
     --prompt "Hello world" \\
     --algorithm LowConfidenceVerbose

4. Compare algorithms:
   python show_decoding_order.py --simple

For Sequential algorithm, the decoding order is ALWAYS:
  [0, 1, 2, 3, ...] (left to right)

For LowConfidence algorithm, the decoding order VARIES:
  [2, 0, 3, 1, ...] (highest confidence first)
        """)

        show_decoding_order_simple()


if __name__ == "__main__":
    main()
