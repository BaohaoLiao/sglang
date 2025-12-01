"""
Example: Per-Request DLLM Configuration

This example demonstrates how to specify DLLM algorithm and block size
per-request, allowing you to dynamically change algorithms without
reinitializing the engine.

Usage:
    python example_per_request_dllm.py --model /path/to/llada2

Features:
    - Override DLLM algorithm per request
    - Override block size per request
    - Compare different algorithms on same prompts
    - Batch processing with heterogeneous DLLM settings
"""

import argparse
import sglang


def example_1_basic_override():
    """Example 1: Basic per-request algorithm override"""
    print("\n" + "=" * 70)
    print("Example 1: Basic Per-Request Algorithm Override")
    print("=" * 70)

    llm = sglang.Engine(
        model_path=args.model,
        dllm_algorithm="Sequential",  # Default algorithm
        dllm_block_size=4
    )

    prompt = "Translate to French: Hello"

    # Use Sequential (engine default)
    print("\n1. Using engine default (Sequential, block_size=4):")
    outputs1 = llm.generate(prompt, {"temperature": 1.0, "max_new_tokens": 4})
    print(f"   Text: {outputs1['text']}")
    print(f"   Decoding order: {outputs1['dllm_decoding_order']}")

    # Override to LowConfidence
    print("\n2. Override to LowConfidence:")
    outputs2 = llm.generate(
        prompt,
        {
            "temperature": 1.0,
            "max_new_tokens": 4,
            "dllm_algorithm": "LowConfidence"
        }
    )
    print(f"   Text: {outputs2['text']}")
    print(f"   Decoding order: {outputs2['dllm_decoding_order']}")

    # Override to LowConfidenceVerbose (with debug output)
    print("\n3. Override to LowConfidenceVerbose (verbose mode):")
    outputs3 = llm.generate(
        prompt,
        {
            "temperature": 1.0,
            "max_new_tokens": 4,
            "dllm_algorithm": "LowConfidenceVerbose"
        }
    )
    print(f"   Text: {outputs3['text']}")
    print(f"   Decoding order: {outputs3['dllm_decoding_order']}")


def example_2_block_size_override():
    """Example 2: Override block size per request"""
    print("\n" + "=" * 70)
    print("Example 2: Per-Request Block Size Override")
    print("=" * 70)

    llm = sglang.Engine(
        model_path=args.model,
        dllm_algorithm="Sequential",
        dllm_block_size=4  # Default block size
    )

    prompt = "Translate to French: Good morning"

    # Use default block size 4
    print("\n1. Block size 4 (default):")
    outputs1 = llm.generate(prompt, {"temperature": 1.0, "max_new_tokens": 8})
    print(f"   Decoding order: {outputs1['dllm_decoding_order']}")
    print(f"   Expected: [0,1,2,3,0,1,2,3] (two blocks of 4)")

    # Override to block size 8
    print("\n2. Block size 8 (override):")
    outputs2 = llm.generate(
        prompt,
        {
            "temperature": 1.0,
            "max_new_tokens": 8,
            "dllm_block_size": 8
        }
    )
    print(f"   Decoding order: {outputs2['dllm_decoding_order']}")
    print(f"   Expected: [0,1,2,3,4,5,6,7] (one block of 8)")

    # Override to block size 2
    print("\n3. Block size 2 (override):")
    outputs3 = llm.generate(
        prompt,
        {
            "temperature": 1.0,
            "max_new_tokens": 8,
            "dllm_block_size": 2
        }
    )
    print(f"   Decoding order: {outputs3['dllm_decoding_order']}")
    print(f"   Expected: [0,1,0,1,0,1,0,1] (four blocks of 2)")


def example_3_comparison():
    """Example 3: Compare different algorithms on same prompt"""
    print("\n" + "=" * 70)
    print("Example 3: Algorithm Comparison on Same Prompt")
    print("=" * 70)

    llm = sglang.Engine(
        model_path=args.model,
        dllm_algorithm="Sequential",
        dllm_block_size=4
    )

    prompt = "Translate to Spanish: Thank you"
    algorithms = ["Sequential", "LowConfidence", "SequentialVerbose", "LowConfidenceVerbose"]

    for algo in algorithms:
        print(f"\n{algo}:")
        outputs = llm.generate(
            prompt,
            {
                "temperature": 1.0,
                "max_new_tokens": 4,
                "dllm_algorithm": algo
            }
        )
        print(f"   Text: {outputs['text']}")
        print(f"   Decoding order: {outputs['dllm_decoding_order']}")


def example_4_batch_heterogeneous():
    """Example 4: Batch processing with different DLLM settings"""
    print("\n" + "=" * 70)
    print("Example 4: Batch with Heterogeneous DLLM Settings")
    print("=" * 70)

    llm = sglang.Engine(
        model_path=args.model,
        dllm_algorithm="Sequential",
        dllm_block_size=4
    )

    prompts = [
        "Translate to French: Hello",
        "Translate to Spanish: Goodbye",
        "Translate to German: Thank you"
    ]

    sampling_params_list = [
        {"temperature": 1.0, "max_new_tokens": 4, "dllm_algorithm": "Sequential", "dllm_block_size": 4},
        {"temperature": 1.0, "max_new_tokens": 4, "dllm_algorithm": "LowConfidence", "dllm_block_size": 4},
        {"temperature": 1.0, "max_new_tokens": 8, "dllm_algorithm": "Sequential", "dllm_block_size": 8}
    ]

    outputs_list = llm.generate(prompts, sampling_params_list)

    for i, outputs in enumerate(outputs_list):
        params = sampling_params_list[i]
        print(f"\nRequest {i}: {prompts[i]}")
        print(f"   Algorithm: {params.get('dllm_algorithm', 'Sequential (default)')}")
        print(f"   Block size: {params.get('dllm_block_size', 4)}")
        print(f"   Text: {outputs['text']}")
        print(f"   Decoding order: {outputs['dllm_decoding_order']}")


def example_5_partial_override():
    """Example 5: Partial overrides (only algorithm or only block size)"""
    print("\n" + "=" * 70)
    print("Example 5: Partial Overrides")
    print("=" * 70)

    llm = sglang.Engine(
        model_path=args.model,
        dllm_algorithm="Sequential",
        dllm_block_size=4
    )

    prompt = "Translate to French: Good evening"

    # Override only algorithm, use default block size
    print("\n1. Override only algorithm (LowConfidence), use default block size (4):")
    outputs1 = llm.generate(
        prompt,
        {
            "temperature": 1.0,
            "max_new_tokens": 4,
            "dllm_algorithm": "LowConfidence"
            # dllm_block_size not specified, uses engine default (4)
        }
    )
    print(f"   Decoding order: {outputs1['dllm_decoding_order']}")
    print(f"   Expected: Adaptive order with 4 positions")

    # Override only block size, use default algorithm
    print("\n2. Override only block size (8), use default algorithm (Sequential):")
    outputs2 = llm.generate(
        prompt,
        {
            "temperature": 1.0,
            "max_new_tokens": 8,
            "dllm_block_size": 8
            # dllm_algorithm not specified, uses engine default (Sequential)
        }
    )
    print(f"   Decoding order: {outputs2['dllm_decoding_order']}")
    print(f"   Expected: [0,1,2,3,4,5,6,7] (sequential with 8 positions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Request DLLM Configuration Examples")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the LLaDA2 model"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Which example to run (default: all)"
    )

    args = parser.parse_args()

    examples = {
        "1": example_1_basic_override,
        "2": example_2_block_size_override,
        "3": example_3_comparison,
        "4": example_4_batch_heterogeneous,
        "5": example_5_partial_override,
    }

    if args.example == "all":
        for example_func in examples.values():
            example_func()
    else:
        examples[args.example]()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
