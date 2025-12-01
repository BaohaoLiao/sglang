#!/usr/bin/env python3
"""
Example: Using Sequential DLLM with SGLang's Offline Engine

This script demonstrates how to use the Sequential algorithm in offline mode
and view the decoding order.
"""

import sglang


def example_basic_sequential():
    """Basic example with Sequential algorithm."""
    print("\n" + "="*70)
    print("Example 1: Basic Sequential Algorithm")
    print("="*70 + "\n")

    # Initialize engine with Sequential algorithm
    llm = sglang.Engine(
        model_path="path/to/llada2-model",
        dllm_algorithm="Sequential",
        dllm_block_size=4,
    )

    # Generate
    prompt = "Translate to French: Hello, how are you?"
    outputs = llm.generate(prompt, {"temperature": 1.0, "max_new_tokens": 20})

    print("\nResult:")
    print(f"Prompt: {prompt}")
    print(f"Output: {outputs}")


def example_verbose_sequential():
    """Example with verbose Sequential that shows decoding order."""
    print("\n" + "="*70)
    print("Example 2: Sequential with Decoding Order Tracking")
    print("="*70 + "\n")

    # Initialize engine with verbose Sequential algorithm
    llm = sglang.Engine(
        model_path="path/to/llada2-model",
        dllm_algorithm="SequentialVerbose",  # Use verbose version
        dllm_block_size=4,
    )

    # Generate - this will print detailed decoding information
    prompt = "Translate to French: Hello"
    outputs = llm.generate(prompt, {"temperature": 1.0, "max_new_tokens": 8})

    print("\nResult:")
    print(f"Prompt: {prompt}")
    print(f"Output: {outputs}")


def example_compare_algorithms():
    """Compare Sequential vs LowConfidence."""
    print("\n" + "="*70)
    print("Example 3: Compare Sequential vs LowConfidence")
    print("="*70 + "\n")

    prompt = "Translate to Spanish: Good morning"

    # Sequential
    print("\n--- Using Sequential Algorithm ---\n")
    llm_seq = sglang.Engine(
        model_path="path/to/llada2-model",
        dllm_algorithm="Sequential",
        dllm_block_size=4,
    )
    output_seq = llm_seq.generate(prompt, {"temperature": 1.0, "max_new_tokens": 20})
    print(f"Sequential output: {output_seq}")

    # LowConfidence
    print("\n--- Using LowConfidence Algorithm ---\n")
    llm_conf = sglang.Engine(
        model_path="path/to/llada2-model",
        dllm_algorithm="LowConfidence",
        dllm_block_size=4,
    )
    output_conf = llm_conf.generate(prompt, {"temperature": 1.0, "max_new_tokens": 20})
    print(f"LowConfidence output: {output_conf}")

    print("\nNote: Sequential will always unmask in order: 0→1→2→3")
    print("      LowConfidence will unmask based on confidence scores")


def example_batch_generation():
    """Example with batch generation."""
    print("\n" + "="*70)
    print("Example 4: Batch Generation with Sequential")
    print("="*70 + "\n")

    llm = sglang.Engine(
        model_path="path/to/llada2-model",
        dllm_algorithm="Sequential",
        dllm_block_size=4,
    )

    prompts = [
        "Translate to French: Hello",
        "Translate to French: Goodbye",
        "Translate to French: Thank you"
    ]

    outputs = llm.generate(prompts, {"temperature": 1.0, "max_new_tokens": 20})

    print("\nResults:")
    for prompt, output in zip(prompts, outputs):
        print(f"{prompt} → {output}")


def example_custom_tracking():
    """Example of custom tracking of decoding order."""
    print("\n" + "="*70)
    print("Example 5: Custom Decoding Order Tracking")
    print("="*70 + "\n")

    # This shows how you might track decoding order programmatically
    # if you need to analyze it

    class DecodingTracker:
        def __init__(self):
            self.orders = []

        def track_generation(self, llm, prompt, params):
            """Track decoding order during generation."""
            # Note: This is a conceptual example
            # Actual implementation would need hooks into the algorithm

            output = llm.generate(prompt, params)

            # In practice, you'd get this from the algorithm
            # For Sequential, it's always [0, 1, 2, 3, ...]
            decoding_order = list(range(params.get("max_new_tokens", 20)))

            self.orders.append({
                "prompt": prompt,
                "output": output,
                "order": decoding_order
            })

            return output

        def analyze(self):
            """Analyze collected decoding orders."""
            print("\nDecoding Order Analysis:")
            for i, record in enumerate(self.orders):
                print(f"\nGeneration {i+1}:")
                print(f"  Prompt: {record['prompt']}")
                print(f"  Order: {record['order']}")
                print(f"  Output: {record['output']}")

    tracker = DecodingTracker()

    llm = sglang.Engine(
        model_path="path/to/llada2-model",
        dllm_algorithm="Sequential",
        dllm_block_size=4,
    )

    # Track multiple generations
    tracker.track_generation(llm, "Hello", {"max_new_tokens": 8})
    tracker.track_generation(llm, "Goodbye", {"max_new_tokens": 8})

    # Analyze
    tracker.analyze()


def main():
    """Run all examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          SGLang Sequential DLLM - Offline Mode Examples             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    print("\nNote: These examples assume you have:")
    print("  1. LLaDA2 model available")
    print("  2. Sequential algorithm installed")
    print("  3. Sufficient GPU memory")
    print("\nReplace 'path/to/llada2-model' with your actual model path.\n")

    # Uncomment the examples you want to run:

    # example_basic_sequential()
    # example_verbose_sequential()
    # example_compare_algorithms()
    # example_batch_generation()
    # example_custom_tracking()

    print("\n" + "="*70)
    print("To run these examples:")
    print("="*70)
    print("1. Uncomment the example functions at the bottom of this file")
    print("2. Update the model_path to your LLaDA2 model")
    print("3. Run: python example_offline_sequential.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
