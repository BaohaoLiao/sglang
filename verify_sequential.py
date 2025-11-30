#!/usr/bin/env python3
"""
Verify that the Sequential DLLM algorithm is properly registered and available.
"""

import sys


def verify_algorithm_registration():
    """Verify Sequential is registered in the algorithm mapping."""
    print("=" * 60)
    print("Verifying Sequential DLLM Algorithm Registration")
    print("=" * 60)
    print()

    try:
        from sglang.srt.dllm.algorithm import algo_name_to_cls

        print("Available DLLM algorithms:")
        for name in sorted(algo_name_to_cls.keys()):
            algo_class = algo_name_to_cls[name]
            print(f"  ✓ {name:20s} -> {algo_class.__module__}.{algo_class.__name__}")

        print()

        # Check specifically for Sequential
        if 'Sequential' in algo_name_to_cls:
            print("✅ SUCCESS: Sequential algorithm is registered!")
            print()
            algo = algo_name_to_cls['Sequential']
            print(f"Class: {algo}")
            print(f"Module: {algo.__module__}")
            print(f"Docstring: {algo.__doc__.strip()}")
            return True
        else:
            print("❌ ERROR: Sequential algorithm not found!")
            print()
            print("Available algorithms:", list(algo_name_to_cls.keys()))
            return False

    except ImportError as e:
        print(f"❌ ERROR: Failed to import SGLang modules: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return False


def verify_algorithm_loading():
    """Verify Sequential can be instantiated."""
    print()
    print("=" * 60)
    print("Verifying Sequential Algorithm Can Be Loaded")
    print("=" * 60)
    print()

    try:
        from sglang.srt.dllm.config import DllmConfig
        from sglang.srt.dllm.algorithm import get_algorithm

        # Create a config
        config = DllmConfig(
            mask_id=156895,
            block_size=4,
            algorithm="Sequential"
        )

        print(f"Config created:")
        print(f"  algorithm: {config.algorithm}")
        print(f"  block_size: {config.block_size}")
        print(f"  mask_id: {config.mask_id}")
        print()

        # Load the algorithm
        algorithm = get_algorithm(config)

        print(f"✅ SUCCESS: Sequential algorithm instantiated!")
        print(f"  Instance: {algorithm}")
        print(f"  Type: {type(algorithm)}")
        print(f"  Block size: {algorithm.block_size}")
        print(f"  Mask ID: {algorithm.mask_id}")

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to load Sequential algorithm: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_examples():
    """Show usage examples."""
    print()
    print("=" * 60)
    print("Usage Examples")
    print("=" * 60)
    print()

    print("1. Start server with Sequential:")
    print("-" * 60)
    print("""
    python -m sglang.launch_server \\
      --model-path path/to/llada2-model \\
      --dllm-algorithm Sequential \\
      --dllm-block-size 4
    """)

    print()
    print("2. Or using sglang CLI:")
    print("-" * 60)
    print("""
    sglang serve \\
      --model path/to/llada2-model \\
      --dllm-algorithm Sequential \\
      --dllm-block-size 4
    """)

    print()
    print("3. In Python code:")
    print("-" * 60)
    print("""
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.dllm.algorithm.base import DllmAlgorithm

    # Create server args
    server_args = ServerArgs(
        model_path="path/to/llada2-model",
        dllm_algorithm="Sequential",
        dllm_block_size=4
    )

    # Algorithm will be automatically loaded
    """)

    print()
    print("4. Available algorithms:")
    print("-" * 60)
    print("  - Sequential     (deterministic, left-to-right)")
    print("  - LowConfidence  (adaptive, confidence-based)")
    print()


if __name__ == "__main__":
    print()
    success = True

    # Run verifications
    if not verify_algorithm_registration():
        success = False

    if not verify_algorithm_loading():
        success = False

    # Show usage examples
    show_usage_examples()

    # Final result
    print("=" * 60)
    if success:
        print("✅ All verifications passed!")
        print("Sequential algorithm is ready to use.")
    else:
        print("❌ Some verifications failed.")
        print("Please check the error messages above.")
    print("=" * 60)
    print()

    sys.exit(0 if success else 1)
