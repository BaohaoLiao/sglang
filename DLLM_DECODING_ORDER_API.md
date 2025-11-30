# DLLM Decoding Order API

## Overview

All DLLM algorithms now return the decoding order as part of the generation output. The decoding order shows which positions were unmasked in which sequence.

## Quick Start

```python
import sglang

# Initialize with any DLLM algorithm
llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="Sequential",  # or "LowConfidence", "SequentialVerbose", "LowConfidenceVerbose"
    dllm_block_size=4
)

# Generate text
outputs = llm.generate("Translate to French: Hello", {"temperature": 1.0})

# Access decoding order
decoding_order = outputs["dllm_decoding_order"]
print(f"Decoding order: {decoding_order}")
```

## Output Structure

The `outputs` dictionary now includes a `dllm_decoding_order` field:

```python
{
    "text": "Bonjour",
    "output_ids": [2345, 6789, 1234, 5678],
    "dllm_decoding_order": [[0, 1, 2, 3]],  # List of lists (one per request)
    # ... other fields ...
}
```

### Field Format

- **Type**: `List[List[int]]` or `None`
- **Structure**: Outer list contains one entry per request in the batch
- **Inner list**: Position indices in the order they were unmasked
- **Value**: `None` for non-DLLM requests, `[[]]` for DLLM without decoding order tracking

## Algorithm-Specific Behavior

### Sequential

Always returns deterministic left-to-right order:

```python
llm = sglang.Engine(model_path="...", dllm_algorithm="Sequential", dllm_block_size=4)
outputs = llm.generate("Hello", {"temperature": 1.0, "max_new_tokens": 8})

# First block: [0, 1, 2, 3]
# Second block: [0, 1, 2, 3] (relative to block)
print(outputs["dllm_decoding_order"])  # [[0, 1, 2, 3, 0, 1, 2, 3]]
```

### LowConfidence

Returns adaptive order based on confidence scores:

```python
llm = sglang.Engine(model_path="...", dllm_algorithm="LowConfidence", dllm_block_size=4)
outputs = llm.generate("Hello", {"temperature": 1.0, "max_new_tokens": 8})

# Order varies based on model confidence
print(outputs["dllm_decoding_order"])  # [[2, 0, 3, 1, 1, 3, 0, 2]]  (example)
```

## Batch Generation

For batch requests, the outer list contains one decoding order per request:

```python
llm = sglang.Engine(model_path="...", dllm_algorithm="LowConfidence", dllm_block_size=4)

prompts = [
    "Translate to French: Hello",
    "Translate to Spanish: Hello"
]

outputs_list = llm.generate(prompts, {"temperature": 1.0, "max_new_tokens": 4})

for i, outputs in enumerate(outputs_list):
    print(f"Request {i}: {outputs['dllm_decoding_order']}")
    # Request 0: [[2, 0, 3, 1]]
    # Request 1: [[1, 2, 0, 3]]
```

## Complete Example

```python
import sglang

# Compare Sequential vs LowConfidence decoding orders
def compare_decoding_orders(model_path, prompt):
    results = {}

    for algorithm in ["Sequential", "LowConfidence"]:
        llm = sglang.Engine(
            model_path=model_path,
            dllm_algorithm=algorithm,
            dllm_block_size=4
        )

        outputs = llm.generate(prompt, {"temperature": 1.0, "max_new_tokens": 4})

        results[algorithm] = {
            "text": outputs["text"],
            "output_ids": outputs["output_ids"],
            "decoding_order": outputs["dllm_decoding_order"][0]  # First request
        }

    return results

# Run comparison
results = compare_decoding_orders("/path/to/llada2", "Translate to French: Hello")

print("Sequential:")
print(f"  Text: {results['Sequential']['text']}")
print(f"  Decoding order: {results['Sequential']['decoding_order']}")

print("\nLowConfidence:")
print(f"  Text: {results['LowConfidence']['text']}")
print(f"  Decoding order: {results['LowConfidence']['decoding_order']}")
```

Output:
```
Sequential:
  Text: Bonjour
  Decoding order: [0, 1, 2, 3]

LowConfidence:
  Text: Bonjour
  Decoding order: [2, 0, 3, 1]
```

## Interpreting the Decoding Order

### Position Indexing

Decoding order positions are **relative to each block**:

```python
# Block size = 4, generated 8 tokens
# Decoding order: [0, 1, 2, 3, 0, 1, 2, 3]
#                 └─ Block 1 ─┘ └─ Block 2 ─┘
```

To get absolute positions:

```python
def get_absolute_positions(decoding_order, block_size=4):
    """Convert relative positions to absolute positions."""
    absolute = []
    for i, pos in enumerate(decoding_order):
        block_num = i // block_size
        absolute_pos = block_num * block_size + pos
        absolute.append(absolute_pos)
    return absolute

decoding_order = [0, 1, 2, 3, 0, 1, 2, 3]
absolute = get_absolute_positions(decoding_order, block_size=4)
print(absolute)  # [0, 1, 2, 3, 4, 5, 6, 7]
```

### Understanding Adaptive Order

For LowConfidence, the decoding order reveals which positions the model was most confident about:

```python
decoding_order = [2, 0, 3, 1]
# Position 2 was unmasked first  → Highest confidence
# Position 1 was unmasked last   → Lowest confidence
```

## Non-DLLM Requests

For non-DLLM requests (standard autoregressive generation), `dllm_decoding_order` is `None`:

```python
# Regular model (no DLLM)
llm = sglang.Engine(model_path="/path/to/llama")
outputs = llm.generate("Hello", {"temperature": 1.0})

print(outputs["dllm_decoding_order"])  # None
```

## API Reference

### Output Dictionary Fields

```python
{
    "text": str,                          # Generated text
    "output_ids": List[int],              # Token IDs
    "dllm_decoding_order": Optional[List[List[int]]],  # Decoding order
    "prompt_tokens": int,                 # Input token count
    "completion_tokens": int,             # Output token count
    # ... other standard fields ...
}
```

### Batch Output Structure

For batch generation, returns a list of dictionaries:

```python
outputs_list: List[Dict] = llm.generate(prompts, sampling_params)

for outputs in outputs_list:
    order = outputs["dllm_decoding_order"]  # List[List[int]] for this request
    # order[0] contains the decoding order for this specific request
```

## Advanced: Analyzing Decoding Patterns

```python
import sglang
import numpy as np

def analyze_decoding_patterns(model_path, prompts, algorithm="LowConfidence"):
    """Analyze decoding order patterns across multiple prompts."""

    llm = sglang.Engine(
        model_path=model_path,
        dllm_algorithm=algorithm,
        dllm_block_size=4
    )

    all_orders = []

    for prompt in prompts:
        outputs = llm.generate(prompt, {"temperature": 1.0, "max_new_tokens": 4})
        order = outputs["dllm_decoding_order"][0]
        all_orders.append(order)

    # Analyze patterns
    print(f"Algorithm: {algorithm}")
    print(f"Number of prompts: {len(prompts)}")
    print()

    for i, order in enumerate(all_orders):
        print(f"Prompt {i}: {prompts[i][:30]}...")
        print(f"  Decoding order: {order}")
        print(f"  First unmasked: position {order[0]}")
        print(f"  Last unmasked: position {order[-1]}")

    # Check if orders are consistent
    if len(set(map(tuple, all_orders))) == 1:
        print("\n✓ All prompts have identical decoding order")
    else:
        print("\n✓ Decoding orders vary by prompt (adaptive)")

# Example usage
prompts = [
    "Translate to French: Hello",
    "Translate to French: Goodbye",
    "Translate to French: Thank you"
]

analyze_decoding_patterns("/path/to/llada2", prompts, "LowConfidence")
```

## Troubleshooting

### Empty Decoding Order

If `dllm_decoding_order` is `[[]]`:

```python
outputs = llm.generate(prompt, {"temperature": 1.0})
assert outputs["dllm_decoding_order"] == [[]], "Empty order list"
```

**Causes**:
1. No tokens were generated (`max_new_tokens=0`)
2. Generation finished before any blocks completed
3. Request was aborted early

### None Value

If `dllm_decoding_order` is `None`:

```python
outputs = llm.generate(prompt, {"temperature": 1.0})
assert outputs["dllm_decoding_order"] is None
```

**Cause**: Not using a DLLM algorithm (standard autoregressive generation)

## Migration from Verbose Algorithms

Previously, verbose algorithms (`SequentialVerbose`, `LowConfidenceVerbose`) printed decoding order. Now it's available programmatically:

### Before (printed output):

```python
llm = sglang.Engine(model_path="...", dllm_algorithm="SequentialVerbose")
outputs = llm.generate("Hello", {"temperature": 1.0})
# Decoding order: [0, 1, 2, 3]  ← Printed to stdout
```

### After (programmatic access):

```python
llm = sglang.Engine(model_path="...", dllm_algorithm="Sequential")  # No need for verbose
outputs = llm.generate("Hello", {"temperature": 1.0})
order = outputs["dllm_decoding_order"][0]
print(f"Decoding order: {order}")  # Your choice: print or process
```

**Benefits**:
- No stdout pollution in production
- Programmatic access for analysis
- Works with all algorithms (Sequential, LowConfidence, and their verbose variants)

## Summary

- **All DLLM algorithms** now return decoding order in `outputs["dllm_decoding_order"]`
- **Sequential**: Always `[0, 1, 2, 3, ...]` (deterministic)
- **LowConfidence**: Varies based on confidence scores (adaptive)
- **Format**: `List[List[int]]` - outer list for batch, inner list for positions
- **Non-DLLM**: Returns `None`

For more examples, see:
- `OFFLINE_DECODING_ORDER.md` - Offline mode usage
- `VERBOSE_ALGORITHMS.md` - Verbose algorithm details
- `compare_decoding_orders.py` - Comparison tool
