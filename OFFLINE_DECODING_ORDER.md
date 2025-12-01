# Viewing Decoding Order in SGLang Offline Mode

> **✨ NEW**: Decoding order is now available programmatically! See `DLLM_DECODING_ORDER_API.md` for the new API.

## Quick Answer

### Sequential Algorithm
The decoding order is **always deterministic**:
```
Block 1: [0, 1, 2, 3]
Block 2: [4, 5, 6, 7]
Block 3: [8, 9, 10, 11]
...
```

It always unmasks **left-to-right**, one position at a time.

### LowConfidence Algorithm
The decoding order is **adaptive** based on confidence scores:
```
Block 1: [2, 0, 3, 1]  (unmasks highest confidence first)
Block 2: [5, 7, 4, 6]
...
```

It unmasks the position with the **highest confidence** each iteration.

## Your Code

```python
import sglang

# Use verbose version to see decoding order
# Option 1: Sequential (deterministic order)
llm = sglang.Engine(
    model_path="path/to/llada2",
    dllm_algorithm="SequentialVerbose",  # ← Verbose Sequential
    dllm_block_size=4
)

# Option 2: LowConfidence (adaptive order by confidence)
llm = sglang.Engine(
    model_path="path/to/llada2",
    dllm_algorithm="LowConfidenceVerbose",  # ← Verbose LowConfidence
    dllm_block_size=4
)

outputs = llm.generate(prompt, {"temperature": 1.0})
```

## What You'll See

### SequentialVerbose Output

When you run with `SequentialVerbose`, you'll see output like:

```
============================================================
SEQUENTIAL DLLM DECODING - Verbose Mode
============================================================
Block size: 4
Mask token ID: 156895
Total masked tokens: 4
Start position: 12

Iteration 0:
  Masked count: 4
  Selected position: 0
  Predicted token ID: 2345
  Remaining positions: [1, 2, 3]
  Current state: ['2345', 'M', 'M', 'M']

Iteration 1:
  Masked count: 3
  Selected position: 1
  Predicted token ID: 6789
  Remaining positions: [2, 3]
  Current state: ['2345', '6789', 'M', 'M']

Iteration 2:
  Masked count: 2
  Selected position: 2
  Predicted token ID: 1234
  Remaining positions: [3]
  Current state: ['2345', '6789', '1234', 'M']

Iteration 3:
  Masked count: 1
  Selected position: 3
  Predicted token ID: 5678
  Remaining positions: []
  Current state: ['2345', '6789', '1234', '5678']

Final forward pass with all tokens unmasked...

============================================================
DECODING SUMMARY
============================================================
Decoding order: [0, 1, 2, 3]
Number of iterations: 4
Generated tokens: [2345, 6789, 1234, 5678]
============================================================
```

### LowConfidenceVerbose Output

When you run with `LowConfidenceVerbose`, you'll see output like:

```
============================================================
LOW CONFIDENCE DLLM DECODING - Verbose Mode
============================================================
Block size: 4
Mask token ID: 156895
Total masked tokens: 4
Start position: 12
Strategy: Unmask HIGHEST confidence token first

Iteration 0:
  Masked count: 4
  Confidence scores (masked positions):
    Position 2: 0.8542
    Position 0: 0.7123
    Position 3: 0.6891
    Position 1: 0.5234
  → Selected position: 2 (confidence: 0.8542)
  → Predicted token ID: 1234
  → Current state: ['M', 'M', '1234', 'M']

Iteration 1:
  Masked count: 3
  Confidence scores (masked positions):
    Position 0: 0.8901
    Position 3: 0.7456
    Position 1: 0.6234
  → Selected position: 0 (confidence: 0.8901)
  → Predicted token ID: 2345
  → Current state: ['2345', 'M', '1234', 'M']

Iteration 2:
  Masked count: 2
  Confidence scores (masked positions):
    Position 3: 0.8123
    Position 1: 0.7012
  → Selected position: 3 (confidence: 0.8123)
  → Predicted token ID: 5678
  → Current state: ['2345', 'M', '1234', '5678']

Iteration 3:
  Masked count: 1
  Confidence scores (masked positions):
    Position 1: 0.9234
  → Selected position: 1 (confidence: 0.9234)
  → Predicted token ID: 6789
  → Current state: ['2345', '6789', '1234', '5678']

Final forward pass with all tokens unmasked...

============================================================
DECODING SUMMARY
============================================================
Decoding order: [2, 0, 3, 1]
Confidence scores: ['0.8542', '0.8901', '0.8123', '0.9234']
Number of iterations: 4
Generated tokens: [2345, 6789, 1234, 5678]

Analysis:
  - Highest confidence: 0.9234 (position 1)
  - Lowest confidence: 0.8123 (position 3)
  - Average confidence: 0.8425
============================================================
```

**Key Difference**: Notice how LowConfidence unmasks `[2, 0, 3, 1]` based on confidence scores, while Sequential always unmasks `[0, 1, 2, 3]` in order.

## Comparison: Sequential vs LowConfidence

### Sequential (Deterministic)
```python
llm = sglang.Engine(model_path="...", dllm_algorithm="Sequential")
outputs = llm.generate(prompt, {"temperature": 1.0})

# Decoding order: ALWAYS [0, 1, 2, 3, 4, 5, ...]
```

### LowConfidence (Adaptive)
```python
llm = sglang.Engine(model_path="...", dllm_algorithm="LowConfidence")
outputs = llm.generate(prompt, {"temperature": 1.0})

# Decoding order: Variable, e.g., [2, 0, 3, 1, 5, 4, ...]
# Depends on confidence scores
```

## Files Created

I've created several tools for you:

### 1. SequentialVerbose Algorithm
**File**: `sglang/python/sglang/srt/dllm/algorithm/sequential_verbose.py`

Automatically outputs decoding order during generation.

### 2. Example Script
**File**: `example_offline_sequential.py`

Contains 5 complete examples:
- Basic Sequential usage
- Verbose mode with decoding order
- Algorithm comparison
- Batch generation
- Custom tracking

### 3. Decoding Order Tool
**File**: `show_decoding_order.py`

Command-line tool to visualize decoding order:

```bash
# Show pattern (no model needed)
python show_decoding_order.py --simple

# Run with model
python show_decoding_order.py \
  --model /path/to/llada2 \
  --prompt "Hello world" \
  --algorithm SequentialVerbose
```

## API Reference

### Basic Usage
```python
import sglang

# Initialize
llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="Sequential",  # or "SequentialVerbose"
    dllm_block_size=4,
    # Other args...
)

# Generate
outputs = llm.generate(
    prompt="Your prompt here",
    sampling_params={
        "temperature": 1.0,
        "max_new_tokens": 20,
        "top_p": 0.9,
    }
)

print(outputs)
```

### Batch Generation
```python
prompts = [
    "Translate to French: Hello",
    "Translate to French: Goodbye",
]

outputs = llm.generate(
    prompts,
    {"temperature": 1.0, "max_new_tokens": 20}
)

for prompt, output in zip(prompts, outputs):
    print(f"{prompt} → {output}")
```

### With Context Manager
```python
with sglang.Engine(model_path="...", dllm_algorithm="Sequential") as llm:
    outputs = llm.generate("Hello", {"temperature": 1.0})
    print(outputs)
```

## Programmatic Access to Decoding Order

If you need to access the decoding order programmatically:

```python
import sglang
from sglang.srt.dllm.algorithm import algo_name_to_cls

# Initialize with verbose algorithm
llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="SequentialVerbose",
    dllm_block_size=4
)

# The algorithm instance is accessible (advanced usage)
# Note: This requires accessing internal attributes

outputs = llm.generate("Hello", {"temperature": 1.0})

# For Sequential, you can predict the order:
def get_sequential_order(num_tokens, block_size=4):
    """Get decoding order for Sequential algorithm."""
    return list(range(num_tokens))

# Example
num_generated = 16
order = get_sequential_order(num_generated)
print(f"Decoding order: {order}")  # [0, 1, 2, ..., 15]

# For LowConfidence, the order depends on the model's confidence scores
# You cannot predict it in advance, but you can see it after generation
# by using LowConfidenceVerbose which prints:
#   - Decoding order: [2, 0, 3, 1, ...]
#   - Confidence scores for each position
#   - Analysis (highest, lowest, average confidence)
```

## Understanding the Output

### What Each Part Means

```
Iteration 2:
  Masked count: 2           ← 2 tokens still masked
  Selected position: 2      ← Unmasking position 2 (0-indexed)
  Predicted token ID: 1234  ← Model predicted token 1234
  Remaining positions: [3]  ← Only position 3 left
  Current state: [..., '1234', 'M']  ← Current block state
```

### Decoding Summary

```
Decoding order: [0, 1, 2, 3]      ← Order positions were unmasked
Number of iterations: 4            ← Took 4 iterations (= block_size)
Generated tokens: [2345, ...]      ← Actual token IDs generated
```

## Verification

To verify Sequential is working correctly:

```python
import sglang

# Run same prompt twice
llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="Sequential",
    dllm_block_size=4
)

# With temperature=0 (deterministic sampling)
out1 = llm.generate("Hello", {"temperature": 0.0})
out2 = llm.generate("Hello", {"temperature": 0.0})

assert out1 == out2, "Sequential should be deterministic with temp=0!"
print("✓ Verified: Sequential is deterministic")

# Decoding order is always [0, 1, 2, 3, ...]
print("Decoding order: [0, 1, 2, 3, ...]")
```

## Advanced: Custom Tracking

If you need more control:

```python
import sglang
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="SequentialVerbose",
    dllm_block_size=4
)

# Now you'll see detailed logs including decoding order
outputs = llm.generate("Hello", {"temperature": 1.0})
```

## Summary

| Feature | Sequential | SequentialVerbose | LowConfidence | LowConfidenceVerbose |
|---------|------------|-------------------|---------------|----------------------|
| **Decoding order** | [0,1,2,3,...] | [0,1,2,3,...] | Adaptive (e.g., [2,0,3,1,...]) | Adaptive (e.g., [2,0,3,1,...]) |
| **Strategy** | Left-to-right | Left-to-right | Highest confidence first | Highest confidence first |
| **Output** | Silent | Prints details + order | Silent | Prints confidence + order + analysis |
| **Use for** | Production | Debugging deterministic | Production | Debugging adaptive |
| **Performance** | Fast | Slightly slower (I/O) | Fast | Slightly slower (I/O) |
| **Predictable** | ✓ Yes | ✓ Yes | ✗ No (depends on model) | ✗ No (depends on model) |

**For your offline use case:**

```python
# Production (silent)
llm = sglang.Engine(model_path="...", dllm_algorithm="Sequential")  # or "LowConfidence"

# Debugging Sequential (verbose with order)
llm = sglang.Engine(model_path="...", dllm_algorithm="SequentialVerbose")

# Debugging LowConfidence (verbose with order + confidence scores)
llm = sglang.Engine(model_path="...", dllm_algorithm="LowConfidenceVerbose")
```

**Key Takeaways:**
- **Sequential**: Always unmasks **[0, 1, 2, 3, ...]** (deterministic, left-to-right)
- **LowConfidence**: Unmasks based on **confidence scores** (adaptive, varies per run)
