# SGLang DLLM Verbose Algorithms - Complete Guide

This document explains the verbose versions of SGLang's DLLM algorithms, which output detailed decoding information for debugging and analysis.

## Overview

SGLang now supports **verbose versions** of both DLLM algorithms:

| Algorithm | Decoding Order | Verbose Version | What It Shows |
|-----------|----------------|-----------------|---------------|
| **Sequential** | [0, 1, 2, 3, ...] | `SequentialVerbose` | Position-by-position unmasking |
| **LowConfidence** | Adaptive (e.g., [2, 0, 3, 1, ...]) | `LowConfidenceVerbose` | Confidence scores + decoding order |

## Quick Start

### Sequential with Verbose Output

```python
import sglang

llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="SequentialVerbose",  # ← Verbose Sequential
    dllm_block_size=4
)

outputs = llm.generate("Translate to French: Hello", {"temperature": 1.0})
```

**Output:**
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

...

DECODING SUMMARY
============================================================
Decoding order: [0, 1, 2, 3]
Number of iterations: 4
Generated tokens: [2345, 6789, 1234, 5678]
============================================================
```

### LowConfidence with Verbose Output

```python
import sglang

llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="LowConfidenceVerbose",  # ← Verbose LowConfidence
    dllm_block_size=4
)

outputs = llm.generate("Translate to French: Hello", {"temperature": 1.0})
```

**Output:**
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

...

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

## Key Differences

### SequentialVerbose
- Shows **which position** is unmasked each iteration
- Shows **remaining positions** to unmask
- Decoding order is always **deterministic**: [0, 1, 2, 3, ...]
- Useful for: Debugging position-specific issues, verifying left-to-right behavior

### LowConfidenceVerbose
- Shows **confidence scores** for all masked positions
- Shows **which position has highest confidence**
- Shows **analysis** of confidence distribution
- Decoding order is **adaptive**: varies based on model confidence
- Useful for: Understanding model uncertainty, analyzing confidence patterns

## Use Cases

### When to Use SequentialVerbose

```python
# Debugging: Why is position 2 always getting the wrong token?
llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="SequentialVerbose",
    dllm_block_size=4
)

outputs = llm.generate(prompt, {"temperature": 1.0})
# Look at the output for position 2 specifically
```

**Best for:**
- Debugging position-specific behavior
- Verifying deterministic decoding
- Educational purposes (understanding DLLM step-by-step)
- Reproducing issues (same order every time)

### When to Use LowConfidenceVerbose

```python
# Analysis: Which positions does the model struggle with?
llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="LowConfidenceVerbose",
    dllm_block_size=4
)

outputs = llm.generate(prompt, {"temperature": 1.0})
# Look at confidence scores to identify uncertain positions
```

**Best for:**
- Analyzing model confidence patterns
- Identifying positions with low confidence
- Understanding adaptive unmasking behavior
- Quality assessment (higher confidence = better quality?)

## Command-Line Tools

### show_decoding_order.py

View decoding order patterns:

```bash
# Show pattern (no model)
python show_decoding_order.py --simple

# Run with Sequential
python show_decoding_order.py \
  --model /path/to/llada2 \
  --prompt "Hello world" \
  --algorithm SequentialVerbose

# Run with LowConfidence
python show_decoding_order.py \
  --model /path/to/llada2 \
  --prompt "Hello world" \
  --algorithm LowConfidenceVerbose
```

### compare_decoding_orders.py

Compare Sequential vs LowConfidence:

```bash
# Show pattern difference (no model)
python compare_decoding_orders.py --pattern-only

# Run actual comparison
python compare_decoding_orders.py \
  --model /path/to/llada2 \
  --prompt "Your prompt here"
```

## Performance Considerations

| Version | Speed | Output | Use Case |
|---------|-------|--------|----------|
| `Sequential` | ⚡ Fast | Silent | Production |
| `SequentialVerbose` | 🐌 Slower (I/O) | Detailed | Debugging |
| `LowConfidence` | ⚡ Fast | Silent | Production |
| `LowConfidenceVerbose` | 🐌 Slower (I/O) | Very detailed | Analysis |

**Recommendation:**
- **Production**: Use `Sequential` or `LowConfidence` (silent versions)
- **Debugging**: Use `SequentialVerbose` or `LowConfidenceVerbose`

The verbose versions are slower because they print information at each iteration. This is acceptable for debugging but not recommended for production workloads.

## Interpreting the Output

### Sequential Output

```
Iteration 2:
  Masked count: 2           ← 2 tokens still masked
  Selected position: 2      ← Unmasking position 2 (always next in sequence)
  Predicted token ID: 1234  ← Model predicted token ID 1234
  Remaining positions: [3]  ← Only position 3 left to unmask
  Current state: [..., '1234', 'M']  ← Current block state (M = masked)
```

**Pattern**: Always unmasks position 0, then 1, then 2, then 3, etc.

### LowConfidence Output

```
Iteration 0:
  Masked count: 4
  Confidence scores (masked positions):
    Position 2: 0.8542  ← Highest confidence
    Position 0: 0.7123
    Position 3: 0.6891
    Position 1: 0.5234  ← Lowest confidence
  → Selected position: 2 (confidence: 0.8542)  ← Unmask highest first
```

**Pattern**: Always unmasks the position with highest confidence (varies each time).

**Analysis Section:**
```
Analysis:
  - Highest confidence: 0.9234 (position 1)  ← Most confident prediction
  - Lowest confidence: 0.8123 (position 3)   ← Least confident prediction
  - Average confidence: 0.8425               ← Overall confidence level
```

- **High average confidence (>0.9)**: Model is very confident about this block
- **Low average confidence (<0.7)**: Model is uncertain, output quality may vary
- **Wide range**: Some positions are easy (high conf), others hard (low conf)

## Advanced: Programmatic Access

While the verbose algorithms automatically print output, you can also access the internal state if needed:

```python
import sglang

# The verbose algorithms track decoding order internally
llm = sglang.Engine(
    model_path="/path/to/llada2",
    dllm_algorithm="LowConfidenceVerbose",
    dllm_block_size=4
)

# After generation, the algorithm instance contains:
# - self.decoding_order: list of positions in order unmasked
# - self.confidence_scores: list of confidence values (LowConfidenceVerbose only)

# Note: Accessing internal state requires digging into SGLang internals
# For most use cases, the printed output is sufficient
```

## Comparison Table

| Feature | Sequential | SequentialVerbose | LowConfidence | LowConfidenceVerbose |
|---------|------------|-------------------|---------------|----------------------|
| **Order** | [0,1,2,3,...] | [0,1,2,3,...] | Adaptive | Adaptive |
| **Shows positions** | ✗ | ✓ | ✗ | ✓ |
| **Shows confidence** | ✗ | ✗ | ✗ | ✓ |
| **Shows analysis** | ✗ | ✗ | ✗ | ✓ |
| **Deterministic** | ✓ | ✓ | ✗ | ✗ |
| **Production** | ✓ | ✗ | ✓ | ✗ |
| **Debugging** | ✗ | ✓ | ✗ | ✓ |
| **Performance** | Fast | Slower | Fast | Slower |

## FAQ

### Q: Will verbose output slow down my generation?

**A:** Yes, slightly. The verbose versions print to stdout at each iteration, which adds I/O overhead. The algorithmic complexity is identical, but expect ~5-10% slower due to printing.

### Q: Can I redirect the output to a file?

**A:** Yes! The verbose output goes to stdout, so you can redirect it:

```bash
python your_script.py > decoding_log.txt 2>&1
```

### Q: Why is LowConfidence order different each time?

**A:** LowConfidence selects the position with the highest confidence score, which depends on the model's predictions. These predictions can vary due to:
- Different sampling (if temperature > 0)
- Floating-point variations
- Model state

With `temperature=0`, the order should be more stable but may still vary due to numerical precision.

### Q: Can I turn off verbose output temporarily?

**A:** Yes, just switch back to the non-verbose version:

```python
# Verbose (for debugging)
llm = sglang.Engine(model_path="...", dllm_algorithm="SequentialVerbose")

# Silent (for production)
llm = sglang.Engine(model_path="...", dllm_algorithm="Sequential")
```

### Q: Which algorithm is better for production?

**A:**
- **LowConfidence**: Generally better quality (unmasks confident positions first)
- **Sequential**: More predictable, easier to debug, deterministic

For production, try both and measure quality on your specific use case.

## Summary

**Verbose algorithms are for debugging and analysis only.**

- ✅ Use `SequentialVerbose` to debug position-specific issues
- ✅ Use `LowConfidenceVerbose` to analyze confidence patterns
- ❌ Don't use verbose versions in production (too slow)
- 🎯 Use the provided tools (`show_decoding_order.py`, `compare_decoding_orders.py`) for quick analysis

**Next Steps:**
1. Try `show_decoding_order.py --simple` to see patterns
2. Run `compare_decoding_orders.py --pattern-only` to understand differences
3. Use verbose algorithms to debug specific issues
4. Switch back to silent versions for production

---

For more information, see:
- `OFFLINE_DECODING_ORDER.md` - Offline mode usage
- `SEQUENTIAL_DLLM.md` - Sequential algorithm details
- `DLLM_ALGORITHMS_COMPARISON.md` - Algorithm comparison
