# Sequential DLLM Algorithm for SGLang

## Overview

I've added a **Sequential** unmasking strategy to SGLang's DLLM implementation. This provides a deterministic, left-to-right token unmasking approach as an alternative to the confidence-based `LowConfidence` algorithm.

## Implementation

### File Location
```
sglang/python/sglang/srt/dllm/algorithm/sequential.py
```

### How It Works

The Sequential algorithm unmasks tokens from left to right in a predictable order:

```python
class Sequential(DllmAlgorithm):
    def run(self, model_runner, forward_batch):
        for iteration in range(self.block_size):
            # 1. Check if any tokens are still masked
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            # 2. Forward pass to get predictions
            logits = model_runner.forward(forward_batch)
            predicted_tokens = torch.argmax(logits)

            # 3. Find the FIRST (leftmost) masked position
            masked_positions = torch.where(mask_index)[0]
            first_masked_idx = masked_positions[0]

            # 4. Unmask that position
            forward_batch.input_ids[first_masked_idx] = predicted_tokens[first_masked_idx]

        # Final forward pass
        return model_runner.forward(forward_batch)
```

## Usage

### Start SGLang Server with Sequential Algorithm

```bash
python -m sglang.launch_server \
  --model-path path/to/llada2-model \
  --dllm-algorithm Sequential \
  --dllm-block-size 4
```

Or using the CLI:

```bash
sglang serve \
  --model path/to/llada2-model \
  --dllm-algorithm Sequential \
  --dllm-block-size 4
```

### Supported Algorithm Names

The algorithm discovery system now recognizes:
- `LowConfidence` - Original confidence-based algorithm
- `Sequential` - New deterministic left-to-right algorithm

## Comparison: Sequential vs LowConfidence

### Sequential Strategy

**Generation Flow** (block_size=4):
```
Initial:     [M, M, M, M]

Iteration 0: Forward pass
             → Select position 0 (leftmost)
             → [U, M, M, M]

Iteration 1: Forward pass
             → Select position 1 (leftmost remaining)
             → [U, U, M, M]

Iteration 2: Forward pass
             → Select position 2 (leftmost remaining)
             → [U, U, U, M]

Iteration 3: Forward pass
             → Select position 3 (leftmost remaining)
             → [U, U, U, U]

Final:       Forward pass → Return results
```

**Characteristics:**
- ✅ **Deterministic**: Same input always produces same unmasking order
- ✅ **Predictable**: Easy to understand and debug
- ✅ **Simple**: No confidence computation needed
- ✅ **Fast**: Minimal overhead for position selection
- ⚠️ **Less adaptive**: Doesn't use model uncertainty

### LowConfidence Strategy

**Generation Flow** (block_size=4):
```
Initial:     [M, M, M, M]

Iteration 0: Forward pass
             → Confidences: [0.9, 0.7, 0.85, 0.6]
             → Select position 0 (highest: 0.9)
             → [U, M, M, M]

Iteration 1: Forward pass
             → Confidences: [-, 0.88, 0.82, 0.75]
             → Select position 1 (highest: 0.88)
             → [U, U, M, M]

Iteration 2: Forward pass
             → Confidences: [-, -, 0.91, 0.78]
             → Select position 2 (highest: 0.91)
             → [U, U, U, M]

Iteration 3: Forward pass
             → Confidences: [-, -, -, 0.95]
             → Select position 3
             → [U, U, U, U]

Final:       Forward pass → Return results
```

**Characteristics:**
- ✅ **Adaptive**: Uses model confidence to guide unmasking
- ✅ **Quality-driven**: Unmasks most confident predictions first
- ⚠️ **Non-deterministic**: Different runs may unmask in different orders
- ⚠️ **More complex**: Requires confidence computation

## When to Use Each Strategy

### Use Sequential When:
1. **Debugging**: You need predictable, reproducible behavior
2. **Testing**: You want to verify model behavior consistently
3. **Benchmarking**: You need fair comparison across runs
4. **Simplicity**: You prefer straightforward, interpretable unmasking
5. **Development**: You're iterating on model architecture

### Use LowConfidence When:
1. **Production**: You want best quality output
2. **Quality-critical**: Generated text quality is paramount
3. **Adaptive behavior**: You want the model to guide its own generation
4. **Uncertainty-aware**: You want to leverage model confidence

## Performance Characteristics

| Metric | Sequential | LowConfidence |
|--------|------------|---------------|
| **Determinism** | ✅ Always same order | ❌ Varies by confidence |
| **Reproducibility** | ✅ Perfect | ⚠️ Depends on random seed |
| **Computation** | ✅ Minimal overhead | ⚠️ Confidence calculation |
| **Quality** | ⚠️ May not be optimal | ✅ Usually better |
| **Speed** | ✅ Slightly faster | ⚠️ Slightly slower |
| **Debugging** | ✅ Easy to trace | ⚠️ Harder to predict |

## Example Output Comparison

Given the same prompt: `"Translate to French: Hello"`

### Sequential (deterministic):
```
Input:  "Translate to French: [M][M][M][M]"

Iteration 0: "Translate to French: [Bon][M][M][M]"     (position 0)
Iteration 1: "Translate to French: [Bon][jour][M][M]"  (position 1)
Iteration 2: "Translate to French: [Bon][jour][!][M]"  (position 2)
Iteration 3: "Translate to French: [Bon][jour][!][.]"  (position 3)

Output: "Bonjour!."
```

### LowConfidence (adaptive):
```
Input:  "Translate to French: [M][M][M][M]"

Iteration 0: "Translate to French: [M][jour][M][M]"    (conf: 0.92, pos 1)
Iteration 1: "Translate to French: [Bon][jour][M][M]"  (conf: 0.89, pos 0)
Iteration 2: "Translate to French: [Bon][jour][M][.]"  (conf: 0.95, pos 3)
Iteration 3: "Translate to French: [Bon][jour][!][.]"  (conf: 0.88, pos 2)

Output: "Bonjour!."
```

Note: LowConfidence may unmask high-confidence end tokens (like ".") before uncertain middle tokens.

## Implementation Details

### Automatic Discovery

The algorithm is automatically discovered by SGLang's import system:

```python
# sglang/srt/dllm/algorithm/__init__.py
def import_algorithms():
    for module in pkgutil.iter_modules(package.__path__):
        if hasattr(module, "Algorithm"):
            mapping[module.Algorithm.__name__] = module.Algorithm
    return mapping

# Result:
algo_name_to_cls = {
    'LowConfidence': <class LowConfidence>,
    'Sequential': <class Sequential>  # ← Automatically added!
}
```

### Configuration

The algorithm is selected via `DllmConfig`:

```python
# sglang/srt/dllm/config.py
config = DllmConfig(
    mask_id=156895,
    block_size=4,
    algorithm="Sequential"  # ← Use new algorithm
)
```

## Testing

To verify the implementation:

```bash
# 1. Check algorithm is registered
python -c "
from sglang.srt.dllm.algorithm import algo_name_to_cls
print('Available algorithms:', list(algo_name_to_cls.keys()))
"
# Expected output: ['LowConfidence', 'Sequential']

# 2. Try loading the algorithm
python -c "
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.algorithm.base import DllmAlgorithm

config = DllmConfig(mask_id=156895, block_size=4, algorithm='Sequential')
algo = DllmAlgorithm.from_server_args(type('Args', (), {
    'dllm_algorithm': 'Sequential',
    'dllm_block_size': 4
})())
print('Algorithm loaded:', algo.__class__.__name__)
"
# Expected output: Algorithm loaded: Sequential
```

## Future Enhancements

Potential improvements to the Sequential algorithm:

1. **Configurable stride**: Unmask N tokens per iteration
   ```python
   denoise_num = config.get('denoise_num', 1)
   indices = masked_positions[:denoise_num]
   ```

2. **Right-to-left variant**: Reverse sequential unmasking
   ```python
   first_masked_idx = masked_positions[-1]  # Last instead of first
   ```

3. **Bidirectional**: Unmask from both ends
   ```python
   if iteration % 2 == 0:
       idx = masked_positions[0]   # Left
   else:
       idx = masked_positions[-1]  # Right
   ```

4. **Random sequential**: Random order but deterministic with seed
   ```python
   torch.manual_seed(iteration)
   idx = masked_positions[torch.randperm(len(masked_positions))[0]]
   ```

## Contributing More Algorithms

To add additional unmasking strategies:

1. Create `sglang/python/sglang/srt/dllm/algorithm/your_algorithm.py`
2. Implement your algorithm class extending `DllmAlgorithm`
3. Export it as `Algorithm = YourAlgorithm`
4. It will be automatically discovered!

Example template:

```python
from sglang.srt.dllm.algorithm.base import DllmAlgorithm

class YourAlgorithm(DllmAlgorithm):
    def run(self, model_runner, forward_batch):
        # Your unmasking logic here
        pass

Algorithm = YourAlgorithm
```

## Conclusion

The Sequential algorithm provides a valuable alternative to LowConfidence, offering:
- Deterministic, reproducible behavior
- Simplified debugging and testing
- Faster execution with minimal overhead
- Easy-to-understand unmasking pattern

It's particularly useful during development and testing, while LowConfidence remains the better choice for production deployments where quality is paramount.
