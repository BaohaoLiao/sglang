# Sequential DLLM Cheat Sheet

## TL;DR - Quick Commands

```bash
# Start server with Sequential
sglang serve --model /path/to/llada2 --dllm-algorithm Sequential --dllm-block-size 4

# Verify it works
python verify_sequential.py

# Test generation
curl http://localhost:30000/generate -H "Content-Type: application/json" \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 20}}'
```

## Command Comparison

### Sequential (Deterministic)
```bash
python -m sglang.launch_server \
  --model-path /path/to/llada2 \
  --dllm-algorithm Sequential \    # ← Use this
  --dllm-block-size 4
```

### LowConfidence (Adaptive - Default)
```bash
python -m sglang.launch_server \
  --model-path /path/to/llada2 \
  --dllm-algorithm LowConfidence \  # ← Or this
  --dllm-block-size 4
```

## Python One-Liners

### Start Runtime
```python
import sglang as sgl
runtime = sgl.Runtime(model_path="/path/to/llada2", dllm_algorithm="Sequential", dllm_block_size=4)
```

### Generate
```python
output = runtime.generate("Translate: Hello", max_new_tokens=50)
```

### Check Algorithm
```python
from sglang.srt.dllm.algorithm import algo_name_to_cls
print(list(algo_name_to_cls.keys()))  # ['LowConfidence', 'Sequential']
```

## Key Parameters

| Flag | Value | Purpose |
|------|-------|---------|
| `--dllm-algorithm` | `Sequential` | Use deterministic left-to-right |
| `--dllm-algorithm` | `LowConfidence` | Use confidence-based (default) |
| `--dllm-block-size` | `4` | Tokens per block (common) |
| `--dllm-block-size` | `8` | More parallel generation |
| `--dllm-block-size` | `2` | Less parallel, may be faster |

## When to Use What

| Scenario | Use This |
|----------|----------|
| 🔧 Debugging | `Sequential` |
| 📊 Testing | `Sequential` |
| 🎓 Learning | `Sequential` |
| 🚀 Production | `LowConfidence` |
| 🎯 Quality-critical | `LowConfidence` |
| ⚡ Speed-critical | `Sequential` |

## File Locations

```
sglang/python/sglang/srt/dllm/algorithm/
├── __init__.py              # Auto-discovery
├── base.py                  # Base class
├── low_confidence.py        # Confidence-based
└── sequential.py            # Sequential (NEW!)
```

## Troubleshooting One-Liners

```bash
# Check algorithm is registered
python -c "from sglang.srt.dllm.algorithm import algo_name_to_cls; print(list(algo_name_to_cls.keys()))"

# Load Sequential
python -c "from sglang.srt.dllm.config import DllmConfig; from sglang.srt.dllm.algorithm import get_algorithm; print(get_algorithm(DllmConfig(mask_id=156895, block_size=4, algorithm='Sequential')))"

# Check model architecture
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('/path/to/model').architectures)"
```

## HTTP API Examples

### Generate
```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 50}}'
```

### Health Check
```bash
curl http://localhost:30000/health
```

### Model Info
```bash
curl http://localhost:30000/get_model_info
```

## Differences at a Glance

```
Sequential:     [M M M M] → [U M M M] → [U U M M] → [U U U M] → [U U U U]
                    ↓            ↓            ↓            ↓            ↓
                  pos 0        pos 1        pos 2        pos 3       done
                (always left-to-right, predictable)

LowConfidence:  [M M M M] → [M U M M] → [U U M M] → [U U M U] → [U U U U]
                    ↓            ↓            ↓            ↓            ↓
                conf 0.9     conf 0.88    conf 0.91    conf 0.85     done
                (highest confidence first, adaptive)
```

## Performance Comparison

| Metric | Sequential | LowConfidence |
|--------|------------|---------------|
| Speed | ⚡⚡⚡⚡⚡ (5/5) | ⚡⚡⚡⚡ (4/5) |
| Quality | ⭐⭐⭐ (3/5) | ⭐⭐⭐⭐ (4/5) |
| Determinism | ✅ Yes | ❌ No |
| Complexity | 😊 Simple | 😐 Medium |

## Common Configurations

### Development
```bash
--dllm-algorithm Sequential --dllm-block-size 4 --mem-fraction-static 0.7
```

### Production
```bash
--dllm-algorithm LowConfidence --dllm-block-size 4 --mem-fraction-static 0.9
```

### High Throughput
```bash
--dllm-algorithm Sequential --dllm-block-size 8 --max-running-requests 64
```

### High Quality
```bash
--dllm-algorithm LowConfidence --dllm-block-size 4 --max-running-requests 16
```

## Quick Reference Links

- Full docs: `SEQUENTIAL_DLLM.md`
- Quick start: `SEQUENTIAL_QUICKSTART.md`
- Comparison: `DLLM_ALGORITHMS_COMPARISON.md`
- Implementation: `IMPLEMENTATION_SUMMARY.md`

---

**Remember**: Sequential = Predictable, LowConfidence = Quality
