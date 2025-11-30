# Quick Start: Using Sequential DLLM in SGLang

## Prerequisites

- SGLang installed with DLLM support
- LLaDA2 model (or another DLLM-compatible model)
- Sequential algorithm file at: `sglang/python/sglang/srt/dllm/algorithm/sequential.py`

## Method 1: Command Line (Recommended)

### Using `launch_server`

```bash
python -m sglang.launch_server \
  --model-path /path/to/llada2-model \
  --dllm-algorithm Sequential \
  --dllm-block-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

### Using `sglang` CLI

```bash
sglang serve \
  --model /path/to/llada2-model \
  --dllm-algorithm Sequential \
  --dllm-block-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

### Full Example with All Options

```bash
python -m sglang.launch_server \
  --model-path /path/to/llada2-model \
  --dllm-algorithm Sequential \
  --dllm-block-size 4 \
  --tp 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.8
```

## Method 2: Python API

### Basic Usage

```python
import sglang as sgl

# Initialize with Sequential algorithm
runtime = sgl.Runtime(
    model_path="/path/to/llada2-model",
    dllm_algorithm="Sequential",
    dllm_block_size=4
)

# Generate text
output = runtime.generate(
    "Translate to French: Hello, how are you?",
    max_new_tokens=50
)

print(output)
```

### Advanced Configuration

```python
from sglang.srt.server_args import ServerArgs
from sglang import Runtime

# Create server args with Sequential
args = ServerArgs(
    model_path="/path/to/llada2-model",
    dllm_algorithm="Sequential",
    dllm_block_size=4,
    tp_size=1,
    mem_fraction_static=0.8,
    port=30000
)

# Initialize runtime
runtime = Runtime.from_server_args(args)

# Use it
response = runtime.generate(
    prompt="What is the capital of France?",
    max_new_tokens=100,
    temperature=0.7
)

print(response)
```

## Method 3: HTTP API

### 1. Start the Server

```bash
sglang serve \
  --model /path/to/llada2-model \
  --dllm-algorithm Sequential \
  --dllm-block-size 4 \
  --port 30000
```

### 2. Send Requests

#### Using curl

```bash
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Translate to French: Hello",
    "sampling_params": {
      "max_new_tokens": 50,
      "temperature": 0.7
    }
  }'
```

#### Using Python requests

```python
import requests
import json

url = "http://localhost:30000/generate"

payload = {
    "text": "Translate to French: Hello",
    "sampling_params": {
        "max_new_tokens": 50,
        "temperature": 0.7
    }
}

response = requests.post(url, json=payload)
result = response.json()

print(result["text"])
```

#### Using OpenAI-compatible API

```python
from openai import OpenAI

# Point to SGLang server
client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

# Generate
response = client.completions.create(
    model="llada2-model",
    prompt="Translate to French: Hello",
    max_tokens=50,
    temperature=0.7
)

print(response.choices[0].text)
```

## Method 4: Programmatic (Advanced)

### Direct Algorithm Usage

```python
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.algorithm import get_algorithm

# Create config
config = DllmConfig(
    mask_id=156895,      # LLaDA2 mask token
    block_size=4,
    algorithm="Sequential"
)

# Get algorithm instance
algorithm = get_algorithm(config)

print(f"Using algorithm: {algorithm.__class__.__name__}")
print(f"Block size: {algorithm.block_size}")
print(f"Mask ID: {algorithm.mask_id}")

# Use in your inference pipeline
# (requires model_runner and forward_batch setup)
```

## Configuration Options

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model-path` | Path to LLaDA2 model | `/models/llada2-7b` |
| `--dllm-algorithm` | Algorithm name | `Sequential` |
| `--dllm-block-size` | Tokens per block | `4` |

### Optional Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--tp` | Tensor parallelism | `1` | `--tp 2` |
| `--port` | Server port | `30000` | `--port 8080` |
| `--host` | Server host | `127.0.0.1` | `--host 0.0.0.0` |
| `--mem-fraction-static` | GPU memory fraction | `0.9` | `--mem-fraction-static 0.8` |

## Switching Between Algorithms

### Sequential (Deterministic)

```bash
sglang serve \
  --model /path/to/llada2 \
  --dllm-algorithm Sequential \  # ← Deterministic
  --dllm-block-size 4
```

**Use when**: Debugging, testing, reproducibility

### LowConfidence (Adaptive)

```bash
sglang serve \
  --model /path/to/llada2 \
  --dllm-algorithm LowConfidence \  # ← Adaptive
  --dllm-block-size 4
```

**Use when**: Production, quality-critical applications

## Verification

### Check Algorithm is Loaded

```bash
# Run verification script
cd sglang
python verify_sequential.py
```

Expected output:
```
✅ SUCCESS: Sequential algorithm is registered!
✅ SUCCESS: Sequential algorithm instantiated!
```

### Check Server Logs

When you start the server, look for:

```
[INFO] Loading DLLM algorithm: Sequential
[INFO] DLLM block size: 4
[INFO] DLLM mask token: 156895
```

## Troubleshooting

### Algorithm Not Found

**Error**: `Unknown diffusion LLM algorithm: Sequential`

**Solution**:
1. Verify file exists: `sglang/python/sglang/srt/dllm/algorithm/sequential.py`
2. Check file has `Algorithm = Sequential` at the end
3. Restart Python/server to reload modules

### Import Error

**Error**: `ModuleNotFoundError: No module named 'sglang.srt.dllm.algorithm.sequential'`

**Solution**:
1. Check file permissions: `chmod 644 sequential.py`
2. Verify `__init__.py` exists in algorithm directory
3. Reinstall SGLang if needed: `pip install -e .`

### Model Not Compatible

**Error**: `Unknown diffusion LLM: <model_name>`

**Solution**:
- Sequential only works with DLLM models (currently LLaDA2)
- Check `config.py:29` for supported architectures
- Use a compatible model

## Examples

### Example 1: Simple Generation

```bash
# Start server
sglang serve \
  --model /models/llada2-7b \
  --dllm-algorithm Sequential \
  --dllm-block-size 4

# In another terminal
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "sampling_params": {"max_new_tokens": 20}}'
```

### Example 2: Batch Processing

```python
import sglang as sgl

runtime = sgl.Runtime(
    model_path="/models/llada2-7b",
    dllm_algorithm="Sequential",
    dllm_block_size=4
)

prompts = [
    "Translate to French: Good morning",
    "Translate to Spanish: Thank you",
    "Translate to German: Goodbye"
]

outputs = runtime.generate(
    prompts,
    max_new_tokens=30,
    temperature=0.7
)

for prompt, output in zip(prompts, outputs):
    print(f"{prompt} -> {output}")
```

### Example 3: Streaming

```python
import sglang as sgl

runtime = sgl.Runtime(
    model_path="/models/llada2-7b",
    dllm_algorithm="Sequential",
    dllm_block_size=4
)

# Stream generation
for token in runtime.generate_stream(
    "Write a short story:",
    max_new_tokens=100
):
    print(token, end="", flush=True)
```

## Performance Tips

1. **Block Size**: Start with 4, increase if quality is good
   ```bash
   --dllm-block-size 4  # Good default
   --dllm-block-size 8  # More parallel, may reduce quality
   ```

2. **Memory**: Adjust based on GPU size
   ```bash
   --mem-fraction-static 0.9  # 90% GPU memory (default)
   --mem-fraction-static 0.7  # 70% if running multiple processes
   ```

3. **Batching**: Enable for throughput
   ```bash
   --max-running-requests 32  # Concurrent requests
   ```

## Comparison

| Aspect | Sequential | LowConfidence |
|--------|------------|---------------|
| **Determinism** | ✅ Always same | ❌ Varies |
| **Speed** | ⚡ Slightly faster | ⚡ Slightly slower |
| **Quality** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Better |
| **Debugging** | ✅ Easy | ⚠️ Harder |
| **Use Case** | Development, testing | Production |

## Next Steps

- Read `SEQUENTIAL_DLLM.md` for detailed documentation
- Check `DLLM_ALGORITHMS_COMPARISON.md` for algorithm comparison
- Experiment with different block sizes
- Compare with LowConfidence for your use case

## Support

If you encounter issues:
1. Check server logs for error messages
2. Run `verify_sequential.py` to verify installation
3. Compare with LowConfidence to isolate DLLM vs algorithm issues
4. Check SGLang documentation for model compatibility
