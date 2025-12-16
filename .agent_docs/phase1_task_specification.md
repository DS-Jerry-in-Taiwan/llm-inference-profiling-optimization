# Phase 1 Task Specification: Baseline & KV-Cache Analysis

## 🎯 Objective
Implement a minimal LLM inference pipeline using `gpt2` (or `distilgpt2`) to establish a performance baseline and verify the impact of KV-cache.

## 📦 Deliverables
1. ✅ `src/baseline.py`: Fully implemented inference script.
2. ✅ `results/baseline_results.json`: Latency metrics.
3. ✅ `results/charts/baseline_comparison.png`: Chart comparing With/Without KV-cache.

## 📐 Execution Steps

### Step 1: Implement `src/baseline.py`
**Goal**: Create a script that loads a model and performs text generation.

**Requirements**:
- Use `AutoModelForCausalLM` and `AutoTokenizer`.
- Implement a class `BaselineInference` with methods:
  - `load_model(model_name)`
  - `generate(prompt, max_new_tokens, use_cache)`
  - `measure_latency(prompt, max_new_tokens, use_cache, num_runs)`

**Key Logic**:
- **Device**: Auto-detect (`cuda` if available, else `cpu`).
- **Timing**: Use `time.perf_counter()` for precision.
- **Metric**: Record `First Token Latency` (prefill) and `Per Token Latency` (decode).

### Step 2: Run Experiments
**Goal**: Measure latency for two scenarios:
1. `use_cache=False`: Recomputes attention for every token (Slow).
2. `use_cache=True`: Uses KV-cache (Fast).

**Parameters**:
- Model: `gpt2` (124M) or `distilgpt2` (for faster CPU testing).
- Prompt: "The future of artificial intelligence is"
- Max New Tokens: 50
- Runs: 3 (take average).

### Step 3: Save Results & Visualize
**Goal**: Save data and generate a comparison chart.

**Actions**:
- Save metrics to `results/baseline_results.json`.
- Use `matplotlib` to create a bar chart comparing "Total Time" and "Per Token Latency".
- Save chart to `results/charts/baseline_comparison.png`.

## 🧪 Verification
Run `python src/baseline.py`.
Expected Output:
- Logs showing model loading.
- Latency for No-Cache vs Cache.
- "Baseline experiment completed successfully." message.
