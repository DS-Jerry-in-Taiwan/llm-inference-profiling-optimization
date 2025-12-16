# Phase 2 Task Specification: Deep Profiling

## 🎯 Objective
Use PyTorch Profiler to capture execution traces of the inference process and identify bottlenecks.

## 📦 Deliverables
1. ✅ `src/profiling.py`: Script with PyTorch Profiler integration.
2. ✅ `results/traces/trace_no_cache.json`: Trace file for `use_cache=False`.
3. ✅ `results/traces/trace_with_cache.json`: Trace file for `use_cache=True`.
4. ✅ `results/profiling_report.md`: Analysis report summarizing the findings.

## 📐 Execution Steps

### Step 1: Implement `src/profiling.py`
**Goal**: Create a script that runs inference inside a profiler context.

**Requirements**:
- Import `torch.profiler`.
- Setup `torch.profiler.profile` with:
  - `activities=[ProfilerActivity.CPU]` (add CUDA if available).
  - `record_shapes=True`, `profile_memory=True`, `with_stack=True`.
- Wrap the `model.generate` (or the internal loop) with `record_function` labels (e.g., "Inference Step").
- Export traces using `prof.export_chrome_trace()`.

### Step 2: Run Profiling
**Action**:
- Run `python src/profiling.py`.
- Ensure it runs BOTH scenarios (With and Without Cache).
- **Limit**: Profile only 1-2 generation steps (not the full 50 tokens) to keep traces readable.

### Step 3: Analyze & Report
**Action**:
- Programmatically analyze the profiler output (using `prof.key_averages().table()`).
- Extract top 5 most time-consuming operators for both scenarios.
- Write a summary to `results/profiling_report.md`.
- **Key Question to Answer**: "How much time does the Attention mechanism take in No-Cache vs. Cache?"

## 🧪 Verification
- Check if `results/traces/*.json` files exist and are > 0 bytes.
- Check if `results/profiling_report.md` contains a comparison table.
