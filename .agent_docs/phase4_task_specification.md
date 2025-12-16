# Phase 4 Task Specification: Dynamic Quantization

## 🎯 Objective
Apply INT8 dynamic quantization to the PyTorch GPT-2 model and benchmark the performance against the FP32 baseline and ONNX.

## 📦 Deliverables
1. ✅ `src/optimize_quantization.py`: Script for quantization and benchmarking.
2. ✅ `results/quantization_results.json`: Metrics for the quantized model.
3. ✅ `results/charts/final_comparison.png`: Comprehensive chart (PyTorch vs. ONNX vs. Quantized).
4. ✅ `models/gpt2_int8.pt`: The quantized PyTorch model.

## 📐 Execution Steps

### Step 1: Implement `src/optimize_quantization.py`
**Goal**: Quantize the model and run inference.

**Requirements**:
- Load the baseline PyTorch model.
- Use `torch.quantization.quantize_dynamic` to convert `nn.Linear` layers to `qint8`.
- Save the quantized model to `models/gpt2_int8.pt`.
- Check model size reduction (file size).

### Step 2: Benchmark Quantized Model
**Action**:
- Measure latency (First Token & Per Token) for the INT8 model.
- Generate text to manually verify quality (ensure it's not gibberish).
- Compare speedup vs. Baseline FP32.

### Step 3: Final Comparison & Visualization
**Action**:
- Aggregate all results:
  1. PyTorch FP32 (Baseline)
  2. PyTorch FP32 + KV-Cache (from Phase 1)
  3. ONNX FP32 (from Phase 3)
  4. PyTorch INT8 (Quantized)
- Create a final bar chart `results/charts/final_comparison.png`.
- Write a summary of all optimizations.

## 🧪 Verification
- `models/gpt2_int8.pt` is significantly smaller than the original model.
- Inference runs without error.
- Final chart shows the performance hierarchy.
