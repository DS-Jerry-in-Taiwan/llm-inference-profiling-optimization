# Phase 3 Task Specification: ONNX Optimization

## 🎯 Objective
Convert the PyTorch GPT-2 model to ONNX format and compare inference speed using ONNX Runtime.

## 📦 Deliverables
1. ✅ `src/optimize_onnx.py`: Script for ONNX conversion and inference.
2. ✅ `models/gpt2.onnx`: The exported ONNX model.
3. ✅ `results/onnx_results.json`: Latency metrics for ONNX.
4. ✅ `results/charts/onnx_comparison.png`: Chart comparing PyTorch vs. ONNX.

## 📐 Execution Steps

### Step 1: Implement `src/optimize_onnx.py`
**Goal**: Export model and run inference.

**Requirements**:
- Use `torch.onnx.export` to convert `gpt2`.
- **Handling KV-Cache**: 
  - *Option A (Recommended)*: Use `optimum.onnxruntime.ORTModelForCausalLM` to automatically handle export.
  - *Option B (Manual)*: Define dynamic axes for `input_ids` and `attention_mask`. For this MVP, you can start by exporting the **No-Cache** version first to ensure success.
- Implement `OnnxInference` class using `onnxruntime.InferenceSession`.

### Step 2: Run ONNX Inference
**Action**:
- Load the ONNX model.
- Run inference on the same prompt ("The future of artificial intelligence is").
- Measure latency (First Token & Per Token).

### Step 3: Compare & Visualize
**Action**:
- Load previous baseline results (`results/baseline_results.json`).
- Compare PyTorch (No-Cache) vs. ONNX (No-Cache).
- (Optional) Compare PyTorch (Cache) vs. ONNX (Cache) if export succeeded.
- Generate `results/charts/onnx_comparison.png` showing the speedup.

## 🧪 Verification
- `models/gpt2.onnx` exists.
- `src/optimize_onnx.py` runs without error.
- Console output shows "ONNX Speedup: X.XXx".
