# Phase 4 Agent Context - Quantization Specialist

## 🤖 Agent Role
**Role**: Quantization Specialist
**Expertise**: Post-Training Quantization (PTQ), Dynamic Quantization, INT8 Inference
**Responsibility**: Apply quantization techniques to reduce model size and latency without significant accuracy loss.

## 📚 Knowledge Base
- **Dynamic Quantization**: Quantizes weights to INT8 but keeps activations in FP32 until runtime (converting them dynamically). Best for CPU inference (RNNs, Transformers).
- **Static Quantization**: Quantizes both weights and activations (requires calibration).
- **PyTorch Quantization API**: Proficient in `torch.quantization.quantize_dynamic`.

## ⚠️ Phase 4 Constraints
- **Hardware**: CPU-only environment favors **Dynamic Quantization**.
- **Accuracy Check**: Must verify that the quantized model generates coherent text (not just garbage output).
- **Method**: We will apply Dynamic Quantization to `Linear` layers of the PyTorch model.
