# Methodology: LLM Inference Optimization

## KV-Cache

**What is KV-Cache?**  
KV-Cache (Key-Value Cache) is a technique used in transformer-based language models to store the intermediate key and value tensors computed during previous decoding steps.  
- **Without KV-Cache**: Each new token requires recomputing attention over the entire sequence, resulting in $O(N^2)$ complexity.
- **With KV-Cache**: Only the new token attends to previous tokens, reducing complexity to $O(N)$ for each new token.

**Benefit**:  
- Dramatically reduces decode-time latency, especially for long sequences.

---

## ONNX Optimization

**What is ONNX?**  
ONNX (Open Neural Network Exchange) is an open standard for representing machine learning models.  
- **Export**: PyTorch models are converted to ONNX format.
- **Inference**: ONNX Runtime applies graph optimizations (e.g., operator fusion, constant folding) for faster execution.

**Graph Fusion**:  
- Combines multiple operations into a single kernel, reducing memory access and improving CPU utilization.

**Benefit**:  
- Achieves significant speedup on CPU inference without changing model accuracy.

---

## Quantization (INT8)

**What is Quantization?**  
Quantization reduces the precision of model weights and/or activations from FP32 (float) to INT8 (integer).  
- **Dynamic Quantization**: Only weights are quantized to INT8; activations are converted on-the-fly during inference.  
- **Static Quantization**: Both weights and activations are quantized, but requires calibration.

**Why INT8 is Faster on CPU**:  
- Integer operations are faster and use less memory bandwidth.
- Reduces model size, improving cache locality.

**Trade-offs**:  
- Slight accuracy loss is possible, but for LLMs, dynamic quantization of Linear layers typically preserves output quality.

---

## Trade-offs & Observations

| Method      | Latency | Model Size | Notes                        |
|-------------|---------|------------|------------------------------|
| Baseline    | High    | Large      | No optimization              |
| KV-Cache    | Low     | -          | Best for long sequences      |
| ONNX        | Low     | -          | Requires export, no accuracy loss |
| Quantized   | Lowest  | Smallest   | Slight accuracy trade-off    |

- **Accuracy**: All methods produced coherent text; quantization did not degrade output quality in our tests.
- **Speed**: Quantization and ONNX both provided >2x speedup over baseline.
- **Memory**: Quantization reduced model size by ~11MB (INT8 weights).

---

## References

- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Transformers KV-Cache](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2#kv-cache)
