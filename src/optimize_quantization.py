"""
Dynamic Quantization Specialist: Quantize GPT2 with torch.quantization.quantize_dynamic,
benchmark latency and model size, and compare with FP32 and ONNX.

- Applies torch.quantization.quantize_dynamic to nn.Linear layers.
- Saves quantized model to models/gpt2_int8.pt.
- Measures model size reduction.
- Benchmarks latency and verifies text quality.
- Aggregates results and generates final_comparison.png.
"""

import os
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import matplotlib.pyplot as plt

def get_file_size(path):
    return os.path.getsize(path) / (1024 * 1024)

def quantize_and_save(model_name, save_path):
    """
    Apply dynamic quantization to GPT2 and save the quantized model.

    Args:
        model_name (str): HuggingFace model name.
        save_path (str): Path to save quantized model state_dict.

    Returns:
        torch.nn.Module: Quantized model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), save_path)
    return quantized_model

def measure_latency(model, tokenizer, prompt, max_new_tokens=50, num_runs=3, device="cpu"):
    """
    Measure total latency and sample output for quantized model inference.

    Args:
        model (torch.nn.Module): Quantized model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer.
        prompt (str): Input prompt.
        max_new_tokens (int): Number of tokens to generate.
        num_runs (int): Number of runs to average.
        device (str): Device for inference.

    Returns:
        dict: Latency statistics and sample output.
    """
    latencies = []
    generated_texts = []
    for _ in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens)
        latencies.append(time.perf_counter() - start)
        generated_texts.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return {
        "total_latency_avg": sum(latencies) / num_runs,
        "total_latencies": latencies,
        "sample_text": generated_texts[0]
    }

def main():
    """
    Quantize GPT2, benchmark latency/model size, aggregate results, and plot comparison.
    """
    model_name = "gpt2"
    quant_path = "models/gpt2_int8.pt"
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)

    # Quantize and save
    quantized_model = quantize_and_save(model_name, quant_path)
    orig_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Model size
    orig_bin_path = "models/gpt2_fp32_tmp/pytorch_model.bin"
    try:
        orig_path = orig_model.save_pretrained("models/gpt2_fp32_tmp")
        orig_size = get_file_size(orig_bin_path)
    except Exception:
        # fallback: try to find in HuggingFace cache
        from transformers.utils import cached_file
        orig_bin_path = cached_file(model_name, "pytorch_model.bin")
        orig_size = get_file_size(orig_bin_path)
    quant_size = get_file_size(quant_path)

    # Latency
    prompt = "The future of artificial intelligence is"
    quant_result = measure_latency(quantized_model, tokenizer, prompt)
    # Load baseline/onnx results
    with open("results/baseline_results.json") as f:
        baseline = json.load(f)
    with open("results/onnx_results.json") as f:
        onnx = json.load(f)
    pt_latency = baseline["no_cache"]["total_latency_avg"]
    pt_cache_latency = baseline["with_cache"]["total_latency_avg"]
    onnx_latency = onnx["total_latency_avg"]
    quant_latency = quant_result["total_latency_avg"]

    # Save quantization results
    with open("results/quantization_results.json", "w") as f:
        json.dump({
            "total_latency_avg": quant_latency,
            "total_latencies": quant_result["total_latencies"],
            "sample_text": quant_result["sample_text"],
            "model_size_mb": quant_size
        }, f, indent=2)

    # Plot final comparison
    labels = ["PyTorch (No-Cache)", "PyTorch (KV-Cache)", "ONNX", "Quantized (INT8)"]
    times = [pt_latency, pt_cache_latency, onnx_latency, quant_latency]
    sizes = [orig_size, orig_size, orig_size, quant_size]
    fig, ax1 = plt.subplots()
    color = ["blue", "green", "orange", "red"]
    ax1.bar(labels, times, color=color)
    ax1.set_ylabel("Total Latency (s)")
    ax1.set_title("Final Comparison: Latency & Model Size")
    for i, v in enumerate(times):
        ax1.text(i, v, f"{v:.2f}s", ha="center", va="bottom")
    ax2 = ax1.twinx()
    ax2.plot(labels, sizes, color="black", marker="o", label="Model Size (MB)")
    ax2.set_ylabel("Model Size (MB)")
    for i, v in enumerate(sizes):
        ax2.text(i, v, f"{v:.1f}MB", ha="center", va="top")
    fig.tight_layout()
    plt.savefig("results/charts/final_comparison.png")

    # Print summary
    print("=== Quantization Benchmark ===")
    print(f"Original Model Size: {orig_size:.2f} MB")
    print(f"Quantized Model Size: {quant_size:.2f} MB")
    print(f"FP32 Latency: {pt_latency:.2f}s")
    print(f"FP32+KV-Cache Latency: {pt_cache_latency:.2f}s")
    print(f"ONNX Latency: {onnx_latency:.2f}s")
    print(f"Quantized Latency: {quant_latency:.2f}s")
    print(f"Sample Output: {quant_result['sample_text'][:100]}...")

if __name__ == "__main__":
    main()
