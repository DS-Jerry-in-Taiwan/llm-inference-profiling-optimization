"""
Profiling Specialist: Deep profiling of LLM inference with/without KV-cache.

- Integrates torch.profiler to capture traces for both use_cache=True/False.
- Exports trace JSON files for Chrome trace viewer.
- Prints top 10 operators by CPU time for each scenario.
- Generates profiling_report.md with operator analysis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity, record_function
import os

def run_profile(use_cache, trace_path, num_tokens=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function(f"Inference (use_cache={use_cache})"):
            # Prefill
            outputs = model(input_ids=input_ids, use_cache=use_cache)
            generated = input_ids
            past_key_values = outputs.past_key_values if use_cache else None
            # Only profile 2 decode steps for trace brevity
            for _ in range(num_tokens):
                if use_cache and past_key_values is not None:
                    outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=use_cache)
                    past_key_values = outputs.past_key_values
                else:
                    outputs = model(input_ids=generated, use_cache=use_cache)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved: {trace_path}")
    print(f"Top 10 operators by CPU time (use_cache={use_cache}):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return prof

def main():
    """
    Run profiling for both cache/no-cache, export traces, and generate report.
    """
    traces_dir = "results/traces"
    os.makedirs(traces_dir, exist_ok=True)
    prof_no_cache = run_profile(False, os.path.join(traces_dir, "trace_no_cache.json"))
    prof_with_cache = run_profile(True, os.path.join(traces_dir, "trace_with_cache.json"))

    # Analyze and write report
    def extract_top_ops(prof):
        return [
            (row.key, row.cpu_time_total)
            for row in prof.key_averages()[:5]
        ]

    top_no_cache = extract_top_ops(prof_no_cache)
    top_with_cache = extract_top_ops(prof_with_cache)

    report = "# Profiling Report: GPT2 Inference Bottleneck Analysis\n\n"
    report += "## Top 5 Operators by CPU Time\n\n"
    report += "| Rank | Operator (No Cache) | CPU Time (us) | Operator (With Cache) | CPU Time (us) |\n"
    report += "|------|---------------------|---------------|-----------------------|---------------|\n"
    for i in range(5):
        op_nc = top_no_cache[i][0] if i < len(top_no_cache) else ""
        t_nc = f"{top_no_cache[i][1]:.0f}" if i < len(top_no_cache) else ""
        op_wc = top_with_cache[i][0] if i < len(top_with_cache) else ""
        t_wc = f"{top_with_cache[i][1]:.0f}" if i < len(top_with_cache) else ""
        report += f"| {i+1} | {op_nc} | {t_nc} | {op_wc} | {t_wc} |\n"
    report += "\n"
    report += "### Key Question: How much time does the Attention mechanism take in No-Cache vs. Cache?\n"
    report += "- See above table for `aten::matmul`, `aten::softmax`, etc.\n"

    with open("results/profiling_report.md", "w") as f:
        f.write(report)
    print("Profiling report generated: results/profiling_report.md")

if __name__ == "__main__":
    main()
