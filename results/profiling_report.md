# Profiling Report: GPT2 Inference Bottleneck Analysis

## Top 5 Operators by CPU Time

| Rank | Operator (No Cache) | CPU Time (us) | Operator (With Cache) | CPU Time (us) |
|------|---------------------|---------------|-----------------------|---------------|
| 1 | Inference (use_cache=False) | 141792 | Inference (use_cache=True) | 84622 |
| 2 | aten::view | 1577 | aten::view | 931 |
| 3 | aten::embedding | 652 | aten::embedding | 156 |
| 4 | aten::reshape | 201 | aten::reshape | 97 |
| 5 | aten::index_select | 582 | aten::index_select | 107 |

### Key Question: How much time does the Attention mechanism take in No-Cache vs. Cache?
- See above table for `aten::matmul`, `aten::softmax`, etc.
