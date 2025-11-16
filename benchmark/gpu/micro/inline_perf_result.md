
# CUDA Benchmark Results

**Device:** Tesla P40  
**Timestamp:** 2025-11-17T00:37:07.170095  

## Workload Configuration

| Workload | Binary | Elements | Iterations | Threads | Blocks |
|----------|--------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/vec_add | 100000 | 1000 | 512 | 196 |
| medium | benchmark/gpu/workload/vec_add | 10000 | 10000 | 256 | 40 |
| small | benchmark/gpu/workload/vec_add | 1000 | 10000 | 256 | 4 |
| tiny | benchmark/gpu/workload/vec_add | 32 | 10000 | 32 | 1 |
| xlarge | benchmark/gpu/workload/vec_add | 1000000 | 1000 | 512 | 1954 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (tiny) | tiny | 9.28 | - | - |
| Baseline (small) | small | 7.88 | - | - |
| Baseline (medium) | medium | 8.42 | - | - |
| Baseline (large) | large | 9.54 | - | - |
| Empty probe - Inline (tiny) | tiny | 10.22 | 9.28 | 1.10x (+10.1%) |
| Empty probe - Legacy (tiny) | tiny | 9.24 | 9.28 | 0.99x (-0.4%) |
| Empty probe - Inline (small) | small | 9.74 | 7.88 | 1.24x (+23.6%) |
| Empty probe - Legacy (small) | small | 9.27 | 7.88 | 1.18x (+17.6%) |
| Empty probe - Inline (medium) | medium | 9.33 | 8.42 | 1.11x (+10.8%) |
| Empty probe - Legacy (medium) | medium | 9.93 | 8.42 | 1.18x (+17.9%) |
| Empty probe - Inline (large) | large | 11.80 | 9.54 | 1.24x (+23.7%) |
| Empty probe - Legacy (large) | large | 12.80 | 9.54 | 1.34x (+34.2%) |
| Entry+Exit - Inline (tiny) | tiny | 10.23 | 9.28 | 1.10x (+10.2%) |
| Entry+Exit - Legacy (tiny) | tiny | 10.29 | 9.28 | 1.11x (+10.9%) |
| Entry+Exit - Inline (small) | small | 10.14 | 7.88 | 1.29x (+28.7%) |
| Entry+Exit - Legacy (small) | small | 10.24 | 7.88 | 1.30x (+29.9%) |
| Entry+Exit - Inline (medium) | medium | 10.42 | 8.42 | 1.24x (+23.8%) |
| Entry+Exit - Legacy (medium) | medium | 10.33 | 8.42 | 1.23x (+22.7%) |
| Entry+Exit - Inline (large) | large | 12.87 | 9.54 | 1.35x (+34.9%) |
| Entry+Exit - Legacy (large) | large | 12.76 | 9.54 | 1.34x (+33.8%) |

