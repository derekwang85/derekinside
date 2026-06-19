# DereInside Benchmarks

> 测试日期: 2026-06-19 | 硬件: Intel Xeon 8352S (32核) / 62GB RAM / CPU-only ollama

## Summary

| Metric | Value | Comparison |
|:-------|:-----:|:-----------|
| CLI Search (cold, 12 queries) | avg **2.04s** | Bottleneck: embedding |
| CLI Search (warm cache) | avg **~50ms** | After EmbeddingCache |
| HTTP API (warm cache, 5 queries) | avg **336ms** | Includes network + pool latency |
| Cache Speedup | **40x** | 2.04s → 50ms |
| Concurrency (5 workers x 2 rounds) | ✅ | Connection pool works |
| Time Travel (before/after filter) | ✅ | p.created_at filter |
| Knowledge Graph | 687 entities, 1427 links | Regex-extracted |

### Competitor Comparison (LongMemEval Recall@5)

| System | Score | Notes |
|:-------|:-----:|:------|
| MemPalace | 96.6% | SOTA, uses reranker + spaces |
| Graphiti (Zep) | 63.8% | Real-time temporal graphs |
| Mem0 | 49.0% | Hybrid vector+graph+KV |
| **DereInside** | **TBD** | Need to run LongMemEval |

## Detailed Results

### Latency by Query

| Query | CLI (cold) | HTTP (warm) | Cache |
|:------|:----------:|:-----------:|:-----:|
| KYC approval flow | 2.04s | 4.6ms | ✅ |
| buyer risk assessment | 0.93s | 5.1ms | ✅ |
| trade order processing | 9.79s | 5.0ms | ✅ |
| document management | 1.88s | 5.3ms | ✅ |
| user authentication | 2.47s | 4.8ms | ✅ |
| KYCService | 1.98s | 5.0ms | ✅ |
| VaTransactions | 1.05s | 4.9ms | ✅ |
| approveKYC | 2.22s | 4.7ms | ✅ |
| database schema | 1.54s | 5.2ms | ✅ |
| REST API endpoints | 1.78s | 5.0ms | ✅ |
| security configuration | 1.46s | 4.8ms | ✅ |
| logging and audit | 1.14s | 5.0ms | ✅ |

### Throughput

| Scenario | Queries | Total Time | Avg/query |
|:---------|:-------:|:----------:|:---------:|
| Sequential CLI (12 queries) | 12 | ~24.5s | 2.04s |
| Concurrent HTTP (5 workers) | 10 | ~2.8s | 0.28s |

### Embedding Cache

| Metric | Value |
|:-------|:-----:|
| Cache size | 256 entries (LRU) |
| Cold request | ~2.3s |
| Warm request | ~5ms |
| Speedup | 460x |

## Running

```bash
python3 tests/benchmark.py
```
