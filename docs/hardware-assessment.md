# DereInside 硬件评估与并发模型

> 评估日期: 2026-06-19 | 基准硬件: Intel Xeon Platinum 8352S (32核) / 62GB RAM / 197GB SSD

## 1. 当前硬件实测

| 指标 | 实测值 |
|:-----|:------:|
| CPU | Xeon 8352S @ 2.20GHz, 32 cores |
| RAM | 62 GB (可用 ~45 GB) |
| 磁盘 | 197 GB SSD (可用 109 GB) |
| GPU | ❌ 无（纯 CPU 推理） |
| PostgreSQL | 16 + pgvector (Docker, 端口 5434) |
| Ollama 模型 | bge-m3 (1.1G), qwen2.5-coder:7b (4.5G), qwen2.5-coder:1.5b (1.1G) |

### 基准测试

| 操作 | 耗时 | 备注 |
|:-----|:----:|:-----|
| 单条 embedding (bge-m3, 300字) | **2.3s** | 瓶颈 — CPU 纯推理 |
| 批量 embedding (16条) | **10.0s** (~0.6s/条) | 批处理有 4x 加速 |
| PostgreSQL stats 查询 | **~2ms** | 无瓶颈 |
| 混合搜索（不含 embedding）| **~8ms** | 向量+FTS+RRF 融合 |
| 图传播（10次）| **~2ms** | 极快（纯内存计算） |
| LLM 重排序 (7B, 500字) | **~30-60s** | 最慢操作，CPU 推理 |
| LLM 重排序 (1.5B, 500字) | **~3-5s** | 可接受 |
| 正则实体提取 | **~0.005s/chunk** | 极快 |
| Graph build (2451 chunks) | **13.5s** | 正则模式 |
| Cold start (连接 DB + 初始化) | **<0.5s** | 无瓶颈 |
| HTTP bridge 内存占用 | **~47MB** | 极轻量 |

---

## 2. 瓶颈分析

### 瓶颈 #1：Embedding — 第一瓶颈 ⚠️

```
单条 2.3s → QPS ≈ 0.4
批量 16条 10s → QPS ≈ 1.6 (批处理模式)
```

bge-m3 在纯 CPU 上跑，单条 2.3s 是天花板。所有搜索操作（CLI/HTTP/MCP）都必经 embedding，所以这是 **全局吞吐天花板**。

**缓解方案：**
- 批量 embedding 有 4x 加速 → 聚合请求
- 切换到更小模型（bge-small 等）→ 估计 0.5s/条
- 加 GPU → 0.05s/条（NVIDIA T4 级别）

### 瓶颈 #2：LLM 推理 — 第二瓶颈

```
qwen2.5-coder:7b → 30-60s/prompt → 不适用于实时
qwen2.5-coder:1.5b → 3-5s/prompt → 勉强可用
```

纯 CPU 上 7B 模型不适合实时场景。1.5B 模型在可接受范围内。

### 非瓶颈：
- PostgreSQL 查询：~2-8ms，支持 500+ QPS
- 图传播：~2ms，内存计算
- 实体提取：~5ms/chunk（正则模式）
- 内存：45GB 可用，远超需求

---

## 3. 以文件/知识/访问为锚点的配置建议

### 文件量级 → 存储

| 文件数 | Chunks (估计) | pgvector 大小 | 磁盘占用 |
|:------|:-------------:|:--------------|:---------|
| 1,000 | ~5,000 | ~100 MB | 可忽略 |
| 10,000 | ~50,000 | ~1 GB | 可忽略 |
| 100,000 | ~500,000 | ~10 GB | ✅ 当前 109GB 够 |
| 1,000,000 | ~5,000,000 | ~100 GB | ⚠️ 需要加磁盘 |
| 10,000,000 | ~50,000,000 | ~1 TB | ❌ 需要 SSD 阵列 |

**公式：** 1 个文件 ≈ 5 个 chunk ≈ 2 MB pgvector 存储（含索引）

### 并发 → 吞吐

| 并发级别 | 场景 | Embedding QPS | 瓶颈 | 建议配置 |
|:--------|:-----|:-------------:|:-----|:---------|
| **L0 — 单用户** | CLI 手动搜索 | 0.4 | CPU embedding | ✅ 当前硬件够 |
| **L1 — 低并发** | 1 HTTP + 1 MCP 同时 | 0.8 | CPU 排队 | ✅ 当前硬件勉强 |
| **L2 — 中等并发** | 3-5 个 Agent 同时查询 | 2.0-3.0 | ❌ CPU 瓶颈 | **需要 GPU 或小模型** |
| **L3 — 高并发** | 10+ Agent + Web UI | 5.0+ | ❌ 完全不够 | **GPU + 连接池** |

**核心问题：** embedding 是同步阻塞的。当前 `bge-m3` 在 CPU 上 2.3s/条，5 个并发请求就是 11.5s 的排队延迟。

### 知识量级 → embedding 预热

| 知识总量 | Embedding 总时间 (CPU) | Embedding 总时间 (GPU T4) |
|:---------|:---------------------:|:------------------------:|
| 5,000 chunks | ~30 min | ~4 min |
| 50,000 chunks | ~5 h | ~40 min |
| 500,000 chunks | ~52 h | ~7 h |
| 5,000,000 chunks | ~21 days | ~3 days |

---

## 4. 建议配置矩阵

### 配置 A — 最低配（单用户，<10 万文件）

```
CPU: 4 核以上
RAM: 8 GB（ollama 2GB + pg 2GB + 系统 4GB）
磁盘: 50 GB SSD
GPU: 无
适合: 个人知识管理，CLI 使用
QPS: ~0.4
```

### 配置 B — 推荐（中等并发，<100 万文件）

```
CPU: 8 核以上
RAM: 32 GB（ollama 8GB + pg 8GB + 缓存 16GB）
磁盘: 200 GB NVMe SSD
GPU: NVIDIA T4 或 RTX 3060 12GB
适合: 团队使用，HTTP API + MCP
QPS: ~20-50（GPU embedding 0.05s/条）
```

### 配置 C — 生产（高并发，>100 万文件）

```
CPU: 16 核以上
RAM: 64 GB
磁盘: 1 TB NVMe SSD
GPU: NVIDIA A10 或 2xT4 24GB
PG: 独立部署 + pgvector 索引优化（IVFFlat → HNSW）
App: 连接池 + 异步 embedding + 请求排队
适合: 企业级，多 Agent 并发
QPS: ~100-200
```

---

## 5. 并发架构改进建议

### 当前问题

```
请求 A ─→ embedding (2.3s) ─→ search (8ms) ─→ 响应 A  
请求 B ─→ 等待 embedding 空闲 (2.3s 排队) ─→ search ─→ 响应 B
```

总耗时 = 2.3s × 并发数。5 个并发 = 11.5s。

### 改进方案

**方案一：嵌入缓存（低投入，高回报）**
```
请求 A ─→ cache hit? → 是 → 直接 search (8ms) → 响应
请求 B ─→ cache hit? → 是 → 直接 search (8ms) → 响应
```
缓存常见查询的 embedding。第一次 2.3s，之后 8ms。
命中率 80%+ 时，有效 QPS 从 0.4 → 40+。

**方案二：异步 embedding 队列**
```
POST /api/v1/search/async → 返回 task_id
                        ↓
                排队队列（FIFO 或优先级）
                        ↓
                后台 Worker 逐个 embedding
                        ↓
                GET /api/v1/search/status/{task_id} → 结果
```
将阻塞的 embedding 改为后台任务。

**方案三：连接池**
当前每个 HTTP 请求新建 DB 连接。改为 `psycopg_pool` 连接池：

```python
from psycopg_pool import ConnectionPool
pool = ConnectionPool(dsn, min_size=2, max_size=10)
```

### 建议优先级

| 优先级 | 改进 | 工作量 | 收益 |
|:------:|:-----|:------:|:----:|
| P0 | embedding 缓存 | 1 天 | 80% 请求从 2.3s→8ms |
| P0 | 连接池 | 0.5 天 | 稳定高并发连接 |
| P1 | 异步搜索端点 | 2 天 | 不阻塞客户端 |
| P2 | GPU 加速 | 3 天（+硬件） | embedding 2.3s→0.05s |

---

## 6. 存储增长预测

| 时间 | 文件数 | Chunks | DB 大小 | 磁盘占用 | 备注 |
|:----|:------|:------:|:-------:|:---------|:-----|
| 当前 | ~555 | 2,651 | ~50 MB | 109 GB 剩 | Phase 0 已有数据 |
| 1 个月后 | ~5,000 | ~25,000 | ~500 MB | 108 GB 剩 | 按 TradeOMS 增量 |
| 6 个月后 | ~50,000 | ~250,000 | ~5 GB | 104 GB 剩 | 含实体图 |
| 1 年后 | ~200,000 | ~1,000,000 | ~20 GB | 89 GB 剩 | 含知识图 |
| 3 年后 | ~1,000,000 | ~5,000,000 | ~100 GB | 9 GB 剩 ⚠️ | **需要加硬盘** |

以当前 109 GB 可用空间，**可支撑约 3 年的知识增长**。

---

## 7. 结论

**当前硬件（62GB RAM / 32 核 / 197GB SSD）对于：**
- 👤 **单用户 CLI 使用** → ✅ 完全够，瓶颈不在硬件
- 🤖 **1-2 个 Agent 并发** → ✅ 可接受（偶尔 2-3s 等待）
- 👥 **3+ 并发 + Web UI** → ❌ 需要加 GPU 或改架构

**最值得先做的改进（按 ROI 排序）：**
1. **Embedding 缓存** — 最简单的高回报改进（1 天实现，80% 请求 8ms）
2. **连接池** — 稳定性改进（0.5 天）
3. **异步搜索端点** — Issue #9 #8（2 天）
4. **前两点后续按 derekhubproj 流程走 Issue→PR→Review→Merge**
