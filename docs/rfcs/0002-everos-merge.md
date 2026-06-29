# RFC-0002: DereInside × EverOS 合并方案

> **Status**: Draft — 暂存待议
> **Date**: 2026-06-26
> **Author**: aITMS01
> **Related**: RFC-0001 (Fact Logging), EverOS (EverMind-AI, ⭐9k)

## 背景

调研 getzep/graphiti 后产生 RFC-0001（在 derekinside 上加事实日志）。
调研 EverMind-AI/EverOS 后发现它已完整实现 RFC-0001 设想的功能，且远超之。
Derek 提出：能否把 EverOS 的优势吸收进 DereInside，打造一个超级知识库。

## 规模对比

| 项目 | 代码量 | 存储 | 索引 |
|------|--------|------|------|
| derekinside | ~8,273 行 Python | PostgreSQL + pgvector | 向量 + BM25 + 知识图谱 |
| EverOS | ~33,215 行 Python (memory 模块 11,481) | Markdown 文件 + SQLite + LanceDB | 向量 + BM25 + 标量过滤 |
| 差距 | 4x | 不同技术栈 | 不同索引方案 |

EverOS 单独 memory 模块就比整个 derekinside 大。

## 两种合并路径

### 路径 A：依赖式合并（推荐）

```
derekinside (port 18890)
├── /api/v1/memory/*     → 等效调用 everos (port 8000)
├── /api/v1/knowledge/*  → 等效调用 everos (port 8000)
├── /api/v1/search       → derekinside 原生（代码搜索）
├── /api/v1/graph/*      → derekinside 原生（知识图谱）
├── /api/v1/mine         → derekinside 原生（文件摄入）
└── /api/v1/facts        → derekinside 原生（RFC-0001 或弃用为 everos 别名）
```

**实现**：pip install everos → 在 derekinside HTTP bridge 中 mount everos 的路由。

```python
from everos.entrypoints.api.routes.memorize import router as everos_memory
app.mount("/api/v1/memory", everos_memory)
```

**好处**：
- 0 代码重写，33k 行直接用
- 持续受益于上游更新（Apache-2.0）
- 1-2 天完成主体集成

**代价**：
- 多一个 Python 依赖 + 多一个服务进程
- 两个索引系统：derekinside (pgvector) + everos (LanceDB)
- 运维复杂度 +1

---

### 路径 B：吸收式合并

把 EverOS 的设计思想移植到 derekinside 的 PostgreSQL + pgvector 栈上。

#### 功能清单与工作量

| # | 功能 | 对标 EverOS | 工作量 | 依赖 |
|---|------|------------|--------|------|
| 1 | Markdown 文件写入 + Cascade Watcher | 编辑 .md → 自动同步索引 | ~3 天 | 新增 `derekinside` wing 管理 + inotify watcher |
| 2 | 正交检索（5 维 slice） | user×agent×app×project×session | ~2 天 | storage 层加过滤维度 + search 接口扩展 |
| 3 | Agent Case + Skill 表 | 结构化 Agent 记忆 | ~2 天 | 新表 + 提取 pipeline |
| 4 | Knowledge Wiki | 文档 CRUD + 分类 + 主题树 | ~3 天 | 新表 + CRUD API |
| 5 | Boundary Detection | 对话边界自动检测 | ~2 天 | LLM + 时序窗口 |
| 6 | Reflection | 离线聚类 + 再提取 + skill 蒸馏 | ~5 天 | 后台任务调度 + LLM pipeline |
| 7 | Memory Root 路径管理器 | `~/.derekinside/` 目录布局 | ~1 天 | config 层 |
| **合计** | | | **~18 天** | |

**好处**：
- 单一系统、统一技术栈（PostgreSQL）
- 完全控制，不依赖外部项目
- derekinside 的代码知识图谱 + EverOS 的记忆管理在同一个 API 下

**代价**：
- 18 天开发，其中 Reflection（5 天）难度最高、最易出 bug
- 同步上游 EverOS 的新功能需要自己重新实现

---

## 两条路径对比

| 维度 | 路径 A（依赖） | 路径 B（吸收） |
|------|---------------|---------------|
| 上线时间 | 1-2 天 | 18 天 |
| 代码量 | ±0（复用 33k） | +8k~12k（翻倍） |
| 维护负担 | 追踪上游更新 | 完全自维护 |
| 技术栈 | pgvector + LanceDB 双索引 | PostgreSQL 单栈 |
| 进程数 | 2 个（derekinside + everos） | 1 个（derekinside） |
| Reflection | 现成 | 自研 |
| 代码知识图谱 | derekinside 原生 | derekinside 原生 |

## 建议路线

> **Phase 1 → 路径 A（1-2 天）**：拿到 90% 的价值
> **Phase 2（可选）**：逐步按需吸收核心子模块到 derekinside

Phase 1 完成后就能：
```
# 同时拥有代码搜索和 Agent 记忆
curl localhost:18890/api/v1/search?q=NCC认证        # derekinside（代码）
curl localhost:8000/api/v1/memory/search             # everos（记忆）
```

## References

- [EverOS 源码分析 — 架构全景 & 数据模型](everos-analysis.md)
- [RFC-0001: Agent Fact Logging & Cross-Agent Shared Memory](0001-fact-logging.md)
- [derekinside HTTP bridge](../../src/derekinside/bridge/http.py)
