# RFC-0001: Agent Fact Logging & Cross-Agent Shared Memory

> **Status**: Draft — 暂存待议
> **Date**: 2026-06-26
> **Author**: aITMS01
> **Inspired by**: Graphiti (getzep/graphiti, ⭐27k) + openclaw-graphiti-memory (clawdbrunner)

## Motivation

Derek 在调研 getzep/graphiti 后发现：
- 现有 Agent 之间没有结构化共享内存。每次子 Agent（sessions_spawn）看到的都是干净 session，前一个 Agent 的知识丢失
- derekinside 已有 KnowledgeGraph（5,485 entities, 10,817 links），但它是静态索引，不是动态的时序事实存储
- 不用引入 Neo4j / OpenAI API key，在 derekinside 现有基础设施上就能实现"lite 版共享内存"

## Design

### 新增数据模型

```python
@dataclass
class Fact:
    id: int
    fact_text: str          # "NCC API 401: OAuth token 7天过期未自动续期"
    valid_from: datetime    # 事实生效时间 (一般是 NOW)
    valid_to: datetime | None  # None = 当前仍真
    agent_id: str           # "aITMS01" / "tradeoms_worker"
    episode_type: str       # "decision" | "discovery" | "preference" | "fix"
    metadata: dict          # {"original_query": "...", "source_issue": "ISS-xxx"}
```

### 新增表

- `facts` (id, fact_text, valid_from, valid_to, agent_id, episode_type, metadata, created_at)
- `fact_entities` (fact_id, entity_id) — 事实 ↔ 实体多对多链接

### 新增 API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/facts` | 记录事实（自动调用 EntityExtractor 提取实体并链接） |
| POST | `/api/v1/facts/search` | 搜索事实（带时序过滤：valid_at / valid_between） |

### Agent 使用模式

```bash
# 搜
~/.openclaw/scripts/fact-search.sh "NCC API 认证问题"

# 记（关键发现/决策后）
~/.openclaw/scripts/fact-log.sh agent_id "NCC API 401 根因 = OAuth token 7天过期未自动续期" --type discovery

# 开工前上下文（组装所有相关 facts）
~/.openclaw/scripts/fact-context.sh "修复 NCC API" agent_id
```

### Agent 指令模式

在每个 AGENTS.md 中加两行：

```
search before ask: 开工前先 `fact-search` 查一次相关事实
log after decision: 关键发现/决策后 `fact-log` 记一次
```

## Effort Estimate

| Component | Lines | Difficulty |
|-----------|-------|------------|
| `storage/pgvector.py`: `facts` + `fact_entities` schema | ~25 | Low |
| `storage/graph.py`: Fact dataclass + CRUD methods | ~60 | Medium |
| `indexer/entity.py`: `extract_fact_entities()` | ~15 | Low |
| `bridge/http.py`: `POST /api/v1/facts` + `/facts/search` | ~60 | Medium |
| Shell wrappers: `fact-log.sh`, `fact-search.sh`, `fact-context.sh` | ~55 | Low |
| Agent instructions update | ~20 | Low |
| **Total** | **~235** | **Low-Medium** |

## Risks

1. **Temporal invalidation accuracy** — Auto-invalidating old facts when new ones are logged depends on LLM accuracy to detect conflicting facts. Proposal: start with manual invalidation only.
2. **Entity linking recall** — Unstructured fact text may extract entity names that don't match existing KnowledgeGraph names. Mitigation: fuzzy matching.
3. **No additional infrastructure needed** — Reuses existing PostgreSQL + pgvector.

## References

- [getzep/graphiti](https://github.com/getzep/graphiti) — temporal knowledge graph engine
- [clawdbrunner/openclaw-graphiti-memory](https://github.com/clawdbrunner/openclaw-graphiti-memory) — OpenClaw wrapper for Graphiti
- [derekinside HTTP bridge](../../src/derekinside/bridge/http.py) — existing API
- [derekinside KnowledgeGraph](../../src/derekinside/storage/graph.py) — existing entity/relation store
- [derekinside EntityExtractor](../../src/derekinside/indexer/entity.py) — existing entity extraction
