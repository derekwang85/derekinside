#!/usr/bin/env python3
"""
📦 Phase 0 — gbrain → derekinside 数据迁移

Usage:
  python3 scripts/migrate-from-gbrain.py            # 全量迁移
  python3 scripts/migrate-from-gbrain.py --dry-run  # 预览

What it does:
  1. 在 gbrain 的 PostgreSQL 实例上创建 derekinside 数据库
  2. 创建 derekinside schema（含 wing/room 分层）
  3. 从 gbrain 的 pages + content_chunks 读取数据
  4. 按 source_path 映射到 wing/room
  5. 写入 derekinside 数据库（含 embedding 向量）
"""

import argparse
import os
import re

GBRAIN_DSN = "dbname=gbrain host=localhost port=5434 user=postgres"
TARGET_DSN = "dbname=derekinside host=localhost port=5434 user=postgres"


def wing_room_from_path(source_path: str) -> tuple[str, str]:
    """从 source_path 推断 wing（项目域）和 room（主题域）"""
    if not source_path:
        return ("uncategorized", "uncategorized")

    # Agent conversation logs
    if re.match(r"^agent-[a-f0-9]+", source_path):
        return ("agent", "conversations")

    # Root-level workspace files
    if "/" not in source_path:
        return ("openclaw", "workspace")

    parts = source_path.split("/")
    top = parts[0]

    # skills/{skill_name}/...
    if top == "skills" and len(parts) >= 2:
        return ("skills", parts[1])

    # OpenClaw workspace
    if top in ("memory", "tasks", "docs", "audits", "guides", "archives"):
        return ("openclaw", top)
    if top in (
        "02-architecture",
        "04-adr",
        "06-runbook",
        "11-product-manual",
        "12-development-guide",
        "13-operations",
        "14-quality-gates",
        "15-api-contracts",
        "16-skills",
    ):
        return ("openclaw", top[:2])  # use prefix

    knowledge_map = {
        "knowledge": "gbrain",
        "learning": "gbrain",
        "guides": "guides",
    }
    if top in knowledge_map:
        return (knowledge_map[top], top)

    # Fallback: top-level dir is wing, second-level is room
    if len(parts) >= 2:
        return (top, parts[1])
    return ("uncategorized", top)


def create_schema(cur):
    """创建 derekinside 数据库 schema"""
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS wings (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            id SERIAL PRIMARY KEY,
            wing_id INTEGER REFERENCES wings(id),
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(wing_id, name)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id SERIAL PRIMARY KEY,
            room_id INTEGER REFERENCES rooms(id),
            gbrain_page_id INTEGER,
            slug TEXT,
            title TEXT,
            source_path TEXT,
            source_kind TEXT,
            page_kind TEXT,
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ,
            metadata JSONB DEFAULT '{}'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            page_id INTEGER REFERENCES pages(id),
            gbrain_chunk_id INTEGER,
            chunk_index INTEGER,
            chunk_text TEXT,
            token_count INTEGER,
            embedding vector(1024),
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id)
    """)


def run_migration(dry_run: bool = False):
    print(f"{'─' * 50}")
    print("📦 gbrain → derekinside Phase 0")
    print(f"   Mode: {'DRY RUN' if dry_run else 'LIVE 迁移'}")
    print(f"{'─' * 50}")

    import psycopg

    src_conn = psycopg.connect(
        GBRAIN_DSN, password=os.environ.get("PGPASSWORD", "postgres")
    )
    tgt_conn = psycopg.connect(
        TARGET_DSN, password=os.environ.get("PGPASSWORD", "postgres")
    )
    src = src_conn.cursor()
    tgt = tgt_conn.cursor()

    # 1. 创建 schema
    if not dry_run:
        create_schema(tgt)
        tgt_conn.commit()
        print("\n✅ Schema 创建完成")
    else:
        print("\n📋 Schema: 将创建 wings/rooms/pages/chunks 表")

    # 2. 读取 gbrain 数据
    src.execute(
        "SELECT id, slug, title, source_path, source_kind, page_kind, created_at, updated_at FROM pages ORDER BY id"
    )
    rows = src.fetchall()
    print(f"\n📄 gbrain: {len(rows)} pages")

    # 统计 wing/room 分布
    wing_count = {}
    for row in rows:
        source_path = row[3]
        wing, room = wing_room_from_path(source_path)
        if wing not in wing_count:
            wing_count[wing] = {}
        wing_count[wing][room] = wing_count[wing].get(room, 0) + 1

    print("\n🏛️  Wing/Room 分布:")
    for wing, rooms in sorted(wing_count.items()):
        total = sum(rooms.values())
        print(f"  {wing}/ ({total} pages):")
        for room, count in sorted(rooms.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {room}: {count}")
        if len(rooms) > 5:
            print(f"    ... +{len(rooms)-5} more rooms")

    # 3. 写入目标数据库
    if dry_run:
        print("\n📋 [DRY-RUN] 迁移预览:")
        print(
            f"   创建 {len(set((w,r) for _,_,_,p,_,_,_,_ in rows for w,r in [wing_room_from_path(p)]))} 个 wing/room"
        )
        print(f"   复制 {len(rows)} 个 pages")
        src_conn.close()
        tgt_conn.close()
        return

    wing_cache = {}
    room_cache = {}
    pages_migrated = 0
    chunks_migrated = 0

    for row in rows:
        (
            page_id,
            slug,
            title,
            source_path,
            source_kind,
            page_kind,
            created_at,
            updated_at,
        ) = row
        wing_name, room_name = wing_room_from_path(source_path)

        # Get or create wing
        if wing_name not in wing_cache:
            tgt.execute(
                "INSERT INTO wings (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
                (wing_name,),
            )
            r = tgt.fetchone()
            if r:
                wing_cache[wing_name] = r[0]
            else:
                tgt.execute("SELECT id FROM wings WHERE name = %s", (wing_name,))
                wing_cache[wing_name] = tgt.fetchone()[0]

        # Get or create room
        room_key = (wing_cache[wing_name], room_name)
        if room_key not in room_cache:
            tgt.execute(
                "INSERT INTO rooms (wing_id, name) VALUES (%s, %s) ON CONFLICT (wing_id, name) DO NOTHING RETURNING id",
                (wing_cache[wing_name], room_name),
            )
            r = tgt.fetchone()
            if r:
                room_cache[room_key] = r[0]
            else:
                tgt.execute(
                    "SELECT id FROM rooms WHERE wing_id = %s AND name = %s", room_key
                )
                room_cache[room_key] = tgt.fetchone()[0]

        # Copy page
        tgt.execute(
            """INSERT INTO pages (room_id, gbrain_page_id, slug, title, source_path, source_kind, page_kind, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT DO NOTHING""",
            (
                room_cache[room_key],
                page_id,
                slug,
                title,
                source_path,
                source_kind,
                page_kind,
                created_at,
                updated_at,
            ),
        )
        pages_migrated += 1

        # Copy chunks for this page
        src.execute(
            "SELECT id, chunk_index, chunk_text, token_count, embedding FROM content_chunks WHERE page_id = %s ORDER BY chunk_index",
            (page_id,),
        )
        for chunk_row in src.fetchall():
            chunk_id, chunk_index, chunk_text, token_count, embedding = chunk_row
            tgt.execute(
                """INSERT INTO chunks (page_id, gbrain_chunk_id, chunk_index, chunk_text, token_count, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    pages_migrated,
                    chunk_id,
                    chunk_index,
                    chunk_text,
                    token_count,
                    embedding,
                ),
            )
            chunks_migrated += 1

        if pages_migrated % 100 == 0:
            tgt_conn.commit()
            print(
                f"  ⏳ {pages_migrated}/{len(rows)} pages, {chunks_migrated} chunks...",
                end="\r",
            )

    tgt_conn.commit()
    print("\n\n✅ 迁移完成:")
    print(
        f"   {pages_migrated} pages → {len(wing_cache)} wings / {len(room_cache)} rooms"
    )
    print(f"   {chunks_migrated} chunks with embeddings")

    # 4. 验证
    tgt.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = tgt.fetchone()[0]
    tgt.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
    embed_count = tgt.fetchone()[0]
    print(
        f"\n📊 验证: {chunk_count} chunks, {embed_count} 带嵌入 ({embed_count*100//chunk_count if chunk_count else 0}%)"
    )

    src_conn.close()
    tgt_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gbrain → derekinside migration")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run)
