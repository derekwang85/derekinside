#!/usr/bin/env python3
"""
derekinside Phase 0 — gbrain → derekinside 数据迁移

从 gbrain (PostgreSQL:5434/gbrain) 读取现有 pages + chunks + embeddings，
迁移到 derekinside (PostgreSQL:5435/derekinside)。

迁移过程：
  1. 读取 gbrain 的 pages 表
  2. 读取 gbrain 的 chunks 表 (含 embeddings)
  3. 写入 derekinside 的 pages + chunks + embeddings
  4. 输出迁移报告

用法:
  python3 scripts/migrate-from-gbrain.py           # 全量迁移
  python3 scripts/migrate-from-gbrain.py --dry-run  # 预览不写

依赖:
  pip install psycopg[binary] numpy
"""

import argparse
import sys

# TODO: Phase 0 implementation
# - Connect to gbrain DB (localhost:5434/gbrain)
# - Connect to derekinside DB (localhost:5435/derekinside)
# - Read pages table → map to wing/room
# - Read chunks table → map to drawer
# - Copy embeddings as-is
# - Write migration report


def main():
    parser = argparse.ArgumentParser(description="Migrate gbrain → derekinside")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument(
        "--gbrain-db", default="postgres://postgres:***@localhost:5434/gbrain"
    )
    parser.add_argument(
        "--target-db", default="postgres://postgres:***@localhost:5435/derekinside"
    )
    args = parser.parse_args()

    print("📦 gbrain → derekinside 迁移")
    print(f"   Source: {args.gbrain_db.split('@')[1]}")
    print(f"   Target: {args.target_db.split('@')[1]}")
    print(f"   Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()
    print("   ➖ gbrain: 15,812 chunks across 265 pages")
    print("   ➕ derekinside: adding wing/room metadata")
    print()
    print("   ⏳ Phase 0 — 待实现")
    print("   实现后: python3 scripts/migrate-from-gbrain.py --dry-run")
    sys.exit(0)


if __name__ == "__main__":
    main()
