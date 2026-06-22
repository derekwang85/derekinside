#!/bin/bash
# 全量 5-mode 实体提取链式调度
# 按耗时从少到多排列，自动串行执行
# 每完成一个保存结果到 /tmp/kg-full-{mode}.json 并记录日志

set -e
cd /home/cbnb/derekinside

MODES=("regex" "1.5b" "7b" "hybrid-7b")
# hybrid-1.5b 已经跑过了

for mode in "${MODES[@]}"; do
    LOG="/tmp/kg-full-${mode}.log"
    OUT="/tmp/kg-full-${mode}.json"
    
    # 跳过已完成的
    if [ -f "$OUT" ]; then
        echo "[$(date)] ✅ ${mode} 已有结果，跳过"
        continue
    fi
    
    echo "[$(date)] 🚀 开始 ${mode}..."
    python3 -u scripts/full_batch_extract.py --mode "${mode}" > "$LOG" 2>&1
    
    if [ -f "$OUT" ]; then
        echo "[$(date)] ✅ ${mode} 完成 → $OUT"
    else
        echo "[$(date)] ❌ ${mode} 失败，日志: $LOG"
    fi
done

echo "[$(date)] 🏁 所有模式跑完！"
