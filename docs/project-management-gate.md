# 项目管理门禁 — GitHub Issue 与代码状态一致性

## 问题

Phase 2 完成了但 GitHub 上没有对应的 Issue，项目管理和代码交付脱节。

## 防退化规则

### 规则 1：创建 Issue 后必须验证

```
gh issue create ...  →  成功？✅  →  继续
                      →  失败？❌  →  重试一次，再失败则报给 Derek
```

### 规则 2：发版/阶段交付前查 Issue

```bash
# 检查当前阶段的所有 open issue 是否存在
gh issue list --label phase-<N> --json number,title,state
# 预期：至少一条 open 或最近 closed
# 如果为空 → 项目管缺失，补建 issue
```

### 规则 3：WBS 与 Issue 一一映射

每个 Phase 至少对应一个 GitHub Issue。
Issue 的 checkbox 清单对应 WBS 的子任务。
完成一个 checkbox → 更新 issue 注释。

### 快速审计命令

```bash
# 显示无 issue 的 phase（检查漏洞）
cd /home/cbnb/derekinside
for phase in 0 1 2 3; do
  count=$(gh issue list --label "phase-${phase}" --state all 2>/dev/null | wc -l)
  echo "Phase ${phase}: ${count} issues"
done
```
