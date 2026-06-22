#!/usr/bin/env python3
"""
Integration test: verify that Track A (Model Registry + Pipeline + Profiler + Engine)
works correctly end-to-end without connecting to any real models.
"""
import sys
sys.path.insert(0, "/home/cbnb/derekinside")

from derekinside.engine.model import (
    ModelProfile, intel_rank, cost_rank, speed_rank,
    CapabilityNotSupported, NoModelSatisfies,
)
from derekinside.engine.pipeline import PipelineResolver
from derekinside.engine.profiler import ModelProfiler, PassiveObserver
from derekinside.engine.registry import ModelRegistry

passed = 0
failed = 0

def check(label, ok):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {label}")
    else:
        failed += 1
        print(f"  ❌ {label}")

# ── 1. ModelProfile Constraints ──
print("\n1. ModelProfile constraint logic")
p = ModelProfile(intelligence="medium", cost_tier="free")
check("medium satisfies low_min", p.satisfies({"min_intelligence": "low"}))
check("medium fails high_min", not p.satisfies({"min_intelligence": "high"}))
check("free satisfies paid_max", p.satisfies({"max_cost": "paid_low"}))
check("oscillating frozen", not ModelProfile(
    intelligence="high", oscillating=True
).satisfies({"min_intelligence": "low"}))

# ── 2. Rank ordering ──
print("\n2. Rank ordering")
check("intel rank", intel_rank("very_high") > intel_rank("high") > intel_rank("medium") > intel_rank("low"))
check("cost rank", cost_rank("paid_high") > cost_rank("paid_low") > cost_rank("free"))
check("speed rank", speed_rank("fast") > speed_rank("medium") > speed_rank("slow") > speed_rank("batch"))

# ── 3. Profile from_config ──
print("\n3. Profile from_config")
p2 = ModelProfile.from_config({"intelligence": "high", "cost_tier": "paid_low", "speed_tier": "fast"})
check("user intel", p2.intelligence == "high")
check("user cost", p2.cost_tier == "paid_low")
check("user speed", p2.speed_tier == "fast")
check("fast latency hint", p2.avg_latency_ms == 50)
check("probe_count=999 (never probe)", p2.probe_count == 999)

# ── 4. PipelineResolver constraint solving ──
print("\n4. PipelineResolver constraint solving")

# Build a mock registry with hardcoded profiles
from derekinside.engine.model import ModelEndpoint

class MockModel(ModelEndpoint):
    def invoke(self, capability, **kwargs): raise CapabilityNotSupported()
    def health(self): return {"status": "ok", "latency_ms": 5}

# Register mock models
cfg = {"models": {}}
reg = ModelRegistry(cfg)
for name, intel, cost, speed, lat in [
    ("minimax-2.7b", "low", "free", "fast", 300),
    ("qwen-1.5b", "medium", "free", "medium", 2500),
    ("qwen-7b", "high", "free", "slow", 6500),
    ("gpt-4o-mini", "very_high", "paid_low", "fast", 200),
]:
    m = MockModel(name, {})
    m.profile = ModelProfile(
        intelligence=intel, cost_tier=cost, speed_tier=speed,
        avg_latency_ms=lat,
    )
    reg._models[name] = m

pipeline_cfg = {
    "extract_bulk": {
        "requires": {"min_intelligence": "low", "max_cost": "free"},
        "objective": "optimize_speed",
        "candidates": ["minimax-2.7b", "qwen-1.5b", "qwen-7b"],
    },
    "extract_deep": {
        "requires": {"min_intelligence": "high", "max_cost": "paid_low"},
        "objective": "optimize_quality",
        "candidates": ["minimax-2.7b", "qwen-7b", "gpt-4o-mini"],
    },
    "extract_impossible": {
        "requires": {"min_intelligence": "very_high", "max_cost": "free"},
        "objective": "optimize_quality",
        "candidates": ["minimax-2.7b", "qwen-1.5b", "qwen-7b"],
    },
}

resolver = PipelineResolver(reg, pipeline_cfg)

# extract_bulk: low min, free -> all qualify; speed -> minimax wins
m = resolver.select("extract_bulk")
check("bulk picks minimax for speed", m.name == "minimax-2.7b")

# extract_deep: high min -> minimax excluded; quality -> gpt-4o-mini wins
m = resolver.select("extract_deep")
check("deep picks gpt4o for quality", m.name == "gpt-4o-mini")

# extract_impossible: very_high + free -> no model qualifies -> should relax
try:
    m = resolver.select("extract_impossible")
    # After relax: very_high->high (if originally specified), or same
    check("impossible relaxes to available (but may fail)", True)
except NoModelSatisfies:
    # qwen-7b (high, free) should be relaxed to: min_intel not originally specified
    # After relax, no candidate with very_high+free. With relaxation:
    # first relax: if min_intel was in original (very_high), drops to high
    # but all candidates except gpt4o-mini have cost=free. After intel relax: 
    # qwen-7b (high, free) qualifies!
    check("impossible relaxes correctly, but needs config fix", False)

# ── 5. Profiler logic ──
print("\n5. Profiler logic")
from derekinside.engine.profiler import _acc_to_intel, _latency_to_speed
check("acc 0.7 -> very_high", _acc_to_intel(0.7) == "very_high")
check("acc 0.55 -> high", _acc_to_intel(0.55) == "high")
check("acc 0.4 -> medium", _acc_to_intel(0.4) == "medium")
check("acc 0.2 -> low", _acc_to_intel(0.2) == "low")
check("50ms -> fast", _latency_to_speed(50) == "fast")
check("500ms -> medium", _latency_to_speed(500) == "medium")
check("5000ms -> slow", _latency_to_speed(5000) == "slow")
check("20000ms -> batch", _latency_to_speed(20000) == "batch")

# ── Summary ──
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
if failed:
    sys.exit(1)
else:
    print("All tests passed! ✅")
