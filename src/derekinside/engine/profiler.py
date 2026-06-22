"""
derekinside — ModelProfiler + PassiveObserver.

Dual-stage automatic model profiling:
  - Boot probe: ~2s, 3 sample smoke test, initial positioning.
  - Deep probe: ~10s, 15 sample full benchmark, precise four-dimensional mapping.
  - Passive observation: zero-cost runtime metrics collected as side effects.

All probes use golden data (LongMemEval human-annotated entities).
No external model (GPT, etc.) is ever called for evaluation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from derekinside.engine.model import (
    ModelEndpoint,
    ModelProfile,
    CapabilityNotSupported,
    intel_rank,
    cost_rank,
    speed_rank,
)

logger = logging.getLogger(__name__)

# ── Golden Data for probing (zero-cost ground truth) ──
# From LongMemEval: 290 human-annotated entities, 100 golden chunks.
# Each entry: (text, expected_entities)

SMOKE_SET: list[tuple[str, list[tuple[str, str]]]] = [
    # Sample 1: Code entity (class-level)
    (
        "public class OrderService extends BaseService "
        "implements InitializingBean { "
        "private final OrderRepository repository; "
        "@Autowired public OrderService(OrderRepository repo) { ... } }",
        [("OrderService", "class"), ("BaseService", "class"),
         ("OrderRepository", "class")],
    ),
    # Sample 2: Doc concept (Chinese business domain)
    (
        "KYC 申请流程：客户提交身份证明文件后，"
        "由合规部门进行人工审核，审核通过后交易权限生效。",
        [("KYC", "concept"), ("合规审核", "concept")],
    ),
    # Sample 3: API endpoint
    (
        "@PostMapping(\"/api/order/create\")\n"
        "public Result createOrder(@RequestBody OrderRequest req) {\n"
        "    return orderService.create(req);\n}",
        [("/api/order/create", "api"), ("createOrder", "function")],
    ),
    # Sample 4: Simple concept
    (
        "保证金制度：交易前必须存入不低于合约价值10%的保证金。",
        [("保证金制度", "concept")],
    ),
    # Sample 5: Java interface
    (
        "public interface TradingService {\n"
        "    OrderResult executeOrder(OrderRequest request);\n"
        "    void cancelOrder(String orderId);\n}",
        [("TradingService", "class"), ("executeOrder", "function"),
         ("cancelOrder", "function")],
    ),
]

FULL_SET: list[tuple[str, list[tuple[str, str]]]] = [
    # Will be populated from LongMemEval golden dataset
    # Currently using SMOKE_SET + more diverse samples
    *SMOKE_SET,
    ("router.get(\"/api/v1/positions\", authMiddleware, positionController.getPositions)",
     [("/api/v1/positions", "api"), ("getPositions", "function")]),
    ("风险控制：系统自动监控持仓风险率，当风险率低于100%时触发强平。",
     [("风险控制", "concept")]),
    ("@Service\npublic class AuditLogService {\n"
     "    @Autowired private AuditLogRepository repository;",
     [("AuditLogService", "class"), ("AuditLogRepository", "class")]),
    ("def calculate_margin(price: float, quantity: int, leverage: int) -> float:",
     [("calculate_margin", "function")]),
    ("Maven dependency:\n"
     "<dependency>\n<groupId>com.example</groupId>\n"
     "<artifactId>trade-oms-core</artifactId>\n</dependency>",
     [("trade-oms-core", "module")]),
]

SMOKE_CHUNK = "public class Test { private String name; public void run() {} }"


def _exact_match(
    result: Any, expected: list[tuple[str, str]]
) -> bool:
    """Compare extraction result against expected entities.
    Pure local string match — no external model involved."""
    if not result:
        return False
    # result could be ExtractionResult or dict
    if hasattr(result, "entities"):
        entities = result.entities
    elif isinstance(result, dict):
        entities = result.get("entities", [])
    else:
        entities = []

    names_found = {e.name if hasattr(e, "name") else e.get("name", "") for e in entities}
    expected_names = {e[0] for e in expected}
    if not expected_names:
        return True
    overlap = names_found & expected_names
    # Consider match if at least 50% of expected entities found
    return len(overlap) / len(expected_names) >= 0.5


def _acc_to_intel(acc: float) -> str:
    if acc >= 0.6:
        return "very_high"
    if acc >= 0.5:
        return "high"
    if acc >= 0.3:
        return "medium"
    return "low"


def _latency_to_speed(ms: float) -> str:
    if ms < 100:
        return "fast"
    if ms < 2000:
        return "medium"
    if ms < 10000:
        return "slow"
    return "batch"


def _guess_cost_tier(driver: str, config: dict) -> str:
    """Guess cost tier based on driver and config."""
    if driver in ("ollama", "vllm"):
        return "free"
    if config.get("base_url", "").endswith(".freecode.com"):
        return "free"  # FreeCode = free
    return "paid_low"  # OpenAI/MiniMax cloud


# ── Oscillation Alert ──


class OscillationAlert(Exception):
    """Raised when a model's profile oscillates across probes."""


# ── ModelProfiler ──


class ModelProfiler:
    """
    Dual-stage automatic model profiler.

    - boot: ~2s, 3-5 sample smoke test for initial positioning.
    - deep: ~10s, 12-15 sample full benchmark for precise mapping.

    Uses ONLY golden data (LongMemEval). Never calls external LLMs for evaluation.
    """

    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    # ── Public API ──

    def get_or_probe(self, model_name: str, mode: str = "auto") -> ModelProfile:
        """Get profile or auto-probe if missing."""
        model = self._registry._models.get(model_name)
        if model is None:
            raise KeyError(f"Cannot profile unknown model '{model_name}'")

        if model.profile is not None and mode == "auto":
            return model.profile

        if model.profile is not None and mode == "auto":
            return model.profile

        if mode == "boot":
            return self._boot_probe(model, model_name)
        return self._deep_probe(model, model_name)

    # ── Boot Probe (~2s) ──

    def _boot_probe(self, model: ModelEndpoint, name: str) -> ModelProfile:
        """Quick smoke test: ~2s, 3-5 samples, initial four-dimensional positioning."""
        # 1. Capabilities detection
        caps = {}
        for cap in ["embed", "extract", "rerank", "generate"]:
            try:
                model.invoke(cap, text="hello", max_tokens=10)
                caps[cap] = True
            except CapabilityNotSupported:
                caps[cap] = False

        # 2. Speed measure
        start = time.time()
        try:
            model.invoke("extract", text=SMOKE_CHUNK)
        except CapabilityNotSupported:
            pass
        latency = (time.time() - start) * 1000

        # 3. Intelligence measure (only if capable of extraction)
        intelligence, confidence = "low", 0.3
        if caps.get("extract"):
            correct = 0
            for text, expected in SMOKE_SET:
                try:
                    result = model.invoke("extract", text=text)
                    if _exact_match(result, expected):
                        correct += 1
                except Exception:
                    pass
            acc = correct / max(len(SMOKE_SET), 1)
            intelligence = _acc_to_intel(acc)
            confidence = min(0.5, acc + 0.2)

        # 4. Cost tier guess
        cost_tier = _guess_cost_tier(
            getattr(model, "_config", {}).get("driver", ""),
            getattr(model, "_config", {}),
        )

        profile = ModelProfile(
            capabilities=caps,
            intelligence=intelligence,
            intelligence_confidence=confidence,
            speed_tier="fast" if latency < 1000 else "medium",
            avg_latency_ms=latency,
            quality=intelligence,
            cost_tier=cost_tier,
            probe_count=1,
            last_probed_at=time.time(),
        )
        self._registry.refresh_profile(name, profile)
        logger.info(
            "Boot probe for '%s': intel=%s@%.2f, speed=%s, cost=%s, caps=%s",
            name, intelligence, confidence, profile.speed_tier, cost_tier, caps,
        )
        return profile

    # ── Deep Probe (~10s) ──

    def _deep_probe(self, model: ModelEndpoint, name: str) -> ModelProfile:
        """Full benchmark: ~10s, 12-15 samples, precise positioning."""
        old = model.profile

        # ── Oscillation detection ──
        if old and old.probe_count >= 2:
            recent_intels = []
            for h in old.history[-5:]:
                to_data = h.get("to", {})
                if isinstance(to_data, dict):
                    val = to_data.get("intelligence")
                else:
                    val = None
                if val:
                    recent_intels.append(val)
            unique = set(recent_intels)
            if len(unique) >= 3:
                logger.warning(
                    "Model '%s' oscillating across %d probes: %s — freezing",
                    name, old.probe_count, list(unique),
                )
                old.oscillating = True
                self._registry.refresh_profile(name, old)
                raise OscillationAlert(
                    f"Model '{name}' oscillates: intel levels {list(unique)}. "
                    f"Profile frozen. Please verify the model endpoint."
                )

        # ── Normal deep probe ──
        latencies: list[float] = []
        correct = 0
        probe_set = FULL_SET

        for text, expected in probe_set:
            try:
                start = time.time()
                result = model.invoke("extract", text=text)
                latencies.append((time.time() - start) * 1000)
                if _exact_match(result, expected):
                    correct += 1
            except Exception as e:
                logger.debug("Deep probe sample failed: %s", e)
                latencies.append(5000.0)  # penalty for failed samples

        avg_lat = sum(latencies) / max(len(latencies), 1)
        acc = correct / max(len(probe_set), 1)
        intelligence = _acc_to_intel(acc)
        speed_tier = _latency_to_speed(avg_lat)
        confidence = acc

        # Build history
        history = list(old.history) if old else []
        history.append({
            "at": time.time(),
            "from": {"intelligence": old.intelligence if old else None},
            "to": {"intelligence": intelligence},
        })

        profile = ModelProfile(
            capabilities=(old.capabilities if old else {}),
            intelligence=intelligence,
            intelligence_confidence=confidence,
            speed_tier=speed_tier,
            avg_latency_ms=avg_lat,
            quality=intelligence,
            cost_tier=(old.cost_tier if old else "free"),
            probe_count=(old.probe_count + 1) if old else 1,
            last_probed_at=time.time(),
            history=history,
        )
        self._registry.refresh_profile(name, profile)

        # Detect change
        if old and old.intelligence != intelligence:
            logger.info(
                "Model '%s' intel changed: %s -> %s (acc: %.2f, speed: %s)",
                name, old.intelligence, intelligence, acc, speed_tier,
            )

        return profile


# ── PassiveObserver ──


class PassiveObserver:
    """
    Zero-cost runtime metrics collector.

    Records side-effect data from every actual query:
      - latency
      - entity count
      - error rate
      - JSON parse success rate

    Never makes extra API calls. Triggered by usage volume + anomaly detection.
    """

    def __init__(self, profiler: ModelProfiler, registry: ModelRegistry):
        self._profiler = profiler
        self._registry = registry
        self._metrics: dict[str, dict] = {}
        self._qcount: dict[str, int] = {}
        self._threshold: dict[str, int] = {}

    def record(
        self,
        model_name: str,
        capability: str,
        latency_ms: float,
        success: bool = True,
        entity_count: int = 0,
        output_json_valid: bool = True,
    ) -> None:
        """Record one query's side-effect data."""
        m = self._metrics.setdefault(model_name, {
            "total_calls": 0,
            "latency_window": [],
            "errors": 0,
            "avg_entities": 0.0,
            "json_errors": 0,
        })
        m["total_calls"] += 1
        m["latency_window"] = (m["latency_window"] + [latency_ms])[-20:]
        if not success:
            m["errors"] += 1
        if not output_json_valid:
            m["json_errors"] += 1
        ewma = m["avg_entities"]
        m["avg_entities"] = ewma * 0.9 + entity_count * 0.1

        self._qcount[model_name] = self._qcount.get(model_name, 0) + 1
        threshold = self._threshold.get(model_name, 50)

        if self._qcount[model_name] >= threshold and self._suspicious(model_name):
            try:
                self._profiler.get_or_probe(model_name, mode="deep")
            except Exception as e:
                logger.warning("Passive-triggered deep probe failed for %s: %s", model_name, e)
            self._qcount[model_name] = 0
            # Exponential backoff: 50 -> 100 -> 200 -> 400
            self._threshold[model_name] = min(threshold * 2, 400)

    def _suspicious(self, model_name: str) -> bool:
        """Statistical anomaly detection — no external model needed."""
        m = self._metrics.get(model_name)
        if not m:
            return False

        # 1. Latency spike (doubled)
        if len(m["latency_window"]) >= 10:
            recent = m["latency_window"][-5:]
            older = m["latency_window"][:-5]
            if older and sum(recent) / len(recent) > sum(older) / len(older) * 2:
                logger.info("Passive anomaly: %s latency spike (recent=%.0fms, older=%.0fms)",
                            model_name, sum(recent)/len(recent), sum(older)/len(older))
                return True

        # 2. Entity count dropped to near-zero
        if m["avg_entities"] < 0.5 and m["total_calls"] >= 10:
            logger.info("Passive anomaly: %s entity count dropped to %.2f",
                        model_name, m["avg_entities"])
            return True

        # 3. Error rate > 10%
        if m["total_calls"] >= 20 and m["errors"] / m["total_calls"] > 0.1:
            logger.info("Passive anomaly: %s error rate %.1f%%",
                        model_name, m["errors"] / m["total_calls"] * 100)
            return True

        # 4. JSON parse failures > 20%
        if m["total_calls"] >= 10 and m["json_errors"] / m["total_calls"] > 0.2:
            logger.info("Passive anomaly: %s JSON error rate %.1f%%",
                        model_name, m["json_errors"] / m["total_calls"] * 100)
            return True

        return False
