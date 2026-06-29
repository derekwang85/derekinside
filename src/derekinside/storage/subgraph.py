"""
derekinside — Subgraph Query on KnowledgeGraph.

Returns a subgraph centered on an entity, up to N depth.
Output as adjacency list or JSON graph structure.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from derekinside.storage.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class SubgraphNode:
    id: int
    name: str
    entity_type: str
    description: str = ""
    depth: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class SubgraphEdge:
    source_id: int
    target_id: int
    relation_type: str
    weight: float = 1.0


@dataclass
class Subgraph:
    center: SubgraphNode
    nodes: list[SubgraphNode] = field(default_factory=list)
    edges: list[SubgraphEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "center": {
                "id": self.center.id,
                "name": self.center.name,
                "type": self.center.entity_type,
            },
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.entity_type,
                    "depth": n.depth,
                    "description": n.description,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.relation_type,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }

    def to_ascii(self) -> str:
        """Render subgraph as ASCII tree."""
        lines = [f"🔍 {self.center.name} ({self.center.entity_type})"]
        if self.center.description:
            lines.append(f"   {self.center.description}")

        # Group nodes by depth
        by_depth: dict[int, list[SubgraphNode]] = defaultdict(list)
        for n in self.nodes:
            if n.id != self.center.id:
                by_depth[n.depth].append(n)

        for depth in sorted(by_depth.keys()):
            indent = "  " * depth
            for node in by_depth[depth]:
                edges_from = [
                    e
                    for e in self.edges
                    if e.source_id == node.id or e.target_id == node.id
                ]
                rel_strs = set()
                for e in edges_from:
                    if e.source_id == self.center.id:
                        rel_strs.add(f"→ {e.relation_type}")
                    elif e.target_id == self.center.id:
                        rel_strs.add(f"← {e.relation_type}")
                    else:
                        rel_strs.add(f"↔ {e.relation_type}")
                rel = ",".join(sorted(rel_strs)) if rel_strs else ""
                desc = f" — {node.description[:60]}" if node.description else ""
                lines.append(f"{indent}├─ {node.name} ({node.entity_type}) {rel}{desc}")
        return "\n".join(lines)


def build_subgraph(
    graph: KnowledgeGraph,
    entity_name: str,
    max_depth: int = 2,
    max_nodes: int = 50,
) -> Subgraph | None:
    """
    Build a subgraph centered on an entity.

    Uses BFS to traverse relations up to max_depth.
    Returns None if entity not found.
    """
    entity = graph.get_entity_by_name(entity_name)
    if not entity:
        return None

    visited_entities: set[int] = {entity.id}
    visited_edges: set[tuple[int, int, str]] = set()
    nodes: dict[int, SubgraphNode] = {}
    edges: list[SubgraphEdge] = []

    # Get description
    desc = (entity.metadata or {}).get("description", "") if entity.metadata else ""
    center = SubgraphNode(
        id=entity.id,
        name=entity.name,
        entity_type=entity.entity_type,
        description=desc,
        depth=0,
    )
    nodes[entity.id] = center

    queue: deque[tuple[int, int]] = deque([(entity.id, 0)])

    while queue and len(nodes) < max_nodes:
        current_id, depth = queue.popleft()
        if depth >= max_depth:
            continue

        relations = graph.get_relations_for_entity(current_id)
        for rel in relations:
            edge_key = (rel.source_entity_id, rel.target_entity_id, rel.relation_type)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            neighbor_id = (
                rel.target_entity_id
                if rel.source_entity_id == current_id
                else rel.source_entity_id
            )

            if neighbor_id not in nodes and len(nodes) < max_nodes:
                neighbor = graph.get_entity(neighbor_id)
                if neighbor:
                    ndesc = (
                        (neighbor.metadata or {}).get("description", "")
                        if neighbor.metadata
                        else ""
                    )
                    node = SubgraphNode(
                        id=neighbor.id,
                        name=neighbor.name,
                        entity_type=neighbor.entity_type,
                        description=ndesc,
                        depth=depth + 1,
                    )
                    nodes[neighbor.id] = node
                    visited_entities.add(neighbor.id)

                    if neighbor.id not in [e[0] for e in queue]:
                        queue.append((neighbor.id, depth + 1))

            if neighbor_id in nodes:
                edges.append(
                    SubgraphEdge(
                        source_id=rel.source_entity_id,
                        target_id=rel.target_entity_id,
                        relation_type=rel.relation_type,
                        weight=rel.weight,
                    )
                )

    # Determine depth for nodes not visited in BFS
    return Subgraph(
        center=center,
        nodes=list(nodes.values()),
        edges=edges,
    )
