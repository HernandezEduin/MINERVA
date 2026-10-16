"""Build reproducible evidence subgraphs from independent KG trajectories.

A trajectory is an ordered sequence of (head, relation, tail) integer triples.
Graph-control actions must not be represented as evidence edges.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

Edge = Tuple[int, int, int]
Path = Tuple[Edge, ...]


@dataclass(frozen=True)
class EvidenceSubgraph:
    edges: FrozenSet[Edge]
    nodes: FrozenSet[int]
    trajectories: Tuple[Path, ...]
    trajectory_scores: Tuple[float, ...]

    def as_dict(self) -> dict:
        return {
            "edges": [list(edge) for edge in sorted(self.edges)],
            "nodes": sorted(self.nodes),
            "trajectories": [[list(edge) for edge in path] for path in self.trajectories],
            "trajectory_scores": list(self.trajectory_scores),
        }


def canonical_edge(edge: Sequence[int], inverse_relation_map: Optional[Dict[int, int]] = None) -> Edge:
    """Canonicalize an inverse traversal into its original forward triple.

    The map must contain inverse-relation-ID -> forward-relation-ID pairs.
    Never infer inverse IDs arithmetically or by string prefix.
    """
    if len(edge) != 3:
        raise ValueError("Evidence edges must contain exactly (head, relation, tail)")
    h, r, t = (int(value) for value in edge)
    if inverse_relation_map and r in inverse_relation_map:
        return (t, inverse_relation_map[r], h)
    return (h, r, t)


def build_evidence_subgraph(
    trajectories: Iterable[Sequence[Sequence[int]]],
    scores: Optional[Sequence[float]] = None,
    *,
    top_k: Optional[int] = None,
    control_relations: Iterable[int] = (),
    inverse_relation_map: Optional[Dict[int, int]] = None,
    deduplicate_paths: bool = True,
) -> EvidenceSubgraph:
    """Union the graph edges of complete, independently decoded trajectories.

    Trajectories are ranked by provided cumulative log-scores, if any, before
    applying top_k. Control relations are filtered, but a STOP/RESTART must
    already have been handled by the decoder's segmentation policy.
    """
    paths = list(trajectories)
    if scores is None:
        ranked = [(float(-i), i, p) for i, p in enumerate(paths)]
    else:
        if len(scores) != len(paths):
            raise ValueError("scores must align one-to-one with trajectories")
        ranked = [(float(s), i, p) for i, (s, p) in enumerate(zip(scores, paths))]
        ranked.sort(key=lambda item: (-item[0], item[1]))
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        ranked = ranked[:top_k]
    controls = set(int(r) for r in control_relations)
    selected: List[Path] = []
    selected_scores: List[float] = []
    seen = set()
    for score, _, trajectory in ranked:
        path = tuple(
            canonical_edge(edge, inverse_relation_map)
            for edge in trajectory
            if int(edge[1]) not in controls
        )
        if deduplicate_paths and path in seen:
            continue
        seen.add(path)
        selected.append(path)
        selected_scores.append(score)
    edges = frozenset(edge for path in selected for edge in path)
    nodes = frozenset(node for h, _, t in edges for node in (h, t))
    return EvidenceSubgraph(
        edges=edges,
        nodes=nodes,
        trajectories=tuple(selected),
        trajectory_scores=tuple(selected_scores),
    )
