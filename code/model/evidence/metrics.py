"""Evidence retrieval metrics; independent of answer endpoint scoring."""
from typing import Iterable, Mapping, Sequence, Set, Tuple

from .retrieval import Edge, EvidenceSubgraph, canonical_edge


def _prf(predicted: Set[Edge], gold: Set[Edge]) -> dict:
    correct = len(predicted & gold)
    # Explicitly define the empty/empty case as exact agreement.
    precision = correct / len(predicted) if predicted else float(not gold)
    recall = correct / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_evidence(
    retrieved: EvidenceSubgraph,
    gold_paths: Iterable[Sequence[Sequence[int]]],
    *,
    evidence_groups: Iterable[Iterable[Sequence[Sequence[int]]]] = (),
) -> dict:
    """Evaluate the union of retrieved triples.

    gold_paths is a union-reference edge set, not an equivalence class of
    alternative proofs. evidence_groups, when present, are REQUIRED groups
    of ALTERNATIVE paths: a group is covered if any one entire path in it
    is contained in the retrieved edge set.
    """
    gold = {canonical_edge(edge) for path in gold_paths for edge in path}
    predicted = set(retrieved.edges)
    result = _prf(predicted, gold)
    result.update({
        "retrieved_edges": len(predicted),
        "retrieved_nodes": len(retrieved.nodes),
        "retrieved_paths": len(retrieved.trajectories),
        "gold_edges": len(gold),
    })
    groups = list(evidence_groups)
    if groups:
        hits = []
        for group in groups:
            alternatives = [{canonical_edge(edge) for edge in path} for path in group]
            if not alternatives:
                raise ValueError("Evidence group has no alternative paths")
            hits.append(any(path.issubset(predicted) for path in alternatives))
        result["group_coverage"] = sum(hits) / len(hits)
        result["complete_evidence"] = float(all(hits))
    else:
        result["group_coverage"] = None
        result["complete_evidence"] = None
    total_path_edges = sum(len(path) for path in retrieved.trajectories)
    result["redundancy"] = (
        1.0 - len(predicted) / total_path_edges if total_path_edges else 0.0
    )
    return result
