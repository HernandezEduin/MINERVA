"""Convert supervised evidence paths into exact KG action indices.

The action space is represented by aligned next_entities/next_relations arrays,
matching the current EpisodeNLQ state interface.
"""
from typing import Iterable, List, Sequence, Tuple

Edge = Tuple[int, int, int]


def lookup_gold_action(
    current_entity: int,
    gold_edge: Sequence[int],
    next_entities: Sequence[int],
    next_relations: Sequence[int],
) -> int:
    """Find a gold action by its destination AND relation, never relation alone."""
    if len(gold_edge) != 3:
        raise ValueError("Gold edges must be (head, relation, tail) triples")
    head, relation, tail = (int(v) for v in gold_edge)
    if head != int(current_entity):
        raise ValueError("Gold path is not contiguous with current navigation state")
    if len(next_entities) != len(next_relations):
        raise ValueError("Candidate relation/entity arrays are misaligned")
    matches = [
        i for i, (entity, rel) in enumerate(zip(next_entities, next_relations))
        if int(entity) == tail and int(rel) == relation
    ]
    if not matches:
        raise ValueError(
            "Gold edge absent from the available KG actions; check graph "
            "direction, vocabulary, and max_num_actions truncation"
        )
    return matches[0]


def validate_gold_path(start_entity: int, path: Iterable[Sequence[int]]) -> List[Edge]:
    """Require a contiguous path rooted at the specified source entity."""
    current = int(start_entity)
    validated = []
    for edge in path:
        if len(edge) != 3:
            raise ValueError("Malformed evidence triple")
        head, relation, tail = (int(v) for v in edge)
        if head != current:
            raise ValueError("Evidence path contains a disconnected hop")
        validated.append((head, relation, tail))
        current = tail
    return validated
