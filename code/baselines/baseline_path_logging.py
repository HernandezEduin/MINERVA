"""Human-readable trajectory reports compatible with MINERVA's test_paths.txt.

The formatter is deliberately side-effect-free: it neither samples actions nor
changes the evaluated path. Baselines call it only for explicitly requested logs.
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


Edge = Tuple[int, int, int]


def _entity_name(entity_id: int, id2ent: Mapping[int, str], ent2name: Mapping[str, str]) -> str:
    entity_key = id2ent.get(int(entity_id), str(entity_id))
    return str(ent2name.get(entity_key, entity_key))


def _relation_name(relation_id: int, id2rel: Mapping[int, str], rel2name: Mapping[str, str]) -> str:
    relation_key = id2rel.get(int(relation_id), str(relation_id))
    return str(rel2name.get(relation_key, relation_key))


def _path_text(
    start: int,
    edges: Optional[Sequence[Edge]],
    id2ent: Mapping[int, str],
    id2rel: Mapping[int, str],
    ent2name: Mapping[str, str],
    rel2name: Mapping[str, str],
) -> str:
    if edges is None:
        return "n/a"
    text = _entity_name(start, id2ent, ent2name)
    for _, relation, target in edges:
        text += f" --[{_relation_name(relation, id2rel, rel2name)}]--> {_entity_name(target, id2ent, ent2name)}"
    return text


def _metric_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def format_path_report(
    *,
    question_id: Any,
    question_tokens: Any,
    tokenizer: Any,
    start: int,
    answers: Sequence[int],
    episode: Any,
    question_index: int,
    raw_path: Optional[Sequence[Edge]],
    cleaned_path: Optional[Sequence[Edge]],
    metrics: Dict[str, Optional[float]],
    reference_length: Optional[int],
    id2ent: Mapping[int, str],
    id2rel: Mapping[int, str],
    ent2name: Mapping[str, str],
    rel2name: Mapping[str, str],
    walk_number: Optional[int] = None,
    num_walks: Optional[int] = None,
) -> str:
    """Format one *evaluated* trajectory; never resample or choose another path.

    ``None`` signifies an unreachable oracle answer, while ``[]`` is a valid
    zero-hop path. A random walk report describes one sampled trajectory, not
    the Monte Carlo mean reported in the statistics JSON.
    """
    if isinstance(question_tokens, str):
        question = question_tokens
    else:
        question = tokenizer.decode([int(token) for token in question_tokens], skip_special_tokens=True)

    answer_names = [_entity_name(entity, id2ent, ent2name) for entity in sorted(answers)]
    gold_answer = answer_names[0] if len(answer_names) == 1 else str(answer_names)
    final_entity = (raw_path[-1][2] if raw_path else start) if raw_path is not None else None
    solved = final_entity is not None and int(final_entity) in answers
    predicted_answer = _entity_name(final_entity, id2ent, ent2name) if final_entity is not None else "n/a"

    reference_path = episode.get_path(question_index) if episode.paths_exists else None
    if reference_path is not None:
        gold_path = _path_text(start, reference_path, id2ent, id2rel, ent2name, rel2name)
    else:
        reference_key = episode.get_path_key(question_index) if episode.path_key_exists else None
        gold_path = " --> ".join(f"[{_relation_name(r, id2rel, rel2name)}]" for r in reference_key) if reference_key else None

    lines = [
        f"Question ID:      {question_id}",
        f"Question:         {question.strip().capitalize()}",
        f"Start Entity:     {_entity_name(start, id2ent, ent2name)}",
        f"Gold Answer:      {gold_answer}",
        f"Predicted Ans:    {predicted_answer}",
    ]
    if gold_path is not None:
        lines.append(f"Gold Path:        {gold_path}")
    lines.extend([
        f"Predicted Path:   {_path_text(start, cleaned_path, id2ent, id2rel, ent2name, rel2name)}",
        f"Path (Raw):       {_path_text(start, raw_path, id2ent, id2rel, ent2name, rel2name)}",
    ])
    if walk_number is not None:
        lines.append(f"Walk Selected:    {walk_number} of {num_walks} (sample order; not ranked)")
    if metrics.get("F1_SG") is not None or metrics.get("PED") is not None:
        lines.extend([
            f"Path F1 Score(↑): {_metric_text(metrics.get('F1_SG'))}",
            f"Edit Distance(↓): {_metric_text(metrics.get('PED'))}",
        ])
    if metrics.get("F1_Rel") is not None or metrics.get("RED") is not None:
        lines.extend([
            f"Rel F1 Score(↑):  {_metric_text(metrics.get('F1_Rel'))}",
            f"Rel Edit Dist(↓): {_metric_text(metrics.get('RED'))}",
        ])
    lines.extend([
        f"Gold Hop Count:   {reference_length if reference_length is not None else 'n/a'}",
        f"Agent Hop Count:  {len(cleaned_path) if cleaned_path is not None else 'n/a'}",
        f"Solved (Hit@1):   {bool(solved)}",
        "",
        "=" * 40,
        "",
    ])
    return "\n".join(lines)
