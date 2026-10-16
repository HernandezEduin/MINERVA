"""Normalize evidence annotations from Path (singular) or Multi-Paths (plural).

The evidence mode reads raw rows directly, keeping the original QA loader and
its legacy Paths field unchanged. Only explicit graph triples count as evidence.
"""
import ast
from typing import Any, List


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Evidence annotation must be a Python/JSON list of triples") from exc
    return value


def _is_edge(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(
        not isinstance(x, (list, tuple, dict)) for x in value
    )


def _is_path(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and all(_is_edge(x) for x in value)


def _collect_paths(value: Any) -> List[list]:
    if _is_path(value):
        return [[list(edge) for edge in value]]
    if not isinstance(value, (list, tuple)):
        raise ValueError("Evidence must contain paths consisting of triples")
    result = []
    for child in value:
        result.extend(_collect_paths(child))
    return result


def get_evidence_paths(row: dict) -> List[list]:
    """Return evidence paths; Multi-Paths takes priority when both exist.

    Path: one sequence of triples. Multi-Paths: one or more paths, potentially
    nested in relation-chain groups. Structure is flattened ONLY for the
    path set; no semantic alternative/required group labels are inferred.
    Empty/invalid annotations fail explicitly instead of falling back to QA
    answers or reconstructing paths from a relation key.
    """
    key = "Multi-Paths" if "Multi-Paths" in row else "Path" if "Path" in row else None
    if key is None:
        raise ValueError("Evidence mode requires a Path or Multi-Paths column")
    parsed = _parse(row[key])
    if parsed is None:
        raise ValueError(f"{key} is empty")
    if key == "Path" and not _is_path(parsed):
        raise ValueError("Path must contain exactly one contiguous sequence of triples")
    paths = _collect_paths(parsed)
    if not paths:
        raise ValueError(f"{key} contains no evidence paths")
    return paths
