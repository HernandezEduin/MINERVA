"""Standalone evidence retrieval evaluation (no answer-ranking trainer import).

Input is JSON or JSONL with one record per claim. This evaluator deliberately
consumes decoded evidence trajectories rather than running MINERVA checkpoints;
checkpoint decoding will be wired into this module separately.
"""
import argparse
import json
from pathlib import Path

from code.model.evidence.metrics import evaluate_evidence
from code.model.evidence.retrieval import build_evidence_subgraph


def evaluate_record(record: dict, top_k: int) -> dict:
    if "gold_paths" not in record or "trajectories" not in record:
        raise ValueError("Every record needs trajectories and gold_paths")
    subgraph = build_evidence_subgraph(
        record["trajectories"],
        record.get("scores"),
        top_k=top_k,
        control_relations=record.get("control_relations", ()),
        inverse_relation_map={
            int(k): int(v) for k, v in record.get("inverse_relation_map", {}).items()
        },
    )
    # Reference edges must be in the same canonical orientation as predictions.
    inverse_map = {int(k): int(v) for k, v in record.get("inverse_relation_map", {}).items()}
    from code.model.evidence.retrieval import canonical_edge
    gold_paths = [
        [canonical_edge(edge, inverse_map) for edge in path]
        for path in record["gold_paths"]
    ]
    groups = [
        [[canonical_edge(edge, inverse_map) for edge in path] for path in group]
        for group in record.get("evidence_groups", [])
    ]
    return {
        "claim_id": record.get("claim_id"),
        "metrics": evaluate_evidence(subgraph, gold_paths, evidence_groups=groups),
        "subgraph": subgraph.as_dict(),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Decoded claims JSON/JSONL")
    parser.add_argument("--output", required=True, help="Evaluation output JSON")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    source = Path(args.input)
    if source.suffix == ".jsonl":
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text())
        records = payload if isinstance(payload, list) else [payload]
    results = [evaluate_record(record, args.top_k) for record in records]
    metrics = [result["metrics"] for result in results]
    names = ["precision", "recall", "f1", "group_coverage", "complete_evidence", "redundancy"]
    averages = {
        name: sum(m[name] for m in metrics if m[name] is not None)
        / sum(m[name] is not None for m in metrics)
        if any(m[name] is not None for m in metrics) else None
        for name in names
    }
    output = {"count": len(results), "macro": averages, "examples": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
