"""Dependency-light evidence behavior tests."""
import unittest

from code.model.evidence.evaluation import evaluate_record
from code.model.evidence.retrieval import build_evidence_subgraph
from code.model.evidence.supervision import lookup_gold_action, validate_gold_path


class EvidenceTests(unittest.TestCase):
    def test_union_deduplicates_paths_and_edges(self):
        paths = [[(1, 2, 3)], [(1, 2, 3)], [(1, 4, 5)]]
        result = build_evidence_subgraph(paths, [-0.1, -0.2, -0.3])
        self.assertEqual(len(result.edges), 2)
        self.assertEqual(len(result.trajectories), 2)

    def test_inverse_canonicalization(self):
        result = build_evidence_subgraph([[(3, 7, 1)]], inverse_relation_map={7: 2})
        self.assertEqual(result.edges, frozenset({(1, 2, 3)}))

    def test_missing_gold_action_is_explicit(self):
        with self.assertRaises(ValueError):
            lookup_gold_action(1, (1, 2, 3), [3], [4])
        with self.assertRaises(ValueError):
            validate_gold_path(1, [(9, 2, 3)])

    def test_evaluation_alternative_group(self):
        record = {
            "claim_id": "x",
            "trajectories": [[[1, 2, 3]], [[1, 4, 5]]],
            "scores": [-0.1, -0.3],
            "gold_paths": [[[1, 2, 3]]],
            "evidence_groups": [[[[1, 2, 3]], [[1, 4, 5]]]],
        }
        result = evaluate_record(record, 2)["metrics"]
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["complete_evidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
