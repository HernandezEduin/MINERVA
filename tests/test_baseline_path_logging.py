"""Regression tests for optional MINERVA-compatible baseline path logs."""

import unittest

from code.baselines.baseline_path_logging import format_path_report


class _Tokenizer:
    def decode(self, tokens, skip_special_tokens=False):
        assert tokens == [10, 20]
        assert skip_special_tokens
        return "who is the parent of person a?"


class _Episode:
    paths_exists = True
    path_key_exists = True

    def get_path(self, idx):
        assert idx == 0
        return [(1, 7, 2)]

    def get_path_key(self, idx):
        return [7]


class PathReportTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            question_id=123,
            question_tokens=[10, 20],
            tokenizer=_Tokenizer(),
            start=1,
            answers={2},
            episode=_Episode(),
            question_index=0,
            metrics={"PED": 0.0, "RED": 0.0, "F1_SG": 1.0, "F1_Rel": 1.0},
            reference_length=1,
            id2ent={1: "p1", 2: "p2"},
            id2rel={7: "r7", 9: "noop"},
            ent2name={"p1": "Person A", "p2": "Person B"},
            rel2name={"r7": "parent", "noop": "No Operation"},
        )

    def test_minerva_style_text_and_cleaned_raw_paths(self):
        kwargs = dict(self.kwargs)
        kwargs.update(raw_path=[(1, 7, 2), (2, 9, 2)], cleaned_path=[(1, 7, 2)],
                      walk_number=1, num_walks=20)
        report = format_path_report(**kwargs)
        self.assertIn("Question ID:      123\n", report)
        self.assertIn("Question:         Who is the parent of person a?\n", report)
        self.assertIn("Gold Path:        Person A --[parent]--> Person B", report)
        self.assertIn("Predicted Path:   Person A --[parent]--> Person B", report)
        self.assertIn("Path (Raw):       Person A --[parent]--> Person B --[No Operation]--> Person B", report)
        self.assertIn("Path F1 Score(↑): 1.0000", report)
        self.assertIn("Rel Edit Dist(↓): 0.0000", report)
        self.assertIn("Walk Selected:    1 of 20 (sample order; not ranked)", report)
        self.assertIn("Solved (Hit@1):   True", report)
        self.assertEqual(report, format_path_report(**kwargs))

    def test_unreachable_oracle_path_is_distinct_from_zero_hops(self):
        kwargs = dict(self.kwargs)
        kwargs.update(raw_path=None, cleaned_path=None,
                      metrics={"PED": None, "RED": None, "F1_SG": None, "F1_Rel": None})
        report = format_path_report(**kwargs)
        self.assertIn("Predicted Ans:    n/a", report)
        self.assertIn("Predicted Path:   n/a", report)
        self.assertIn("Agent Hop Count:  n/a", report)
        self.assertIn("Solved (Hit@1):   False", report)

        kwargs.update(raw_path=[], cleaned_path=[], answers={1})
        report = format_path_report(**kwargs)
        self.assertIn("Predicted Ans:    Person A", report)
        self.assertIn("Predicted Path:   Person A", report)
        self.assertIn("Agent Hop Count:  0", report)
        self.assertIn("Solved (Hit@1):   True", report)

    def test_relation_only_reference(self):
        class RelationOnly:
            paths_exists = False
            path_key_exists = True

            def get_path_key(self, idx):
                return [7, 7]

        kwargs = dict(self.kwargs)
        kwargs.update(episode=RelationOnly(), raw_path=[], cleaned_path=[])
        report = format_path_report(**kwargs)
        self.assertIn("Gold Path:        [parent] --> [parent]", report)


if __name__ == "__main__":
    unittest.main()
