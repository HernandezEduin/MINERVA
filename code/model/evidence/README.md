# Evidence retrieval (experimental)

This package is independent of the existing answer-navigation trainer and
evaluator. **The end-to-end supervised MINERVA checkpoint trainer and decoder
are not implemented yet.** The first implementation covers gold-action lookup,
trajectory union, evidence metrics, a teacher-forced target/loss utility, and
an offline decoded-trajectory evaluator.

## Evaluation of decoded trajectories

```bash
python -m code.model.evidence.evaluation --input decoded.json --output evidence_results.json --top-k 10
```

Evidence annotations are read from the existing dataset columns: `Path` for one trajectory or `Multi-Paths` for multiple trajectories. A singular `Path` contains triples directly; `Multi-Paths` may contain nested path groups. When both are present, `Multi-Paths` takes precedence. These annotations are gold references, never inference inputs. The old `gold_paths` key remains accepted for decoded-only compatibility. No `Evidence-Paths` column is required.\n\nEach decoded input record can have this shape:

```json
{
  "claim_id": "c1",
  "trajectories": [
    [[1, 10, 2], [2, 11, 3]],
    [[1, 12, 4]]
  ],
  "scores": [-0.2, -0.4],
  "Multi-Paths": [
    [[1, 10, 2], [2, 11, 3]],
    [[1, 12, 4]]
  ],
  "evidence_groups": [
    [
      [[1, 10, 2], [2, 11, 3]],
      [[1, 12, 4]]
    ]
  ]
}
```

Each edge is `[head_entity_id, relation_id, tail_entity_id]`.
`scores` are cumulative path log probabilities; higher is better.
`evidence_groups` is optional: every outer group is required, and
its inner paths are alternatives. `Path`/`Multi-Paths` (or the legacy decoded-only `gold_paths` key) specifies the union-reference
edge set for edge precision/recall/F1. Alternative group coverage is
reported separately to avoid conflating alternative and complementary
evidence. Nested path grouping is flattened for the reference path set; it does not imply which paths are alternatives or jointly required. Only explicit `evidence_groups` specifies that semantics. All source labels must be annotated independently from
question-answer correctness.

The optional `control_relations` list prevents NO_OP/STOP/RESTART
triples from entering evidence. The optional `inverse_relation_map`
maps inverse relation IDs to their canonical forward IDs; the map
should be generated from the active relation vocabulary.

**Inference contract:** callers must segment STOP/RESTART trajectories
before passing them to `build_evidence_subgraph`. The initial evidence
pipeline should disable RESTART and use STOP for termination. Gold
annotations are used only for training or post-hoc evaluation, never
for inference path selection.
