# Evidence-claim task formulation (FACTKG reference)

Reference: Kim et al., *FACTKG: Fact Verification via Reasoning on Knowledge Graphs* (2023), https://arxiv.org/abs/2305.06590 — especially Table 1, Sections 3.1 and 4.3.2.

## Goal and separation of responsibility

The MINERVA evidence mode is a **claim-conditioned KG evidence retriever**. Given a natural-language claim, an explicitly supplied starting entity, and a KG, it outputs K independently decoded trajectories. Their canonical KG edge union is the retrieved evidence subgraph. A separate verifier consumes the claim and this retrieved subgraph to predict SUPPORTED / REFUTED (or INSUFFICIENT_EVIDENCE if the evaluation task provides such a label). MINERVA must not infer a verification label solely from terminal entity correctness.

FACTKG's baseline predicts relevant relations and hop count and collects paths before downstream verification; our research alternative learns explicit stepwise navigation. This is a proposed MINERVA architecture, **not** a claim that FACTKG uses MINERVA or that the full pipeline is implemented here.

## Reasoning requirements

| Claim pattern | Retrieval supervision / evaluation consideration |
| --- | --- |
| One-hop | Identify the relevant evidence triple |
| Conjunction | Several required facts; a single retrieved path may be insufficient |
| Existence | Retrieve an instance of the asserted relation, where available; failure to retrieve is not proof of nonexistence |
| Multi-hop | Recover relation-ordered connected evidence path(s), possibly with unspecified intermediate nodes |
| Negation | Retrieve relevant evidence/counterevidence; interpretation depends on the negated proposition and verifier |

A negative or refuted label does not imply a gold *negative KG edge*. In open-world KGs, absence of an edge is not automatically evidence of falsity. FACTKG's dataset-specific refutation construction and negation semantics must not be generalized into a closed-world assumption.

## Gold annotation contract

- `Path`: one entity-level evidence trajectory composed of (head, relation, tail) triples.
- `Multi-Paths`: multiple annotated evidence trajectories, potentially nested. The existing parser flattens these **only** for path extraction.
- If both exist, `Multi-Paths` is selected.
- `Path-Key` and endpoints are not silently substituted as semantic gold evidence.
- Nesting in `Multi-Paths` does **not** tell us whether the paths are interchangeable alternative witnesses or jointly required subclaims. Explicit evidence group semantics (when available) must remain separate.
- The current retrieval metric's union-of-gold-edges precision/recall/F1 is a structural reference score, **not** proof that the retrieved graph semantically validates the claim. Group coverage is reported only when explicitly annotated.

## Supervision and scoring

Teacher forcing trains on explicit annotated edges under the *training* graph action space, preserving the shared policy and leaving the original REINFORCE trainer unchanged. Multiple paths may be trained as independently weighted references; marginal likelihood is appropriate for *alternative* witnesses only. A conjunction with complementary evidence needs coverage-aware supervision and set-based evaluation. Inference cannot consume labels, gold paths, relation-chain hints, or gold endpoints.

For retrieval, report edge precision/recall/F1, path exact match, group coverage when labels exist, redundancy across K paths, and subgraph size. Stratify results by claim reasoning type. For end-to-end comparison, measure verdict accuracy of a fixed downstream verifier separately from retrieval metrics.

## Implementation status

The evidence package currently implements normalization, gold action lookup, teacher-force target preparation, decoded-path subgraph union, and offline evidence metrics. The end-to-end TensorFlow teacher-forced training loop, checkpoint decoding, and claim verifier are **not implemented yet**. This document constrains those upcoming changes rather than claiming their completion.

## Compatibility

Only `evidence-claim` is the development branch. Keep the original `code/model/trainer.py` and `code/model/evaluation.py` behavior unchanged. Evidence-specific training/evaluation live in `code/model/evidence/` and reuse the same policy weights where shape-compatible.
