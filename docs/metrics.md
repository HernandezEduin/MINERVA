# Evaluation Metrics

This document describes the metrics implemented by the current natural-language MINERVA evaluator. The terminology is aligned with the accompanying submission, where answer correctness is evaluated together with the explicit navigation trajectory.

The authoritative implementations are in:

- <code>code/model/trainer.py</code> for rollout ranking and dataset-level aggregation;
- <code>code/model/environment.py</code> for path cleanup, path/relation overlap, multi-answer path expansion, and navigation diagnostics; and
- <code>code/model/metrics.py</code> for Levenshtein edit distance, entropy, and set precision/recall/F1.

## 1. Which rollout is evaluated?

During evaluation, each question produces multiple rollouts. With beam search enabled, the number of evaluated rollouts is:

~~~text
min(test_rollouts, max_num_actions)
~~~

Each rollout has a cumulative policy log-probability. Rollouts are sorted from highest to lowest score.

Two different views of those rollouts are used:

- **Answer ranking metrics** consider the ranked candidate endpoints across rollouts.
- **Trajectory-fidelity metrics and top-rollout diagnostics** use the single highest-scoring rollout for each question, whether or not that rollout reaches a correct answer.

This distinction is important: F1_SG, F1_REL, PED, and RED do not select the best matching gold trajectory after looking at all predicted rollouts. They evaluate the policy's top-scoring predicted trajectory. The only exception is how a multi-answer example chooses among multiple **valid reference paths**, described below.

## 2. Answer ranking: Hits@K and MRR

The evaluator reports:

- Hits@1
- Hits@3
- Hits@5
- Hits@10
- Hits@20
- Mean Reciprocal Rank (MRR)

These are **rollout-induced endpoint rankings**, not exhaustive full-entity rankings such as those commonly used for knowledge-graph embeddings.

### Default max pooling

With <code>pool: max</code>, rollouts are processed in descending log-probability order. Duplicate terminal entities do not consume additional rank positions. Therefore, the effective ranking is the sequence of **unique endpoint entities ordered by their highest-scoring occurrence**.

For a question, let rank(q) be the 1-based rank of the first gold endpoint in that unique endpoint ranking. Then:

~~~text
Hits@K(q) = 1 if rank(q) <= K, otherwise 0
MRR(q)    = 1 / rank(q), or 0 if no gold answer is reached
~~~

Dataset values are macro-averages over questions.

For multi-answer questions, reaching **any** gold endpoint counts as a correct ranked answer.

### Sum pooling

With <code>pool: sum</code>, rollout log-probabilities are grouped by endpoint and combined with log-sum-exp before endpoint ranking.

The current trainer contains an explicit TODO for correcting this branch for multi-answer questions. For multi-answer experiments, <code>pool: max</code> should therefore be treated as the supported/default interpretation unless the sum-pooling code is updated.

### Per-hop answer metrics

The evaluator also groups questions by their annotated <code>Hops</code> value and reports per-hop:

- Hits@1
- MRR

These are useful for separating overall answer quality from reasoning-depth effects.

## 3. Answer-set endpoint coverage

Endpoint coverage compares the **set of unique terminal entities reached across all rollouts** for a question with its set of gold answers.

For predicted endpoint set A and gold answer set G:

~~~text
Precision = |A ∩ G| / |A|
Recall    = |A ∩ G| / |G|
F1        = harmonic mean of Precision and Recall
~~~

The evaluator computes these values per question and macro-averages them.

This metric is especially useful for multi-answer KGQA because Hits@1 can be perfect after finding only one valid answer, whereas answer-set recall exposes whether the rollout population covers the other valid answers.

For single-answer datasets, the same code evaluates against a singleton gold set.

## 4. Which predicted path is compared?

The top-scoring rollout is first converted into an edge sequence:

~~~text
(head_entity, relation, tail_entity)
~~~

The configured <code>path_segment_policy</code> is then applied. The CLI/configuration default is <code>final_segment_truncate</code>.

| Policy | Behavior |
| --- | --- |
| <code>raw</code> | Keep the raw trajectory |
| <code>truncate_at_stop</code> | Remove NO_OP and cut before the first STOP |
| <code>final_segment</code> | Remove NO_OP and keep only edges after the final RESTART |
| <code>final_segment_truncate</code> | Keep only the final post-RESTART segment and cut it before STOP |

Metric functions additionally ignore special NO_OP, STOP, and RESTART relations where appropriate.

Inverse traversal relations are canonicalized before path/relation comparison: an inverse edge is mapped to its original forward relation and its head/tail orientation is swapped.

## 5. Subgraph overlap: F1_SG

The code names these values **SubGraph Overlap Metrics (SG)**.

For the cleaned predicted path P and a gold path P*, both are converted to sets of canonical directed KG edges. Edge order and multiplicity are ignored.

~~~text
E_pred = set of predicted canonical edges
E_gold = set of gold directed edges
~~~

Precision_SG, Recall_SG, and F1_SG are ordinary set-overlap precision, recall, and F1.

### Interpretation

F1_SG answers:

> Did the predicted trajectory traverse the same evidence edges as a valid reference path?

It is permutation-invariant, so it complements PED, which is order-sensitive.

## 6. Path Edit Distance (PED)

PED is the standard Levenshtein edit distance between the **ordered canonical edge sequences** of the predicted and gold paths.

Insertions, deletions, and substitutions each have unit cost.

**Important:** the current implementation returns the **raw edit distance**. It is **not divided by path length**.

Per-question PED is therefore an integer, while the reported dataset-level average is generally fractional because distances are averaged across questions.

Lower is better; PED = 0 means exact edge-sequence agreement after the configured trajectory cleanup and inverse-edge canonicalization.

The evaluator reports both:

- overall average PED; and
- PED grouped by annotated hop count.

## 7. Relation overlap: F1_REL

The code names these values **Relation-Set Overlap Metrics (REL)**.

The cleaned predicted relations are canonicalized so inverse relation tokens map back to their original relation IDs. Special actions are ignored. Predicted and gold relations are then compared as sets.

This yields Precision_REL, Recall_REL, and F1_REL.

F1_REL is less strict than F1_SG because it ignores the entities connected by each relation and ignores relation order.

Relation-level metrics can be computed when either:

- a full entity-level gold <code>Paths</code> annotation is available; or
- a relation-chain <code>Path-Key</code> is available.

## 8. Relation Edit Distance (RED)

RED is the standard Levenshtein edit distance between the **ordered relation sequences** of the cleaned predicted path and the gold relation sequence.

As with PED:

- inverse relation tokens are canonicalized;
- special actions are ignored; and
- the distance is **raw, not length-normalized**.

Lower is better; RED = 0 means exact relation-sequence agreement.

The evaluator reports overall and per-hop averages.

## 9. Node-set overlap

When full entity-level reference paths are available, the evaluator also compares:

- the set of entities visited by the cleaned predicted path; and
- the set of head/tail entities in the gold path.

It reports node precision, recall, and F1.

This is a looser diagnostic than F1_SG because a trajectory may visit the correct nodes through the wrong relations or edges.

## 10. Multi-answer path fidelity

A multi-answer question can have more than one entity-level path that is semantically valid. The current test evaluator handles the case where the dataset provides:

- multiple gold answer entities;
- a gold relation-chain <code>Path-Key</code>; and
- no single explicit entity-level <code>Paths</code> annotation.

For that case, the evaluator enumerates all paths in the evaluator graph that:

1. start at the question's source entity;
2. follow the annotated relation chain exactly; and
3. terminate at a valid gold answer.

Call this set of valid reference paths G(q).

The top-scoring predicted trajectory is still fixed. Only the **reference** is allowed to vary:

~~~text
PED_multi(q)   = min over P* in G(q) of PED(P_pred, P*)
F1_SG_multi(q) = max over P* in G(q) of F1_SG(P_pred, P*)
~~~

For F1_SG, the precision/recall pair reported for the example comes from the same valid reference path that maximizes F1.

This semantic expansion is used only during test evaluation and only when the required multi-answer and Path-Key information is available.

## 11. Navigation diagnostics

Diagnostics are computed on the raw highest-scoring rollout before trajectory cleanup.

| Metric | Implementation |
| --- | --- |
| Special Step Rate | Fraction of raw steps whose relation is NO_OP, STOP, or RESTART |
| Restart Rate | Fraction of raw steps that are RESTART |
| No-Op Rate | Fraction of raw steps that are NO_OP |
| Cycle Rate | Fraction of non-special steps whose tail entity was already visited |
| Backtrack Rate | Fraction of non-special steps that immediately reverse the previous edge through its inverse relation |
| Unique Edges | Number of distinct canonical non-special edges traversed |
| Redundancy | 1 - unique_edges / non_special_steps |
| Avg Segment Hops | Number of edges remaining after applying <code>path_segment_policy</code>, averaged across questions |

These metrics describe policy behavior; they are not used as trajectory-fidelity rewards.

## 12. STOP diagnostics

When STOP is enabled, the evaluator reports both rollout-level and top-rollout statistics.

For rollout-level metrics, the denominator is all evaluated rollouts:

- **Stop Rate:** P(stopped)
- **Correct Stop Rate:** P(stopped and final endpoint is a gold answer)
- **Incorrect Stop Rate:** P(stopped and final endpoint is not a gold answer)
- **Hit without Stop Rate:** P(hit and did not stop)
- **Termination Step:** mean tracked termination step

Top-rollout versions apply the same definitions to the highest-scoring rollout for each question.

## 13. RESTART diagnostics

When RESTART is enabled, the evaluator reports:

- **Restart Any Rate:** P(rollout used RESTART at least once)
- **Post Restart Success Rate:** P(hit | rollout used RESTART)
- **Restart and Hit Rate:** P(rollout used RESTART and hit)

Rollout-level and top-rollout versions are both reported.

## 14. Search-space and entropy diagnostics

The evaluator additionally reports:

- valid action count at each step and its overall mean;
- average action-distribution entropy by step; and
- a question-level entropy summary derived from the same stored per-step action entropies.

These are primarily debugging/analysis statistics rather than headline task metrics.


## 15. Structural calibration references

The path-fidelity numbers in the accompanying submission are accompanied by two **non-learned structural calibration references**. They are intended to make the scale of PED, RED, F1_SG, and F1_REL easier to interpret; they are not competing KGQA models.

The implementations are in [<code>code/baselines/</code>](../code/baselines/), and the dataset-level launchers are in <code>scripts/baselines/</code>. The definitions follow Appendix A.4 of the accompanying submission.

### RW-Ans_MC and the unbiased random-walk reference

<code>code/baselines/random_walk_stats.py</code> samples actual trajectories from the evaluator graph. At each step, the walk chooses uniformly among the valid navigation actions exposed by the grapher.

The provided experiment wrappers use:

- the benchmark's fixed mixed-hop navigation horizon;
- **100 Monte Carlo walks per question**; and
- seeds **0, 42, and 100**.

For one question, the answer statistic is the fraction of sampled walks whose terminal entity is in the valid answer set. Averaging this quantity over evaluation questions gives the Monte Carlo random-walk answer reachability reported in the paper as **RW-Ans_MC**. The JSON summary uses the key <code>RW_Ans</code>.

Crucially, the same sampled random-walk trajectories are also passed through the repository's path-fidelity evaluator. Their PED, RED, F1_SG, and F1_REL therefore provide an **unguided-navigation calibration** for the learned agents rather than an analytical proxy based only on graph degree or density.

### Shortest Path Oracle

<code>code/baselines/shortcut_oracle_stats.py</code> implements the paper's **Shortest Path Oracle**. The oracle is given the valid answer entity set and performs breadth-first search on the evaluator graph to find a shortest path from the topic entity to any valid answer.

It is deliberately **question-agnostic**: the natural-language question is never used to choose the path. Its resulting trajectory is scored with the same PED, RED, F1_SG, and F1_REL implementations as a learned agent trajectory.

This reference answers a different calibration question from the random walk:

- the random walk measures path fidelity under **unguided traversal**;
- the Shortest Path Oracle measures path fidelity when **answer reachability is known** and graph-theoretic path length is minimized, but the question's intended reasoning structure is ignored.

The Shortest Path Oracle is therefore **not a strict upper or lower bound** on path fidelity. A shortest route to a correct answer can differ from the evidence path implied by the question.

### Released wrappers

| Setting | Random walk / RW-Ans_MC | Shortest Path Oracle |
| --- | :---: | :---: |
| Kinship | Yes | Yes |
| MQuAKE-ST single-answer | Yes | Yes |
| MQuAKE-ST multi-answer | Yes | Yes |
| MetaQA | Yes | — |

The random-walk and oracle scripts share the same path-fidelity helper functions in <code>baseline_path_fidelity.py</code>, including inverse-edge canonicalization, multi-answer semantic path expansion where available, and the configured path-segment cleanup policy.

## 16. Output files

The evaluator writes a <code>scores.txt</code> containing the available sections for the selected dataset/configuration:

~~~text
Answer Metrics
Per Hop Answer Metrics
Valid Action Count at Each Step
Entropy Metrics
Reasoning Diagnostics (Top-Rollout)
Stop Quality                         # when enabled
Restart Quality                      # when enabled
Answer(s) Endpoint Coverage Metrics
SubGraph Overlap Metrics (SG)        # when path references are available
Node-Set Overlap Metrics             # when full paths are available
Path Edit Distance Metrics (PED)     # when path references are available
Relation-Set Overlap Metrics (REL)   # when paths or Path-Key are available
Relation Edit Distance Metrics (RED) # when paths or Path-Key are available
~~~

When <code>print_paths: True</code>, per-question trajectory logs include the question, source entity, gold answer(s), predicted answer, predicted path, raw path, and the available per-example fidelity values.

## 17. Metric summary

| Metric | Scope | Order-sensitive | Entity-sensitive | Better |
| --- | --- | ---: | ---: | --- |
| Hits@K | Ranked unique endpoints | No | Yes | Higher |
| MRR | Ranked unique endpoints | No | Yes | Higher |
| Answer-set F1 | Endpoint set across all rollouts | No | Yes | Higher |
| F1_SG | Top-rollout edge set vs reference | No | Yes | Higher |
| PED | Top-rollout edge sequence vs reference | Yes | Yes | Lower |
| F1_REL | Top-rollout relation set vs reference | No | No | Higher |
| RED | Top-rollout relation sequence vs reference | Yes | No | Lower |
| Node F1 | Top-rollout node set vs reference | No | Yes | Higher |

For paper-aligned trajectory reporting, the core path-fidelity names are **F1_SG**, **F1_REL**, **PED**, and **RED**. Hits@1 remains the endpoint-success measure; the fidelity metrics determine whether the agent reached its result through an evidence trajectory consistent with the annotated reasoning structure.
