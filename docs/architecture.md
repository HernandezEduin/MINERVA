# Architecture

## Task formulation

The natural-language branch of this repository treats multi-hop KGQA as **question-conditioned graph navigation**.

For each example, the agent receives:

- a natural-language question;
- a specified topic/source entity; and
- a knowledge graph.

Starting from the topic entity, the policy repeatedly selects an outgoing graph action and produces an explicit trajectory. The terminal entity is interpreted as the predicted answer.

This differs from the original MINERVA knowledge-graph-completion setup, where the policy is conditioned on a symbolic query relation in a query of the form (h, r, ?). The adaptation here conditions navigation on an embedding of the full natural-language question.

## High-level pipeline

~~~text
Natural-language question
        │
        ▼
Tokenizer + EmbeddingServer
        │
        ▼
Question representation
        │
        ▼
Projection adapter
        │
        ├──────────────┐
        ▼              │
Recurrent path state   │
        │              │
        └──────┬───────┘
               ▼
      Action scoring policy
               │
               ▼
 Current KG action space
               │
               ▼
     Next relation/entity
               │
          repeat by hop
               ▼
 Answer endpoint + trajectory
~~~

## Question representation

<code>code/data/embedding_server.py</code> provides transformer-based question representations outside the TensorFlow 1-style graph used by the MINERVA implementation. This avoids placing the transformer encoder directly inside the legacy TensorFlow execution graph.

The question representation is projected into the policy space. The current code supports configurable projection adapters, including linear, MLP, and residual variants.

## Navigation policy

The main policy implementation is in <code>code/model/agent.py</code>. Its decision at each hop is conditioned on the question representation and the recurrent path state.

The environment in <code>code/model/environment.py</code> exposes the valid relation/entity actions at the current node. Depending on configuration, the action space can include inverse graph edges and special navigation actions.

At evaluation time, beam search can keep multiple high-scoring trajectories per question. Rollout scores are cumulative policy log-probabilities and are used to rank candidate endpoints.

## Special actions

The vocabulary reserves three special navigation relations:

**NO_OP**

Keeps the rollout at its current entity. It is treated as a special action rather than a knowledge-graph evidence edge.

**STOP**

Allows the policy to terminate before exhausting the configured hop budget. Once a rollout stops, the environment tracks its termination step and endpoint.

**RESTART**

Returns a rollout to the original source entity while retaining the remaining hop budget, allowing another navigation attempt.

STOP and RESTART are optional and controlled by <code>use_stop_signal</code> and <code>use_restart_signal</code>.

## Trajectory cleanup for evaluation

Raw trajectories may contain NO_OP, STOP, RESTART, or multiple attempts. The evaluator therefore supports a configurable <code>path_segment_policy</code>:

| Policy | Evaluated trajectory |
| --- | --- |
| <code>raw</code> | Use the recorded trajectory as-is before metric-specific special-token filtering |
| <code>truncate_at_stop</code> | Remove NO_OP and discard STOP plus all later steps |
| <code>final_segment</code> | Remove NO_OP and keep only steps after the final RESTART |
| <code>final_segment_truncate</code> | Keep the final attempt after RESTART and truncate it at STOP |

The CLI/configuration default is <code>final_segment_truncate</code>. See [Evaluation metrics](metrics.md) for how the cleaned top-scoring rollout is compared with reference paths.

## Multi-answer questions

For multi-answer examples, a rollout is successful when its terminal entity is any valid gold answer.

The evaluator also measures answer-set coverage across all rollouts. When an example provides a gold relation-chain <code>Path-Key</code> but not a single entity-level <code>Paths</code> annotation, test-time path-fidelity evaluation can enumerate all graph paths that:

1. begin at the source entity;
2. follow the annotated relation chain exactly; and
3. end at any gold answer.

This prevents multi-answer fidelity from depending on an arbitrarily chosen entity-level reference path.

## Graph direction

<code>create_graph.py</code> adds inverse traversal relations to the generated graph. The runtime can expose a directed or undirected navigation view according to <code>use_directed_graph</code>.

For evaluation, inverse traversals are canonicalized back into their original forward relation and edge orientation before edge- and relation-level fidelity metrics are computed.

## Main code paths

| File | Responsibility |
| --- | --- |
| <code>code/model/agent.py</code> | Question-conditioned recurrent navigation policy |
| <code>code/model/trainer.py</code> | Reinforcement-learning training, beam evaluation, logging, and metric aggregation |
| <code>code/model/environment.py</code> | KG transitions, rewards, special actions, path cleanup, and per-example metrics |
| <code>code/model/metrics.py</code> | Shared edit-distance, entropy, and set-overlap utilities |
| <code>code/model/evaluation.py</code> | Standalone checkpoint evaluation |
| <code>code/data/feed_data.py</code> | QA batching, question formats, and ID/label translation |
| <code>code/data/data_utils.py</code> | QA parsing, tokenization, split handling, and cache creation |
| <code>code/data/embedding_server.py</code> | Transformer question-embedding service |
| <code>code/data/preprocessing_scripts/</code> | Graph, vocabulary, and readable-label preprocessing |

## Relation to the original MINERVA implementation

The navigation principle remains MINERVA's: learn a policy that walks a knowledge graph and reaches an answer entity. The principal task-level change in this branch is the conditioning signal and evaluation target:

| Original symbolic setup | Natural-language navigation setup |
| --- | --- |
| Query: (h, r, ?) | Question + topic entity |
| Condition on one query relation | Condition on the full question representation |
| Primarily endpoint ranking for KGC | Endpoint quality plus explicit trajectory fidelity |
| Alternative path around a masked target edge | Question-driven multi-hop navigation through the KG |

The symbolic version maintained in this repository is available on the <code>minerva_tf1</code> branch.
