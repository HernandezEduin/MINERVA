# Data Format

MINERVA operates on two linked inputs:

1. a knowledge graph used to construct the navigation action space; and
2. a natural-language QA table containing the topic entity, valid answer entity or entities, and optional reference-path information.

The preprocessing wrappers in <code>scripts/preprocessing/</code> prepare the released datasets into the format expected by the current code.

## Processed dataset layout

A typical processed dataset looks like:

~~~text
datasets/nlq/<dataset>/
├── triplets.txt
├── full_graph.txt
├── <dataset>_qa_nhop.csv
├── vocab/
│   ├── entity_vocab.json
│   └── relation_vocab.json
├── node_data.csv          # when human-readable metadata is available
├── relation_data.csv      # when human-readable metadata is available
├── README.md
└── LICENSE
~~~

MQuAKE-ST has separate single-answer and multi-answer QA CSV files. Exact filenames are set in the corresponding YAMLs under <code>configs/&lt;dataset&gt;/</code>.

## Knowledge graph

The source graph file is tab-separated, one directed triple per line:

~~~text
head_entity	relation	tail_entity
~~~

For example:

~~~text
entity_a	relation_x	entity_b
entity_b	relation_y	entity_c
~~~

The graph-building utility adds inverse traversal edges using relation names prefixed with an underscore. During fidelity evaluation, inverse relation tokens are canonicalized back to the corresponding forward directed edge.

The common preprocessing wrapper runs <code>create_graph.py</code> with the full-graph option, producing <code>full_graph.txt</code>. The vocabulary builder then assigns integer IDs to entities and relations.

Special relation IDs are also reserved for:

- <code>PAD</code>
- <code>DUMMY_START_RELATION</code>
- <code>NO_OP</code>
- <code>STOP</code>
- <code>RESTART</code>
- <code>UNK</code>

STOP and RESTART only become available navigation actions when enabled by the configuration.

## Natural-language QA CSV

The current loader uses the following columns directly for the navigation datasets:

| Column | Purpose |
| --- | --- |
| <code>Question-Number</code> | Stable question/example identifier |
| <code>Question</code> | Natural-language question |
| <code>Source</code> | Human-readable topic/source label |
| <code>Source-Entity</code> | Topic entity key present in the graph vocabulary |
| <code>Answer</code> | Human-readable answer label, or a list for multi-answer data |
| <code>Answer-Entity</code> | Gold entity key, or a Python-list representation for multi-answer data |
| <code>Hops</code> | Annotated reasoning depth used for per-hop evaluation |

Supported optional columns include:

| Column | Purpose |
| --- | --- |
| <code>Question-Paraphrased</code> | Python-list representation of alternate question wordings |
| <code>Question-Disambiguated</code> | Alternate disambiguated question text |
| <code>Paths</code> | Python-list representation of an entity-level evidence path |
| <code>Paths-Label</code> | Human-readable path description |
| <code>Path-Key</code> | Gold relation chain, represented as relation names separated by <code>-&gt;</code> |
| <code>SplitLabel</code> | Explicit train/dev/test split assignment |

The released navigation datasets already provide their expected schema. For custom data, use the same column names because the current preprocessing code maps them directly.

### Multi-annotated reference schema

Some datasets provide multiple valid path annotations and/or relation-chain
annotations for the same question. MINERVA supports this richer
**multi-annotated reference schema** while remaining backward-compatible with
datasets that provide a single reference path or relation chain.

The preprocessing layer normalizes the following fields into MINERVA's internal
representation:

| Input column | Internal MINERVA representation | Use |
| --- | --- | --- |
| <code>Multi-Paths</code> | <code>Paths</code> containing a list of entity-level reference paths | Best-reference PED, F1<sub>SG</sub>, and node-set overlap |
| <code>Multi-Paths-Key</code> | <code>Path-Key</code> containing a list of relation chains | Best-reference RED and F1<sub>REL</sub> |
| <code>Question-Family-ID</code> | Preserved | Family-aware downstream analysis |
| <code>Question-Family-Size</code> | Preserved | Family-size metadata and analysis |
| <code>Graph-Answer-Entity</code> | Mapped to entity IDs and preserved | Auxiliary graph-expanded answer set |
| <code>Path-Count</code> | Preserved | Number of released reference paths |
| <code>Graph-Path-Count</code> | Preserved | Number of graph-consistent path realizations |

Datasets using the legacy schema continue to provide <code>Paths</code> and
<code>Path-Key</code> directly. Their existing single-reference representation
is preserved, so the same preprocessing and evaluation pipeline remains
compatible with KINSHIP, MQuAKE-ST, MetaQA, and other datasets following that
format.

Regardless of the reference-path schema, the default answer reward and standard
Hits@K evaluation use the released <code>Answer-Entity</code> field.
<code>Graph-Answer-Entity</code>, when available, is retained as auxiliary
metadata and does not replace the released answer target unless an evaluation
protocol explicitly requests the graph-expanded answer set.

When multiple released entity-level reference paths are available, path-fidelity
metrics use best-reference matching:

```math
\mathrm{PED}(q)
=
\min_{P^\star \in \mathcal{R}(q)}
\mathrm{PED}\!\left(P_{\mathrm{pred}}(q), P^\star\right)
```

```math
F1_{\mathrm{SG}}(q)
=
\max_{P^\star \in \mathcal{R}(q)}
F1_{\mathrm{SG}}\!\left(P_{\mathrm{pred}}(q), P^\star\right)
```

where $\mathcal{R}(q)$ is the set of released entity-level reference paths for
question $q$.

Likewise, when multiple released relation chains are available:

```math
\mathrm{RED}(q)
=
\min_{R^\star \in \mathcal{R}_{\mathrm{rel}}(q)}
\mathrm{RED}\!\left(R_{\mathrm{pred}}(q), R^\star\right)
```

```math
F1_{\mathrm{REL}}(q)
=
\max_{R^\star \in \mathcal{R}_{\mathrm{rel}}(q)}
F1_{\mathrm{REL}}\!\left(R_{\mathrm{pred}}(q), R^\star\right)
```

This prevents an agent from being penalized for following one released-valid
reasoning trajectory when several valid annotations are provided.

When entity-level reference paths are unavailable but annotated relation chains
are present, MINERVA can instead reconstruct answer-consistent entity-level
reference paths from the source entity, annotated relation chain(s), released
answer set, and active evaluator graph. These reconstructed references are
treated separately from directly released path annotations.

This prevents an agent from being penalized for following one released-valid
reasoning trajectory when several valid annotations are provided.

When entity-level reference paths are unavailable but annotated relation chains
are present, MINERVA can instead reconstruct answer-consistent entity-level
reference paths from the source entity, annotated relation chain(s), released
answer set, and active evaluator graph. These reconstructed references are
treated separately from directly released path annotations.

## Single-answer and multi-answer encoding

A single-answer row stores one graph entity in <code>Answer-Entity</code>:

~~~csv
Question-Number,Question,Source,Source-Entity,Answer,Answer-Entity,Hops
1,"Example question?",Source label,source_id,Answer label,answer_id,2
~~~

For multi-answer data, <code>Answer-Entity</code> and <code>Answer</code> are stored as Python-list literals in the CSV:

~~~text
['answer_id_1', 'answer_id_2']
~~~

The loader detects a multi-answer dataset when every <code>Answer-Entity</code> value is a list literal and converts each list to a set/list of entity IDs for navigation and evaluation.

## Reference paths

When an entity-level evidence path is available, <code>Paths</code> is represented as a Python-list literal containing triples:

~~~text
[['entity_a', 'relation_x', 'entity_b'],
 ['entity_b', 'relation_y', 'entity_c']]
~~~

The loader converts each entity and relation to its vocabulary ID.

When only the relation chain is annotated, <code>Path-Key</code> stores a relation sequence:

~~~text
relation_x->relation_y
~~~

This is sufficient for relation-level metrics. In multi-answer test evaluation, the current evaluator can also enumerate entity-level paths that follow the annotated relation chain from the source entity to any valid gold answer and use those paths for semantic path-fidelity evaluation.

## Splits and caching

If <code>SplitLabel</code> is present, the data loader can use the provided split labels. Otherwise it has a fallback automatic split.

Tokenized questions and mapped entity/relation IDs are cached as Parquet files. The YAML field <code>cached_QAMetaData_path</code> points to a JSON metadata file that records those cache locations. If the cache does not exist, the raw CSV is processed and a new cache is created.

When changing the source CSV, tokenizer, or vocabulary, use a new cache path or force preprocessing so an incompatible old cache is not reused.

## Question input modes

The batcher supports several question representations:

- <code>full_text</code>: the natural-language question.
- <code>paraphrased</code>: randomly select a stored paraphrase during training.
- <code>relation_only</code>: construct the input from the annotated relation sequence.
- <code>graph_only</code>: omit linguistic question content.

The public dataset-specific configurations use <code>full_text</code> by default.

## Adding a custom dataset

A minimal custom workflow is:

1. Create <code>datasets/nlq/&lt;name&gt;/triplets.txt</code>.
2. Create a QA CSV following the schema above.
3. Run the common graph/vocabulary preprocessing:
   
   ~~~bash
   bash scripts/preprocessing/dataset.sh <name>
   ~~~
4. Add <code>configs/&lt;name&gt;/train.yaml</code> and <code>configs/&lt;name&gt;/evaluate.yaml</code>, using an existing dataset as a template.
5. Set <code>data_input_dir</code>, <code>raw_QAData_path</code>, <code>cached_QAMetaData_path</code>, graph direction, path length, and model dimensions for the new dataset.
6. Verify that every source/answer entity and every annotated path relation exists in the generated vocabulary.

For datasets with external integer IDs and human-readable metadata, <code>create_vocab_title.py</code> can additionally build readable mappings; the released MQuAKE-ST and MetaQA preprocessing wrappers demonstrate that workflow.
