# Getting Started

This guide covers the supported workflow for the natural-language MINERVA adaptation in this repository. All commands below assume that your current directory is the repository root.

## 1. Environment

The provided Conda environment uses Python 3.9, TensorFlow 2.11.1, CUDA Toolkit 11.2, and cuDNN 8.1.

~~~bash
conda env create -f environment.yml
conda activate minerva
~~~

If you do not need the Conda-managed CUDA stack, you can install the Python dependencies directly:

~~~bash
python -m pip install -r requirements.txt
~~~

The launcher scripts accept an optional GPU index. If no usable GPU index is supplied, they set CUDA_VISIBLE_DEVICES to an empty value and run on CPU.

~~~bash
# GPU 0
bash scripts/run_nlq.sh configs/kinship/train.yaml 0

# CPU
bash scripts/run_nlq.sh configs/kinship/train.yaml
~~~

## 2. Dataset preparation

The repository provides preprocessing wrappers for the navigation datasets. They copy the navigation-ready source files into <code>datasets/nlq/&lt;dataset&gt;/</code>, build the graph representation, and create the entity/relation vocabularies required by MINERVA.

For blind evaluation, use the anonymized dataset mirrors below.

### Kinship

**Blind dataset:** [Kaggle](https://www.kaggle.com/datasets/anonymousexpert/kinship)

Download the dataset archive from the page above and extract it so that the repository contains:

~~~text
raw_data/
└── kinship_hinton/
    ├── kg/
    ├── qa/
    ├── README.md
    └── LICENSE
~~~

Then preprocess it:

~~~bash
bash scripts/preprocessing/kinship.sh
~~~

The processed dataset is written under:

~~~text
datasets/nlq/kinship/
~~~

including:

~~~text
kinship_qa_nhop.csv
~~~

### MQuAKE-ST

**Blind dataset:** [Kaggle](https://www.kaggle.com/datasets/anonymousexpert/mquake-st)

Download the dataset archive from the page above and extract it so that the repository contains:

~~~text
raw_data/
└── mquake_st_dataset/
    ├── kg/
    ├── metadata/
    ├── qa/
    ├── README.md
    └── LICENSE
~~~

Then preprocess it:

~~~bash
bash scripts/preprocessing/mquake_st.sh
~~~

The processed dataset is written under:

~~~text
datasets/nlq/mquake_st/
~~~

and includes both QA settings used by the provided configurations:

~~~text
mquake_sa_qa_nhop.csv
mquake_ma_qa_nhop.csv
~~~

### What preprocessing creates

The dataset wrappers under <code>scripts/preprocessing/</code> copy the required source files and invoke the shared preprocessing utilities in <code>code/data/preprocessing_scripts/</code>.

For the supplied datasets, preprocessing creates a layout similar to:

~~~text
datasets/nlq/<dataset>/
├── triplets.txt
├── full_graph.txt
├── <qa-file>.csv
├── vocab/
│   ├── entity_vocab.json
│   └── relation_vocab.json
└── ...
~~~

See [Data format](data_format.md) for the complete schema and instructions for adding a custom dataset.

## 3. Dataset-specific configurations

Current public examples are organized by dataset rather than in one shared NLQ configuration directory.

| Dataset | Train config | Evaluation config |
| --- | --- | --- |
| Kinship | <code>configs/kinship/train.yaml</code> | <code>configs/kinship/evaluate.yaml</code> |
| MQuAKE-ST single-answer | <code>configs/mquake_st/train_single.yaml</code> | <code>configs/mquake_st/evaluate_single.yaml</code> |
| MQuAKE-ST multi-answer | <code>configs/mquake_st/train_multi.yaml</code> | <code>configs/mquake_st/evaluate_multi.yaml</code> |
| MetaQA | <code>configs/metaqa/train.yaml</code> | <code>configs/metaqa/evaluate.yaml</code> |

Older experiment YAMLs may remain under <code>configs/nlq/</code>, but the dataset-specific directories above are the maintained entry points documented for public use.

## 4. Training

Use <code>scripts/run_nlq.sh</code> with a training YAML:

~~~bash
# Kinship
bash scripts/run_nlq.sh configs/kinship/train.yaml 0

# MQuAKE-ST, single-answer
bash scripts/run_nlq.sh configs/mquake_st/train_single.yaml 0

# MQuAKE-ST, multi-answer
bash scripts/run_nlq.sh configs/mquake_st/train_multi.yaml 0

# MetaQA
bash scripts/run_nlq.sh configs/metaqa/train.yaml 0
~~~

The launcher calls <code>code/model/trainer.py</code> with the selected YAML.

The supplied training configurations use <code>load_model: False</code>. Checkpoints and logs are written below the configured <code>base_output_dir</code>, with run-specific paths assembled by the option loader.

## 5. Evaluation

Evaluation configurations use <code>load_model: True</code> and contain a <code>model_load_dir</code>. Point that field to the checkpoint you want to evaluate.

~~~bash
# Kinship
bash scripts/run_eval.sh configs/kinship/evaluate.yaml 0

# MQuAKE-ST, single-answer
bash scripts/run_eval.sh configs/mquake_st/evaluate_single.yaml 0

# MQuAKE-ST, multi-answer
bash scripts/run_eval.sh configs/mquake_st/evaluate_multi.yaml 0

# MetaQA
bash scripts/run_eval.sh configs/metaqa/evaluate.yaml 0
~~~

The evaluation launcher calls <code>code/model/evaluation.py</code>. When <code>print_paths: True</code>, human-readable trajectory logs are also written for qualitative inspection.

See [Evaluation metrics](metrics.md) for the exact ranking, path-fidelity, answer-coverage, and diagnostic metrics reported by the evaluator.


## 6. Pretrained checkpoints

Pretrained checkpoints for the blind evaluation are available from the anonymized model page:

**Checkpoints:** [Kaggle](https://www.kaggle.com/models/anonymousexpert/minerva-kgqa)

Each released setting provides **three random seeds: 0, 42, and 100**.

| Dataset / setting | Expected checkpoint prefix |
| --- | --- |
| Kinship | <code>checkpoints/kinship/qa_nhop_reason_3hop_seed&lt;seed&gt;/model/model.ckpt</code> |
| MQuAKE-ST single-answer | <code>checkpoints/mquake_st/sa_qa_nhop_reason_4hop_seed&lt;seed&gt;/model/model.ckpt</code> |
| MQuAKE-ST multi-answer | <code>checkpoints/mquake_st/ma_qa_nhop_reason_4hop_seed&lt;seed&gt;/model/model.ckpt</code> |
| MetaQA | <code>checkpoints/metaqa/qa_nhop_reason_3hop_seed&lt;seed&gt;/model/model.ckpt</code> |

### Downloading the checkpoints

Download the model archive from the Kaggle page above and extract it **at the repository root without creating an additional directory**.

The archive already contains the expected `checkpoints/` hierarchy and run-specific directories. After extraction, the repository should contain paths such as:

~~~text
checkpoints/
├── kinship/
│   └── qa_nhop_reason_3hop_seed0/
│       └── model/
│           └── model.ckpt.*
├── mquake_st/
│   ├── sa_qa_nhop_reason_4hop_seed0/
│   │   └── model/
│   │       └── model.ckpt.*
│   └── ma_qa_nhop_reason_4hop_seed0/
│       └── model/
│           └── model.ckpt.*
└── metaqa/
    └── qa_nhop_reason_3hop_seed0/
        └── model/
            └── model.ckpt.*
~~~

The evaluation configurations use <code>seed: 0</code> by default. Their <code>model_load_dir</code> values interpolate the configured seed and path length, so changing <code>seed</code> automatically changes the checkpoint location expected by the evaluator.

For example:

~~~yaml
seed: 42
path_length: 3
model_load_dir: "checkpoints/kinship/qa_nhop_reason_${path_length}hop_seed${seed}/model/model.ckpt"
~~~

To evaluate a different released seed, change the <code>seed</code> field in the evaluation YAML to `42` or `100`. The corresponding checkpoint directory must be present in the extracted archive.

With the checkpoints in place, run evaluation normally:

~~~bash
bash scripts/run_eval.sh configs/kinship/evaluate.yaml 0
~~~

## 7. Structural calibration baselines

The repository includes the non-learned structural calibration references used in the accompanying submission, including the definitions discussed in Appendix A.4. These references operate on the **actual evaluator navigation graph and action space** rather than serving as learned KGQA systems.

- **RW-Ans_MC / unbiased random walk:** samples uniform random navigation trajectories. The supplied scripts use **100 walks per question** and, by default, the three seeds **0, 42, and 100**. The terminal answer-hit rate is the Monte Carlo <code>RW-Ans_MC</code> calibration; the same sampled trajectories are also evaluated with PED, RED, F1_SG, and F1_REL when the required references are available.
- **Shortest Path Oracle:** is given the valid answer set and finds a shortest graph path from the topic entity to a valid answer, but it does **not** use the natural-language question. Its trajectory is evaluated with the same path-fidelity metrics. It is a structural reference, not a path-fidelity upper or lower bound.

Dataset wrappers are provided under <code>scripts/baselines/</code>:

| Setting | Command | Calibration references |
| --- | --- | --- |
| Kinship | <code>bash scripts/baselines/run_kinship.sh</code> | RW-Ans_MC + Shortest Path Oracle |
| MQuAKE-ST single-answer | <code>bash scripts/baselines/run_mquake_st_sa.sh</code> | RW-Ans_MC + Shortest Path Oracle |
| MQuAKE-ST multi-answer | <code>bash scripts/baselines/run_mquake_st_ma.sh</code> | RW-Ans_MC + Shortest Path Oracle |
| MetaQA | <code>bash scripts/baselines/run_metaqa.sh</code> | RW-Ans_MC |

With no seed arguments, the wrappers run <code>0 42 100</code>. To run only selected seeds, pass them explicitly:

~~~bash
bash scripts/baselines/run_kinship.sh 42
bash scripts/baselines/run_mquake_st_sa.sh 0 100
~~~

Machine-readable results are written below <code>output/&lt;dataset&gt;/baselines/</code>. In the random-walk JSON output, the paper's <code>RW-Ans_MC</code> quantity is stored under the summary key <code>RW_Ans</code>.

See [Evaluation metrics](metrics.md#15-structural-calibration-references) for the interpretation of these references and [<code>code/baselines/</code>](../code/baselines/) for the implementations.

## 8. Important configuration groups

The YAML files expose the same options as <code>code/options.py</code>. The most commonly changed groups are:

**Data and question input**

- <code>data_input_dir</code>
- <code>raw_QAData_path</code>
- <code>cached_QAMetaData_path</code>
- <code>question_tokenizer_name</code>
- <code>question_format</code>
- <code>evaluate_paraphrases</code>

**Graph navigation**

- <code>use_full_graph</code>
- <code>use_directed_graph</code>
- <code>max_num_actions</code>
- <code>path_length</code>
- <code>use_stop_signal</code>
- <code>use_restart_signal</code>

**Model**

- <code>embedding_size</code>
- <code>hidden_size</code>
- <code>use_entity_embeddings</code>
- <code>train_entity_embeddings</code>
- <code>train_relation_embeddings</code>
- <code>projection_adapter</code>
- <code>projection_layers</code>
- <code>projection_hidden</code>

**Training and decoding**

- <code>num_rollouts</code>
- <code>test_rollouts</code>
- <code>learning_rate</code>
- <code>gamma</code>
- <code>beta</code>
- <code>use_beam</code>
- <code>pool</code>

**Trajectory evaluation**

- <code>path_segment_policy</code>
- <code>print_paths</code>
- <code>print_predictions</code>

The default trajectory policy in <code>code/options.py</code> is <code>final_segment_truncate</code>: evaluation keeps the final attempt after the last RESTART and stops the evaluated path at STOP.

## 9. Running a folder of configurations

<code>scripts/bulk_nlq.sh</code> launches every YAML in a directory with a configurable maximum number of concurrent jobs:

~~~bash
bash scripts/bulk_nlq.sh path/to/config_folder 4
~~~

Use a directory containing only training configurations that you actually want to launch; dataset directories in <code>configs/</code> may contain both training and evaluation YAMLs.

## 10. Where to look next

- [Architecture](architecture.md): model and code organization.
- [Data format](data_format.md): graph and QA schemas.
- [Evaluation metrics](metrics.md): exact evaluator behavior, metric definitions, and structural calibration references.
