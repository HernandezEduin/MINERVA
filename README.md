# MINERVA: Natural-Language KG Navigation

**Blind-evaluation implementation**

This repository adapts MINERVA to **natural-language multi-hop knowledge graph question answering**. Instead of conditioning the agent on a symbolic query of the form (h, r, ?), the agent receives a **natural-language question**, a **topic entity**, and a **knowledge graph**, then navigates an explicit sequence of graph edges toward an answer.

![MINERVA KG Navigation](images/minerva_navigation.gif)

*At each hop, MINERVA uses the natural-language question together with its recurrent path state to score the executable relation–entity actions available from the current node. The selected edge moves the agent to the next entity, producing an explicit reasoning path whose terminal entity is used as the predicted answer.*

This branch is prepared for anonymous evaluation. Blind dataset and checkpoint resources are linked from the documentation; identifying project and author links are intentionally omitted.

## What is included

- Natural-language question conditioning with transformer embeddings.
- Reinforcement-learning graph navigation with beam-search evaluation.
- Single-answer and multi-answer KGQA.
- Directed and undirected graph settings.
- Optional STOP and RESTART actions.
- Explicit trajectory logging and path-fidelity evaluation.
- Dataset-specific configurations for **Kinship**, **MQuAKE-ST**, and **MetaQA**.
- Pretrained MINERVA checkpoints for three released seeds.
- Structural calibration baselines: **RW-Ans_MC** and the **Shortest Path Oracle**.
- Preprocessing and baseline scripts under <code>scripts/</code>.

## Quick start

Run commands from the repository root.

### 1. Create the environment

~~~bash
conda env create -f environment.yml
conda activate minerva
~~~

The provided environment uses Python 3.9 and includes the CUDA 11.2 / cuDNN 8.1 runtime required by TensorFlow 2.11.1.

A pip-only installation is also possible:

~~~bash
python -m pip install -r requirements.txt
~~~

If you are using a custom Conda environment instead of `environment.yml` and want GPU support, install the matching CUDA runtime:

~~~bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
~~~

### 2. Prepare a dataset

Kinship is the smallest bundled example workflow. Download the blind dataset from [Kaggle](https://www.kaggle.com/datasets/anonymousexpert/kinship), extract it under <code>raw_data/kinship_hinton/</code>, then run:

~~~bash
bash scripts/preprocessing/kinship.sh
~~~

The preprocessing scripts create the graph and vocabularies expected by MINERVA under <code>datasets/nlq/</code>. See [Getting started](docs/getting_started.md) for the complete blind dataset setup.

### 3. Train

~~~bash
bash scripts/run_nlq.sh configs/kinship/train.yaml 0
~~~

The final argument is an optional GPU ID. Omit it to run on CPU.

### 4. Evaluate

Set <code>model_load_dir</code> in the corresponding evaluation YAML to the checkpoint you want to load, then run:

~~~bash
bash scripts/run_eval.sh configs/kinship/evaluate.yaml 0
~~~

## Current experiment configurations

| Dataset | Training | Evaluation |
| --- | --- | --- |
| Kinship | <code>configs/kinship/train.yaml</code> | <code>configs/kinship/evaluate.yaml</code> |
| MQuAKE-ST single-answer | <code>configs/mquake_st/train_single.yaml</code> | <code>configs/mquake_st/evaluate_single.yaml</code> |
| MQuAKE-ST multi-answer | <code>configs/mquake_st/train_multi.yaml</code> | <code>configs/mquake_st/evaluate_multi.yaml</code> |
| MetaQA | <code>configs/metaqa/train.yaml</code> | <code>configs/metaqa/evaluate.yaml</code> |

## Documentation

| Guide | Contents |
| --- | --- |
| [Getting started](docs/getting_started.md) | Environment setup, datasets, pretrained checkpoints, training, evaluation, and baseline scripts |
| [Data format](docs/data_format.md) | Graph files, QA CSV schema, multi-answer data, paths, and custom datasets |
| [Architecture](docs/architecture.md) | How MINERVA is adapted from symbolic queries to natural-language graph navigation |
| [Evaluation metrics](docs/metrics.md) | Hits@K, MRR, F1_SG, F1_REL, PED, RED, and structural calibration references |

## Repository layout

~~~text
MINERVA/
├── code/                  # Model, environment, data loading, and preprocessing
├── configs/
│   ├── kinship/
│   ├── mquake_st/
│   └── metaqa/
├── docs/                  # User and evaluation documentation
├── images/
├── scripts/
│   ├── baselines/
│   ├── preprocessing/
│   ├── run_nlq.sh
│   └── run_eval.sh
├── environment.yml
└── requirements.txt
~~~

## Blind-review note

Citation and identifying project information are intentionally omitted from this branch during blind evaluation.

## License

This repository is released under the [Apache License 2.0](LICENSE).
