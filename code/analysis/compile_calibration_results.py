#!/usr/bin/env python3

"""
Extract structural calibration-reference results for Table I.

Expected directory structure, e.g.:

output/
├── kinship/
│   └── baselines/
│       ├── random_walk_stats_kinship_seed0.json
│       ├── random_walk_stats_kinship_seed42.json
│       ├── random_walk_stats_kinship_seed100.json
│       ├── shortcut_oracle_stats_kinship_seed0.json
│       ├── shortcut_oracle_stats_kinship_seed42.json
│       └── shortcut_oracle_stats_kinship_seed100.json
│
└── mquake_st/
    └── baselines/
        └── ...

The script extracts:

    Ans. Rate
    RED
    PED
    F1_Rel
    F1_SG

Results are grouped by dataset, answer type, navigation-graph scope,
reference scope, baseline, and seed. This keeps the filtered/source graph
intervention orthogonal to released/graph-expanded evaluation references.
Legacy result files that predate reference_scope are interpreted as using
released references.

For the unbiased random walk, three seed values are emitted through
\\numstats{seed0,seed42,seed100}.

For the answer-oracle shortest-path reference, the script checks whether
the stored results are identical across seeds. If they are, it emits a
single deterministic value. If they differ, it prints a warning and
falls back to \\numstats{...}.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("output")

SEED_ORDER = [0, 42, 100]

METRIC_KEYS = {
    "RED": "average_RED",
    "PED": "average_PED",
    "F1_Rel": "average_F1_Rel",
    "F1_SG": "average_F1_SG",
}

DISPLAY_ORDER = [
    ("KINSHIP", "Single"),
    ("MQuAKE-ST", "Single"),
    ("MQuAKE-ST", "Multi"),
    ("MetaQA", "Multi"),
    ("PQ", "Multi"),
    ("PQL", "Multi"),
    ("WC2014", "Multi"),
]

GRAPH_SCOPE_ORDER = {
    "filtered": 0,
    "source": 1,
    "full": 2,
    "unknown": 99,
}

REFERENCE_SCOPE_ORDER = {
    "released": 0,
    "graph": 1,
    "unknown": 99,
}


# -------------------------------------------------------------------------
# JSON loading
# -------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(path: Path) -> Dict[str, Any]:
    data = load_json(path)

    if "summary" not in data:
        raise KeyError(
            f"{path} does not contain a top-level 'summary' object."
        )

    summary = data["summary"]

    if not isinstance(summary, dict):
        raise TypeError(
            f"{path}: 'summary' is not a JSON object."
        )

    return summary


# -------------------------------------------------------------------------
# Metadata inference
# -------------------------------------------------------------------------

def extract_seed(path: Path, summary: Dict[str, Any]) -> int:
    """
    Prefer the seed recorded inside the JSON.
    Fall back to parsing seedXXX from the filename.
    """

    if summary.get("seed") is not None:
        return int(summary["seed"])

    match = re.search(r"seed[_-]?(\d+)", path.stem, re.IGNORECASE)

    if match:
        return int(match.group(1))

    raise ValueError(
        f"Could not determine seed for {path}"
    )


def infer_dataset(path: Path, summary: Dict[str, Any]) -> str:
    """
    Determine whether this result belongs to KINSHIP or MQuAKE-ST.
    """

    # First try metadata.
    for key in ("dataset", "dataset_name", "benchmark"):
        value = summary.get(key)

        if value is None:
            continue

        value = str(value).lower()

        if "kinship" in value:
            return "KINSHIP"

        if "mquake" in value:
            return "MQuAKE-ST"

        if "metaqa" in value:
            return "MetaQA"

        if "pql" in value:
            return "PQL"

        if "pq" in value:
            return "PQ"

        if "wc2014" in value:
            return "WC2014"

    # Fall back to the path.
    path_text = str(path).lower()

    if "kinship" in path_text:
        return "KINSHIP"

    if "mquake" in path_text:
        return "MQuAKE-ST"

    if "metaqa" in path_text:
        return "MetaQA"

    if "pql" in path_text:
        return "PQL"

    if "pq" in path_text:
        return "PQ"

    if "wc2014" in path_text:
        return "WC2014"

    return "UNKNOWN"


def infer_answer_type(
    path: Path,
    summary: Dict[str, Any],
    dataset: str,
) -> str:
    """
    Infer Single vs Multi answer.

    KINSHIP is treated as Single.

    For MQuAKE-ST, metadata is preferred. Filename/path conventions
    such as sa_..., ma_..., single, and multi are also supported.
    """

    if dataset == "KINSHIP":
        return "Single"

    if dataset == "PQL" or dataset == "PQ":
        return "Single"

    if dataset == "WC2014":
        return "Single"

    if dataset == "MetaQA":
        return "Multi"

    # -------------------------------------------------------------
    # Explicit boolean fields
    # -------------------------------------------------------------

    for key in (
        "multi_answer",
        "is_multi_answer",
        "multianswer",
    ):
        if key in summary:
            value = summary[key]

            if isinstance(value, bool):
                return "Multi" if value else "Single"

    # -------------------------------------------------------------
    # String-valued metadata
    # -------------------------------------------------------------

    for key in (
        "answer_type",
        "answer_setting",
        "setting",
        "qa_setting",
        "dataset_setting",
        "mode",
        "split_type",
    ):
        value = summary.get(key)

        if value is None:
            continue

        value = str(value).lower()

        if "multi" in value:
            return "Multi"

        if "single" in value:
            return "Single"

        if value == "ma":
            return "Multi"

        if value == "sa":
            return "Single"

    # -------------------------------------------------------------
    # Filename / path fallback
    # -------------------------------------------------------------

    name = str(path).lower()

    multi_patterns = [
        r"(?:^|[/_\-])ma(?:[/_\-.]|$)",
        r"(?:^|[/_\-])multi(?:[/_\-.]|$)",
        r"multi[_\-]?answer",
    ]

    single_patterns = [
        r"(?:^|[/_\-])sa(?:[/_\-.]|$)",
        r"(?:^|[/_\-])single(?:[/_\-.]|$)",
        r"single[_\-]?answer",
    ]

    for pattern in multi_patterns:
        if re.search(pattern, name):
            return "Multi"

    for pattern in single_patterns:
        if re.search(pattern, name):
            return "Single"

    return "UNKNOWN"


def normalize_baseline(
    path: Path,
    summary: Dict[str, Any],
) -> str:
    """
    Normalize baseline names to the two labels used in the paper.
    """

    baseline = str(summary.get("baseline", "")).lower()
    name = path.name.lower()

    combined = baseline + " " + name

    if "random" in combined and "walk" in combined:
        return "Random Walk"

    if (
        "oracle" in combined
        or "shortcut_oracle" in combined
        or "shortest_path" in combined
        or "shortest-path" in combined
    ):
        return "Answer-Oracle Shortest Path"

    return summary.get("baseline", path.stem)


def infer_graph_scope(
    path: Path,
    summary: Dict[str, Any],
    dataset: str,
) -> str:
    """Infer the navigation graph used by a baseline result.

    For PQ/PQL, ``full_graph.txt`` corresponds to the source graph and
    ``graph.txt`` to the filtered graph. For other datasets we retain the
    generic ``full`` / ``filtered`` terminology. Filename tokens are used as
    fallbacks for older result files.
    """

    path_text = str(path).lower()
    graph_value = str(summary.get("graph", "")).lower()

    # Prefer explicit source/filtered tokens in the filename/path when present.
    if re.search(r"(?:^|[/_\-])source(?:[/_\-.]|$)", path_text):
        return "source"
    if re.search(r"(?:^|[/_\-])filtered(?:[/_\-.]|$)", path_text):
        return "filtered"

    if "full_graph" in graph_value or graph_value.endswith("full_graph.txt"):
        return "source" if dataset in {"PQ", "PQL"} else "full"

    if graph_value.endswith("graph.txt") or graph_value == "graph.txt":
        return "filtered"

    return "unknown"


def infer_reference_scope(
    path: Path,
    summary: Dict[str, Any],
) -> str:
    """Infer released vs graph-expanded reference annotations.

    New baseline outputs store ``reference_scope`` directly. Older outputs did
    not have this field and are therefore treated as ``released``, matching the
    pre-reference-scope evaluator behavior. Filename tokens are used as a
    fallback for newly named result files whose summary metadata is incomplete.
    """

    value = summary.get("reference_scope")
    if value is not None:
        value = str(value).lower()
        if value in {"released", "graph"}:
            return value
        raise ValueError(
            f"Unsupported reference_scope={value!r} in {path}"
        )

    name = path.stem.lower()
    if re.search(r"(?:^|[_\-])graph(?:[_\-]|$)", name):
        return "graph"
    if re.search(r"(?:^|[_\-])released(?:[_\-]|$)", name):
        return "released"

    # Backward compatibility: before reference_scope existed, all baseline
    # answer/path evaluation used the released benchmark annotations.
    return "released"


# -------------------------------------------------------------------------
# Metric extraction
# -------------------------------------------------------------------------

def get_answer_rate(summary: Dict[str, Any]) -> Optional[float]:
    """
    Extract the answer-reaching rate.

    The evaluator currently appears to store this as RW_Ans.

    For random walk:
        fraction of sampled walks that terminate at a valid answer.

    For the oracle:
        fraction of questions for which the oracle reaches a valid answer.

    Multiple possible field names are supported so the script survives
    small evaluator naming changes.
    """

    candidate_keys = (
        "answer_rate",
        "average_answer_rate",
        "Ans_Rate",
        "ans_rate",
        "RW_Ans",
        "rw_ans",
        "answer_success_rate",
        "answer_probability",
        "answer_prob",
    )

    for key in candidate_keys:
        if key in summary and summary[key] is not None:
            return float(summary[key])

    return None


def extract_metrics(
    path: Path,
    summary: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}

    result["AnswerRate"] = get_answer_rate(summary)

    for display_name, json_key in METRIC_KEYS.items():
        value = summary.get(json_key)

        result[display_name] = (
            float(value)
            if value is not None
            else None
        )

    return result


# -------------------------------------------------------------------------
# Collection
# -------------------------------------------------------------------------

def collect_baselines(output_dir: Path):
    """
    Returns:

        results[(dataset, answer_type, graph_scope, reference_scope, baseline)][seed] = metrics
    """

    results = defaultdict(dict)

    files = sorted(output_dir.rglob("baselines/*.json"))

    if not files:
        raise FileNotFoundError(
            f"No JSON files were found under:\n"
            f"    {output_dir.resolve()}/**/baselines/*.json"
        )

    print(f"Found {len(files)} baseline JSON files.\n")

    for path in files:
        try:
            summary = load_summary(path)

            baseline = normalize_baseline(path, summary)

            # Ignore unrelated JSONs inside baselines/.
            if baseline not in (
                "Random Walk",
                "Answer-Oracle Shortest Path",
            ):
                print(
                    f"[SKIP] Unknown baseline: {path}"
                )
                continue

            seed = extract_seed(path, summary)

            dataset = infer_dataset(path, summary)

            answer_type = infer_answer_type(
                path,
                summary,
                dataset,
            )

            graph_scope = infer_graph_scope(
                path,
                summary,
                dataset,
            )

            reference_scope = infer_reference_scope(
                path,
                summary,
            )

            metrics = extract_metrics(path, summary)

            key = (
                dataset,
                answer_type,
                graph_scope,
                reference_scope,
                baseline,
            )

            if seed in results[key]:
                raise ValueError(
                    f"Duplicate seed {seed} for:\n"
                    f"    {key}\n"
                    f"Existing: {results[key][seed]['path']}\n"
                    f"New:      {path}"
                )

            results[key][seed] = {
                "path": path,
                "graph_scope": graph_scope,
                "reference_scope": reference_scope,
                **metrics,
            }

        except Exception as exc:
            print(f"[ERROR] {path}")
            print(f"        {exc}\n")

    return results


# -------------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------------

def format_number(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "--"

    return f"{value:.{digits}f}"


def latex_numstats(values) -> str:
    """
    Produce:

        \\numstats{seed0,seed42,seed100}

    The user's existing LaTeX macro can then compute/display mean ± std.
    """

    if any(value is None for value in values):
        return "--"

    return (
        r"\numstats{"
        + ",".join(f"{value:.6f}" for value in values)
        + "}"
    )


def values_identical(values, tolerance: float = 1e-12) -> bool:
    """
    Check deterministic results across multiple stored seed files.
    """

    valid = [v for v in values if v is not None]

    if len(valid) <= 1:
        return True

    reference = valid[0]

    return all(
        abs(value - reference) <= tolerance
        for value in valid[1:]
    )


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

def validate_seed_set(seed_results, expected=SEED_ORDER):
    available = sorted(seed_results.keys())

    missing = [
        seed
        for seed in expected
        if seed not in seed_results
    ]

    extra = [
        seed
        for seed in available
        if seed not in expected
    ]

    return available, missing, extra


def print_raw_group(
    dataset,
    answer_type,
    graph_scope,
    reference_scope,
    baseline,
    seed_results,
):
    print("=" * 100)
    print(
        f"{dataset} | {answer_type} | graph={graph_scope} | "
        f"references={reference_scope} | {baseline}"
    )
    print("=" * 100)

    available, missing, extra = validate_seed_set(
        seed_results
    )

    print(f"Available seeds: {available}")

    if missing:
        print(f"WARNING: missing expected seeds: {missing}")

    if extra:
        print(f"NOTE: extra seeds present: {extra}")

    print()

    header = (
        f"{'Seed':>6}  "
        f"{'Ans.Rate':>12}  "
        f"{'RED':>12}  "
        f"{'PED':>12}  "
        f"{'F1_Rel':>12}  "
        f"{'F1_SG':>12}"
    )

    print(header)
    print("-" * len(header))

    for seed in available:
        x = seed_results[seed]

        print(
            f"{seed:>6}  "
            f"{format_number(x['AnswerRate']):>12}  "
            f"{format_number(x['RED']):>12}  "
            f"{format_number(x['PED']):>12}  "
            f"{format_number(x['F1_Rel']):>12}  "
            f"{format_number(x['F1_SG']):>12}"
        )

    print()


# -------------------------------------------------------------------------
# LaTeX generation
# -------------------------------------------------------------------------

def random_walk_latex_row(seed_results) -> str:
    """
    Random walk is stochastic, so emit the three trials through numstats.
    """

    _, missing, _ = validate_seed_set(seed_results)

    if missing:
        return (
            "% ERROR: Random walk is missing seeds "
            + str(missing)
        )

    def values(field):
        return [
            seed_results[seed][field]
            for seed in SEED_ORDER
        ]

    answer_rate = latex_numstats(
        values("AnswerRate")
    )

    red = latex_numstats(
        values("RED")
    )

    ped = latex_numstats(
        values("PED")
    )

    f1_rel = latex_numstats(
        values("F1_Rel")
    )

    f1_sg = latex_numstats(
        values("F1_SG")
    )

    return (
        r"\textsc{Unbiased Random Walk}"
        "\n"
        r"& -- & -- "
        f"& {answer_rate} "
        f"& {red} "
        f"& {ped} "
        f"& {f1_rel} "
        f"& {f1_sg} "
        r"\\"
    )


def oracle_latex_row(seed_results) -> str:
    """
    The shortest-path oracle should be deterministic.

    If all stored seeds match:
        emit one scalar value.

    If they unexpectedly differ:
        warn in a LaTeX comment and emit numstats instead.
    """

    available, missing, _ = validate_seed_set(
        seed_results
    )

    if not available:
        return "% ERROR: No oracle results found."

    fields = [
        "AnswerRate",
        "RED",
        "PED",
        "F1_Rel",
        "F1_SG",
    ]

    deterministic = True

    for field in fields:
        vals = [
            seed_results[seed][field]
            for seed in available
        ]

        if not values_identical(vals):
            deterministic = False
            break

    if deterministic:
        reference = seed_results[available[0]]

        return (
            r"\textsc{Answer-Oracle Shortest Path}"
            "\n"
            r"& -- & -- "
            f"& {format_number(reference['AnswerRate'], 4)} "
            f"& {format_number(reference['RED'], 4)} "
            f"& {format_number(reference['PED'], 4)} "
            f"& {format_number(reference['F1_Rel'], 4)} "
            f"& {format_number(reference['F1_SG'], 4)} "
            r"\\"
        )

    # -------------------------------------------------------------
    # Unexpected case: oracle differs by seed.
    # -------------------------------------------------------------

    print(
        "WARNING: Answer-oracle shortest-path values differ "
        "across seeds."
    )
    print(
        "Check shortest-path tie-breaking / evaluator "
        "determinism."
    )

    if missing:
        return (
            "% ERROR: Non-deterministic oracle and missing "
            f"seeds {missing}"
        )

    def values(field):
        return [
            seed_results[seed][field]
            for seed in SEED_ORDER
        ]

    return (
        "% WARNING: Oracle unexpectedly differs across seeds.\n"
        r"\textsc{Answer-Oracle Shortest Path}"
        "\n"
        r"& -- & -- "
        f"& {latex_numstats(values('AnswerRate'))} "
        f"& {latex_numstats(values('RED'))} "
        f"& {latex_numstats(values('PED'))} "
        f"& {latex_numstats(values('F1_Rel'))} "
        f"& {latex_numstats(values('F1_SG'))} "
        r"\\"
    )



# -------------------------------------------------------------------------
# Optional compact summary
# -------------------------------------------------------------------------

def _result_sort_key(key):
    dataset, answer_type, graph_scope, reference_scope, baseline = key

    try:
        dataset_order = DISPLAY_ORDER.index((dataset, answer_type))
    except ValueError:
        dataset_order = len(DISPLAY_ORDER)

    return (
        dataset_order,
        dataset,
        answer_type,
        GRAPH_SCOPE_ORDER.get(graph_scope, 99),
        graph_scope,
        REFERENCE_SCOPE_ORDER.get(reference_scope, 99),
        reference_scope,
        baseline,
    )


def print_compact_summary(results):
    print("\n")
    print("#" * 100)
    print("RAW EXTRACTED RESULTS")
    print("#" * 100)
    print()

    for key in sorted(results, key=_result_sort_key):
        dataset, answer_type, graph_scope, reference_scope, baseline = key

        print_raw_group(
            dataset,
            answer_type,
            graph_scope,
            reference_scope,
            baseline,
            results[key],
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract unbiased-random-walk and "
            "answer-oracle shortest-path calibration "
            "results for Table I."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root output directory. "
            "Default: ./output"
        ),
    )

    parser.add_argument(
        "--graph-scope",
        choices=["all", "filtered", "source", "full", "unknown"],
        default="all",
        help=(
            "Optionally restrict results to one navigation-graph scope. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--reference-scope",
        choices=["all", "released", "graph", "unknown"],
        default="all",
        help=(
            "Optionally restrict results to released or graph-expanded "
            "references. Default: all."
        ),
    )

    args = parser.parse_args()

    results = collect_baselines(
        args.output_dir
    )

    if args.graph_scope != "all" or args.reference_scope != "all":
        results = {
            key: value
            for key, value in results.items()
            if (
                (args.graph_scope == "all" or key[2] == args.graph_scope)
                and (
                    args.reference_scope == "all"
                    or key[3] == args.reference_scope
                )
            )
        }

    if not results:
        raise RuntimeError(
            "No recognized calibration baseline results "
            "were found."
        )

    print_compact_summary(results)


if __name__ == "__main__":
    main()