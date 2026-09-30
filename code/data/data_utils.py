"""
Data utilities for the MINERVA project.

This module provides comprehensive utilities for data loading, preprocessing, and
transformation for knowledge graph question answering tasks. It handles the complete
pipeline from raw CSV data to processed, tokenized datasets ready for training.

Key functionalities:
- JSON file loading and processing
- Text literal extraction from string representations
- QA dataset preprocessing with tokenization and entity/relation mapping
- Dataset splitting with caching for efficient re-use
- Vocabulary loading for entities and relations
- TensorFlow-based text embedding generation with attention masking

The module supports both cached and fresh data processing, with intelligent
fallback mechanisms and comprehensive metadata tracking for reproducibility.

Functions:
    load_json: Load JSON data from file paths
    extract_literals: Extract Python literals from string representations
    process_and_cache_triviaqa_data: Process and cache QA datasets with splits
    load_qa_data: Load QA data with caching support
    load_dictionary: Load entity and relation vocabularies
    ids_to_embeddings_tf: Generate embeddings from token IDs using TensorFlow models
"""

import ast
import json
import logging
import os
from datetime import datetime

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizer

from code.data.itl_typing import DFSplit
from code.data.setup import get_git_root

from typing import Any, Dict, List, Optional, Tuple, Union

QA_CACHE_SCHEMA_VERSION = 2

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file to load
        
    Returns:
        Dictionary containing the loaded JSON data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_literals(column: Union[str, pd.Series], flatten: bool = False) -> Union[pd.Series, List[str]]:
    """
    Extract Python literals from string representations in pandas columns.
    
    Safely evaluates string representations of Python literals (lists, dicts, etc.)
    using ast.literal_eval. Optionally flattens nested lists into a single flat list.
    This is commonly used for processing path data stored as string representations
    of lists in CSV files.
    
    Args:
        column: Pandas Series containing string representations of Python literals,
               or a single string representation
        flatten: If True, flattens all extracted lists into a single list.
                If False, returns a Series of individual lists
                
    Returns:
        If flatten=False: Pandas Series where each element is the evaluated literal
        If flatten=True: Single flattened list containing all elements from all lists
        
    Example:
        >>> import pandas as pd
        >>> data = pd.Series(['[1, 2, 3]', '[4, 5]', '[6]'])
        >>> result = extract_literals(data, flatten=False)
        >>> print(result.tolist())  # [[1, 2, 3], [4, 5], [6]]
        >>> 
        >>> flat_result = extract_literals(data, flatten=True)
        >>> print(flat_result)  # [1, 2, 3, 4, 5, 6]
        
    Raises:
        ValueError: If any string cannot be safely evaluated as a Python literal
        SyntaxError: If any string contains invalid Python syntax
    """
    # Convert single string input to pandas Series for uniform processing
    if isinstance(column, str):
        column = pd.Series([column])

    # Safely evaluate string representations of Python literals
    evaluated_column = column.apply(ast.literal_eval)

    # Flatten all lists into a single list if requested
    if flatten:
        flattened_result = [item for sublist in evaluated_column for item in sublist]
        return flattened_result
        
    return evaluated_column

def _is_triplet_like(value: Any) -> bool:
    """
    Check whether a value has the structure of one graph edge triplet.

    A triplet is expected to be a flat three-element sequence representing:

        [head, relation, tail]

    The outer container may be a list or tuple. Each of the three elements
    must be scalar-like with respect to the path representation; nested lists,
    tuples, or dictionaries are therefore rejected.

    This helper is used only to identify the nesting level of path annotations.
    It does not verify that the head, relation, and tail values exist in the
    knowledge graph or correspond to valid entity/relation identifiers.

    Args:
        value:
            Object to inspect.

    Returns:
        bool:
            True if ``value`` structurally resembles one
            ``[head, relation, tail]`` graph edge; otherwise False.
    """
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and not any(isinstance(x, (list, tuple, dict)) for x in value)
    )


def _is_path_like(value: Any) -> bool:
    """
    Check whether a value has the structure of one entity-level graph path.

    An entity-level path is represented as a non-empty sequence of graph-edge
    triplets:

        [
            [head_1, relation_1, tail_1],
            [head_2, relation_2, tail_2],
            ...
        ]

    Every element must satisfy ``_is_triplet_like``. This function checks only
    the structural nesting of the annotation. It does not verify path
    continuity, relation validity, endpoint correctness, or membership in the
    active knowledge graph.

    Args:
        value:
            Object to inspect.

    Returns:
        bool:
            True if ``value`` structurally resembles one entity-level path;
            otherwise False.
    """
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(_is_triplet_like(edge) for edge in value)
    )


def _flatten_reference_paths(value: Any) -> List[List[List[Any]]]:
    """
    Flatten grouped multi-annotated reference paths into individual paths.

    Multi-annotated reference datasets may organize ``Multi-Paths`` using
    additional grouping levels, for example:

        relation-chain group
            -> entity-level reference path
                -> [head, relation, tail] triplets

    Conceptually, an annotation may therefore look like:

        [
            [path_A1, path_A2],
            [path_B1],
        ]

    where the outer groups correspond to different annotated relation chains.

    MINERVA's best-reference entity-level fidelity metrics require the complete
    collection of valid reference paths rather than the grouping itself. This
    helper recursively traverses the nested annotation and returns:

        [
            path_A1,
            path_A2,
            path_B1,
        ]

    Each detected path is preserved intact; only the higher-level grouping is
    removed. Triplets within a path are not flattened or reordered.

    The function performs structural normalization only. It does not validate
    path continuity, remove duplicate paths, check endpoints, or confirm that
    the annotated edges occur in the active knowledge graph.

    Args:
        value:
            Parsed ``Multi-Paths`` annotation. The value may contain one or
            more nested list/tuple levels above the entity-level paths.

    Returns:
        List[List[List[Any]]]:
            Flat list of entity-level reference paths. Each path is a list of
            ``[head, relation, tail]`` triplets.

    Note:
        The association between a reference path and its original
        relation-chain group is intentionally discarded. Relation-chain
        annotations are retained separately through ``Multi-Paths-Key``.
    """
    paths: List[List[List[Any]]] = []

    def visit(obj: Any) -> None:
        if _is_path_like(obj):
            paths.append([list(edge) for edge in obj])
            return
        if isinstance(obj, (list, tuple)):
            for child in obj:
                visit(child)

    visit(value)
    return paths


def _normalize_multi_path_keys(value: Any) -> List[str]:
    """
    Normalize multi-annotated relation-chain keys to a list of strings.

    ``Multi-Paths-Key`` may represent either a single annotated relation chain:

        "relation_a->relation_b"

    or multiple annotated chains:

        [
            "relation_a->relation_b",
            "relation_c->relation_d",
        ]

    This helper normalizes both cases to:

        List[str]

    so downstream preprocessing can uniformly map every relation chain to its
    sequence of relation IDs.

    The relation-chain strings themselves are not parsed or validated here.
    Splitting on ``->`` and mapping relation names to vocabulary IDs is handled
    later in preprocessing.

    Args:
        value:
            A single relation-chain string or a list/tuple of relation-chain
            values.

    Returns:
        List[str]:
            One string for each annotated relation chain.

    Raises:
        TypeError:
            If ``value`` is neither a string nor a list/tuple.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(key) for key in value]
    raise TypeError(f"Unsupported Multi-Paths-Key value: {type(value)!r}")


def paraphrase2question(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand grouped question paraphrases into individual evaluation instances.

    Some QA datasets store multiple surface realizations of the same underlying
    question in a ``Question-Paraphrased`` column. This function converts that
    grouped representation into one DataFrame row per surface form.

    For each input row:

      1. ``Question-Paraphrased`` is expected to contain a list of question
         strings.
      2. Each listed paraphrase becomes its own output row.
      3. All other columns are copied unchanged, preserving information such
         as source entity, answers, path annotations, family identifiers,
         hop count, and split assignment.
      4. The ``Question`` column is replaced by the selected paraphrase for
         that output row.

    For example:

        Question = "Who is X?"
        Question-Paraphrased = [
            "Who is X?",
            "What person is X?",
            "Identify X."
        ]

    becomes three rows sharing the same structural QA annotations but having
    different ``Question`` values.

    If ``Question-Paraphrased`` is missing or invalid for an individual row,
    that row falls back to a one-element list containing its original
    ``Question`` value. The input DataFrame is not modified in place.

    Args:
        df:
            QA DataFrame containing ``Question`` and
            ``Question-Paraphrased`` columns.

    Returns:
        pd.DataFrame:
            A new DataFrame containing one row per paraphrased question
            instance. ``Question-Paraphrased`` is scalar after the explode
            operation, while ``Question`` contains the corresponding surface
            form.

    Raises:
        ValueError:
            If the DataFrame does not contain a
            ``Question-Paraphrased`` column.

    Note:
        The original canonical question is included as an output instance only
        if it is already present in ``Question-Paraphrased`` or if the
        paraphrase field is empty/invalid and the fallback is used. This
        function does not automatically prepend the canonical question to a
        valid paraphrase list.
    """

    paraphrase_col = "Question-Paraphrased"

    if paraphrase_col not in df.columns:
        raise ValueError(
            f"DataFrame does not contain the expected paraphrase column "
            f"'{paraphrase_col}'"
        )

    df_out = df.copy()

    # Replace empty / invalid paraphrase lists with [original question]
    mask = ~df_out[paraphrase_col].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df_out.loc[mask, paraphrase_col] = df_out.loc[mask, "Question"].apply(lambda x: [x])

    # Expand rows
    df_out = df_out.explode(paraphrase_col, ignore_index=True)

    # Replace Question with paraphrase
    df_out["Question"] = df_out[paraphrase_col]

    return df_out

def process_and_cache_triviaqa_data(
    raw_QAData_path: str,
    cached_toked_qatriples_metadata_path: str,
    question_tokenizer: PreTrainedTokenizer,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    seed: Optional[int] = None,
    override_split: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[DFSplit, Dict[str, Any]]:
    """
    Process and cache question-answer dataset with entity/relation mapping.
    
    Loads raw QA data from CSV, tokenizes questions, maps entities and relations
    to their integer IDs, creates train/dev/test splits, and caches the processed
    data for future use. Supports both automatic splitting and label-guided splitting.
    
    The function expects CSV data with specific column structure:
    - Question: Natural language questions
    - Source-Entity: Starting entity for reasoning
    - Answer-Entity: Target answer entity
    - Paths: (Optional) Reasoning paths as string representations of lists
    - Hops: (Optional) Number of reasoning hops
    - SplitLabel: (Optional) Predefined split labels ('train', 'dev', 'test')
    
    Args:
        raw_QAData_path: Path to the raw CSV file containing QA data
        cached_toked_qatriples_metadata_path: Path where processed metadata will be saved
        question_tokenizer: HuggingFace tokenizer for question text processing
        entity2id: Mapping from entity names to integer IDs
        relation2id: Mapping from relation names to integer IDs
        seed: Optional seed for random number generation
        override_split: If True, use SplitLabel column for splitting when available
        logger: Optional logger for progress tracking and warnings
        
    Returns:
        Tuple containing:
            - DFSplit: Object with train/dev/test DataFrames
            - Dict: Metadata including tokenizer info, column mappings, and file paths
            
    Raises:
        AssertionError: If CSV file has fewer than 3 columns
        ValueError: If git root cannot be determined
        RuntimeError: If data loading fails or DataFrames are invalid
        KeyError: If required entities/relations are missing from vocabularies
        
    Note:
        - Questions are tokenized without special tokens ([CLS], [SEP])
        - Entity and relation names are mapped to integer IDs
        - Paths are converted from string representations to lists of [head, rel, tail] triples
        - Automatic splitting uses 80/10/10 train/dev/test if no SplitLabel column
        - Small test sets (<100 samples) are used as dev sets with 50/50 dev/test split
    """

    # Load and validate CSV data
    csv_df = pd.read_csv(raw_QAData_path)
    assert len(csv_df.columns) > 2, \
        "CSV file must have at least 3 columns (Question, Source-Entity, Answer-Entity)"
    
    is_multi_answer = bool(csv_df["Answer-Entity"].apply(lambda x: x.startswith('[') and x.endswith(']')).all())

    # Extract required columns
    question_number = csv_df["Question-Number"]
    questions = csv_df["Question"]
    source_ent = csv_df["Source-Entity"] 
    answer_ent = csv_df["Answer-Entity"] if not is_multi_answer else extract_literals(
        csv_df["Answer-Entity"], flatten=False
    )
    source_label = csv_df["Source"]
    answer_label = csv_df["Answer"] if not is_multi_answer else extract_literals(
        csv_df["Answer"], flatten=False
    )
    
    # Extract optional columns. Multi-annotated reference datasets use Multi-Paths / 
    # Multi-Paths-Key fields; single-reference datasets continue to use Paths / Path-Key.
    questions_paraphrased = extract_literals(csv_df["Question-Paraphrased"]) if 'Question-Paraphrased' in csv_df.columns else None
    questions_disambiguated = csv_df["Question-Disambiguated"] if 'Question-Disambiguated' in csv_df.columns else None

    raw_paths_column = (
        "Multi-Paths" if "Multi-Paths" in csv_df.columns
        else "Paths" if "Paths" in csv_df.columns
        else None
    )
    paths_are_multi_reference = raw_paths_column == "Multi-Paths"
    if raw_paths_column == "Multi-Paths":
        paths = extract_literals(csv_df[raw_paths_column]).map(_flatten_reference_paths)
    elif raw_paths_column == "Paths":
        paths = extract_literals(csv_df[raw_paths_column])
    else:
        paths = None

    raw_paths_label_column = (
        "Multi-Paths-Label" if "Multi-Paths-Label" in csv_df.columns
        else "Paths-Label" if "Paths-Label" in csv_df.columns
        else None
    )
    paths_label = (
        csv_df[raw_paths_label_column].rename("Paths-Label")
        if raw_paths_label_column is not None else None
    )

    raw_path_keys_column = (
        "Multi-Paths-Key" if "Multi-Paths-Key" in csv_df.columns
        else "Path-Key" if "Path-Key" in csv_df.columns
        else None
    )
    path_keys_are_multi_reference = raw_path_keys_column == "Multi-Paths-Key"
    if raw_path_keys_column == "Multi-Paths-Key":
        path_keys = extract_literals(csv_df[raw_path_keys_column]).map(_normalize_multi_path_keys)
    elif raw_path_keys_column == "Path-Key":
        path_keys = csv_df[raw_path_keys_column]
    else:
        path_keys = None

    question_family_id = csv_df["Question-Family-ID"] if "Question-Family-ID" in csv_df.columns else None
    question_family_size = csv_df["Question-Family-Size"] if "Question-Family-Size" in csv_df.columns else None

    # Graph-expanded answers are retained as auxiliary metadata for multi-reference datasets, but
    # Answer-Entity remains the training/evaluation target by default.
    graph_answer_ent = (
        extract_literals(csv_df["Graph-Answer-Entity"], flatten=False)
        if "Graph-Answer-Entity" in csv_df.columns else None
    )
    graph_answer_label = (
        extract_literals(csv_df["Graph-Answer"], flatten=False)
        if "Graph-Answer" in csv_df.columns else None
    )

    path_count = csv_df["Path-Count"] if "Path-Count" in csv_df.columns else None
    graph_path_count = csv_df["Graph-Path-Count"] if "Graph-Path-Count" in csv_df.columns else None
    split_label = csv_df["SplitLabel"] if 'SplitLabel' in csv_df.columns else None
    hops = csv_df["Hops"] if 'Hops' in csv_df.columns else None

    # Ensure output directory exists
    dir_name = os.path.dirname(cached_toked_qatriples_metadata_path)
    os.makedirs(dir_name, exist_ok=True)

    # Tokenize questions (without special tokens for later processing)
    tokenized_questions = questions.map(
        lambda x: question_tokenizer.encode(x, add_special_tokens=False)
    )
    if questions_paraphrased is not None:
        # paraphrased Questions are List[str], so we tokenize each paraphrase and keep as list of token lists
        tokenized_questions_paraphrased = questions_paraphrased.map(
            lambda paraphrase_list: [question_tokenizer.encode(q, add_special_tokens=False) for q in paraphrase_list]
        )

    if questions_disambiguated is not None:
        tokenized_questions_disambiguated = questions_disambiguated.map(
            lambda x: question_tokenizer.encode(x, add_special_tokens=False)
        )

    # Map entities and relations to integer IDs
    mapped_source_ent = source_ent.map(lambda ent: entity2id[ent])
    mapped_answer_ent = answer_ent.map(lambda ent: entity2id[ent]) if not is_multi_answer else answer_ent.map(
        lambda ans_list: [entity2id[ans] for ans in ans_list]
    )
    if graph_answer_ent is not None:
        mapped_graph_answer_ent = graph_answer_ent.map(
            lambda ans_list: [entity2id[ans] for ans in ans_list]
        ).rename("Graph-Answer-Entity")

    if paths is not None:
        if paths_are_multi_reference:
            mapped_paths = paths.map(
                lambda reference_paths: [
                    [
                        [entity2id[head], relation2id[rel], entity2id[tail]]
                        for head, rel, tail in path
                    ]
                    for path in reference_paths
                ]
            )
        else:
            mapped_paths = paths.map(
                lambda path: [
                    [entity2id[head], relation2id[rel], entity2id[tail]]
                    for head, rel, tail in path
                ]
            )
        mapped_paths = mapped_paths.rename("Paths")

    if path_keys is not None:
        if path_keys_are_multi_reference:
            mapped_path_keys = path_keys.map(
                lambda keys: [
                    [relation2id[rel] for rel in key.split("->")]
                    for key in keys
                ]
            )
        else:
            mapped_path_keys = path_keys.map(
                lambda keys: [relation2id[rel] for rel in keys.split("->")]
            )
        mapped_path_keys = mapped_path_keys.rename("Path-Key")

    # Generate unique timestamp for file naming
    timestamp = str(int(datetime.now().timestamp()))
    cached_split_locations: Dict[str, str] = {
        name: cached_toked_qatriples_metadata_path.replace(".json", "") + 
              f"_Split-{name}_date-{timestamp}.parquet"
        for name in ["train", "dev", "test"]
    }

    # Get repository root for relative path generation
    repo_root = get_git_root()
    if repo_root is None:
        raise ValueError("Cannot determine git root path. Ensure you're in a git repository.")

    # Convert to relative paths
    cached_split_locations = {
        key: val.replace(repo_root + "/", "") 
        for key, val in cached_split_locations.items()
    }

    # Combine all processed data into final DataFrame
    data_columns = [question_number, tokenized_questions, mapped_source_ent, mapped_answer_ent, source_label, answer_label]
    if question_family_id is not None:
        data_columns.append(question_family_id)
    if question_family_size is not None:
        data_columns.append(question_family_size)
    if questions_paraphrased is not None:
        data_columns.append(tokenized_questions_paraphrased)
    if questions_disambiguated is not None:
        data_columns.append(tokenized_questions_disambiguated)
    if graph_answer_ent is not None:
        data_columns.append(mapped_graph_answer_ent)
    if graph_answer_label is not None:
        data_columns.append(graph_answer_label)
    if paths is not None:
        data_columns.append(mapped_paths)
    if paths_label is not None:
        data_columns.append(paths_label)
    if path_keys is not None:
        data_columns.append(mapped_path_keys)
    if path_count is not None:
        data_columns.append(path_count)
    if graph_path_count is not None:
        data_columns.append(graph_path_count)
    if hops is not None:
        data_columns.append(hops)
    if split_label is not None:
        data_columns.append(split_label)
        
    new_df = pd.concat(data_columns, axis=1)
    new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle data with fixed seed

    # Create train/dev/test splits
    dev_splitted = False
    if (override_split and 'SplitLabel' in new_df.columns and 
        new_df['SplitLabel'].notna().any() and not new_df['SplitLabel'].eq('').all()):
        # Use predefined split labels
        train_df = new_df[new_df['SplitLabel'] == 'train'].reset_index(drop=True)

        if 'test' in new_df["SplitLabel"].values and 'dev' in new_df["SplitLabel"].values:
            test_df = new_df[new_df['SplitLabel'] == 'test'].reset_index(drop=True)
            dev_df = new_df[new_df['SplitLabel'] == 'dev'].reset_index(drop=True)
            dev_splitted = True
            if logger: logger.info("Using SplitLabel column for dev/test splitting")
        else:
            test_df = new_df[new_df['SplitLabel'] != 'train'].reset_index(drop=True)

        if logger: 
            logger.info("Using SplitLabel column for data splitting")
    else:
        # Automatic splitting
        train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=seed)

    # Handle dev set creation
    if len(test_df) < 100:
        # Use entire test set as dev set for small datasets
        dev_df = test_df
        if logger: 
            logger.warning("Test set too small (<100 samples), using as dev set")
    elif not dev_splitted:
        # Automatic splitting
        dev_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed)
        if logger: logger.info("Automatically splitting test set into dev/test")

    # Validate DataFrame creation
    if not all(isinstance(df, pd.DataFrame) for df in [train_df, dev_df, test_df]):
        raise RuntimeError("Data loading failed - invalid DataFrames created")

    # Save processed data to parquet files
    for name, df in {"train": train_df, "dev": dev_df, "test": test_df}.items():
        df.to_parquet(cached_split_locations[name], index=False)

    # Create metadata for reproducibility and documentation
    metadata: Dict[str, Any] = {
        "question_tokenizer": question_tokenizer.name_or_path,
        "question_number_column": "Question-Number",
        "question_column": "Question",
        "question_paraphrased_column": "Question-Paraphrased" if questions_paraphrased is not None else None,
        "question_disambiguated_column": "Question-Disambiguated" if questions_disambiguated is not None else None,
        "question_family_id_column": "Question-Family-ID" if question_family_id is not None else None,
        "question_family_size_column": "Question-Family-Size" if question_family_size is not None else None,
        "source_label_column": "Source",
        "source_entities_column": "Source-Entity",
        "answer_label_column": "Answer",
        "answer_entity_column": "Answer-Entity",
        "graph_answer_label_column": "Graph-Answer" if graph_answer_label is not None else None,
        "graph_answer_entity_column": "Graph-Answer-Entity" if graph_answer_ent is not None else None,
        "paths_column": "Paths" if paths is not None else None,
        "paths_source_column": raw_paths_column,
        "paths_are_multi_reference": paths_are_multi_reference,
        "paths_label_column": "Paths-Label" if paths_label is not None else None,
        "paths_label_source_column": raw_paths_label_column,
        "path_keys_column": "Path-Key" if path_keys is not None else None,
        "path_keys_source_column": raw_path_keys_column,
        "path_keys_are_multi_reference": path_keys_are_multi_reference,
        "path_count_column": "Path-Count" if path_count is not None else None,
        "graph_path_count_column": "Graph-Path-Count" if graph_path_count is not None else None,
        "hops_column": "Hops" if hops is not None else None,
        "splitLabel_column": "SplitLabel" if split_label is not None else None,
        "is_multi_answer": is_multi_answer,
        "cache_schema_version": QA_CACHE_SCHEMA_VERSION,
        "qa_schema": (
            "multi_reference"
            if paths_are_multi_reference or path_keys_are_multi_reference
            else "single_reference"
        ),
        "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "saved_paths": cached_split_locations,
        "timestamp": timestamp,
    }

    # Save metadata to JSON file
    with open(cached_toked_qatriples_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return DFSplit(train=train_df, dev=dev_df, test=test_df), metadata


def load_qa_data(
    cached_metadata_path: str,
    raw_QAData_path: str,
    question_tokenizer_name: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    force_recompute: bool = False,
    override_split: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load QA dataset with intelligent caching and fallback processing.
    
    Attempts to load preprocessed data from cache first. If cache is missing
    or force_recompute is True, processes raw data and creates new cache.
    This function provides a unified interface for QA data loading with
    automatic preprocessing and caching management.
    
    Args:
        cached_metadata_path: Path to cached metadata JSON file
        raw_QAData_path: Path to raw CSV data file (used if cache missing)
        question_tokenizer_name: HuggingFace tokenizer identifier
        entity2id: Entity name to integer ID mapping
        relation2id: Relation name to integer ID mapping
        seed: Optional seed for random number generation
        logger: Optional logger for progress tracking
        force_recompute: If True, ignore cache and reprocess data
        override_split: If True, use SplitLabel column when available
        
    Returns:
        Tuple containing:
            - train_df: Training DataFrame
            - dev_df: Development DataFrame  
            - test_df: Test DataFrame
            - train_metadata: Metadata dictionary with processing information
            
    Raises:
        FileNotFoundError: If raw data file doesn't exist when cache is missing
        json.JSONDecodeError: If cached metadata is corrupted
        KeyError: If required entities/relations missing from vocabularies
        
    Note:
        - Cached data is loaded from parquet files for efficiency
        - Metadata tracks tokenizer, column mappings, and file locations
        - Automatic fallback to raw processing if cache is invalid
    """

    if os.path.exists(cached_metadata_path) and not force_recompute:
        # Load from cache
        print(f"\033[93mFound cached QA data at {cached_metadata_path}, loading instead of "
              f"processing {raw_QAData_path}\033[0m")
              
        # Load metadata and extract file paths
        with open(cached_metadata_path, 'r') as f:
            train_metadata = json.load(f)
        saved_paths: Dict[str, str] = train_metadata["saved_paths"]

        # Load preprocessed DataFrames
        train_df = pd.read_parquet(saved_paths["train"])
        dev_df = pd.read_parquet(saved_paths["dev"])
        test_df = pd.read_parquet(saved_paths["test"])

        print(f"Loaded cached data from \033[93m\033[4m{cached_metadata_path}\033[0m")
        
    else:
        # Process raw data
        print(f"\033[93mCache not found or force_recompute=True. "
              f"Processing raw data from {raw_QAData_path}\033[0m")
              
        # Load tokenizer and process data
        question_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name)
        df_split, train_metadata = process_and_cache_triviaqa_data(
            raw_QAData_path,
            cached_metadata_path,
            question_tokenizer,
            entity2id,
            relation2id,
            seed=seed,
            override_split=override_split,
            logger=logger,
        )
        
        # Extract DataFrames from split object
        train_df, dev_df, test_df = df_split.train, df_split.dev, df_split.test
        print(f"Processing complete. Data saved to:\n"
              f"\033[93m\033[4m{train_metadata['saved_paths']}\033[0m")

    return train_df, dev_df, test_df, train_metadata

def load_dictionary(data_dir: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str], Dict[str, str], Dict[str, str]]:
    """
    Load entity and relation vocabularies with optional title mappings from JSON files.
    
    Loads the entity and relation vocabularies from the standard MINERVA
    directory structure and creates both forward (name->ID) and reverse 
    (ID->name) mappings for efficient lookup operations. Additionally loads
    optional entity and relation title mappings if available.
    
    Args:
        data_dir: Directory containing the vocab subdirectory with JSON files
                 Required files: vocab/entity_vocab.json, vocab/relation_vocab.json
                 Optional files: vocab/entity_title.json, vocab/relation_title.json
                 
    Returns:
        Tuple containing:
            - ent2id: Entity name to integer ID mapping
            - rel2id: Relation name to integer ID mapping
            - id2ent: Integer ID to entity name mapping
            - id2rel: Integer ID to relation name mapping
            - ent2name: Entity ID to human-readable title mapping (empty dict if not available)
            - rel2name: Relation ID to human-readable title mapping (empty dict if not available)
            
    Raises:
        FileNotFoundError: If required vocabulary JSON files don't exist
        json.JSONDecodeError: If vocabulary files contain invalid JSON
        
    Example:
        >>> ent2id, rel2id, id2ent, id2rel, ent2name, rel2name = load_dictionary("datasets/kinship/")
        >>> print(ent2id["person_1"])  # 42
        >>> print(id2ent[42])          # "person_1"
        >>> print(ent2name.get("person_1", "No title"))  # Human-readable title or fallback
    """
    # Load vocabulary mappings from JSON files
    ent2id = load_json(os.path.join(data_dir, "vocab/entity_vocab.json"))
    rel2id = load_json(os.path.join(data_dir, "vocab/relation_vocab.json"))

    # Load Entity/Relation to Title mappings if available
    if os.path.exists(os.path.join(data_dir, "vocab/entity_title.json")):
        ent2name = load_json(os.path.join(data_dir, "vocab/entity_title.json"))
    else:
        ent2name = {}

    if os.path.exists(os.path.join(data_dir, "vocab/relation_title.json")):
        rel2name = load_json(os.path.join(data_dir, "vocab/relation_title.json"))
    else:
        rel2name = {}

    # Create reverse mappings for efficient ID->name lookup
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    return ent2id, rel2id, id2ent, id2rel, ent2name, rel2name
