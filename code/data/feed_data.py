import os
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from code.data.data_utils import load_dictionary, load_qa_data, paraphrase2question
from code.data.embedding_server import EmbeddingServer

class QuestionBatcher:
    """
    A data batcher for Natural Language Question answering over Knowledge Graphs.
    
    This class handles the loading, preprocessing, and batching of question-answer pairs
    for training and evaluation. It provides functionality to:
    
    1. Load and preprocess QA datasets with entity/relation vocabularies
    2. Generate embeddings for questions using transformer models via EmbeddingServer
    3. Batch data for training (random sampling) and testing (sequential)
    4. Translate between entity/relation IDs and human-readable names
    
    The batcher supports both training and evaluation modes, automatically managing
    data splits and providing appropriate batching strategies for each phase.
    
    Attributes:
        batch_size (int): Number of samples per batch
        mode (str): Current mode ('train', 'dev', or 'test')
        entity_vocab (Dict[str, int]): Entity name to ID mapping
        relation_vocab (Dict[str, int]): Relation name to ID mapping
        rev_entity_vocab (Dict[int, str]): Entity ID to name mapping
        rev_relation_vocab (Dict[int, str]): Relation ID to name mapping
        ent2name (Dict[str, str]): Entity name to human-readable title mapping
        rel2name (Dict[str, str]): Relation name to human-readable title mapping
        train_df (pd.DataFrame): Training dataset
        dev_df (pd.DataFrame): Development/validation dataset
        test_df (pd.DataFrame): Test dataset
        eval_df (pd.DataFrame): Current evaluation dataset based on mode
        embedding_server (EmbeddingServer): Server for generating question embeddings
        question_tokenizer (AutoTokenizer): Tokenizer for question text
        pad_id (int): Padding token ID
        cls_id (int): CLS token ID for BERT-style models
        sep_id (int): SEP token ID for BERT-style models
    """
    def __init__(
        self, 
        input_dir: str,
        batch_size: int,
        test_batch_size: int, 
        question_tokenizer_name: str,
        question_format: str,
        reference_scope: str,
        cached_QAMetaData_path: str,
        raw_QAData_path: str,
        mode: str = "train",
        use_weighted_hop_sampling: bool = False,
        evaluate_paraphrases: bool = False,
        seed: Optional[int] = None,
        force_data_prepro: bool = False,
        embedding_server: Optional[EmbeddingServer] = None,
    ) -> None:
        """
        Initialize the QuestionBatcher with data loading and preprocessing.
        
        Args:
            input_dir: Directory containing entity and relation vocabularies
            batch_size: Number of samples per batch
            test_batch_size: Number of samples per batch during evaluation
            question_tokenizer_name: HuggingFace model name for question tokenization
            question_format: Format of the question input ('full_text', 'relation_only', 'graph_only')
            reference_scope: Scope of the reference annotations ('released', 'graph')
            cached_QAMetaData_path: Path to cached preprocessed QA metadata JSON
            raw_QAData_path: Path to raw QA dataset CSV file
            mode: Initial mode ('train', 'dev', or 'test')
            use_weighted_hop_sampling: Whether to use weighted hop-based sampling for training batches
            evaluate_paraphrases: Whether to evaluate on paraphrased questions instead of original text
            seed: Optional seed for random number generation
            force_data_prepro: Whether to force reprocessing of cached data
            embedding_server: Optional pre-initialized embedding server
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.batch_size: int = batch_size
        self.test_batch_size: int = test_batch_size

        # Load knowledge graph vocabularies
        ent2id, rel2id, id2ent, id2rel, ent2name, rel2name = load_dictionary(input_dir)
        self.entity_vocab: Dict[str, int] = ent2id
        self.relation_vocab: Dict[str, int] = rel2id
        self.rev_entity_vocab: Dict[int, str] = id2ent
        self.rev_relation_vocab: Dict[int, str] = id2rel
        self.ent2name: Dict[str, str] = ent2name
        self.rel2name: Dict[str, str] = rel2name

        # Load and preprocess QA datasets
        self.train_df: pd.DataFrame
        self.dev_df: pd.DataFrame
        self.test_df: pd.DataFrame
        self.train_metadata: Dict
        self.question_format: str = question_format
        self.reference_scope: str = reference_scope
        self.evaluate_paraphrases: bool = evaluate_paraphrases
        self.train_df, self.dev_df, self.test_df, self.train_metadata = load_qa_data(
            cached_metadata_path=cached_QAMetaData_path,
            raw_QAData_path=raw_QAData_path,
            question_tokenizer_name=question_tokenizer_name,
            entity2id=ent2id,
            relation2id=rel2id,
            seed=seed,
            logger=None,
            force_recompute=force_data_prepro,
        )

        self.question_column = self.train_metadata.get("question_column")
        self.question_paraphrased_column = self.train_metadata.get("question_paraphrased_column")
        self.question_number_column = self.train_metadata.get("question_number_column")
        self.question_family_id_column = self.train_metadata.get("question_family_id_column")
        self.question_family_size_column = self.train_metadata.get("question_family_size_column")
        self.source_label_column = self.train_metadata.get("source_label_column")
        self.source_entity_column = self.train_metadata.get("source_entities_column")
        self.hops_column = self.train_metadata.get("hops_column")
        self.path_keys_column = self.train_metadata.get("path_keys_column")

        if question_format == 'paraphrased' and self.question_paraphrased_column is None:
            raise ValueError("Paraphrased questions are requested but not available in the dataset.")
        if evaluate_paraphrases and self.question_paraphrased_column is None:
            raise ValueError("Paraphrased questions are requested but not available in the dataset.")
        if reference_scope not in ['released', 'graph']:
            raise ValueError(f"Invalid reference_scope: {reference_scope}. Must be 'released' or 'graph', got {reference_scope}.")
        
        if self.reference_scope == "released":
            self.answer_entity_column = "Answer-Entity"
            self.answer_label_column = "Answer"
            self.paths_column = ("Paths" if self.train_metadata.get("paths_column") is not None else None)
            self.paths_label_column = (
                "Paths-Label"
                if self.train_metadata.get("paths_label_column") is not None
                else None
            )
            self.multi_answers = self.train_metadata.get("is_multi_answer", False)

        else:
            self.answer_entity_column = self.train_metadata.get("graph_answer_entity_column")
            self.answer_label_column = self.train_metadata.get("graph_answer_label_column")
            self.paths_column = self.train_metadata.get("graph_multi_paths_column")
            self.paths_label_column = self.train_metadata.get("graph_multi_paths_label_column")
            self.multi_answers = self.train_metadata.get("is_multi_answer_graph", False)

            # check that all are valid if reference_scope is 'graph'
            if (self.answer_entity_column is None or self.answer_label_column is None or self.paths_column is None or self.paths_label_column is None):
                raise ValueError(
                    "reference_scope='graph' was requested, but the dataset "
                    "does not provide Graph-Answer-Entity, Graph-Answer, "
                    "Graph-Multi-Paths, or Graph-Multi-Paths-Label."
                )

        self.path_exists: bool = self.paths_column is not None
        self.path_key_exists: bool = True if self.train_metadata.get("path_keys_column") is not None else False

        if question_format == 'relation_only' and not (self.path_exists or self.path_key_exists):
            raise ValueError("Relation-only format is requested but no path/path-key information is available in the dataset.")

        if evaluate_paraphrases:
            # For evaluation with paraphrased questions, we will use the paraphrased questions as the main 'Question' column for consistency in batching and embedding generation.
            self.dev_df = paraphrase2question(self.dev_df)
            self.test_df = paraphrase2question(self.test_df)

        # Set initial mode and evaluation dataset
        self.mode: str = mode
        self.eval_df: pd.DataFrame
        self.set_mode(mode)

        # Initialize embedding server for question processing
        self.embedding_server: EmbeddingServer = embedding_server or EmbeddingServer(question_tokenizer_name)

        # Initialize tokenizer and special token IDs
        self.question_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name)
        self.pad_id: int = self.question_tokenizer.pad_token_id or 0
        self.cls_id: int = self.question_tokenizer.cls_token_id or 101
        self.sep_id: int = self.question_tokenizer.sep_token_id or 102
        
        # Cache embedding dimensions for easy access
        self._embedding_dim: Optional[int] = None
        self.calculate_embedding_dimensions()  # Pre-fetch dimensions

        # Precompute training sampling probabilities from the Hops column
        self.use_weighted_hop_sampling: bool = use_weighted_hop_sampling
        self.train_sampling_probs: Optional[np.ndarray] = self._build_train_sampling_probs() if self.use_weighted_hop_sampling else None

    def _build_train_sampling_probs(self) -> Optional[np.ndarray]:
        """
        Build per-row sampling probabilities for the training set using the Hops column.

        Strategy:
            inverse-frequency weighting by hop class, so each hop count contributes
            roughly equally in expectation.

        Returns:
            np.ndarray of shape [len(train_df)] summing to 1, or None if Hops is missing.
        """
        if self.hops_column is None:
            return None

        hops = pd.to_numeric(self.train_df[self.hops_column], errors="coerce")
        if hops.isna().any():
            bad_rows = hops[hops.isna()].index.tolist()[:10]
            raise ValueError(
                f"Found non-numeric or missing values in train_df['{self.hops_column}'] at rows like: {bad_rows}"
            )

        hop_counts = hops.value_counts()
        row_weights = hops.map(lambda h: 1.0 / hop_counts.loc[h]).to_numpy(dtype=np.float64)

        total = row_weights.sum()
        if total <= 0:
            raise ValueError("Training sampling weights sum to zero.")

        return row_weights / total

    def calculate_embedding_dimensions(self) -> Tuple[int, int]:
        """
        Get the embedding dimensions by testing the embedding server.

        Returns:
            Tuple of (batch_size, embedding_dim) where batch_size is 1 for the test
        """
        # Test with a simple question to get dimensions
        test_question = [[0, 0, 0]]
        test_embedding = self.embedding_server.embed(
            token_id_batches=test_question,
            pad_id=self.pad_id,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
        )
        # Cache the embedding dimension for future use
        if self._embedding_dim is None:
            self._embedding_dim = test_embedding.shape[1]
        return test_embedding.shape  # Returns (1, embedding_dim)

    def get_embedding_dim(self) -> int:
        """
        Get just the embedding dimension (not the batch size).

        Returns:
            The embedding dimension as an integer
        """
        if self._embedding_dim is None:
            self.calculate_embedding_dimensions()
        return self._embedding_dim


    def set_mode(self, mode: str) -> None:
        """
        Change the batcher mode and set the corresponding evaluation dataset.
        
        Args:
            mode: New mode ('train', 'dev', or 'test')
            
        Raises:
            AssertionError: If mode is not one of the valid options
        """
        assert mode in ['train', 'dev', 'test'], "Mode must be one of ['train', 'dev', 'test']"
        self.mode = mode
        if mode == 'train':
            self.eval_df = self.train_df
        elif mode == 'dev':
            self.eval_df = self.dev_df
        else:
            self.eval_df = self.test_df

    def set_batch_size(self, batch_size: int) -> None:
        """
        Update the batch size for subsequent batching operations.
        
        Args:
            batch_size: New batch size
        """
        self.batch_size = batch_size

    def set_test_batch_size(self, test_batch_size: int) -> None:
        """
        Update the test batch size for subsequent evaluation batching operations.
        
        Args:
            test_batch_size: New test batch size
        """
        self.test_batch_size = test_batch_size

    def get_mode(self) -> str:
        """
        Get the current batcher mode.
        
        Returns:
            Current mode string ('train', 'dev', or 'test')
        """
        return self.mode

    def get_question_num(self) -> int:
        """
        Get the total number of questions in the current evaluation dataset.
        
        Returns:
            Number of questions in current mode's dataset
        """
        return len(self.eval_df)
    
    def has_multi_answers(self) -> bool:
        """
        Check if the current dataset contains questions with multiple answers.
        
        Returns:
            True if the dataset is multi-answer, False otherwise
        """
        return self.multi_answers
    
    def has_paths(self) -> bool:
        """
        Check if the current dataset contains path information.
        
        Returns:
            True if path information is available, False otherwise
        """
        return self.path_exists
    
    def has_path_keys(self) -> bool:
        """
        Check if the current dataset contains path key information.
        
        Returns:
            True if path key information is available, False otherwise
        """
        return self.path_key_exists

    def _relation_only_questions(
        self,
        paths: Optional[List],
        path_keys: Optional[List],
    ) -> List[List[int]]:
        """Build relation-only question tokens for legacy and multi-reference schemas.

        Single-reference datasets provide one Path-Key relation sequence per question.
        Multi-reference datasets may provide multiple valid relation chains. Relation-only mode
        is inherently single-input, so the first released chain is selected
        deterministically; full-text training/evaluation is unaffected.

        Args:
            paths: List of paths for each question, where each path is a list of triples
            path_keys: List of path keys for each question, where each path key is a list of relations
        Returns:
            List of tokenized relation-only questions, where each question is a list of token IDs
        """
        relations_only: List[str] = []

        row_count = len(path_keys) if path_keys is not None else len(paths)
        for i in range(row_count):
            rel_seq = None

            if path_keys is not None:
                value = path_keys[i]
                if isinstance(value, list) and value:
                    rel_seq = value[0] if isinstance(value[0], list) else value

            if rel_seq is None and paths is not None:
                value = paths[i]
                if isinstance(value, list) and value:
                    if (
                        isinstance(value[0], list)
                        and len(value[0]) == 3
                        and not isinstance(value[0][0], list)
                    ):
                        path = value
                    else:
                        path = value[0]
                    rel_seq = [triple[1] for triple in path]

            if rel_seq is None:
                raise ValueError("Unable to construct relation-only input for a QA row.")

            relations_only.append(
                " ".join([f"[{rel}]" for rel in self.translate_relations(rel_seq)])
            )

        return self.tokenize_questions(relations_only)

    def yield_next_batch_train(self) -> Generator[Tuple[List[str], Union[np.ndarray, List[List[int]]], np.ndarray, np.ndarray], None, None]:
        """
        Generate infinite training batches with random sampling.
        
        Yields batches by randomly sampling questions from the training dataset.
        Each batch contains question texts, embeddings, source entities, and answer entities.
        
        Yields:
            Tuple containing:
                - questions (List[str]): Raw question text strings
                - question_embeddings (np.ndarray): Question embeddings [batch_size, embedding_dim]
                - source_ent (np.ndarray): Source entity IDs [batch_size]
                - answers (Union[np.ndarray, List[List[int]]]): Answer(s) entity IDs [batch_size]
                
        Raises:
            AssertionError: If batcher is not in training mode
        """
        assert self.mode == 'train', "Batcher is not in training mode"
        while True:
            if self.train_sampling_probs is not None:
                # Weighted sampling by Hops for training
                batch_idx = np.random.choice(len(self.eval_df), size=self.batch_size, replace=True, p=self.train_sampling_probs)
            else:
                # Uniformly sample batch indices
                batch_idx = np.random.randint(0, len(self.eval_df), size=self.batch_size)
            
            batch = self.eval_df.iloc[batch_idx]
            
            source_ent: np.ndarray = batch[self.source_entity_column].to_numpy(dtype=int)
            answers: Union[np.ndarray, List[List[int]]] = batch[self.answer_entity_column].to_numpy(dtype=int) if not self.multi_answers else batch[self.answer_entity_column].tolist()
            paths: List[List[List[str, str, str]]] = batch[self.paths_column].tolist() if self.path_exists else None
            path_keys: List[List[str]] = batch[self.path_keys_column].tolist() if self.path_key_exists else None
            hops: List[int] = batch[self.hops_column].tolist()
            ques_ids: List[int] = batch[self.question_number_column].tolist()

            # Extract questions based on the specified format
            if self.question_format == 'full_text':
                questions: List[List[int]] = batch[self.question_column].tolist() # already tokenized
            elif self.question_format == 'paraphrased':
                questions: List[List[int]] = batch[self.question_paraphrased_column] # pandas dataframe where each entry is a list of list of token ids, randomly select one paraphrase for each question
                questions = [q[np.random.randint(0, len(q))] if isinstance(q, list) and len(q) > 0 else [] for q in questions] # handle empty paraphrase lists
            elif self.question_format == 'relation_only':
                questions: List[List[int]] = self._relation_only_questions(paths, path_keys)
            else:  # 'graph_only'
                # add empty questions and 0 vector embeddings
                questions: List[List[int]] = self.tokenize_questions([""] * len(batch)) 
                question_embeddings = np.zeros((len(questions), self.get_embedding_dim()), dtype=np.float32)

                yield questions, question_embeddings, source_ent, answers, paths, path_keys, hops, ques_ids
                continue # skip embedding generation

            # Generate embeddings via the embedding server
            question_embeddings: np.ndarray = self.embedding_server.embed(
                token_id_batches=questions,
                pad_id=self.pad_id,
                cls_id=self.cls_id,
                sep_id=self.sep_id,
                max_length=128,
            )
            yield questions, question_embeddings, source_ent, answers, paths, path_keys, hops, ques_ids

    def yield_next_batch_test(self) -> Generator[Tuple[List[str], Union[np.ndarray, List[List[int]]], np.ndarray, np.ndarray], None, None]:
        """
        Generate sequential test/evaluation batches without repetition.
        
        Iterates through the evaluation dataset sequentially, yielding batches until
        all questions have been processed. Handles partial batches at the end.
        
        Yields:
            Tuple containing:
                - questions (List[str]): Raw question text strings
                - question_embeddings (np.ndarray): Question embeddings [test_batch_size, embedding_dim]
                - source_ent (np.ndarray): Source entity IDs [test_batch_size]
                - answers (Union[np.ndarray, List[List[int]]]): Answer entity IDs [test_batch_size]
        """
        remaining_questions: int = len(self.eval_df)
        current_idx: int = 0
        
        while True:
            if remaining_questions == 0:
                return
            
            # Determine batch indices for current iteration
            if remaining_questions - self.test_batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx + self.test_batch_size)
                current_idx += self.test_batch_size
                remaining_questions -= self.test_batch_size
            else:
                # Handle final partial batch
                batch_idx = np.arange(current_idx, len(self.eval_df))
                remaining_questions = 0

            # Extract batch data
            batch = self.eval_df.iloc[batch_idx]
            source_ent: np.ndarray = batch[self.source_entity_column].to_numpy(dtype=int)
            answers: Union[np.ndarray, List[List[int]]] = batch[self.answer_entity_column].to_numpy(dtype=int) if not self.multi_answers else batch[self.answer_entity_column].tolist()
            paths: List[List[List[str, str, str]]] = batch[self.paths_column].tolist() if self.path_exists else None
            path_keys: List[List[str]] = batch[self.path_keys_column].tolist() if self.path_key_exists else None
            hops: List[int] = batch[self.hops_column].tolist()
            ques_ids: List[int] = batch[self.question_number_column].tolist()
            family_ids = (
                batch[self.question_family_id_column].tolist()
                if self.question_family_id_column is not None else None
            )
            family_sizes = (
                batch[self.question_family_size_column].tolist()
                if self.question_family_size_column is not None else None
            )

            # Extract questions based on the specified format
            if (self.question_format == 'full_text') or (self.question_format == 'paraphrased'):
                # NOTE: For evaluate_paraphrases, each paraphrased questions are copied to the 'Question' column so each are evaluated independently.
                questions: List[List[int]] = batch[self.question_column].tolist() # already tokenized
            elif self.question_format == 'relation_only':
                questions: List[List[int]] = self._relation_only_questions(paths, path_keys)
            else:  # 'graph_only'
                # add empty questions and 0 vector embeddings
                questions: List[List[int]] = self.tokenize_questions([""] * len(batch)) 
                question_embeddings = np.zeros((len(questions), self.get_embedding_dim()), dtype=np.float32)

                yield questions, question_embeddings, source_ent, answers, paths, path_keys, hops, ques_ids, family_ids, family_sizes
                continue # skip embedding generation

            # Generate embeddings via the embedding server
            question_embeddings: np.ndarray = self.embedding_server.embed(
                token_id_batches=questions,
                pad_id=self.pad_id,
                cls_id=self.cls_id,
                sep_id=self.sep_id,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers, paths, path_keys, hops, ques_ids, family_ids, family_sizes

    def translate_entities(self, entity_ids: np.ndarray, dynamic_list: bool = False) -> List[str]:
        """
        Convert entity IDs to their human-readable names.
        
        Args:
            entity_ids: Array of entity IDs to translate
            
        Returns:
            List of entity names corresponding to the input IDs
        """
        if dynamic_list: # assume List[List[int]] (multi-answers)
            result = []
            for sublist in entity_ids:
                if self.ent2name:
                    result.append([self.ent2name.get(self.rev_entity_vocab.get(eid, "Unknown"), "Unknown") for eid in sublist])
                else:
                    result.append([self.rev_entity_vocab.get(eid, "Unknown") for eid in sublist])
            return result
        else: # assuming np.ndarray
            if self.ent2name:
                return [self.ent2name.get(self.rev_entity_vocab.get(eid, "Unknown"), "Unknown") for eid in entity_ids]
            else:
                return [self.rev_entity_vocab.get(eid, "Unknown") for eid in entity_ids]

    def translate_relations(self, relation_ids: np.ndarray) -> List[str]:
        """
        Convert relation IDs to their human-readable names.
        
        Args:
            relation_ids: Array of relation IDs to translate
            
        Returns:
            List of relation names corresponding to the input IDs
        """
        if self.rel2name:
            return [self.rel2name.get(self.rev_relation_vocab.get(rid, "Unknown"), "Unknown") for rid in relation_ids]
        else:
            return [self.rev_relation_vocab.get(rid, "Unknown") for rid in relation_ids]

    def translate_questions(self, questions: Union[List[List[int]], List[str]]) -> List[str]:
        """
        Convert tokenized questions back to human-readable text.
        
        Args:
            questions: List of tokenized questions (as token ID lists) or text strings
            
        Returns:
            List of decoded question text strings
        """
        if isinstance(questions[0], str):
            return questions  # Already decoded
        return [self.question_tokenizer.decode(question) for question in questions]
    
    def tokenize_questions(self, questions: List[str]) -> List[List[int]]:
        """
        Tokenize raw question text into token ID sequences.
        
        Args:
            questions: List of raw question text strings
            
        Returns:
            List of token ID sequences corresponding to the input questions
        """
        if isinstance(questions[0], list):
            return questions  # Already tokenized
        return [self.question_tokenizer.encode(q, add_special_tokens=False) for q in questions]
    